"""FastAPI 基本の実装

GUI の代わりに Web から学習や対戦を実行するためのサーバ"""

from __future__ import annotations

import argparse
import asyncio
from typing import List

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel

from .config import load_config, Config
from .mcts import MCTS
from .network import (
    OtrioNet,
    load_model,
    policy_value,
    create_optimizer,
)
from .training import ReplayBuffer, self_play, self_play_parallel, train_step
from .otrio import GameState, Move, Player


app = FastAPI()
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="ui")

cfg: Config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state: GameState = GameState(num_players=cfg.num_players)
model: OtrioNet = OtrioNet(
    num_players=cfg.num_players,
    num_blocks=cfg.num_blocks,
    channels=cfg.channels,
)
model.to(device)
optimizer = create_optimizer(model, lr=cfg.learning_rate)
buffer = ReplayBuffer(cfg.buffer_capacity, device=device)

mcts: MCTS = MCTS(lambda s: policy_value(model, s), num_simulations=cfg.num_simulations)

train_clients: List[WebSocket] = []
game_clients: List[WebSocket] = []
training_task: asyncio.Task | None = None
train_iteration: int = 0
train_loss: float = 0.0
training_error: str | None = None


def reset(model_path: str | None = None) -> None:
    """\ゲーム状態とモデルを初期化"""

    global state, model, mcts
    state = GameState(num_players=cfg.num_players)
    if model_path:
        loaded = load_model(
            model_path,
            num_players=cfg.num_players,
            num_blocks=cfg.num_blocks,
            channels=cfg.channels,
        )
        if loaded is not None:
            model = loaded
            model.to(device)
    mcts = MCTS(lambda s: policy_value(model, s), num_simulations=cfg.num_simulations)


def _board_to_list(state: GameState) -> list:
    return [[[p.value for p in row] for row in size] for size in state.board]


async def board_state() -> dict:
    return {
        "board": _board_to_list(state),
        "current": state.current_player.name,
        "winner": state.winner.name if state.winner else None,
        "draw": state.draw,
    }


async def broadcast_train(message: dict) -> None:
    living: List[WebSocket] = []
    for ws in train_clients:
        try:
            await ws.send_json(message)
            living.append(ws)
        except WebSocketDisconnect:
            pass
    train_clients[:] = living


async def broadcast_game() -> None:
    data = await board_state()
    living: List[WebSocket] = []
    for ws in game_clients:
        try:
            await ws.send_json(data)
            living.append(ws)
        except WebSocketDisconnect:
            pass
    game_clients[:] = living


@app.post("/start")
async def start_game(data: dict = Body(default={})):  # pragma: no cover - simple wrapper
    path = data.get("model") if data else None
    reset(path)
    await broadcast_game()
    return {"status": "ok"}


class MoveData(BaseModel):
    row: int
    col: int
    size: int


@app.post("/move")
async def player_move(move: MoveData):
    global state
    if not (0 <= move.row < 3 and 0 <= move.col < 3 and 0 <= move.size < 3):
        return JSONResponse({"error": "out_of_range"}, status_code=400)
    if state.board[move.size][move.row][move.col] != Player.NONE:
        return JSONResponse({"error": "occupied"}, status_code=400)

    act = Move(move.row, move.col, move.size, state.current_player)
    state.apply_move(act)

    ai_move = None
    if not state.winner and not state.draw:
        ai_move, _, _ = await asyncio.to_thread(mcts.run, state)
        state.apply_move(ai_move)

    res = await board_state()
    if ai_move:
        res["ai"] = {"row": ai_move.row, "col": ai_move.col, "size": ai_move.size}
    await broadcast_game()
    return res


@app.get("/state")
async def get_state():
    return await board_state()


@app.get("/training_status")
async def training_status():
    status = "running" if training_task and not training_task.done() else "idle"
    return {
        "status": status,
        "iteration": train_iteration,
        "loss": train_loss,
        "error": training_error,
    }


@app.post("/update_training_settings")
async def update_training_settings(settings: dict = Body(...)):
    global cfg, optimizer
    for k, v in settings.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
            if k == "learning_rate":
                for group in optimizer.param_groups:
                    group["lr"] = v
    return {"status": "updated"}


@app.post("/model_save")
async def model_save(data: dict = Body(default={"path": "model.pt"})):
    from .training import save_training_state

    path = data.get("path", "model.pt")
    await asyncio.to_thread(save_training_state, model, optimizer, buffer, path)
    return {"status": "saved", "path": path}


@app.post("/model_load")
async def model_load(data: dict = Body(default={"path": "model.pt"})):
    from .training import load_training_state

    path = data.get("path", "model.pt")
    global model, optimizer, buffer, mcts
    model, optimizer, buffer = await asyncio.to_thread(
        load_training_state,
        path,
        num_players=cfg.num_players,
        learning_rate=cfg.learning_rate,
        buffer_capacity=cfg.buffer_capacity,
        num_blocks=cfg.num_blocks,
        channels=cfg.channels,
        device=device,
    )
    mcts = MCTS(lambda s: policy_value(model, s), num_simulations=cfg.num_simulations)
    return {"status": "loaded", "path": path}


@app.post("/stop")
async def stop_training():
    global training_task
    if training_task:
        training_task.cancel()
    training_task = None
    return {"status": "stopped"}


class TrainRequest(BaseModel):
    iterations: int = 1


async def train_loop(iterations: int) -> None:
    global train_iteration, train_loss, training_error
    training_error = None
    try:
        for i in range(iterations):
            if cfg.parallel_games > 1:
                data = await asyncio.to_thread(
                    self_play_parallel,
                    model,
                    num_games=cfg.parallel_games,
                    num_simulations=cfg.num_simulations,
                    num_players=cfg.num_players,
                    max_moves=cfg.max_moves,
                    resign_threshold=cfg.resign_threshold,
                )
            else:
                data = await asyncio.to_thread(
                    self_play,
                    model,
                    num_simulations=cfg.num_simulations,
                    num_players=cfg.num_players,
                    max_moves=cfg.max_moves,
                    resign_threshold=cfg.resign_threshold,
                )
            buffer.add(data)
            loss = await asyncio.to_thread(
                train_step,
                model,
                optimizer,
                buffer,
                cfg.batch_size,
                cfg.value_loss_weight,
            )
            train_iteration = i + 1
            train_loss = loss
            await broadcast_train({"iteration": train_iteration, "loss": loss})
    except Exception as e:  # pragma: no cover - runtime safety
        training_error = str(e)
    
    


@app.post("/train")
async def start_train(req: TrainRequest):
    global training_task
    if training_task and not training_task.done():
        return {"status": "running"}
    training_task = asyncio.create_task(train_loop(req.iterations))
    return {"status": "started"}


@app.websocket("/ws/train")
async def ws_train(ws: WebSocket):
    await ws.accept()
    train_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        train_clients.remove(ws)


@app.websocket("/ws/game")
async def ws_game(ws: WebSocket):
    await ws.accept()
    game_clients.append(ws)
    await ws.send_json(await board_state())
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        game_clients.remove(ws)


def main() -> None:  # pragma: no cover - CLI helper
    parser = argparse.ArgumentParser(description="OtrioAI Web UI Server")
    parser.add_argument("--model", type=str, default=None, help="読み込むモデル")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    reset(args.model)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()

