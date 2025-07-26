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
frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
if frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="ui")
env_dir = Path(__file__).resolve().parents[2] / "env"

cfg: Config = load_config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OtrioApp:
    """グローバル状態をまとめた管理クラス"""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state: GameState = GameState(num_players=self.cfg.num_players)
        self.model: OtrioNet = OtrioNet(
            num_players=self.cfg.num_players,
            num_blocks=self.cfg.num_blocks,
            channels=self.cfg.channels,
        )
        self.model.to(self.device)
        self.optimizer = create_optimizer(self.model, lr=self.cfg.learning_rate)
        self.buffer = ReplayBuffer(self.cfg.buffer_capacity, device=self.device)
        self.mcts: MCTS = MCTS(
            lambda s: policy_value(self.model, s),
            num_simulations=self.cfg.num_simulations,
        )
        self.train_clients: List[WebSocket] = []
        self.game_clients: List[WebSocket] = []
        self.log_clients: List[WebSocket] = []
        self.training_task: asyncio.Task | None = None
        self.train_iteration: int = 0
        self.train_loss: float = 0.0
        self.training_error: str | None = None
        self.stop_training_flag: bool = False

    def reset(self, model_path: str | None = None) -> None:
        self.state = GameState(num_players=self.cfg.num_players)
        if model_path:
            loaded = load_model(
                model_path,
                num_players=self.cfg.num_players,
                num_blocks=self.cfg.num_blocks,
                channels=self.cfg.channels,
            )
            if loaded is not None:
                self.model = loaded
                self.model.to(self.device)
        self.mcts = MCTS(
            lambda s: policy_value(self.model, s),
            num_simulations=self.cfg.num_simulations,
        )

    def new_model(self) -> None:
        self.model = OtrioNet(
            num_players=self.cfg.num_players,
            num_blocks=self.cfg.num_blocks,
            channels=self.cfg.channels,
        )
        self.model.to(self.device)
        self.optimizer = create_optimizer(self.model, lr=self.cfg.learning_rate)
        self.buffer = ReplayBuffer(self.cfg.buffer_capacity, device=self.device)
        self.mcts = MCTS(
            lambda s: policy_value(self.model, s),
            num_simulations=self.cfg.num_simulations,
        )
        self.reset()

    def _board_to_list(self) -> list:
        return [
            [[p.value for p in row] for row in size]
            for size in self.state.board
        ]

    async def board_state(self) -> dict:
        return {
            "board": self._board_to_list(),
            "current": self.state.current_player.name,
            "winner": self.state.winner.name if self.state.winner else None,
            "draw": self.state.draw,
        }

    async def broadcast_train(self, message: dict) -> None:
        living: List[WebSocket] = []
        for ws in self.train_clients:
            try:
                await ws.send_json(message)
                living.append(ws)
            except WebSocketDisconnect:
                pass
        self.train_clients[:] = living

    async def broadcast_game(self) -> None:
        data = await self.board_state()
        living: List[WebSocket] = []
        for ws in self.game_clients:
            try:
                await ws.send_json(data)
                living.append(ws)
            except WebSocketDisconnect:
                pass
        self.game_clients[:] = living

    async def broadcast_log(self, message: str) -> None:
        living: List[WebSocket] = []
        for ws in self.log_clients:
            try:
                await ws.send_json({"log": message})
                living.append(ws)
            except WebSocketDisconnect:
                pass
        self.log_clients[:] = living

    async def train_loop(self, iterations: int) -> None:
        self.training_error = None
        try:
            for i in range(iterations):
                if self.stop_training_flag:
                    break
                if self.cfg.parallel_games > 1:
                    data = await asyncio.to_thread(
                        self_play_parallel,
                        self.model,
                        num_games=self.cfg.parallel_games,
                        num_simulations=self.cfg.num_simulations,
                        num_players=self.cfg.num_players,
                        max_moves=self.cfg.max_moves,
                        resign_threshold=self.cfg.resign_threshold,
                    )
                else:
                    data = await asyncio.to_thread(
                        self_play,
                        self.model,
                        num_simulations=self.cfg.num_simulations,
                        num_players=self.cfg.num_players,
                        max_moves=self.cfg.max_moves,
                        resign_threshold=self.cfg.resign_threshold,
                    )
                self.buffer.add(data)
                if self.stop_training_flag:
                    break
                loss = await asyncio.to_thread(
                    train_step,
                    self.model,
                    self.optimizer,
                    self.buffer,
                    self.cfg.batch_size,
                    self.cfg.value_loss_weight,
                )
                self.train_iteration = i + 1
                self.train_loss = loss
                await self.broadcast_train({"iteration": self.train_iteration, "loss": loss})
                await self.broadcast_log(f"iteration {self.train_iteration}: loss={loss:.4f}")
        except Exception as e:  # pragma: no cover - runtime safety
            self.training_error = str(e)


manager = OtrioApp(cfg)
app.state.manager = manager


def available_models() -> list[str]:
    if not env_dir.exists():
        return []
    return sorted([p.name for p in env_dir.glob("*.pt")])


@app.post("/new_model")
async def new_model():
    manager.new_model()
    return {"status": "created"}


@app.get("/models")
async def get_models():
    return {"models": available_models()}


@app.post("/start")
async def start_game(data: dict = Body(default={})):  # pragma: no cover - simple wrapper
    path = data.get("model") if data else None
    if path:
        p = Path(path)
        if not p.is_absolute() and not p.exists():
            candidate = env_dir / p.name
            if candidate.exists():
                path = str(candidate)
    manager.reset(path)
    await manager.broadcast_game()
    return {"status": "ok"}


class MoveData(BaseModel):
    row: int
    col: int
    size: int


@app.post("/move")
async def player_move(move: MoveData):
    state = manager.state
    if not (0 <= move.row < 3 and 0 <= move.col < 3 and 0 <= move.size < 3):
        return JSONResponse({"error": "out_of_range"}, status_code=400)
    if state.board[move.size][move.row][move.col] != Player.NONE:
        return JSONResponse({"error": "occupied"}, status_code=400)

    act = Move(move.row, move.col, move.size, state.current_player)
    state.apply_move(act)

    ai_move = None
    if not state.winner and not state.draw:
        ai_move, _, _ = await asyncio.to_thread(manager.mcts.run, state)
        state.apply_move(ai_move)

    res = await manager.board_state()
    if ai_move:
        res["ai"] = {"row": ai_move.row, "col": ai_move.col, "size": ai_move.size}
    await manager.broadcast_game()
    return res


@app.get("/state")
async def get_state():
    return await manager.board_state()


@app.get("/training_status")
async def training_status():
    status = "running" if manager.training_task and not manager.training_task.done() else "idle"
    return {
        "status": status,
        "iteration": manager.train_iteration,
        "loss": manager.train_loss,
        "error": manager.training_error,
    }


@app.post("/update_training_settings")
async def update_training_settings(settings: dict = Body(...)):
    for k, v in settings.items():
        if hasattr(manager.cfg, k):
            setattr(manager.cfg, k, v)
            if k == "learning_rate":
                for group in manager.optimizer.param_groups:
                    group["lr"] = v
    return {"status": "updated"}


@app.post("/model_save")
async def model_save(data: dict = Body(default={"path": "model.pt"})):
    from .training import save_training_state

    path = data.get("path", "model.pt")
    p = Path(path)
    if not p.is_absolute():
        p = env_dir / p.name
    await asyncio.to_thread(
        save_training_state, manager.model, manager.optimizer, manager.buffer, str(p)
    )
    return {"status": "saved", "path": str(p)}


@app.post("/model_load")
async def model_load(data: dict = Body(default={"path": "model.pt"})):
    from .training import load_training_state

    path = data.get("path", "model.pt")
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        candidate = env_dir / p.name
        if candidate.exists():
            p = candidate
    path = str(p)
    manager.model, manager.optimizer, manager.buffer = await asyncio.to_thread(
        load_training_state,
        path,
        num_players=manager.cfg.num_players,
        learning_rate=manager.cfg.learning_rate,
        buffer_capacity=manager.cfg.buffer_capacity,
        num_blocks=manager.cfg.num_blocks,
        channels=manager.cfg.channels,
        device=manager.device,
    )
    manager.mcts = MCTS(
        lambda s: policy_value(manager.model, s),
        num_simulations=manager.cfg.num_simulations,
    )
    return {"status": "loaded", "path": path}


@app.post("/stop")
async def stop_training():
    manager.stop_training_flag = True
    if manager.training_task:
        manager.training_task.cancel()
    manager.training_task = None
    return {"status": "stopped"}


class TrainRequest(BaseModel):
    iterations: int = 1


@app.post("/train")
async def start_train(req: TrainRequest):
    manager.stop_training_flag = False
    if manager.training_task and not manager.training_task.done():
        return {"status": "running"}
    manager.training_task = asyncio.create_task(manager.train_loop(req.iterations))
    return {"status": "started"}


@app.websocket("/ws/train")
async def ws_train(ws: WebSocket):
    await ws.accept()
    manager.train_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.train_clients.remove(ws)


@app.websocket("/ws/log")
async def ws_log(ws: WebSocket):
    await ws.accept()
    manager.log_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.log_clients.remove(ws)


@app.websocket("/ws/game")
async def ws_game(ws: WebSocket):
    await ws.accept()
    manager.game_clients.append(ws)
    await ws.send_json(await manager.board_state())
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.game_clients.remove(ws)


def main() -> None:  # pragma: no cover - CLI helper
    parser = argparse.ArgumentParser(description="OtrioAI Web UI Server")
    parser.add_argument("--model", type=str, default=None, help="読み込むモデル")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    manager.reset(args.model)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
