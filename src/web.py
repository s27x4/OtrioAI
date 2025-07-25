"""Flask ベースの対戦用サーバ"""

from __future__ import annotations

import argparse
from flask import Flask, jsonify, request

from .config import load_config
from .mcts import MCTS
from .network import OtrioNet, load_model, policy_value
from .otrio import GameState, Move, Player

app = Flask(__name__)

cfg = load_config()
state: GameState = GameState(num_players=cfg.num_players)
model: OtrioNet = OtrioNet(num_players=cfg.num_players)
mcts: MCTS = MCTS(lambda s: policy_value(model, s), num_simulations=cfg.num_simulations)


def reset(model_path: str | None = None) -> None:
    """ゲーム状態とモデルを初期化"""
    global state, model, mcts
    state = GameState(num_players=cfg.num_players)
    if model_path:
        model = load_model(model_path, num_players=cfg.num_players)
    mcts = MCTS(lambda s: policy_value(model, s), num_simulations=cfg.num_simulations)


@app.post("/start")
def start_game():
    data = request.get_json(silent=True) or {}
    path = data.get("model")
    reset(path)
    return jsonify({"status": "ok"})


def _board_to_list(state: GameState) -> list:
    return [
        [[p.value for p in row] for row in size] for size in state.board
    ]


@app.post("/move")
def player_move():
    global state
    data = request.get_json(silent=True) or {}
    try:
        r, c, s = int(data["row"]), int(data["col"]), int(data["size"])
    except Exception:
        return jsonify({"error": "invalid"}), 400
    if not (0 <= r < 3 and 0 <= c < 3 and 0 <= s < 3):
        return jsonify({"error": "out_of_range"}), 400
    if state.board[s][r][c] != Player.NONE:
        return jsonify({"error": "occupied"}), 400

    move = Move(r, c, s, state.current_player)
    state.apply_move(move)

    ai_move = None
    if not state.winner and not state.draw:
        ai_move, _ = mcts.run(state)
        state.apply_move(ai_move)

    res = {
        "board": _board_to_list(state),
        "winner": state.winner.name if state.winner else None,
        "draw": state.draw,
    }
    if ai_move:
        res["ai"] = {"row": ai_move.row, "col": ai_move.col, "size": ai_move.size}
    return jsonify(res)


@app.get("/state")
def get_state():
    return jsonify(
        board=_board_to_list(state),
        current=state.current_player.name,
        winner=state.winner.name if state.winner else None,
        draw=state.draw,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="OtrioAI Web Server")
    parser.add_argument("--model", type=str, default=None, help="モデルファイル")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    reset(args.model)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
