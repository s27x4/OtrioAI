import json
from typing import Dict, Optional

from .mcts import MCTS
from .network import OtrioNet, policy_value
from .otrio import GameState, Player


def play_game(model_p1: OtrioNet, model_p2: OtrioNet, num_simulations: int = 50) -> GameState:
    """model_p1 を先手、model_p2 を後手として1局プレイ"""
    state = GameState()
    mcts_p1 = MCTS(lambda s: policy_value(model_p1, s), num_simulations)
    mcts_p2 = MCTS(lambda s: policy_value(model_p2, s), num_simulations)

    while not state.winner and not state.draw:
        mcts = mcts_p1 if state.current_player == Player.PLAYER1 else mcts_p2
        move, _ = mcts.run(state)
        state.apply_move(move)
    return state


def evaluate_models(
    new_model: OtrioNet,
    old_model: OtrioNet,
    num_games: int = 10,
    num_simulations: int = 50,
    log_path: Optional[str] = None,
) -> Dict[str, float]:
    """旧モデルと新モデルを対戦させ勝率を評価する"""
    new_wins = 0
    old_wins = 0
    draws = 0
    logs = []
    for i in range(num_games):
        if i % 2 == 0:
            state = play_game(new_model, old_model, num_simulations)
            winner_as_p1 = new_model
        else:
            state = play_game(old_model, new_model, num_simulations)
            winner_as_p1 = old_model
        logs.append(state.log())
        if state.winner == Player.PLAYER1:
            if winner_as_p1 is new_model:
                new_wins += 1
            else:
                old_wins += 1
        elif state.winner == Player.PLAYER2:
            if winner_as_p1 is new_model:
                old_wins += 1
            else:
                new_wins += 1
        else:
            draws += 1
    result = {
        "new_model_wins": new_wins,
        "old_model_wins": old_wins,
        "draws": draws,
        "win_rate": new_wins / num_games if num_games > 0 else 0.0,
    }
    if log_path:
        with open(log_path, "w") as f:
            json.dump({"result": result, "logs": logs}, f, ensure_ascii=False, indent=2)
    return result
