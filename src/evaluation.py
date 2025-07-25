from __future__ import annotations

import datetime
from typing import Dict

from .otrio import GameState, Player
from .mcts import MCTS
from .network import OtrioNet, policy_value, load_model, save_model


def play_match(
    player1: OtrioNet,
    player2: OtrioNet,
    num_simulations: int = 50,
    num_players: int = 2,
) -> Player:
    """player1 が先手、player2 が後手で1局対戦し勝者を返す"""
    state = GameState(num_players=num_players)
    mcts1 = MCTS(lambda s: policy_value(player1, s), num_simulations=num_simulations)
    mcts2 = MCTS(lambda s: policy_value(player2, s), num_simulations=num_simulations)

    while not state.winner and not state.draw:
        if state.current_player == Player.PLAYER1:
            move, _ = mcts1.run(state)
        else:
            move, _ = mcts2.run(state)
        state.apply_move(move)
    return state.winner if state.winner else Player.NONE


def evaluate_models(
    old_model: OtrioNet,
    new_model: OtrioNet,
    num_games: int = 10,
    num_simulations: int = 50,
    num_players: int = 2,
) -> Dict[str, float]:
    """旧モデルと新モデルを対戦させ、勝率を返す"""
    results = {"win": 0, "loss": 0, "draw": 0}
    for i in range(num_games):
        if i % 2 == 0:
            winner = play_match(new_model, old_model, num_simulations, num_players)
            if winner == Player.PLAYER1:
                results["win"] += 1
            elif winner == Player.PLAYER2:
                results["loss"] += 1
            else:
                results["draw"] += 1
        else:
            winner = play_match(old_model, new_model, num_simulations, num_players)
            if winner == Player.PLAYER1:
                results["loss"] += 1
            elif winner == Player.PLAYER2:
                results["win"] += 1
            else:
                results["draw"] += 1
    results["win_rate"] = results["win"] / num_games
    return results


def evaluate_and_select(
    old_model_path: str,
    new_model_path: str,
    num_games: int = 10,
    threshold: float = 0.55,
    log_path: str = "evaluation.log",
    num_players: int = 2,
) -> bool:
    """モデルを読み込み評価し、必要に応じて置き換える"""
    old_model = load_model(old_model_path, num_players)
    new_model = load_model(new_model_path, num_players)
    results = evaluate_models(old_model, new_model, num_games, num_players=num_players)
    adopt = results["win_rate"] > threshold

    with open(log_path, "a") as f:
        f.write(
            f"{datetime.datetime.utcnow().isoformat()} "
            f"win:{results['win']} loss:{results['loss']} draw:{results['draw']} "
            f"rate:{results['win_rate']:.3f} adopt:{adopt}\n"
        )

    if adopt:
        save_model(new_model, old_model_path)
    return adopt
