import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.network import OtrioNet
from src.evaluation import evaluate_models


def test_evaluate_models_runs():
    old = OtrioNet()
    new = OtrioNet()
    results = evaluate_models(old, new, num_games=1, num_simulations=1)
    assert set(results.keys()) == {"win", "loss", "draw", "win_rate"}
    assert results["win"] + results["loss"] + results["draw"] == 1
    assert 0.0 <= results["win_rate"] <= 1.0
