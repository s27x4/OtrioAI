import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.network import OtrioNet
from src.evaluation import evaluate_models


def test_evaluate_models_returns_result_dict():
    new_model = OtrioNet()
    old_model = OtrioNet()
    result = evaluate_models(new_model, old_model, num_games=2, num_simulations=1)
    assert set(result.keys()) == {"new_model_wins", "old_model_wins", "draws", "win_rate"}
    assert result["new_model_wins"] + result["old_model_wins"] + result["draws"] == 2
    assert 0.0 <= result["win_rate"] <= 1.0
