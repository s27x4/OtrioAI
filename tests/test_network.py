import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.network import OtrioNet, state_to_tensor
from src.otrio import GameState


def test_state_to_tensor_shape():
    state = GameState()
    t = state_to_tensor(state)
    assert t.shape == (7, 3, 3)


def test_network_forward_shapes():
    model = OtrioNet()
    state = GameState()
    x = state_to_tensor(state).unsqueeze(0)
    policy, value = model(x)
    assert policy.shape == (1, 27)
    assert value.shape == (1,)
