import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from OtrioAI.network import OtrioNet, state_to_tensor, create_optimizer
from OtrioAI.otrio import GameState


def test_state_to_tensor_shape():
    state = GameState()
    t = state_to_tensor(state)
    assert t.shape == (7, 3, 3)


def test_state_to_tensor_multi_players():
    state3 = GameState(num_players=3)
    t3 = state_to_tensor(state3)
    assert t3.shape == (10, 3, 3)
    state4 = GameState(num_players=4)
    t4 = state_to_tensor(state4)
    assert t4.shape == (13, 3, 3)


def test_network_forward_shapes():
    model = OtrioNet(num_players=2, num_blocks=2)
    state = GameState()
    x = state_to_tensor(state).unsqueeze(0)
    policy, value = model(x)
    assert policy.shape == (1, 27)
    assert value.shape == (1,)


def test_create_optimizer():
    model = OtrioNet(num_players=2, num_blocks=1)
    optim = create_optimizer(model, lr=0.001)
    assert isinstance(optim, torch.optim.Optimizer)
