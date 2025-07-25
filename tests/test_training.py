import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.training import ReplayBuffer, self_play, train_step
from src.network import OtrioNet


def test_self_play_generates_samples():
    model = OtrioNet(num_players=2)
    data = self_play(model, num_simulations=1, num_players=2)
    assert len(data) > 0
    state, policy, value = data[0]
    assert state.shape == (7, 3, 3)
    assert policy.shape == (27,)
    assert isinstance(value.item(), float)


def test_replay_buffer_and_train_step():
    model = OtrioNet(num_players=2)
    buffer = ReplayBuffer(capacity=10)
    buffer.add(self_play(model, num_simulations=1, num_players=2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = train_step(model, optimizer, buffer, batch_size=1)
    assert isinstance(loss, float)
