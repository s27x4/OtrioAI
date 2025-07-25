import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.training import ReplayBuffer, self_play, train_step
from src.network import OtrioNet


def test_self_play_generates_samples():
    model = OtrioNet()
    data = self_play(model, num_simulations=1)
    assert len(data) > 0
    state, policy, value = data[0]
    assert state.shape == (7, 3, 3)
    assert policy.shape == (27,)
    assert isinstance(value.item(), float)


def test_replay_buffer_and_train_step():
    model = OtrioNet()
    buffer = ReplayBuffer(capacity=10)
    buffer.add(self_play(model, num_simulations=1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = train_step(model, optimizer, buffer, batch_size=1)
    assert isinstance(loss, float)


def test_sample_empty_buffer():
    buffer = ReplayBuffer(capacity=10)
    assert buffer.sample(1) is None


def test_train_step_empty_buffer():
    model = OtrioNet()
    buffer = ReplayBuffer(capacity=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = train_step(model, optimizer, buffer, batch_size=1)
    assert loss == 0.0
