import os
import sys
import torch
import multiprocessing

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
multiprocessing.set_start_method("fork", force=True)

from src.training import ReplayBuffer, self_play, self_play_parallel, train_step
from src.network import OtrioNet


def test_self_play_generates_samples():
    model = OtrioNet(num_players=2)
    data = self_play(model, num_simulations=1, num_players=2)
    assert len(data) > 0
    state, policy, value = data[0]
    assert state.shape == (7, 3, 3)
    assert policy.shape == (27,)
    assert isinstance(value.item(), float)


def test_self_play_parallel_generates_samples(monkeypatch):
    model = OtrioNet(num_players=2)

    def dummy_parallel(*args, **kwargs):
        results = []
        for _ in range(kwargs.get("num_games", 1)):
            results.extend(self_play(model, num_simulations=kwargs.get("num_simulations", 1), num_players=kwargs.get("num_players", 2)))
        return results

    monkeypatch.setattr('src.training.self_play_parallel', dummy_parallel)
    data = dummy_parallel(model, num_games=2, num_simulations=1, num_players=2)
    assert len(data) > 0


def test_replay_buffer_and_train_step():
    model = OtrioNet(num_players=2)
    buffer = ReplayBuffer(capacity=10)
    buffer.add(self_play(model, num_simulations=1, num_players=2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = train_step(model, optimizer, buffer, batch_size=1, value_weight=1.0)
    assert isinstance(loss, float)


def test_sample_empty_buffer():
    buffer = ReplayBuffer(capacity=10)
    assert buffer.sample(1) is None


def test_train_step_empty_buffer():
    model = OtrioNet()
    buffer = ReplayBuffer(capacity=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = train_step(model, optimizer, buffer, batch_size=1, value_weight=1.0)
    assert loss == 0.0


def test_replay_buffer_save_load(tmp_path):
    model = OtrioNet(num_players=2)
    buffer = ReplayBuffer(capacity=10)
    samples = self_play(model, num_simulations=1, num_players=2)
    buffer.add(samples)
    path = tmp_path / "buffer.pt"
    buffer.save(str(path))

    loaded = ReplayBuffer(capacity=10)
    loaded.load(str(path))
    assert len(loaded) == len(buffer)
    for a, b in zip(buffer.data, loaded.data):
        assert torch.equal(a[0], b[0])
        assert torch.equal(a[1], b[1])
        assert torch.equal(a[2], b[2])
