import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from OtrioAI.training import (
    ReplayBuffer,
    self_play,
    save_training_state,
    load_training_state,
)
from OtrioAI.network import OtrioNet
from OtrioAI.config import Config


def test_save_and_load_training_state(tmp_path):
    cfg = Config(num_simulations=1, buffer_capacity=10, learning_rate=0.001, batch_size=1, num_players=2)
    model = OtrioNet(num_players=cfg.num_players)
    buffer = ReplayBuffer(cfg.buffer_capacity)
    buffer.add(self_play(model, num_simulations=1, num_players=cfg.num_players))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    path = tmp_path / "state.pt"
    save_training_state(model, optimizer, buffer, str(path))

    loaded_model, loaded_optim, loaded_buffer = load_training_state(
        str(path),
        num_players=cfg.num_players,
        learning_rate=cfg.learning_rate,
        buffer_capacity=cfg.buffer_capacity,
    )

    assert len(loaded_buffer) == len(buffer)
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(p1, p2)
    assert isinstance(loaded_optim.state_dict(), dict)
