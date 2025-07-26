import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.gui import train_gui_loop
from src.config import Config


def test_train_gui_loop_runs_headless(monkeypatch):
    cfg = Config(num_simulations=1, buffer_capacity=10, learning_rate=0.001, batch_size=1, num_players=2)
    called = {}

    async def dummy(msg):
        called["msg"] = msg

    monkeypatch.setattr("src.gui.web.broadcast_train", dummy)
    losses = train_gui_loop(1, cfg=cfg, headless=True, broadcast=True)
    assert len(losses) == 1
    assert isinstance(losses[0], float)
    assert called["msg"]["iteration"] == 1
