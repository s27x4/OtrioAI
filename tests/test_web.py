import importlib
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from src.config import Config


def test_web_endpoints(monkeypatch, tmp_path):
    import src.web as web

    def fake_load_config():
        return Config(num_simulations=1, buffer_capacity=10, learning_rate=0.001, batch_size=1, num_players=2)

    env_path = tmp_path / "env"
    env_path.mkdir()
    (env_path / "foo.pt").write_text("dummy")
    monkeypatch.setattr(web, "load_config", fake_load_config)
    monkeypatch.setattr(web, "env_dir", env_path)
    web = importlib.reload(web)
    monkeypatch.setattr(web, "env_dir", env_path)

    class DummyMCTS:
        def __init__(self, fn, num_simulations=1):
            pass
        def run(self, state):
            return state.legal_moves()[0], None, 0.0

    called = {}
    def fake_load_model(path, num_players=2, **kwargs):
        called["model"] = path
        return None

    monkeypatch.setattr(web, "MCTS", DummyMCTS)
    monkeypatch.setattr(web, "load_model", fake_load_model)

    client = TestClient(web.app)

    res = client.get("/models")
    assert res.json()["models"] == ["foo.pt"]

    res = client.post("/start", json={"model": "foo.pt"})
    assert res.status_code == 200
    assert called["model"] == str(env_path / "foo.pt")

    res = client.post("/move", json={"row": 0, "col": 0, "size": 0})
    data = res.json()
    assert data["board"][0][0][0] == 1
    assert "ai" in data


def test_ws_train(monkeypatch):
    import src.web as web

    def fake_load_config():
        return Config(num_simulations=1, buffer_capacity=10, learning_rate=0.001, batch_size=1, num_players=2)

    monkeypatch.setattr(web, "load_config", fake_load_config)
    web = importlib.reload(web)

    async def dummy_train_loop(*args, **kwargs):
        await web.broadcast_train({"iteration": 1, "loss": 0.5})

    monkeypatch.setattr(web, "train_loop", dummy_train_loop)

    client = TestClient(web.app)
    with client.websocket_connect("/ws/train") as ws:
        res = client.post("/train", json={"iterations": 1})
        assert res.status_code == 200
        data = ws.receive_json()
        assert data["iteration"] == 1


def test_new_model(monkeypatch):
    import src.web as web

    def fake_load_config():
        return Config(num_simulations=1, buffer_capacity=10, learning_rate=0.001, batch_size=1, num_players=2)

    monkeypatch.setattr(web, "load_config", fake_load_config)
    web = importlib.reload(web)

    client = TestClient(web.app)
    res = client.post("/new_model")
    assert res.status_code == 200
    assert res.json()["status"] == "created"


def test_stop_training(monkeypatch):
    import src.web as web

    def fake_load_config():
        return Config(num_simulations=1, buffer_capacity=10, learning_rate=0.001, batch_size=1, num_players=2)

    monkeypatch.setattr(web, "load_config", fake_load_config)
    web = importlib.reload(web)

    client = TestClient(web.app)
    res = client.post("/train", json={"iterations": 10})
    assert res.status_code == 200
    assert res.json()["status"] == "started"

    res = client.post("/stop")
    assert res.status_code == 200
    assert res.json()["status"] == "stopped"
    assert web.stop_training_flag is True

