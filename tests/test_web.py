import importlib
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import Config


def test_web_endpoints(monkeypatch):
    import src.web as web

    def fake_load_config():
        return Config(num_simulations=1, buffer_capacity=10, learning_rate=0.001, batch_size=1, num_players=2)

    monkeypatch.setattr(web, "load_config", fake_load_config)
    web = importlib.reload(web)

    class DummyMCTS:
        def __init__(self, fn, num_simulations=1):
            pass
        def run(self, state):
            return state.legal_moves()[0], None

    called = {}
    def fake_load_model(path, num_players=2):
        called["model"] = path
        return None

    monkeypatch.setattr(web, "MCTS", DummyMCTS)
    monkeypatch.setattr(web, "load_model", fake_load_model)

    client = web.app.test_client()

    res = client.post("/start", json={"model": "foo.pt"})
    assert res.status_code == 200
    assert called["model"] == "foo.pt"

    res = client.post("/move", json={"row": 0, "col": 0, "size": 0})
    data = res.get_json()
    assert data["board"][0][0][0] == 1
    assert "ai" in data
