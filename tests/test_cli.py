import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.cli import main
from src.config import Config


def test_cli_self_play(monkeypatch, capsys):
    def fake_load():
        return Config(num_simulations=1, buffer_capacity=10, learning_rate=0.001, batch_size=1, num_players=2)

    monkeypatch.setattr('src.cli.load_config', fake_load)
    monkeypatch.setattr(sys, 'argv', ['cli', '--self-play'])
    main()
    out = capsys.readouterr().out
    assert "サンプル" in out


def test_cli_train_loop(monkeypatch, capsys):
    def fake_load():
        return Config(num_simulations=1, buffer_capacity=10, learning_rate=0.001, batch_size=1, num_players=2)

    monkeypatch.setattr('src.cli.load_config', fake_load)
    monkeypatch.setattr(sys, 'argv', ['cli', '--train-loop', '2'])
    main()
    out = capsys.readouterr().out
    assert "平均損失" in out or "loss=" in out
