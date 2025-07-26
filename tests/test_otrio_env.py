import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.otrio_env import OtrioEnv


def test_env_legal_moves_and_step():
    env = OtrioEnv(players=2)
    moves = env.legal_moves()
    assert len(moves) == 27

    obs, reward, done, truncated, info = env.step(moves[0])
    assert obs.shape == (108,)
    assert not done
    assert reward == 0


def test_env_win():
    env = OtrioEnv(players=2)
    env.step(0)   # slot0 cell0
    env.step(9)   # slot1 cell0
    env.step(1)   # slot0 cell1
    env.step(10)  # slot1 cell1
    obs, reward, done, _trunc, info = env.step(2)  # slot0 cell2 -> win
    assert done
    assert reward == 1
    assert info["winner"] is not None


def test_otrio_env_interface():
    env = OtrioEnv(players=2)
    obs, _ = env.reset()
    assert obs.shape == (108,)
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    assert obs.shape == (108,)
    assert isinstance(done, bool)
    assert "winner" in info
