import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.otrio_env import OtrioBase, OtrioEnv


def test_otrio_base_moves_and_stash():
    env = OtrioBase(active_colors=[0, 1])
    moves = env.legal_moves()
    assert len(moves) == 27
    assert env.stash[0][0] == 3

    obs, reward, done, _ = env.step(moves[0])
    assert obs.shape == (4, 3, 3, 3)
    assert env.stash[0][moves[0] // 9] == 2
    assert not done
    assert reward == 0


def test_otrio_base_win():
    env = OtrioBase(active_colors=[0, 1])
    env.step(0)   # color0 slot0 cell0
    env.step(9)   # color1 slot1 cell0
    env.step(1)   # color0 slot0 cell1
    env.step(10)  # color1 slot1 cell1
    obs, reward, done, _ = env.step(2)  # color0 slot0 cell2 -> win
    assert done
    assert reward == 1
    assert env.winner == 0


def test_otrio_env_interface():
    env = OtrioEnv(players=2)
    obs, _ = env.reset()
    assert obs.shape == (108,)
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    assert obs.shape == (108,)
    assert isinstance(done, bool)
    assert "winner" in info
