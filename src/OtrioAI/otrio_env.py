"""Gymnasium 向け Otrio 環境実装 (GameState を利用)"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .otrio import GameState, Move, Player

BOARD_CELLS = 9  # 3x3
SLOTS = 3  # small, medium, large
MAX_ACTIONS = BOARD_CELLS * SLOTS


class OtrioEnv(gym.Env):
    """Gymnasium 互換のシンプルな環境"""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, players: int = 4):
        assert players in (2, 3, 4)
        self.num_players = players
        self.state = GameState(num_players=players)
        self.action_space = spaces.Discrete(MAX_ACTIONS)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(4 * SLOTS * 3 * 3,),
            dtype=np.int8,
        )

    # ----------------- helpers -----------------
    def _observe(self) -> np.ndarray:
        planes = np.zeros((4, SLOTS, 3, 3), dtype=np.int8)
        for size in range(SLOTS):
            for r in range(3):
                for c in range(3):
                    p = self.state.board[size][r][c]
                    if p != Player.NONE:
                        planes[p.value - 1, size, r, c] = 1
        return planes

    # ----------------- gym API -----------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        self.state = GameState(num_players=self.num_players)
        obs = self._observe()
        return obs.flatten(), {}

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, bool, dict]:
        size, cell = divmod(action, BOARD_CELLS)
        row, col = divmod(cell, 3)
        if self.state.board[size][row][col] != Player.NONE:
            raise ValueError("Illegal move: occupied slot.")
        move = Move(row, col, size, self.state.current_player)
        self.state.apply_move(move)
        done = self.state.winner is not None or self.state.draw
        reward = 1 if self.state.winner is not None else 0
        obs = self._observe()
        info = {"winner": self.state.winner}
        return obs.flatten(), reward, done, False, info

    def legal_moves(self):
        return [m.size * BOARD_CELLS + m.row * 3 + m.col for m in self.state.legal_moves()]

    def render(self, mode: str = "ansi"):
        grid = [["   "] * 3 for _ in range(3)]
        sym = {Player.NONE: "   ", Player.PLAYER1: "R", Player.PLAYER2: "G", Player.PLAYER3: "B", Player.PLAYER4: "Y"}
        for size, size_char in zip(range(SLOTS), ("s", "m", "L")):
            for cell in range(BOARD_CELLS):
                r, c = divmod(cell, 3)
                p = self.state.board[size][r][c]
                if p != Player.NONE:
                    grid[r][c] = grid[r][c].replace(" ", sym[p] + size_char, 1)
        out = "\n".join(" | ".join(row) for row in grid)
        if mode == "ansi":
            print(out)
        return out
