"""Gymnasium 向け Otrio 環境実装"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

BOARD_CELLS = 9  # 3x3
SLOTS = 3  # small, medium, large
MAX_ACTIONS = BOARD_CELLS * SLOTS

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diags
]


class OtrioBase:
    """ゲームロジックのみを扱う基本クラス"""

    def __init__(self, active_colors: List[int] = (0, 1, 2, 3)):
        self.colors = list(active_colors)
        self.num_colors = len(self.colors)
        self.reset()

    # ---------- core API ----------
    def reset(self) -> np.ndarray:
        # -1 = empty, 0-3 = color id
        self.board = np.full((SLOTS, BOARD_CELLS), -1, dtype=np.int8)
        self.stash = {c: {s: 3 for s in range(SLOTS)} for c in self.colors}
        self.turn = 0
        self.winner: int | None = None
        return self.observe()

    def observe(self) -> np.ndarray:
        planes = np.zeros((4, SLOTS, 3, 3), dtype=np.int8)
        for slot in range(SLOTS):
            for cell in range(BOARD_CELLS):
                col = self.board[slot, cell]
                if col != -1:
                    planes[col, slot, cell // 3, cell % 3] = 1
        return planes

    def legal_moves(self) -> List[int]:
        color = self.colors[self.turn]
        moves: List[int] = []
        for slot in range(SLOTS):
            if self.stash[color][slot] == 0:
                continue
            for cell in range(BOARD_CELLS):
                if self.board[slot, cell] == -1:
                    moves.append(slot * BOARD_CELLS + cell)
        return moves

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, dict]:
        if self.winner is not None:
            raise RuntimeError("Game over.")

        color = self.colors[self.turn]
        slot, cell = divmod(action, BOARD_CELLS)

        if self.board[slot, cell] != -1:
            raise ValueError("Illegal move: occupied slot.")
        if self.stash[color][slot] <= 0:
            raise ValueError("Illegal move: no pieces left.")

        self.board[slot, cell] = color
        self.stash[color][slot] -= 1

        self._check_win(color, slot, cell)

        done = self.winner is not None or all(
            self.board[s].min() != -1 for s in range(SLOTS)
        )
        reward = 0
        if done and self.winner is not None:
            reward = 1

        self.turn = (self.turn + 1) % self.num_colors
        return self.observe(), reward, done, {}

    # ---------- helpers ----------
    def _check_win(self, color: int, slot: int, cell: int) -> None:
        # ① 同サイズ 3 連
        for line in WIN_LINES:
            if cell not in line:
                continue
            if all(self.board[slot, c] == color for c in line):
                self.winner = color
                return

        # ② サイズ順 3 連 (昇順 / 降順)
        for line in WIN_LINES:
            if cell not in line:
                continue
            seq = [self.board[s, line[i]] for i, s in enumerate(range(2, -1, -1))]
            if seq == [color] * 3:
                self.winner = color
                return
            seq = [self.board[s, line[i]] for i, s in enumerate(range(3))]
            if seq == [color] * 3:
                self.winner = color
                return

        # ③ Bullseye
        if all(self.board[s, cell] == color for s in range(SLOTS)):
            self.winner = color


class OtrioEnv(gym.Env):
    """Gymnasium 互換のマルチエージェント環境"""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, players: int = 4):
        assert players in (2, 3, 4)
        active = list(range(players if players != 2 else 4))
        self.core = OtrioBase(active_colors=active)

        self.action_space = spaces.Discrete(MAX_ACTIONS)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(4 * SLOTS * 3 * 3,),
            dtype=np.int8,
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs = self.core.reset()
        return obs.flatten(), {}

    def step(self, action):
        obs, reward, done, _ = self.core.step(action)
        info = {"winner": self.core.winner}
        return obs.flatten(), reward, done, False, info

    def render(self, mode: str = "ansi"):
        grid = [["   "] * 3 for _ in range(3)]
        sym = {-1: "   ", 0: "R", 1: "G", 2: "B", 3: "Y"}
        for slot, size_char in zip(range(SLOTS), ("s", "m", "L")):
            for cell in range(BOARD_CELLS):
                c = self.core.board[slot, cell]
                if c != -1:
                    r, cx = divmod(cell, 3)
                    grid[r][cx] = grid[r][cx].replace(" ", sym[c] + size_char, 1)
        out = "\n".join(" | ".join(row) for row in grid)
        if mode == "ansi":
            print(out)
        return out
