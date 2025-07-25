# coding: utf-8
"""Otrioゲーム環境の簡易実装"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class Player(Enum):
    NONE = 0
    PLAYER1 = 1
    PLAYER2 = 2

    @staticmethod
    def opponent(player: 'Player') -> 'Player':
        if player == Player.PLAYER1:
            return Player.PLAYER2
        elif player == Player.PLAYER2:
            return Player.PLAYER1
        else:
            return Player.NONE


@dataclass(frozen=True)
class Move:
    row: int
    col: int
    size: int  # 0:小, 1:中, 2:大
    player: Player


@dataclass
class GameState:
    board: List[List[List[Player]]] = field(
        default_factory=lambda: [[[Player.NONE for _ in range(3)] for _ in range(3)] for _ in range(3)]
    )
    current_player: Player = Player.PLAYER1
    move_history: List[Move] = field(default_factory=list)
    winner: Optional[Player] = None
    draw: bool = False

    def clone(self) -> 'GameState':
        new_state = GameState(
            board=[[row[:] for row in size] for size in self.board],
            current_player=self.current_player,
            move_history=self.move_history[:],
            winner=self.winner,
            draw=self.draw,
        )
        return new_state

    def legal_moves(self) -> List[Move]:
        if self.winner or self.draw:
            return []
        moves = []
        for size in range(3):
            for r in range(3):
                for c in range(3):
                    if self.board[size][r][c] == Player.NONE:
                        moves.append(Move(r, c, size, self.current_player))
        return moves

    def apply_move(self, move: Move) -> None:
        assert self.board[move.size][move.row][move.col] == Player.NONE
        self.board[move.size][move.row][move.col] = move.player
        self.move_history.append(move)
        self._update_status(move.player)
        if not self.winner and not self.draw:
            self.current_player = Player.opponent(self.current_player)

    def _update_status(self, player: Player) -> None:
        # 勝利判定: サイズ別の3目並べ
        for size in range(3):
            b = self.board[size]
            lines = (
                [b[0][0], b[0][1], b[0][2]],
                [b[1][0], b[1][1], b[1][2]],
                [b[2][0], b[2][1], b[2][2]],
                [b[0][0], b[1][0], b[2][0]],
                [b[0][1], b[1][1], b[2][1]],
                [b[0][2], b[1][2], b[2][2]],
                [b[0][0], b[1][1], b[2][2]],
                [b[0][2], b[1][1], b[2][0]],
            )
            for line in lines:
                if all(p == player for p in line):
                    self.winner = player
                    return
        # スタック勝利（同じマスに3サイズ）
        for r in range(3):
            for c in range(3):
                if all(self.board[size][r][c] == player for size in range(3)):
                    self.winner = player
                    return
        # 引き分け判定
        if not any(
            self.board[size][r][c] == Player.NONE
            for size in range(3)
            for r in range(3)
            for c in range(3)
        ):
            self.draw = True

    def log(self) -> str:
        lines = []
        for move in self.move_history:
            lines.append(
                f"{move.player.name}: size={move.size}, pos=({move.row},{move.col})"
            )
        if self.winner:
            lines.append(f"Winner: {self.winner.name}")
        elif self.draw:
            lines.append("Draw")
        return "\n".join(lines)
