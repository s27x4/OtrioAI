import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.otrio import GameState, Move, Player


def test_line_same_size_horizontal():
    state = GameState()
    state.apply_move(Move(0, 0, 0, Player.PLAYER1))
    state.apply_move(Move(0, 1, 0, Player.PLAYER1))
    state.apply_move(Move(0, 2, 0, Player.PLAYER1))
    assert state.winner == Player.PLAYER1


def test_stack_victory():
    state = GameState()
    state.apply_move(Move(0, 0, 0, Player.PLAYER1))
    state.apply_move(Move(0, 0, 1, Player.PLAYER1))
    state.apply_move(Move(0, 0, 2, Player.PLAYER1))
    assert state.winner == Player.PLAYER1


def test_line_big_medium_small_horizontal():
    state = GameState()
    state.apply_move(Move(0, 0, 2, Player.PLAYER1))
    state.apply_move(Move(0, 1, 1, Player.PLAYER1))
    state.apply_move(Move(0, 2, 0, Player.PLAYER1))
    assert state.winner == Player.PLAYER1


def test_line_big_medium_small_vertical():
    state = GameState()
    state.apply_move(Move(0, 0, 0, Player.PLAYER1))
    state.apply_move(Move(1, 0, 1, Player.PLAYER1))
    state.apply_move(Move(2, 0, 2, Player.PLAYER1))
    assert state.winner == Player.PLAYER1


def test_line_big_medium_small_diagonal():
    state = GameState()
    state.apply_move(Move(0, 0, 2, Player.PLAYER1))
    state.apply_move(Move(1, 1, 1, Player.PLAYER1))
    state.apply_move(Move(2, 2, 0, Player.PLAYER1))
    assert state.winner == Player.PLAYER1
