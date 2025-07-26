import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from OtrioAI.otrio import GameState, Move, Player


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


def test_initial_legal_moves_count():
    state = GameState()
    moves = state.legal_moves()
    assert len(moves) == 27
    assert all(move.player == Player.PLAYER1 for move in moves)


def test_apply_move_and_switch_player():
    state = GameState()
    move = Move(0, 0, 0, state.current_player)
    state.apply_move(move)
    assert state.board[0][0][0] == Player.PLAYER1
    assert state.current_player == Player.PLAYER2
    assert len(state.move_history) == 1


def test_illegal_move_raises():
    state = GameState()
    move = Move(0, 0, 0, state.current_player)
    state.apply_move(move)
    with pytest.raises(AssertionError):
        state.apply_move(Move(0, 0, 0, state.current_player))


def test_draw_game():
    moves = [
        Move(2, 2, 2, Player.PLAYER1),
        Move(1, 0, 1, Player.PLAYER2),
        Move(2, 0, 1, Player.PLAYER1),
        Move(0, 2, 0, Player.PLAYER2),
        Move(0, 1, 0, Player.PLAYER1),
        Move(2, 2, 1, Player.PLAYER2),
        Move(1, 2, 0, Player.PLAYER1),
        Move(1, 1, 2, Player.PLAYER2),
        Move(1, 1, 0, Player.PLAYER1),
        Move(0, 2, 1, Player.PLAYER2),
        Move(2, 2, 0, Player.PLAYER1),
        Move(2, 1, 0, Player.PLAYER2),
        Move(1, 1, 1, Player.PLAYER1),
        Move(1, 0, 0, Player.PLAYER2),
        Move(1, 0, 2, Player.PLAYER1),
        Move(1, 2, 2, Player.PLAYER2),
        Move(0, 1, 2, Player.PLAYER1),
        Move(0, 1, 1, Player.PLAYER2),
        Move(2, 1, 1, Player.PLAYER1),
        Move(2, 1, 2, Player.PLAYER2),
        Move(0, 0, 1, Player.PLAYER1),
        Move(0, 0, 0, Player.PLAYER2),
        Move(2, 0, 0, Player.PLAYER1),
        Move(0, 0, 2, Player.PLAYER2),
        Move(0, 2, 2, Player.PLAYER1),
        Move(2, 0, 2, Player.PLAYER2),
        Move(1, 2, 1, Player.PLAYER1),
    ]
    state = GameState()
    for m in moves:
        state.apply_move(m)
    assert state.winner is None
    assert state.draw
