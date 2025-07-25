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


def test_initial_legal_moves():
    state = GameState()
    moves = state.legal_moves()
    assert len(moves) == 27
    # すべての手がプレイヤー1の手として生成されていること
    assert all(m.player == Player.PLAYER1 for m in moves)
    # 重複がないことを確認
    assert len({(m.row, m.col, m.size) for m in moves}) == 27


def test_apply_move_and_turn_switch():
    state = GameState()
    move = Move(1, 2, 0, state.current_player)
    state.apply_move(move)
    assert state.board[0][1][2] == Player.PLAYER1
    # 勝敗がついていない場合は手番が交代する
    assert state.current_player == Player.PLAYER2
    assert len(state.move_history) == 1


def test_no_legal_moves_after_win():
    state = GameState()
    state.apply_move(Move(0, 0, 0, Player.PLAYER1))
    state.apply_move(Move(0, 1, 0, Player.PLAYER1))
    state.apply_move(Move(0, 2, 0, Player.PLAYER1))
    assert state.winner == Player.PLAYER1
    assert state.legal_moves() == []


def test_draw_detection_and_no_legal_moves():
    # 引き分け状態を直接作成
    board0 = [
        [Player.PLAYER1, Player.PLAYER2, Player.PLAYER1],
        [Player.PLAYER1, Player.PLAYER2, Player.PLAYER2],
        [Player.PLAYER2, Player.PLAYER1, Player.PLAYER1],
    ]
    board1 = [[Player.PLAYER2 if p == Player.PLAYER1 else Player.PLAYER1 for p in row] for row in board0]
    board2 = board0
    state = GameState(board=[board0, board1, board2])
    state._update_status(Player.PLAYER1)
    assert state.draw
    assert state.winner is None
    assert state.legal_moves() == []
