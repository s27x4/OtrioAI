import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.otrio import GameState, Player, Move


def test_next_player_three_players():
    state = GameState(num_players=3)
    assert state.current_player == Player.PLAYER1
    state.apply_move(Move(0,0,0, state.current_player))
    assert state.current_player == Player.PLAYER2
    state.apply_move(Move(0,0,1, state.current_player))
    assert state.current_player == Player.PLAYER3


def test_clone_preserves_num_players():
    state = GameState(num_players=4)
    clone = state.clone()
    assert clone.num_players == 4
