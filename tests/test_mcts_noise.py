import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from OtrioAI.mcts import MCTS, Node
from OtrioAI.otrio import GameState


def test_dirichlet_noise_changes_priors():
    state = GameState()
    mcts = MCTS()
    # ノイズなし
    root = Node(state.clone())
    mcts.expand(root)
    priors_before = [c.prior for c in root.children.values()]

    # ノイズあり
    root_noise = Node(state.clone())
    mcts.expand(root_noise)
    mcts._apply_dirichlet_noise(root_noise)
    priors_after = [c.prior for c in root_noise.children.values()]

    assert priors_before != priors_after
    assert abs(sum(priors_after) - 1.0) < 1e-6
