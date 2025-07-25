from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple
import math

from .otrio import GameState, Move, Player


@dataclass
class Node:
    state: GameState
    parent: Optional['Node'] = None
    prior: float = 1.0
    children: Dict[Move, 'Node'] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0

    @property
    def q(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def expanded(self) -> bool:
        return len(self.children) > 0


class MCTS:
    def __init__(self, policy_value_fn: Optional[Callable[[GameState], Tuple[Dict[Move, float], float]]] = None,
                 num_simulations: int = 100, c_puct: float = 1.4):
        self.policy_value_fn = policy_value_fn or self._uniform_policy_value
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def _uniform_policy_value(self, state: GameState) -> Tuple[Dict[Move, float], float]:
        moves = state.legal_moves()
        if not moves:
            return {}, 0.0
        prob = 1.0 / len(moves)
        return {m: prob for m in moves}, 0.0

    def select_child(self, node: Node) -> Tuple[Move, Node]:
        assert node.expanded()
        best_move = None
        best_score = -float('inf')
        best_child = None
        for move, child in node.children.items():
            u = self.c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
            score = child.q + u
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        return best_move, best_child

    def expand(self, node: Node) -> float:
        policy, value = self.policy_value_fn(node.state)
        for move, p in policy.items():
            next_state = node.state.clone()
            next_state.apply_move(move)
            node.children[move] = Node(state=next_state, parent=node, prior=p)
        return value

    def backup(self, search_path: list[Node], value: float) -> None:
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # switch perspective

    def run(self, state: GameState) -> Move:
        root = Node(state=state.clone())
        # initial expansion
        self.expand(root)
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            # selection
            while node.expanded() and node.children:
                move, node = self.select_child(node)
                search_path.append(node)
            # evaluation & expansion
            if node.state.winner or node.state.draw:
                value = 1.0 if node.state.winner == search_path[0].state.current_player else -1.0 if node.state.winner else 0.0
            else:
                value = self.expand(node)
            # backup
            self.backup(search_path, value)
        # choose move with highest visit count
        best_move, best_child = max(root.children.items(), key=lambda item: item[1].visit_count)
        return best_move
