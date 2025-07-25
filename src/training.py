import random
from typing import List, Tuple

import torch
from torch import optim

from .mcts import MCTS
from .network import OtrioNet, policy_value, state_to_tensor, loss_fn
from .otrio import GameState, Player, Move


class ReplayBuffer:
    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self.data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def add(self, samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> None:
        for sample in samples:
            if len(self.data) >= self.capacity:
                self.data.pop(0)
            self.data.append(sample)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if len(self.data) == 0:
            return None
        if len(self.data) < batch_size:
            indices = random.choices(range(len(self.data)), k=batch_size)
        else:
            indices = random.sample(range(len(self.data)), batch_size)
        states, policies, values = zip(*[self.data[i] for i in indices])
        return (
            torch.stack(states),
            torch.stack(policies),
            torch.stack(values).view(-1),
        )

    def __len__(self) -> int:
        return len(self.data)


def self_play(
    model: OtrioNet, num_simulations: int = 50, num_players: int = 2
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """MCTS を用いた 1 局の自己対戦を行い、学習データを返す"""
    state = GameState(num_players=num_players)
    mcts = MCTS(lambda s: policy_value(model, s), num_simulations=num_simulations)
    history: List[Tuple[GameState, torch.Tensor]] = []

    while not state.winner and not state.draw:
        move, visits = mcts.run(state)
        total = sum(visits.values())
        policy = torch.zeros(27, dtype=torch.float32)
        if total > 0:
            for m, v in visits.items():
                idx = m.size * 9 + m.row * 3 + m.col
                policy[idx] = v / total
        history.append((state.clone(), policy))
        state.apply_move(move)

    samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for s, p in history:
        if state.draw:
            value = torch.tensor(0.0)
        elif s.current_player == state.winner:
            value = torch.tensor(1.0)
        else:
            value = torch.tensor(-1.0)
        samples.append((state_to_tensor(s), p, value))
    return samples


def train_step(
    model: OtrioNet,
    optimizer: optim.Optimizer,
    buffer: ReplayBuffer,
    batch_size: int,
) -> float:
    model.train()
    sample = buffer.sample(batch_size)
    if sample is None:
        return 0.0
    states, policies, values = sample
    optimizer.zero_grad()
    policy_logits, value_pred = model(states)
    loss = loss_fn(policy_logits, value_pred, policies, values)
    loss.backward()
    optimizer.step()
    return loss.item()
