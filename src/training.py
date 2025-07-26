import random
from typing import List, Tuple

import torch
from torch import optim

from .mcts import MCTS
from concurrent.futures import ThreadPoolExecutor
from .network import OtrioNet, policy_value, state_to_tensor, loss_fn
from .augmentation import augment_samples
from .otrio import GameState, Player, Move


class ReplayBuffer:
    def __init__(self, capacity: int = 10000, device: str | torch.device | None = None) -> None:
        self.capacity = capacity
        self.device = torch.device(device or "cpu")
        self.data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def add(self, samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> None:
        augmented = augment_samples(samples)
        for state, policy, value in augmented:
            if len(self.data) >= self.capacity:
                self.data.pop(0)
            self.data.append(
                (
                    state.to(self.device, non_blocking=True),
                    policy.to(self.device, non_blocking=True),
                    value.to(self.device, non_blocking=True),
                )
            )

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

    def save(self, path: str) -> None:
        """ReplayBuffer をファイルに保存する"""
        cpu_data = [
            (
                s.to("cpu"),
                p.to("cpu"),
                v.to("cpu"),
            )
            for s, p, v in self.data
        ]
        torch.save(cpu_data, path)

    def load(self, path: str) -> None:
        """ファイルから ReplayBuffer を読み込む"""
        raw = torch.load(path, map_location=torch.device("cpu"))
        self.data = [
            (
                s.to(self.device),
                p.to(self.device),
                v.to(self.device),
            )
            for s, p, v in raw
        ]


def self_play(
    model: OtrioNet, num_simulations: int = 100, num_players: int = 2
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


def self_play_parallel(
    model: OtrioNet,
    num_games: int,
    num_simulations: int = 100,
    num_players: int = 2,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """複数ゲームを並列に自己対戦しデータを生成"""
    results: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def worker(_: int):
        return self_play(model, num_simulations=num_simulations, num_players=num_players)

    with ThreadPoolExecutor() as ex:
        for data in ex.map(worker, range(num_games)):
            results.extend(data)
    return results


def train_step(
    model: OtrioNet,
    optimizer: optim.Optimizer,
    buffer: ReplayBuffer,
    batch_size: int,
) -> float:
    model.train()
    device = next(model.parameters()).device
    sample = buffer.sample(batch_size)
    if sample is None:
        return 0.0
    states, policies, values = sample
    states = states.to(device)
    policies = policies.to(device)
    values = values.to(device)
    optimizer.zero_grad()
    policy_logits, value_pred = model(states)
    loss = loss_fn(policy_logits, value_pred, policies, values)
    loss.backward()
    optimizer.step()
    return loss.item()


def save_training_state(
    model: OtrioNet,
    optimizer: optim.Optimizer,
    buffer: ReplayBuffer,
    path: str,
) -> None:
    """モデル・オプティマイザ・リプレイバッファをまとめて保存する"""
    cpu_buffer = [
        (s.to("cpu"), p.to("cpu"), v.to("cpu")) for s, p, v in buffer.data
    ]
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "buffer": cpu_buffer,
        },
        path,
    )


def load_training_state(
    path: str,
    num_players: int = 2,
    learning_rate: float = 1e-3,
    buffer_capacity: int = 10000,
    num_blocks: int = 0,
    channels: int = 128,
    device: str | torch.device | None = None,
) -> Tuple[OtrioNet, optim.Optimizer, ReplayBuffer]:
    """保存された学習状態を読み込む"""
    data = torch.load(path, map_location=torch.device("cpu"))
    model = OtrioNet(num_players=num_players, num_blocks=num_blocks, channels=channels)
    model.load_state_dict(data.get("model", {}))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if "optimizer" in data:
        optimizer.load_state_dict(data["optimizer"])
    buffer = ReplayBuffer(buffer_capacity, device=device)
    raw = data.get("buffer", [])
    buffer.data = [
        (s.to(buffer.device), p.to(buffer.device), v.to(buffer.device))
        for s, p, v in raw
    ]
    return model, optimizer, buffer
