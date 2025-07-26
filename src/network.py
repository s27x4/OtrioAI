import torch
import torch.nn as nn
import torch.nn.functional as F
from .otrio import GameState, Player, Move
from typing import Dict, Tuple


def state_to_tensor(state: GameState) -> torch.Tensor:
    """GameState を多チャネル Tensor に変換する."""

    num_players = state.num_players
    tensor = torch.zeros(num_players * 3 + 1, 3, 3, dtype=torch.float32)

    for size in range(3):
        for r in range(3):
            for c in range(3):
                p = state.board[size][r][c]
                if p != Player.NONE:
                    idx = (p.value - 1) * 3 + size
                    tensor[idx, r, c] = 1.0

    if num_players > 1:
        val = 1.0 - (state.current_player.value - 1) / (num_players - 1)
    else:
        val = 1.0
    tensor[-1].fill_(val)

    return tensor


class ResidualBlock(nn.Module):
    """2層の畳み込みで残差接続を行うブロック"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class OtrioNet(nn.Module):
    def __init__(self, num_players: int = 2, num_blocks: int = 0, channels: int = 128):
        super().__init__()
        self.num_players = num_players
        in_channels = num_players * 3 + 1
        layers = [
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
        ]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(channels))
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 3 * 3, 27),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 3, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value.squeeze(-1)


def policy_value(model: OtrioNet, state: GameState) -> Tuple[Dict[Move, float], float]:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        tensor = state_to_tensor(state).unsqueeze(0).to(device)
        policy_logits, value = model(tensor)
        policy = torch.softmax(policy_logits[0].cpu(), dim=0)
    moves = state.legal_moves()
    if not moves:
        return {}, value.item()
    prob = torch.zeros(27)
    for move, p in zip(moves, policy):
        idx = move.size * 9 + move.row * 3 + move.col
        prob[idx] = p
    move_probs = {move: prob[move.size * 9 + move.row * 3 + move.col].item() for move in moves}
    return move_probs, value.item()


def loss_fn(
    policy_logits: torch.Tensor,
    value: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    value_weight: float = 1.0,
) -> torch.Tensor:
    """policy は確率分布を期待する"""
    log_p = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(target_policy * log_p).sum(dim=1).mean()
    value_loss = F.mse_loss(value, target_value)
    return policy_loss + value_weight * value_loss


def save_model(model: OtrioNet, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_model(path: str, num_players: int = 2, num_blocks: int = 0, channels: int = 128) -> OtrioNet:
    model = OtrioNet(num_players=num_players, num_blocks=num_blocks, channels=channels)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model


def create_optimizer(model: OtrioNet, lr: float = 1e-3) -> torch.optim.Optimizer:
    """学習用 Optimizer を作成する"""
    return torch.optim.Adam(model.parameters(), lr=lr)


def to_device(
    model: OtrioNet,
    device: str | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> torch.device:
    """モデルとオプティマイザを指定デバイスへ移動"""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    if optimizer is not None:
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(dev)
    return dev
