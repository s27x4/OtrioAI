import torch
import torch.nn as nn
import torch.nn.functional as F
from .otrio import GameState, Player, Move
from typing import Dict, Tuple


def state_to_tensor(state: GameState) -> torch.Tensor:
    """GameState を多チャネル Tensor に変換する.

    チャネル構成:
      0-2: PLAYER1 の size0-2
      3-5: PLAYER2 の size0-2
      6  : 現在の手番 (全マス同一値)
    """
    tensor = torch.zeros(7, 3, 3, dtype=torch.float32)
    for size in range(3):
        for r in range(3):
            for c in range(3):
                p = state.board[size][r][c]
                if p == Player.PLAYER1:
                    tensor[size, r, c] = 1.0
                elif p == Player.PLAYER2:
                    tensor[size + 3, r, c] = 1.0
    if state.current_player == Player.PLAYER1:
        tensor[6].fill_(1.0)
    elif state.current_player == Player.PLAYER2:
        tensor[6].fill_(0.0)
    else:
        tensor[6].fill_(0.5)
    return tensor


class OtrioNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(7, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 3 * 3, 27),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value.squeeze(-1)


def policy_value(model: OtrioNet, state: GameState) -> Tuple[Dict[Move, float], float]:
    model.eval()
    with torch.no_grad():
        tensor = state_to_tensor(state).unsqueeze(0)
        policy_logits, value = model(tensor)
        policy = torch.softmax(policy_logits[0], dim=0)
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
) -> torch.Tensor:
    """policy は確率分布を期待する"""
    log_p = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(target_policy * log_p).sum(dim=1).mean()
    value_loss = F.mse_loss(value, target_value)
    return policy_loss + value_loss


def save_model(model: OtrioNet, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_model(path: str) -> OtrioNet:
    model = OtrioNet()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model


def create_optimizer(model: OtrioNet, lr: float = 1e-3) -> torch.optim.Optimizer:
    """学習用 Optimizer を作成する"""
    return torch.optim.Adam(model.parameters(), lr=lr)


def to_device(model: OtrioNet, device: str | None = None) -> torch.device:
    """モデルを指定デバイスへ移動"""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    return dev
