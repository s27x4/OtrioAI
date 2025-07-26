from dataclasses import dataclass, fields
from typing import Any, Dict
import yaml

@dataclass
class Config:
    learning_rate: float = 1e-3
    buffer_capacity: int = 10000
    num_simulations: int = 100
    batch_size: int = 128
    num_players: int = 2
    num_blocks: int = 2
    channels: int = 128
    parallel_games: int = 1
    max_moves: int | None = None
    resign_threshold: float | None = None


def load_config(path: str = "config.yaml") -> Config:
    """config.yaml を読み込んで ``Config`` を生成する。

    当初のフラットなキーに加え、セクション分けされた構造のファイルも
    受け付けるようにしている。
    """

    try:
        with open(path, "r") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}
    except FileNotFoundError:
        data = {}

    # nested セクション形式の場合はフラットに変換
    if any(k in data for k in ["model", "training", "mcts", "self_play", "game", "logging"]):
        flat: Dict[str, Any] = {}
        for section in ["model", "training", "mcts", "self_play", "game", "logging"]:
            section_dict = data.get(section, {})
            if isinstance(section_dict, dict):
                flat.update(section_dict)
        data = flat

    cfg_fields = {f.name for f in fields(Config)}
    filtered = {k: v for k, v in data.items() if k in cfg_fields}
    cfg_dict = {**Config().__dict__, **filtered}
    return Config(**cfg_dict)
