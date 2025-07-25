from dataclasses import dataclass
from typing import Any, Dict
import yaml

@dataclass
class Config:
    learning_rate: float = 1e-3
    buffer_capacity: int = 10000
    num_simulations: int = 50
    batch_size: int = 32
    num_players: int = 2


def load_config(path: str = "config.yaml") -> Config:
    try:
        with open(path, "r") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}
    except FileNotFoundError:
        data = {}
    cfg_dict = {**Config().__dict__, **data}
    return Config(**cfg_dict)
