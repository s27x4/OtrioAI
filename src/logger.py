import json
from typing import Any, List
import matplotlib.pyplot as plt


def log_metrics(path: str, step: int, loss: float, win_rate: float) -> None:
    entry = {"step": step, "loss": loss, "win_rate": win_rate}
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def plot_metrics(path: str, output: str) -> None:
    steps: List[int] = []
    losses: List[float] = []
    win_rates: List[float] = []
    with open(path, "r") as f:
        for line in f:
            e = json.loads(line)
            steps.append(e["step"])
            losses.append(e["loss"])
            win_rates.append(e["win_rate"])
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(steps, losses)
    plt.title("Loss")
    plt.subplot(2, 1, 2)
    plt.plot(steps, win_rates)
    plt.title("Win Rate")
    plt.tight_layout()
    plt.savefig(output)
