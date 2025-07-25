import argparse
from typing import List
import matplotlib.pyplot as plt

from .config import load_config, Config
from .training import self_play, ReplayBuffer, train_step
from .network import OtrioNet, create_optimizer


def train_gui_loop(
    num_iterations: int,
    cfg: Config | None = None,
    headless: bool = False,
) -> List[float]:
    """GUIで学習進捗を表示しながら学習を実行する"""
    if cfg is None:
        cfg = load_config()
    if headless:
        import matplotlib
        matplotlib.use("Agg")

    model = OtrioNet(num_players=cfg.num_players)
    optimizer = create_optimizer(model, lr=cfg.learning_rate)
    buffer = ReplayBuffer(cfg.buffer_capacity)
    losses: List[float] = []

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], label="loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Progress")
    ax.legend()

    for i in range(num_iterations):
        data = self_play(
            model, num_simulations=cfg.num_simulations, num_players=cfg.num_players
        )
        buffer.add(data)
        loss = train_step(model, optimizer, buffer, cfg.batch_size)
        losses.append(loss)

        line.set_data(range(1, len(losses) + 1), losses)
        ax.set_xlim(1, len(losses))
        ymin, ymax = min(losses), max(losses)
        if ymin == ymax:
            ymax = ymin + 1e-3
        ax.set_ylim(ymin, ymax)
        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.ioff()
    if not headless:
        plt.show()
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description="GUI付き学習ループ")
    parser.add_argument(
        "N", type=int, help="学習ループ回数")
    args = parser.parse_args()
    train_gui_loop(args.N)


if __name__ == "__main__":
    main()
