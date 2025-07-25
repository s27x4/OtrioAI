import argparse
from .config import load_config
from .training import self_play, ReplayBuffer, train_step
from .network import OtrioNet, create_optimizer


def main() -> None:
    parser = argparse.ArgumentParser(description="OtrioAI CLI")
    parser.add_argument("--self-play", action="store_true", help="自己対戦を1局実行")
    parser.add_argument("--train", action="store_true", help="1ステップ学習を実行")
    args = parser.parse_args()

    cfg = load_config()
    model = OtrioNet()
    optimizer = create_optimizer(model, lr=cfg.learning_rate)
    buffer = ReplayBuffer(cfg.buffer_capacity)

    if args.self_play:
        data = self_play(model, num_simulations=cfg.num_simulations)
        buffer.add(data)
        print(f"{len(data)} 件のサンプルを生成しました")
    if args.train:
        if len(buffer) == 0:
            buffer.add(self_play(model, num_simulations=cfg.num_simulations))
        loss = train_step(model, optimizer, buffer, cfg.batch_size)
        print(f"loss={loss:.4f}")


if __name__ == "__main__":
    main()
