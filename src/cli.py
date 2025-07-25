import argparse
from src.config import load_config
from src.training import self_play, ReplayBuffer, train_step
from src.network import OtrioNet, create_optimizer


def main() -> None:
    parser = argparse.ArgumentParser(description="OtrioAI CLI")
    parser.add_argument("--self-play", action="store_true", help="自己対戦を1局実行")
    parser.add_argument("--train", action="store_true", help="1ステップ学習を実行")
    parser.add_argument(
        "--train-loop",
        type=int,
        default=None,
        metavar="N",
        help="自己対戦と学習を N 回繰り返す",
    )
    parser.add_argument(
        "--train-gui",
        type=int,
        default=None,
        metavar="N",
        help="GUI を表示しつつ N 回学習を行う",
    )
    parser.add_argument(
        "--load-buffer",
        type=str,
        default=None,
        help="リプレイバッファを読み込むファイルパス",
    )
    args = parser.parse_args()

    cfg = load_config()
    model = OtrioNet(num_players=cfg.num_players)
    optimizer = create_optimizer(model, lr=cfg.learning_rate)
    buffer = ReplayBuffer(cfg.buffer_capacity)
    if args.load_buffer:
        buffer.load(args.load_buffer)

    if args.self_play:
        data = self_play(model, num_simulations=cfg.num_simulations, num_players=cfg.num_players)
        buffer.add(data)
        print(f"{len(data)} 件のサンプルを生成しました")
    if args.train:
        if len(buffer) == 0:
            buffer.add(self_play(model, num_simulations=cfg.num_simulations, num_players=cfg.num_players))
        loss = train_step(model, optimizer, buffer, cfg.batch_size)
        print(f"loss={loss:.4f}")
    if args.train_loop:
        total_loss = 0.0
        for i in range(args.train_loop):
            print(f"{i+1}/{args.train_loop} 回目の学習")
            data = self_play(model, num_simulations=cfg.num_simulations, num_players=cfg.num_players)
            buffer.add(data)
            loss = train_step(model, optimizer, buffer, cfg.batch_size)
            total_loss += loss
            print(f"loss={loss:.4f}")
        avg_loss = total_loss / args.train_loop
        print(f"平均損失: {avg_loss:.4f}")
    if args.train_gui:
        from .gui import train_gui_loop

        train_gui_loop(args.train_gui, cfg)


if __name__ == "__main__":
    main()
