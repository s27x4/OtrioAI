import argparse
from src.config import load_config
from src.training import (
    self_play,
    ReplayBuffer,
    train_step,
    save_training_state,
    load_training_state,
)
from src.network import (
    OtrioNet,
    create_optimizer,
    load_model,
    policy_value,
)
from src.mcts import MCTS
from src.otrio import GameState, Move, Player


def _prompt_move(state: GameState) -> Move:
    """ユーザーから合法手を取得する"""
    while True:
        try:
            text = input("row col size > ")
            r, c, s = map(int, text.strip().split())
            if (
                0 <= r < 3
                and 0 <= c < 3
                and 0 <= s < 3
                and state.board[s][r][c] == Player.NONE
            ):
                return Move(r, c, s, state.current_player)
        except Exception:
            pass
        print("無効な入力です。再入力してください")


def play_vs_model(path: str, cfg) -> None:
    """保存済みモデルと1局対戦する"""
    model = load_model(path, num_players=cfg.num_players)
    state = GameState(num_players=cfg.num_players)
    mcts = MCTS(lambda s: policy_value(model, s), num_simulations=cfg.num_simulations)
    while not state.winner and not state.draw:
        if state.current_player == Player.PLAYER1:
            move = _prompt_move(state)
        else:
            move, _ = mcts.run(state)
            print(f"AI: size={move.size} pos=({move.row},{move.col})")
        state.apply_move(move)
    print(state.log())


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
    parser.add_argument(
        "--load-state",
        type=str,
        default=None,
        help="保存した学習状態を読み込むパス",
    )
    parser.add_argument(
        "--save-state",
        type=str,
        default=None,
        help="学習後の状態を保存するパス",
    )
    parser.add_argument(
        "--play-model",
        type=str,
        default=None,
        metavar="PATH",
        help="指定したモデルと対戦する",
    )
    args = parser.parse_args()

    cfg = load_config()
    if args.load_state:
        model, optimizer, buffer = load_training_state(
            args.load_state,
            num_players=cfg.num_players,
            learning_rate=cfg.learning_rate,
            buffer_capacity=cfg.buffer_capacity,
        )
    else:
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

    if args.play_model:
        play_vs_model(args.play_model, cfg)

    if args.save_state:
        save_training_state(model, optimizer, buffer, args.save_state)


if __name__ == "__main__":
    main()
