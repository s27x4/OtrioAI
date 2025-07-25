# OtrioAI

## プロジェクト概要
OtrioAI はボードゲーム **Otrio** を AI 学習させるための実験プロジェクトです。AlphaZero 方式を参考に、
ゲーム環境・MCTS・ニューラルネットワーク・自己対戦による学習ループの構築を目指しています。

## 依存ライブラリ
ゲーム環境や MCTS は Python 標準ライブラリで動作しますが、
ニューラルネットワーク実装には PyTorch を利用しています。
`requirements.txt` に記載のライブラリをインストールしてください。
推奨 Python バージョンは 3.10 以上です。

## 実行方法
まだ CLI や学習スクリプトはありませんが、`src/otrio.py` に簡易的なゲーム環境 `GameState` を用意しています。
以下は利用例です。

```python
from src.otrio import GameState

state = GameState()
moves = state.legal_moves()
state.apply_move(moves[0])
print(state.log())
```

これで一手分の盤面遷移を試すことができます。

## Otrio ルール概要と実装状況
Otrio では次のような勝利条件があります。

1. 同じサイズの自分の駒を縦・横・斜めに 3 つ並べる
2. 1 マスに自分の駒 3 サイズを積み上げる
3. 大中小を順番に並べる（昇順・降順）

`GameState` では上記 3 つの勝利条件をすべて実装済みです。

## 開発者向け情報
開発タスクは `tasks/` 以下の Markdown に整理してあります。MCTS やニューラルネットワーク、
学習ループなどは今後実装していく予定です。

テストは `pytest` を利用する想定です。テスト追加後は以下を実行してください。

```bash
pip install -r requirements.txt  # 依存ライブラリ追加時
pytest
```
