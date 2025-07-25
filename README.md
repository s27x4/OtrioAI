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
CLI は `src/cli.py` にあり、自己対戦や学習ステップを実行できます。`src/otrio.py` には簡易的なゲーム環境 `GameState` も用意しています。
以下は `GameState` の利用例です。

```python
from src.otrio import GameState

state = GameState()
moves = state.legal_moves()
state.apply_move(moves[0])
print(state.log())
```

これで一手分の盤面遷移を試すことができます。

### CLI の基本的な使い方

`src/cli.py` は Python のモジュールとして実行できます。以下のように起動すると
利用可能なオプション一覧が表示されます。

```bash
python -m src.cli --help
```

主なオプションは次の通りです。

- `--self-play` : 自己対戦を 1 局だけ行い、結果をリプレイバッファに追加します。
- `--train` : リプレイバッファを用いて 1 ステップだけ学習します。
- `--train-loop N` : 自己対戦と学習を N 回繰り返します。
- `--train-gui N` : GUI を表示しながら N 回学習を実行します。
- `--load-buffer PATH` : 既存のリプレイバッファを読み込むファイルを指定します。
- `--load-state PATH` : 保存済みのモデルやオプティマイザ状態を読み込みます。
- `--save-state PATH` : 学習後のモデル・バッファ・オプティマイザ状態を保存します。

例えば 5 回学習を回し、その結果を保存する場合は次のように実行します。

```bash
python -m src.cli --train-loop 5 --save-state checkpoint.pt
```

### モデルの保存と学習再開

`src/cli.py` では学習状態の保存・読み込みも行えます。
例えば以下のように実行することで学習後のモデルとリプレイバッファ、
オプティマイザ状態をまとめて保存できます。

```bash
python -m src.cli --train-loop 10 --save-state checkpoint.pt
```

保存したファイルは `--load-state` オプションで読み込めるため、
途中から学習を再開することが可能です。

### GUI を使った学習進捗の表示

`src/gui.py` では学習ループ中の損失をグラフ表示する簡易 GUI を提供しています。
以下のように実行することで、学習状況をリアルタイムで確認できます。

```bash
python -m src.gui 10  # 10 回学習を実行しながらグラフ表示
```

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
