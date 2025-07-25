# ニューラルネットワーク（Policy + Value）

- [x] 入力形式の定義（盤面を多チャネル Tensor に変換）
- [x] 出力:
  - policy：合法手の確率分布（サイズ = 行動数）
  - value：現在の状態の勝率推定（-1〜1）
- [x] ネットワーク構成（PyTorch や TensorFlow で実装）
- [x] 損失関数の定義：
  - policy のクロスエントロピー
  - value の平均二乗誤差（MSE）
- [x] Optimizer・学習率の設定
- [x] モデルの保存・読み込み処理
