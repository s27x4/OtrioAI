# ニューラルネットワーク（Policy + Value）

- [ ] 入力形式の定義（盤面を多チャネル Tensor に変換）
- [ ] 出力: 
  - policy：合法手の確率分布（サイズ = 行動数）
  - value：現在の状態の勝率推定（-1〜1）
- [ ] ネットワーク構成（PyTorch や TensorFlow で実装）
- [ ] 損失関数の定義：
  - policy のクロスエントロピー
  - value の平均二乗誤差（MSE）
- [ ] Optimizer・学習率の設定
- [ ] モデルの保存・読み込み処理
