## 1 ️⃣ ゲームモデル仕様（実装指針）

| 項目                   | 内容                                                                                                                                                                                             |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **盤面**               | 3 × 3 セル (=9)、各セルに **サイズスロット 3 段**（0 = 小, 1 = 中, 2 = 大）                                                                                                                                        |
| **プレイヤー／色**          | 4 色固定 (0–3)。<br> • 4 人戦 … 各色＝各プレイヤー<br> • 3 人戦 … 1 色捨て、残り 3 色＝各プレイヤー<br> • 2 人戦 … **1 人が 2 色ずつ担当**（手番は ‑A→‑B→‑A… と色ごとに回る ← 実装的には *色＝エージェント* として 4 人戦と同型にしておくとラク）([gameon.cafe][1], [Otrio][2]) |
| **持ち駒**              | 各色につき **小/中/大 ×3 = 9 個**                                                                                                                                                                       |
| **行動 (Action)**      | `action_id = size * 9 + cell` → 範囲 **0‑26**<br> • 置ける条件: そのサイズスロットが未使用<br> • 半手パス（打つ場所ゼロのとき）を含めるなら `27` 番を用意                                                                                   |
| **状態 (Observation)** | 例①: `board[4][3][3][3]` (色チャンネル) の 0/1 バイナリ + `current_color` (1 hot)<br>例②: Flatten 27 長の int 配列 (‑1=空, 0‑3=色)<br>お好みで                                                                        |
| **終端判定**             | <u>勝利条件 3 種</u>（色は同一）([vat19.com][3])<br>① 同サイズ 3 連（横縦斜め 8 ライン）<br>② サイズ順 3 連（昇順 or 降順）<br>③ 同一セル三重 (Bullseye)<br>— いずれか成立で **勝者 = 置いた色**<br>— 全駒置き切り & 勝者なし → ドロー                             |
| **報酬設計**             | 強化学習なら: 勝者 +1、敗者 ‑1、ドロー 0 (多人数時は非勝者 0 or ‑1 はお好み)                                                                                                                                              |

---

## 2 ️⃣ Python 3 × Gymnasium スケルトン

```python
# otrio_env.py
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple

BOARD_CELLS = 9            # 3x3
SLOTS = 3                  # small, medium, large
MAX_ACTIONS = BOARD_CELLS * SLOTS  # = 27
WIN_LINES = [
    (0,1,2), (3,4,5), (6,7,8),      # rows
    (0,3,6), (1,4,7), (2,5,8),      # cols
    (0,4,8), (2,4,6),               # diags
]

class OtrioBase:
    """Pure game logic, no RL glue."""
    def __init__(self, active_colors: List[int]=(0,1,2,3)):
        self.colors      = active_colors
        self.num_colors  = len(self.colors)
        self.reset()

    # ---------- core API ----------
    def reset(self):
        # -1 = empty, 0‑3 = color id
        self.board = np.full((SLOTS, BOARD_CELLS), fill_value=-1, dtype=np.int8)
        self.stash = {c: {s:3 for s in range(SLOTS)} for c in self.colors}
        self.turn  = 0  # index in self.colors
        self.winner: int|None = None
        return self.observe()

    def observe(self) -> np.ndarray:
        # 4x3x3x3 binary planes (color, slot, row, col)
        planes = np.zeros((4, SLOTS, 3, 3), dtype=np.int8)
        for slot in range(SLOTS):
            for cell in range(BOARD_CELLS):
                col = self.board[slot, cell]
                if col != -1:
                    planes[col, slot, cell//3, cell%3] = 1
        return planes

    def legal_moves(self) -> List[int]:
        color = self.colors[self.turn]
        moves = []
        for slot in range(SLOTS):
            if self.stash[color][slot] == 0:   # 在庫切れ
                continue
            for cell in range(BOARD_CELLS):
                if self.board[slot, cell] == -1:
                    moves.append(slot * BOARD_CELLS + cell)
        return moves

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, dict]:
        if self.winner is not None:
            raise RuntimeError("Game over.")

        color = self.colors[self.turn]
        slot, cell = divmod(action, BOARD_CELLS)

        # バリデーション
        if self.board[slot, cell] != -1:
            raise ValueError("Illegal move: occupied slot.")
        if self.stash[color][slot] <= 0:
            raise ValueError("Illegal move: no pieces left.")

        # 反映
        self.board[slot, cell] = color
        self.stash[color][slot] -= 1

        # 勝敗チェック
        self._check_win(color, slot, cell)

        done = self.winner is not None or all(
            self.board[s].min() != -1 for s in range(SLOTS)
        )
        reward = 0
        if done:
            if self.winner is not None:
                reward = 1  # 勝者側
            # 敗者は負の報酬 or 0 は上位環境で付与する

        # 手番更新
        self.turn = (self.turn + 1) % self.num_colors
        return self.observe(), reward, done, {}

    # ---------- helpers ----------
    def _check_win(self, color: int, slot: int, cell: int):
        # ① 同サイズ 3 連
        for line in WIN_LINES:
            if cell not in line:
                continue
            if all(self.board[slot, c] == color for c in line):
                self.winner = color
                return

        # ② サイズ順 3 連 (昇順 / 降順)
        for line in WIN_LINES:
            if cell not in line:
                continue
            seq = [self.board[s, line[i]] for i,s in enumerate(range(2,-1,-1))]
            if seq == [color]*3:
                self.winner = color
                return
            seq = [self.board[s, line[i]] for i,s in enumerate(range(3))]
            if seq == [color]*3:
                self.winner = color
                return

        # ③ Bullseye
        row, col = divmod(cell, 3)
        if all(self.board[s, cell] == color for s in range(SLOTS)):
            self.winner = color

# ---------- Gym wrapper ----------
class OtrioEnv(gym.Env):
    """Gymnasium-compatible multi‑agent wrapper (color = agent)."""
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, players:int=4):
        assert players in (2,3,4)
        active = list(range(players if players!=2 else 4))  # 2人＝4色運用
        self.core = OtrioBase(active_colors=active)

        self.action_space  = spaces.Discrete(MAX_ACTIONS)
        # simple flat int observation (27 slots)
        self.observation_space = spaces.Box(low=-1, high=3,
                                            shape=(SLOTS*BOARD_CELLS,),
                                            dtype=np.int8)

    def reset(self, seed=None, options=None):
        obs = self.core.reset()
        return obs.flatten(), {}

    def step(self, action):
        obs, reward, done, _ = self.core.step(action)
        info = {"winner": self.core.winner}
        return obs.flatten(), reward, done, False, info

    def render(self, mode="ansi"):
        grid = [["   "]*3 for _ in range(3)]
        sym = { -1:"   ", 0:"R", 1:"G", 2:"B", 3:"Y" }
        for slot,size_char in zip(range(SLOTS),("s","m","L")):
            for cell in range(BOARD_CELLS):
                c = self.core.board[slot,cell]
                if c != -1:
                    r,cx = divmod(cell,3)
                    grid[r][cx] = grid[r][cx].replace(" ", sym[c]+size_char, 1)
        out = "\n".join(" | ".join(row) for row in grid)
        if mode == "ansi":
            print(out)
        return out
```

> **使い方一例**
>
> ```python
> env = OtrioEnv(players=4)         # 2/3/4 いずれも OK
> obs, _ = env.reset()
> while True:
>     a = env.action_space.sample() # ランダム手
>     obs, r, done, _, info = env.step(a)
>     if done:
>         print("winner:", info["winner"], "reward:", r)
>         break
> ```
>
> これで “まず動く環境” は確保。AlphaZero 風の MCTS + ニューラルネットを噛ませば自己対戦学習に移行できます✌️

---

## 3 ️⃣ 実装／学習 Tips

1. **アクションマスクを NN へ渡す**
      27 手中の非合法ムーブを −inf にしてソフトマックスさせると効率が激変。
2. **シンメトリ・データ拡張**
      Otrio は 90° 回転・左右反転で等価局面が 8 倍。学習時に random flip / rotate をかけて overfit 防止。
3. **2 人戦の取り扱い**
      ● 最速: “色=エージェント” として 4 者同卓 → 同じ NN weights を共有しつつ **先手/後手/同色自己衝突** を全部経験させる。
      ● 厳密: プレイヤー単位で 2 エージェントにまとめ、着手ロジック側で A⇔B 色交代させる。
4. **報酬スケール**
      スパース (+1/‑1) で十分だが、序中盤価値を NN に教えたければ<br> • 末尾だけスパース<br> • 途中で勝勢評価 (±0.5 など) をリーク<br> のどちらかを混ぜると収束が早い。
5. **盤面エンコーディング**
      CNN を使うなら `shape=(C=12, H=3, W=3)` （色4 ×サイズ3）で入力し、3 × 3 Conv を回すと軽量。
