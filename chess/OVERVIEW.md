# Connect 4 AI — 操作說明

一個會玩**四子棋**的 AI,可以在網頁上跟它對弈。本文說明如何開始訓練、如何開始玩,以及畫面上有哪些功能。

(安裝 Python 環境的步驟見 [README.md](README.md)。以下每個指令前,都要先啟用環境:`source venv/bin/activate`)

---

## 如何開始訓練

```bash
python -m alphazero.train
```

- 會自動從目前最強的模型**接著訓練**(不會從零開始)。
- 訓練成果存在 `checkpoints/` 資料夾,`best.pt` 是目前最強的版本。
- 想從零重新訓練:加上 `--fresh`。
- 想訓練久一點 / 快一點:

```bash
python -m alphazero.train --iterations 80     # 練更多輪
python -m alphazero.train --games 50 --sims 100   # 每輪更快
```

不訓練也沒關係——專案已內附訓練好的模型,可以直接玩。

---

## 如何開始玩

```bash
python webplay.py
```

然後在瀏覽器打開 **http://127.0.0.1:8000**。

要停止:在終端機按 `Ctrl+C`。

---

## 畫面功能

![遊戲畫面](docs/screenshot-midgame.png)

- **點欄位落子**:用滑鼠點任一欄,棋子會掉到底;最新落下的棋子會發光標示。
- **先手**(左上):隨機 / 我先 / AI 先。
- **難度**:簡單 / 普通 / 困難 / 超難——數字越大,AI 想得越久、越強。
- **顯示分析**(勾選框):開啟後,輪到你時棋盤上方會顯示——
  - 你目前的**勝率估計**
  - AI 對**每一欄的評分**長條(它最推薦的那欄會高亮)
- **新遊戲**(紅色按鈕):依目前的「先手 / 難度」設定開始新的一局。

開局時開啟分析,AI 會推薦下正中央:

![開局分析](docs/screenshot-analysis.png)
