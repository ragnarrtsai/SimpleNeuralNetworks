# 版本紀錄 — youbike-transformer

> 每個 run 一筆，資料源為 [`versions.json`](versions.json)（由 `log_version.py` 產生）。
> 測試集固定為 live 最後 7 天（2026-06-16 → 06-22）。所有時間量測於 **MacBook Pro M4 (MPS)**。

## 版本比較

| 版本 | 日期 | 主要變更 | Test MAE | vs baseline | Overfit gap | 命中±1 | Δ幅度保留 | r | 訓練時間 |
|------|------|----------|---------:|------------:|------------:|-------:|----------:|--:|---------:|
| **v1** | 2026-07-05 | 基準：歷史+live、最後7天測試、residual/Huber、dropout 0.1 | 4.136 | −3.9% | 0.430 | 19.8% | 30% | 0.28 | 220s |
| **v2** | 2026-07-11 | +weight_decay 1e-4、dropout 0.2、early-stop（保留最佳 checkpoint） | 4.142 | −3.7% | 0.417 | 20.1% | 27% | 0.27 | 223s |
| **v3** | 2026-07-11 | +station embedding(16維) + 「站×星期×時段」歷史平均變化量特徵 | 4.209 | −2.2% | 1.003 | 19.1% | 50% | 0.32 | 253s |
| **v4** | 2026-07-11 | 站點嵌入降維16→8 + head dropout 0.3 + weight decay 1e-4 + 歷史基準改用2026近期資料計算 | 3.985 | −7.4% | 0.474 | 20.4% | 32% | 0.39 | 258s |

> baseline（persistence，預測不變）命中±1 = 29.2%，Test MAE = 4.302。

## 各版本細節

### v1 — `20260705_223205`（2026-07-05 22:32）
- **變更**：基準：歷史+live、最後7天測試、residual/Huber、dropout 0.1
- **目的/備註**：首個以「歷史封存+自爬 live、last-7-days 為測試集」的正式 run，作為後續版本的比較基準。
- **準確度**：Test MAE 4.136（勝 baseline 3.9%）、RMSE 6.083、overfit gap 0.430；命中 ±1/±2/±5 = 19.8% / 38.1% / 72.4%
- **進步方向指標**：Δ幅度保留 30%（越高=越不縮向均值）、r = 0.276（越高=越知道哪筆會動）
- **訓練時間**：總 220s（載入 47+4s、訓練 168s，8.4s/epoch）；參數 72,449；單筆推論 315µs
- **架構**：Transformer encoder ×2 (d_model=64, nhead=4, ff=128, dropout=0.1), seq_len=7, residual/delta 目標
- **保存**：權重 `model/20260705_223205.pt`（294.2 KB）；架構快照 `output/20260705_223205/snapshot/model.py`、`output/20260705_223205/snapshot/data.py`、`output/20260705_223205/snapshot/train_hist_live.py`

### v2 — `20260711_103756`（2026-07-11 10:37）
- **變更**：+weight_decay 1e-4、dropout 0.2、early-stop（保留最佳 checkpoint）
- **目的/備註**：抗 overfit 對照實驗。結論：三帖藥皆雜訊級變動，正則化無實質改善，模型已逼近此特徵集下的資料雜訊天花板。
- **準確度**：Test MAE 4.142（勝 baseline 3.7%）、RMSE 6.087、overfit gap 0.417；命中 ±1/±2/±5 = 20.1% / 38.1% / 72.4%
- **進步方向指標**：Δ幅度保留 27%（越高=越不縮向均值）、r = 0.271（越高=越知道哪筆會動）
- **訓練時間**：總 223s（載入 48+4s、訓練 170s，8.5s/epoch）；參數 72,449；單筆推論 312µs
- **架構**：Transformer encoder ×2 (d_model=64, nhead=4, ff=128, dropout=0.2), seq_len=7, residual/delta 目標
- **保存**：權重 `model/20260711_103756.pt`（294.2 KB）；架構快照 `output/20260711_103756/snapshot/model.py`、`output/20260711_103756/snapshot/data.py`、`output/20260711_103756/snapshot/train_hist_live_v2.py`

### v3 — `20260711_121252`（2026-07-11 12:12）
- **變更**：+station embedding(16維) + 「站×星期×時段」歷史平均變化量特徵
- **目的/備註**：驗證第06節假設。結果:機制對了(變化量保留率 30→50%、r 0.28→0.32、train MAE 3.71→3.21),但泛化變差(test MAE 4.14→4.21、overfit gap 0.43→1.00),新特徵讓模型記住訓練期(2024-25)各站規律但未遷移到2026測試週。下一步需正則化 embedding + 用貼近測試期的資料算基準。
- **準確度**：Test MAE 4.209（勝 baseline 2.2%）、RMSE 6.094、overfit gap 1.003；命中 ±1/±2/±5 = 19.1% / 36.8% / 71.0%
- **進步方向指標**：Δ幅度保留 50%（越高=越不縮向均值）、r = 0.324（越高=越知道哪筆會動）
- **訓練時間**：總 253s（載入 48+4s、訓練 199s，10.0s/epoch）；參數 101,697；單筆推論 316µs
- **架構**：Transformer encoder ×2 (d_model=64, nhead=4, ff=128, dropout=0.1), seq_len=7, residual/delta 目標
- **保存**：權重 `model/20260711_121252.pt`（408.8 KB）；架構快照 `output/20260711_121252/snapshot/model.py`、`output/20260711_121252/snapshot/data.py`、`output/20260711_121252/snapshot/train_v3.py`

### v4 — `20260711_123300`（2026-07-11 12:33）
- **變更**：站點嵌入降維16→8 + head dropout 0.3 + weight decay 1e-4 + 歷史基準改用2026近期資料計算
- **目的/備註**：依 v3 檢討收斂過擬合。結果:四版最佳——test MAE 3.985(勝baseline 7.4%)、RMSE 5.84、overfit gap 回到受控 0.47、r 0.28→0.39、±5 命中率首度追平 baseline。正則化 + 近期基準特徵同時提升準度與泛化,驗證 v3 方向正確、只是需節制。
- **準確度**：Test MAE 3.985（勝 baseline 7.4%）、RMSE 5.840、overfit gap 0.474；命中 ±1/±2/±5 = 20.4% / 39.4% / 73.7%
- **進步方向指標**：Δ幅度保留 32%（越高=越不縮向均值）、r = 0.390（越高=越知道哪筆會動）
- **訓練時間**：總 258s（載入 49+4s、訓練 204s，10.2s/epoch）；參數 87,105；單筆推論 319µs
- **架構**：Transformer encoder ×2 (d_model=64, nhead=4, ff=128, dropout=0.2), seq_len=7, residual/delta 目標
- **保存**：權重 `model/20260711_123300.pt`（351.8 KB）；架構快照 `output/20260711_123300/snapshot/model.py`、`output/20260711_123300/snapshot/data.py`、`output/20260711_123300/snapshot/train_v4.py`

## 模型與架構保存

- 每個 run 的權重都以唯一 `RUN_ID` 存於 `model/<RUN_ID>.pt`，**不覆蓋、全部保留**。
- 訓練當下的 `model.py` + `data.py` 快照存於 `output/<RUN_ID>/snapshot/`，與該 checkpoint 綁定——即使日後 v3 改了架構/特徵，舊權重仍可用其快照重建載入。
- `model/` 與 `output/` 依 `.gitignore` 為本機保存（未進版控）；如需發布特定版本，依 README 的 release 流程手動處理。

## 下一步方向（未執行）

1. **v3｜station embedding + 「站×星期×小時」歷史基準特徵** — 對症下藥：測試集顯示站點波動與誤差相關 0.958、模型 Δ 幅度只保留 30%、r 僅 0.28，代表缺少站點專屬的循環特徵。
2. **v4｜損失改權重/quantile** — 阻止模型縮向「不變」。
3. **v5｜天氣旗標 + 長程 lag（t−24h / t−7d）** — 補外生訊號與長週期。
