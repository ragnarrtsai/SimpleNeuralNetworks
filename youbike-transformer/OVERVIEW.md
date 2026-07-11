# YouBike 剩餘車數預測

給定一個 YouBike 站點 (例如「捷運科技大樓站」) 現在的車輛狀況，**30 分鐘後還會剩幾輛車**？這個小型 Transformer 用過去 1 小時的車數變化 + 時間/星期 + 站點特徵就能估出來。

## 成效（目前最佳 v4）

| 指標 | 數值 |
|------|------|
| 平均誤差 MAE | **3.99 台**（勝「維持現狀」基準 7.4%） |
| 單筆預測延遲 | **~319 µs**（MacBook Pro M4，純 CPU） |
| 模型大小 | 352 KB（87,105 參數） |
| 測試集 | live 最後 7 天（2026-06-16 → 06-22），1,316 站 |

> 從 v1 到 v4 的版本演進、overfit 與準確度分析見 [`VERSIONS.md`](VERSIONS.md) 與 README 附錄。

## 另一個用途：用「預測失準」偵測異常事件

模型學會了「正常規律」，所以它**大幅失準時，往往正是當下有不尋常人潮**。實測中它成功標記出 **大稻埕碼頭在端午連假（2026-06-19）** 傍晚的人潮暴衝——單車被借光又還爆，模型整晚跟不上。這就是這個模型的核心價值：**用預測誤差反推當下發生了什麼。**

![大稻埕 06-19 絕對數量 vs 換回變化量](assets/anomaly_dadaocheng.png)

> 上半是絕對可借數量（預測黏著水位、難讀差異）；下半把數量換回「30 分鐘變化量」後，實際大起大落、模型幾乎貼著 0——可見模型近似「維持現狀」，真正指向事件的是**殘差**而非預測曲線本身。

## 成果展示

### 預測 vs 實際

每個點是一筆 test 樣本，落在對角線 `y = x` 上代表預測完全準確。

![predicted vs actual](assets/appendix_pred_vs_actual.png)

### 訓練曲線（train vs test MAE）

![training curve](assets/appendix_training_curve.png)

### 一站的時序預測

挑一個熱門站，把模型對「30 分鐘後」的預測疊在實際曲線上：

![sample station](assets/appendix_sample_station.png)

### 一天中哪個時段最難預測

![MAE by hour](assets/appendix_mae_by_hour.png)
