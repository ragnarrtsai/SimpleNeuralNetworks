# YouBike 剩餘車數預測

給定一個 YouBike 站點 (例如「捷運科技大樓站」) 現在的車輛狀況，**30 分鐘後還會剩幾輛車**？這個模型用過去 1 小時的車數變化 + 時間/星期 + 站點位置就能估出來。

## 成果展示

> 訓練完成後執行 `python demo.py --weights latest.pt --run-id <RUN_ID>` 產生下方圖表，然後 `cp output/<RUN_ID>/showcase/*.png assets/`。

### 預測 vs 實際

每個點是一筆 test 樣本，落在對角線 `y = x` 上代表預測完全準確。

![predicted vs actual](assets/pred_vs_actual.png)

### 訓練曲線

![training curve](assets/training_curve.png)

### 一站的時序預測

挑一個熱門站，把模型對「30 分鐘後」的預測疊在實際曲線上：

![sample station](assets/sample_station.png)

### 一天中哪個時段最難預測

![MAE by hour](assets/mae_by_hour.png)
