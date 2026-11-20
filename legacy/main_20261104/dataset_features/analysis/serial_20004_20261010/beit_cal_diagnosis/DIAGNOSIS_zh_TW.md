# BEiTv2 CAL 新舊結果檢查

## 已確認的結果

這次 serial 20004 與舊 CSV 的相同 W3/G128 設定，在 15 個共同 dataset 上沒有出現大幅新增的 accuracy drop。
舊平均 accuracy 是 53.0267%，新平均是 53.0358%；
平均絕對差 0.1431 個百分點，最大絕對差 0.4967 個百分點。
FP baseline 最大差僅 0.0165 個百分點。
這是已保存輸出的比對，不是重新執行 nano4 推論；接近的 baseline 不能證明 checkpoint 雜湊相同。

| dataset | 舊 FP % | 新 FP % | 舊 W3/G128 % | 新 W3/G128 % | 新減舊 pp |
|---|---:|---:|---:|---:|---:|
| aircraft | 91.15 | 91.15 | 86.71 | 86.74 | +0.029 |
| cars | 92.15 | 92.14 | 43.92 | 43.85 | -0.070 |
| cub | 89.30 | 89.30 | 63.60 | 64.05 | +0.449 |
| dogs | 83.07 | 83.07 | 42.27 | 41.99 | -0.277 |
| flowers | 94.50 | 94.49 | 54.81 | 54.81 | -0.004 |
| food | 92.53 | 92.53 | 89.91 | 89.94 | +0.027 |
| inat17 | 75.35 | 75.35 | 64.17 | 64.24 | +0.067 |
| moe | 96.33 | 96.33 | 95.92 | 95.96 | +0.037 |
| nabirds | 89.79 | 89.79 | 68.25 | 68.29 | +0.045 |
| pets | 95.12 | 95.12 | 61.52 | 61.57 | +0.050 |
| soyageing | 71.05 | 71.05 | 21.68 | 22.08 | +0.401 |
| soygene | 69.54 | 69.52 | 36.46 | 36.50 | +0.038 |
| soyglobal | 41.19 | 41.19 | 11.68 | 11.63 | -0.053 |
| soylocal | 34.17 | 34.17 | 15.33 | 14.83 | -0.497 |
| vegfru | 90.39 | 90.39 | 39.17 | 39.07 | -0.104 |

Cotton 沒有舊結果，不納入共同 dataset 的平均。這次 16 個 dataset 的 BEiTv2 CAL 平均 accuracy 為 50.11%，
平均 relative loss 為 38.58%；不可直接與舊 15 個 dataset 平均比較。

## 使用者提供的圖

已核對該圖 run_config 所指定的 prepared_accuracy_cells.csv，BEiTv2 CAL／iNat17 的 32 個 bit/group cells
全部與本機舊 backbones_quant.csv 一致。該圖 serial 10000 只含 iNat17；完整舊 CSV 的其他 dataset 使用不同 serial。
左圖 x=3、綠線 g=128 是 64.17%，這次為 64.2375%；FP 為舊 75.35%、新 75.3495%。
舊 W3/G8 是 74.05%，W3/G32 是 70.53%，W4/G128 是 73.72%。
紅色虛線是 90% FP baseline（67.815% absolute accuracy），不是 FP baseline 本身。
此圖 y 軸是 absolute accuracy；之前 FP inter-intra 散點圖 y 軸是 100 × W3 accuracy / FP accuracy，不能直接比高度。

## Code 檢查

- tools/features_analysis/w3_g128_analysis.py 強制 weight-only W3/G128、linear scope、activation_bits=None。
- scripts/submit_hqq_w6a4_all_datasets_nano5.sh 另呼叫獨立 W3/G128 feature analysis；W6A4 的設定沒有傳入這段分析。
- FP 及 W3 在相同 model/checkpoint/test loader 上順序評估；accuracy 由完整 CAL 回傳 logits 的 argmax 計算。
- Hook 只保存第一個 raw-image backbone feature；不以這個 feature 取代 CAL crop/flip 聚合 logits。
- CAL 一般推論使用 raw、flip 與各三個 attention crops，最後平均 logits（cal_ap_only 等模式另有分支）。
- HQQLinearWrapper.weight 由量化權重 dequantize 取得；目前沒有證據顯示該相容路徑造成這次新增下降。

現行程式的設定並非與舊 CSV 完全一致：舊 CSV quant_zero/quant_scale=True、exclude=head；
W3 feature script 預設 quant_zero/quant_scale=False，並明確排除 model.dfsm.fc、model.dfsm.cam_head。
這些是需控制的比較差異，但實際新舊 W3/G128 accuracy 幾乎相同，不能將它們宣稱為大幅下降的原因。
ZIP 中只有來源 manifest 的 hash，沒有原始 manifest 內容，因此上述新設定是目前 code 的設定，尚未核對 nano4 實際生效版本。

## 可以支持與仍未確認的結論

舊資料已顯示 W3/G128 是 BEiTv2 CAL 的敏感設定：共同 15 個 dataset 的舊 accuracy retention 平均 63.00%，
而 W4/G128 為 92.11%。這支持 bit/group 選擇會明顯影響 accuracy；不支持「這次分析新造成嚴重回歸」。
尚不能由 aggregate CSV 指認主要敏感 layer，或證明 attention crop 放大誤差。未執行 GPU/checkpoint 重現。
本機 opencode_env 缺 timm；匯入模型失敗，不能將本機視為 nano4 的推論環境。

若需定位 CAL 的機制，下一輪應在 nano4 固定 checkpoint、資料、精度、HQQ 版本與 metadata quantization，
只改單一變量：W3/G128 對 W4/G128；QKV 保持 FP；MLP 保持 FP；raw-only 與完整 CAL，最後比較量化模型使用 FP 固定 crop 的結果。
這些是待執行的驗證，未據此修改目前量化 script。

## 來源

- `C:\project\quant_analysis\data\backbones_quant.csv`
- `C:\project\quant_analysis\five_dataset_features\data\w3_g128_results\all_pairs.csv`
- `C:\google drive\my_project\quant_analysis\results_all\quant\serial_10000\corr\prepared_accuracy_cells.csv`

原始來源在此檢查前後 SHA256 相同，見 verification.json。
