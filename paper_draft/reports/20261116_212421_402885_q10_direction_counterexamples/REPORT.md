# q10 方向相反的實例與比例

Run ID：`20261116_212421_402885_q10_direction_counterexamples`。時間：2026-11-16T21:24:21.402885+08:00（Asia/Taipei）。狀態：完成本次描述性統計；未重跑 inference。

## 定義

預期方向為 q10 越大、retention＝Q/FP 越高。數值反向定義為兩者 pairwise differences 的乘積小於零。q10 相對差≤5% 沿用前次的「接近」定義，這類極小差異不當作有實質方向差異。本報告的 clear reverse 是描述性門檻，不代表統計顯著性。

只比較相同 backbone、相同 regime、W3/G128 的 datasets；固定量化 accuracy 差≥5 pp，baseline 接近門檻如下。分母是配對數，不是獨立 datasets 或 training repetitions。

| FP gap max (pp) | Pairs | Numeric reverse | q10-different reverse | q10-different reverse (%) |
|---|---:|---:|---:|---:|
| 0.5 | 6 | 1 | 1 | 16.67 |
| 1 | 15 | 2 | 1 | 6.67 |
| 1.5 | 22 | 3 | 2 | 9.09 |
| 2 | 29 | 5 | 4 | 13.79 |

## 詳細數據

A 是 q10 較小的一方，實際卻有更高 retention。前兩組 FP gap≤1.5 pp，後兩組需放寬至≤2 pp。這四組的 Q accuracy 方向也都相反。

| Model/mode | Pair A/B | FP A/B (%) | q10 A/B | Q A/B (%) | Retention A/B (%) | FP gap (pp) |
|---|---|---:|---:|---:|---:|---:|
| BEiTv2/FT | cars/dogs | 88.87/88.81 | 0.483244/0.843316 | 82.63/54.13 | 92.98/60.94 | 0.058 |
| BEiTv2/CAL | cub/vegfru | 89.28/90.39 | 0.311118/0.340139 | 64.03/39.06 | 71.72/43.21 | 1.109 |
| BEiTv2/FT | cub/dogs | 86.95/88.81 | 0.671450/0.843316 | 65.65/54.13 | 75.51/60.94 | 1.859 |
| BEiTv2/FT | cars/cub | 88.87/86.95 | 0.483244/0.671450 | 82.63/65.65 | 92.98/75.51 | 1.918 |

另有 BEiTv2/FT 的 Dogs/NABirds：q10 分別 0.8433160316/0.8433090104，相對差僅約 0.000833%，baseline 88.81%/87.89%，Q 54.13%/83.92%。這應解讀為 q10 幾乎相同卻量化表現不同，不宜靠極小小數差稱為明確方向預測錯誤。

Baseline gap≤1.5 pp、Q gap≥5 pp 時：22 pairs 中數值反向3組（13.64%），扣除 q10 幾乎相同的一組，q10 差>5%的反向為2組（9.09%）。量化差門檻改為≥10 pp 時，分母12、相應計數3與2，詳見 threshold_counts.csv。

反例顯示 q10 不是每對 datasets 都單調有效，不會單獨推翻整體 Spearman association。四組反例集中在 BEiTv2，且 FT 配對共用 Cars/CUB/Dogs，不能稱為四次獨立驗證。此處分析的是單一 W3/G128 條件，不是原 Table 1 的四條件平均 difficulty score。

## 來源與重現

輸入為上一個 immutable run 的 1,080-pair table，另複製成版本化 input 並核對 SHA-256。本次沒有下載、改寫上游資料、載入 checkpoints 或重新 inference。baseline 為 unquantized AMP-FP16。

method.json 保存精確公式，threshold_counts.csv 包含 q-gap≥5/10 pp 的全部門檻結果；clear_direction_counterexamples.csv 與 near_equal_q10_pair.csv 保存詳細來源欄位。run.json 記錄環境、Git 與 input/output hashes。驗證所有選中 pairs 的 retention 與 Q accuracy 反向標記一致，且 q10-close 與 clear-reverse 分組互斥。
