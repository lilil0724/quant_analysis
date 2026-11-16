# short.pdf 論文合理性與資料核對

審閱日期：2026-11-16，Asia/Taipei。Run ID：`20261116_201210_623333_short_paper_review`。

## 判斷

研究問題與主要觀察可以成立，但目前版本仍是需要修正的研究草稿。最大問題是 **full-grid 使用含 classifier 量化的舊版 CAL，而 Methods 宣稱所有實驗只量化 backbone**。另外有錯誤／占位的 train counts、明確標示 synthetic 的 Figure 6，以及不同 estimators 被當作同一個 retention gap 解讀的問題。

不是所有數字都需要推翻：本次從既有原始 W&B JSONL 重建了 141 個 baselines 與 4,490 個 quantized results；W3/G128 的 0.50／0.82、Table 1 的 Kendall's W 可由這份歷史資料重現。Table 2 的九列數值也都符合已驗證 W3/G128 分析。因此需要修正的是 protocol、資料範圍及解釋，不能將「能重現舊數字」等同於「已完成新版 backbone-only 實驗」。

這次沒有修改原稿、模型或原始實驗資料，也沒有重跑 inference。本文位置採 PDF 印刷頁碼與紅色行號。

## 審閱與證據範圍

- 原稿：[short.pdf](C:/Users/kyle/Downloads/short.pdf)，7 頁；SHA-256：`4c405d79e1da6c6c99c0967424fbcbc984302a5031023c07d3125e0f8107a9e0`。
- 已保留原稿快照於 [本次 input](C:/project/quant_analysis/paper_draft/data/20261116_201210_623333_short_paper_review/short.pdf)。逐頁文字與渲染圖用於檢查，完成後清除暫存。
- 新版 W3/G128：`paper_draft/data/20261116_163429_728618_w3_s20004_paper_package/` 的 144-context 主表、matching、coverage 與 bias/competitor scores。
- 舊 full-grid：`quant_acc/data/wandb_weight_only_20261115_125000_675669/runs.jsonl`，4,631 筆 finished runs，serial 10000–10015；以及既有 W3/G128 logs。
- 本次核對表與輸入 hashes：[review result](C:/project/quant_analysis/paper_draft/result/20261116_201210_623333_short_paper_review/)。
- 未取得論文生成程式、新版 CAL full-grid、原始 inference arrays、事前分析計畫、Table 1 permutation/regression 程式。因此不將未驗證項目直接判成錯誤，也不宣稱已核對每張圖的全部像素或所有文獻 metadata。

## A. 必須優先修正

### A1. Full-grid CAL 的 classifier 實際有量化，與 Methods 衝突

**位置：p.2 L81–92；影響 Abstract、Figure 1–3、Table 1／5 與 Conclusion。**

稿件宣稱 HQQ 量化 backbone 的 Linear，classifier 保留 full precision。但本機歷史資料的 4,490 筆 Q results 對應 serial 10000–10015，不包含新版 CAL 10016。在其 46 份可讀 CAL W3/G128 logs 中，均找到：

```text
[HQQ] Replaced model.dfsm.fc: Linear -> HQQLinear
```

具體例子：[inat17 / ViT / CAL log](C:/project/quant_analysis/quant_acc/data/wandb_weight_only_w3logs_20261115_125600_605683/bm89xn4v/output.log:158)。同一 log 顯示排除清單實際是空的，而不只是 folder 名稱看起來像舊版。`hqq_exclude='head'` 並未排除 CAL 的 `model.dfsm.fc`。

本次也重現 p.6–7 所述的 140 個重疊 contexts：FP 最大差 **0.1667 pp**，Q 最大差 **0.4967 pp**。這些數字是正確的，但 accuracy 接近不能證明量化 protocol 相同，也不能證明 checkpoint 權重 bytes 相同。單一 W3/G128 的小差異不能外推到整個 bit/group grid。

**修法：** 主結果改用已驗收的新版 backbone-only CAL grid 與配對 baseline，重算所有依賴它的 group means、difficulty rankings、correlations、plots。若暫時保留舊 CAL，須將其清楚標示為不同 protocol 的歷史結果，避免用單一「只量化 backbone」描述整份實驗。現有 144-context W3/G128 export 的 actual exclusions 仍應由原始 manifests／coverage 證明，不能僅以 serial 20004 推定。

### A2. Table 3 的 Train N 是錯誤／占位數字

**位置：p.6 Table 3。**

Train N 從 1,000、1,037、1,074 一路固定增加 37 到 1,555，全部帶紅色 dagger，頁面上沒有相應說明。它們與現有歷史 training config 不符。例如：

| Dataset | 稿件 Train N | 歷史 config 的 num_images_train |
|---|---:|---:|
| Aircraft | 1,000 | 6,667 |
| Cars | 1,037 | 8,144 |
| CUB | 1,074 | 5,994 |
| iNat17 | 1,222 | 579,184 |
| Cotton | 1,555 | 240 |

後一欄是本次讀取既有 JSONL 的 config，**仍不是對訓練 CSV 的獨立驗證**；不能直接將這些值當作最終真值填入。需依實際 checkpoint training split、是否合併 train/val、是否 subsampling，核對使用的 CSV 與去重計數，再替換整欄。無法確認時先寫 TBD 或移除此欄。

同表的 **Test N、類別數與 FP accuracy** 可與 144-context 主表對上，無須連同它們全部推翻。完整對照見 `table3_crosscheck.csv`。

### A3. Figure 6 是 synthetic placeholder，不能作為實驗證據

**位置：p.7 Figure 6；p.6 L339–340 已在 Methods 引用它。**

圖上有大型 PLACEHOLDER 浮水印；caption 明寫 synthetic values 與缺少 per-image quantized predictions。因此目前無法用它支持「低 margin 的樣本有較高 swap rate」在這批資料上的實證結論。

**修法：** 刪除圖及當成已完成結果的引用，或從 FP/Q matched predictions 與 FP margin 重新計算。建議每個 dataset/model/mode 各自做 margin deciles，再依事先定義的 dataset weighting 彙總；保存每 bin 的樣本數、swap count、swap rate 與不確定性。完整來源在 server 時，需取回或在 server 計算，不能用目前這張 synthetic 圖補位。

### A4. Full-grid 與 matching 的 retention gap 不是同一個 estimator

**位置：p.2 L93–128；Figure 1–2 對照 Table 2。**

原稿其實分別寫出了兩個公式，但 Results 的敘述容易讓讀者以為是同一個 gap 直接做匹配前後比較：

```text
Full-grid：mean_d(aQ_d / aFP_d)
Matching：weighted_mean(aQ) / weighted_mean(aFP)
```

即使 dataset 等權重，平均比值也不等於平均後的比值。後者相當於以 baseline accuracy 再加權各 dataset 的 retention。此外，full-grid 排除 VegFru，matching 的 Table 2 使用全部 16 datasets。

**差異不只是理論問題：** 固定同一份 serial-20004、16-dataset、BEiTv2/FT 資料，前者 raw gap 為 **0.2150**，後者為 **0.4077**。Table 2 的 0.408 對應後者。這個例子排除了來源版本與 15/16 datasets 差異，僅改 aggregation 就差 **0.1927**。

**修法：** 清楚命名並寫出兩個 estimands，補相同 cohort／相同公式的 bridge table。Table 2 的 raw/matched 比較本身沿用同一 matching 公式，因此 27.6–81.3% 的縮減並非因此自動錯誤；問題是不能把它直接解讀為 full-grid 那個 dataset-average gap 被解釋了相同比例。

## B. 結論與統計解釋需要收斂

### B1. 小於 15% 不等於接近 chance

**位置：p.2 L147–149。**

本次確實找到七個 baseline top-1 <15% 的 contexts，但它們有 80–1,938 類。以均勻隨機分類的 `1/C` 為參照：

- SoyGlobal / BEiTv2 / FT：FP 2.98%，chance 約 0.0516%，約為 chance 的 57.8 倍。
- SoyGene / BEiTv2 / FT：FP 9.42%，chance 約 0.0901%，約為 chance 的 104.6 倍。
- Cotton / BEiTv2 / FT：FP 2.92%，chance 1.25%，約為 chance 的 2.34 倍。

因此不能把七個 contexts 一概稱為 near chance。若類別不平衡，還需另報 majority-class baseline，不能用它與 uniform chance 混稱。

**建議改寫：** “Several soy+cotton contexts have low baseline accuracy, which makes relative retention sensitive to its denominator.” 另報 absolute accuracy drop；可選擇增加含類別數的 chance sensitivity analysis，但需說明假設及分母接近零的情況。

### B2. 「margin 不能與 baseline accuracy 分離」與「預測一樣好」過強

**位置：p.1 L18–19；p.2 L60–62；p.3 L167–170；p.4 L203–205。**

Table 1 支持 margin 與 FP accuracy 都有強 association，以及把 q10 加到 FP accuracy＋log C 後，LODO error 只在 3/9 settings 改善。它不能直接證明兩個 predictor 等效、margin 沒有獨立資訊，或原理上不可分離。

例如 ViT/FZ 的 error 從 0.0159 升到 0.1848，需要檢查小樣本 regression、collinearity、fold-wise extrapolation、特徵縮放及 influential datasets。每個 setting 的有效 dataset observations 是 15，不是測試影像總數；九組 settings 也共用 datasets。

**建議改寫：** “Margin q10 is strongly associated with difficulty, but does not consistently improve leave-one-dataset-out prediction beyond baseline accuracy and class count in this dataset collection.”

若保留預測比較，補模型公式、error 的 fold aggregation、regularization／scaling、各 fold residuals，及 paired error difference 的不確定性。FP accuracy 亦參與 retention outcome 的分母，宜並列 absolute drop 的敏感度，避免把 ratio-based association 當成獨立機制。

### B3. 「feature similarity does not track accuracy loss」過於絕對

**位置：p.1 L19–20；p.4 L180–188；p.7 Table 5。**

稿件自己的 Table 5 顯示 Swin/FT 的 final CKA correlation 約 −0.60、ViT/FZ 約 −0.55；BEiTv2/FZ 的 accuracy 與 CKA 同時大幅下降。這不支持完全沒有關係的表述。

可支持的範圍是：association 在 settings 間不一致，median correlation 約 −0.28，global CKA 不足以一致預測 dataset difficulty。高 CKA 亦不保證分類邊界附近的樣本不受影響。

CAL 另有 view 差異：export 的 backbone features 對應 raw image，但 logits 聚合 crop/flip。Appendix 已提到這點，但主文比較 feature drift 與 decision damage 時仍需承認限制；可分別報 FT/FZ 與 CAL，或用相同 decision path 的 feature export 作驗證。

**建議改寫：** “Global feature similarity is not a consistent predictor of dataset-level accuracy retention across the evaluated settings.”

### B4. Matching 不能證明 margin 解釋了多少因果差距

**位置：p.2 L120–128；p.4 L176–179。**

Table 2 的 overlap 只有約 0.19–0.40，且 minimum matched/raw ESS 約 20.9%。它描述的是共同 support 上、指定 weighting 下的關聯，不能擴大到所有影像，也不是 margin 的 causal mediation estimate。

“even at equal margin” 建議改為 “after balancing the binned margin distributions on their shared support”。bin width=0.05 是近似平衡，不是每張影像 exact matching。除 margin 外，class count、class imbalance、baseline correctness、checkpoint quality 等仍可能不同。

表中的 ESS 是 matched ESS / raw weighted ESS，**不是原始影像數的保留比例**。正文 20.8% 對應所有 bin widths 的最小值 20.7976%；主設定 0.05 的最小值是 20.8959%，應寫 20.9%，或明確說 20.8% 包含 sensitivity settings。

### B5. Prespecified 與 permutation inference 需要方法證據

**位置：p.2 L98–119；p.3 Table 1。**

本次未取得有日期的分析計畫，故不能確認 q10、四個 transition conditions、rho threshold、one-sided tests 與 controls 是否在看結果前決定。這不表示它們必然是事後挑選；但若是探索後決定，應改稱 primary/exploratory，不能寫 prespecified。

需提供 permutation 次數、seed、null／交換單位、ties 處理與 p-value 計算程式。四個 soybean datasets 可能具有來源依賴，九組 tests 共用資料；單純把 15 個 datasets 當完全獨立單位，需要說明。建議加 family-aware sensitivity／leave-family-out，並交代多重比較政策。不要把 36/36 或 34/36 重複條件當作獨立實驗次數。

### B6. Transition-band 與 noise 的論證須縮小範圍

**位置：p.2 L132–146；p.4 L199–201。**

“If datasets differed only in noise, the gap would scatter around zero” 沒有定義 null model、noise 或 exchangeability，不能當成已證明的統計推論。低 bit 下接近 floor、高 bit 下接近 ceiling，本來就會壓縮可觀察到的 group gap。

建議直接描述觀察：“In this HQQ grid, the largest group gaps occur at intermediate retention levels.” 不宜推成所有量化方法中的 dataset effect 只可能出現在這個區域。class-offset correction 的結果也宜寫成 “consistent with a class-specific shift component”，避免由 retrospective correction 直接宣稱已找到普遍的 damage cause。

## C. 報告完整性與可重現性

1. **15／16 datasets 的 scope：** Figure 1–3、Table 1／5 是 15-dataset complete grid；Table 2／4 和 feature exports 是 16 datasets。Abstract 應寫清楚 “16 datasets overall, with 15 complete datasets in the grid analysis”，或先補齊新版 grid 再統一敘述。不能讓讀者以為 16×3×3×32 已完整完成。
2. **HQQ 設定：** p.2 的 version TBD、pretrained/regime TBD 必須補齊。交代實際 quantized sites/counts、CAL classifier exclusions、scale/zero quantization、group axis、compute dtype、software commit、pretraining 與 checkpoint 選擇。classifier “full precision” 最好改成 “unquantized; evaluated under the specified mixed-precision protocol”，避免被讀成 classifier 一定是 FP32。
3. **Train／inference seeds：** 保留 p.4 的單 checkpoint／seed 限制；同時列清楚 grid 與 export 各自的 seed。不能因兩者 checkpoint path 一致而宣稱權重 hash 已驗證。
4. **8/1,152 AUROC 的分母：** p.6 L364–366 的 1,152 列實際包含 damage＋swap、LODO＋兩種 family transfer、兩個 risk models。正文方法只描述 LODO damage，需擴充方法或縮小計數。primary LODO damage 是 288 列，其中 4 列 AUROC undefined；完整 archive 是 1,152 列，其中 8 列 undefined。
5. **Figure 5 relative depth：** 說清楚 index 是從 0 還是 1 開始，final 是否重複最後 block。caption 的 index / number of positions 與圖上 final=1 必須一致；目前沒有生成程式，僅列為待核對，未判定已畫錯。
6. **引用與 TBD：** Moe citation 缺少；[9]、[12] 使用不完整 “et al.” 作者欄，[16] 缺作者全名。已查看本機 LAMPQ、Unfairness、CHIL PDFs，這幾篇文獻確實存在，不能因 citation 不完整就稱為虛構。用原 PDF metadata／正式出版來源補齊書目；本次不是 26 篇文獻的逐項完整審核。
7. **研究範圍：** 目前稿件主要是 HQQ empirical grid＋margin／CKA 診斷，沒有完整呈現原規劃的 dataset static features、PCA／t-SNE 分析。這些不是每篇 paper 的硬性必要條件；但若目標仍是找「哪些 dataset 特徵與 robustness 有關」，目前結果主要停留在 dataset grouping、margin 與 baseline quality 的關聯，尚未回答完整的 static-feature 問題。依主線選擇補分析或縮小論文貢獻，不要寫成已完成。

## 已核對成立的數值

| 項目 | 本次核對結果 | 適用限制 |
|---|---|---|
| 舊 full-grid 規模 | 141 baselines、4,490 Q、140 complete contexts；另 1 context 有 10 個 Q | 是歷史 serial 10000–10015；不是新版 protocol 完成證明 |
| W3/G128 group retention | other=0.82048、soy+cotton=0.49574、gap=0.32474 | 舊 grid、15 datasets、mean of dataset ratios |
| Table 1 Kendall W | 約 0.75536–0.95580，九列依稿件精度對得上 | 舊 grid 的四個 primary conditions |
| margin／difficulty correlations | 多數依稿件精度重現；個別 rho 有約 0.01 rounding／tie 差異 | 精確定義與生成程式仍需保存；未驗證 permutation p 或 regression error |
| Table 2 九列 | Raw、Matched、Reduction、Overlap、ESS 五欄皆符合刊載精度 | 已驗證 16-dataset matching 分析 |
| matching reduction | 27.6166–81.2883% | bin width=0.05，matching estimator |
| Table 3 Test N／C／FP | 16 datasets 與主表相符 | Train N 不成立 |
| BEiTv2/FZ collapse | mean FP=62.9214%、Q=12.2837%；median final CKA=0.15979 | 與 Table 4 一致 |
| bias correction | 九組 mean gain 皆正，約 0.64–11.31 pp | retrospective、cross-fitted 診斷，非部署效能證明 |
| competitor prediction | primary damage ΔBrier 在 7/9 settings <0 | 保留另外兩組未改善的 negative results |
| grid/export accuracy discrepancy | 140 overlap；FP max=0.1667 pp、Q max=0.4967 pp | 不能據此宣稱 quantization/checkpoint identity 相同 |

## 建議修改順序與需要的資料

1. **先驗收新版 CAL full-grid。** 需要 Q／FP summary、classifier exclusion／actual quantized module coverage、checkpoint 身分與 hashes、seed／precision／preprocessing。之後重算所有 grid-dependent 圖表，避免在舊 protocol 上繼續打磨結論。
2. **替換 Table 3 Train N，移除或完成 Figure 6。** 前者需要實際 training CSV／split 設定；後者需要同樣本 FP/Q predictions 與 FP margins，或 server 端計算的可追溯分 bin tables。
3. **統一 cohort 與 estimand。** 既有摘要表已足以顯示公式差異；若要重跑 margin matching 的 15-dataset sensitivity，需要 server 的 sample-level inputs。
4. **收斂三項主張。** 將 margin 不可分離、CKA 無關、margin 解釋差距等句子，改成此資料集與 protocol 下的關聯與預測結果，保留 overlap／ESS／negative cases。
5. **補統計生成程式與 provenance。** 包括 p-values、LODO folds／errors、prespecification 記錄、圖表來源表與完整設定；若要推論 dataset effects，下一階段優先補多個 checkpoint／training seeds 與 family-aware sensitivity。

## 重現與保存

本次 audit 使用 `C:/Users/Public/miniconda3/envs/opencode_env/python.exe`，從既有 JSONL／CSV 讀取並產生 review tables，未下載或刷新 W&B。本次 review 的 source snapshot、指令、環境、Git 狀態、輸入／輸出 hashes 與核對結果存於 [run.json](C:/project/quant_analysis/paper_draft/result/20261116_201210_623333_short_paper_review/run.json)。

提供的 review audit 程式存於 result 的 `provenance/`，只作本次核對的可重現附件，不是新增正式分析模組。程式固定本次 run ID，既有輸出存在時會拒絕覆寫；重現時須選擇新的 run ID。完成審閱不代表新版 full-grid、原始 arrays 或缺少的統計方法已驗收。
