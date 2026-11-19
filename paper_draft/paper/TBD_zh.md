# 待補項目與作者待決事項

紅字加 † 的數字是假資料；紅字 `[TBD: ...]` 是缺少的文字。以下逐項列出位置與所需資料。

## 假資料（需替換）

1. **Table 1 的 Train N 欄**：16 個 dataset 的訓練集大小全為假值。需要每個 dataset 的 train split 樣本數與來源；不能由 test N 推算。
2. **Figure 6（swap rate by margin decile）**：以合成數值繪製並加 PLACEHOLDER 浮水印。需要 W3/G128 每張測試影像的 FP 與量化後預測（或是否 swap），目前只在 nano4 的原始 exports 中。

## 缺少的文字（Section 3）

3. **FT／FZ／CAL 定義**：目前寫成「FT 全網路微調、FZ 凍結 backbone 只訓練 classifier、CAL 以 counterfactual attention learning 微調」，依 TGDA 程式命名推得，未經訓練紀錄確認。需要確認定義、pretrained weights 來源（ViT-B/16 的預訓練資料）與訓練超參數。
4. **量化層數**：HQQ 版本為 0.2.8.post1（作者確認）；仍需每個 backbone 被量化的 Linear 層數。
5. **Moe 資料集引用**：Moe（動漫角色）沒有引用來源。

## 未核對的引用

6. `main.bib` 的欄位依記憶撰寫，尚未逐筆對照官方來源。以下作者欄不完整：LampQ（Kim et al.）、Ozdemir and Li、Liu et al.（ICML 2025）；Fang et al.（TCSVT 2024）作者需核對。HQQ 目前引用官方 blog。
7. Related Work 對各論文的描述依 `paper_draft/INTRODUCTION.md` 的 citation map 撰寫；本機沒有 `paper_draft/pdfs/`，未重讀原文。

## 作者待決事項

8. **投稿 track**：目前設為 Algorithms Track；本文是分析型研究，也可選 Applications Track。
9. **題目**：目前為 "Dataset-Dependent Accuracy Retention under Weight-Only Post-Training Quantization"；另兩個候選保留在 `main.tex` 的註解。
10. **Full grid 來源**：grid 使用 `quant_acc/legacy/`（審核快照 `20261115_weight_only_s10000_10016_audit`，4,490 格、141 contexts）。若本機新版 grid 完成，需替換 `inputs/grid.csv` 並重跑 `build.sh`。
11. **VegFru**：grid 缺 Swin-B 全部與 ViT-B/16 CAL 的 22 格，因此 grid 分析只用 15 個 datasets。補跑後可改為 16 個。
12. **頁數**：`main.pdf` 本文 6 頁多（上限 8 頁）；`short.pdf` 本文 3.5 頁（workshop 上限 4 頁，不含 references 與 appendix）。
13. **4 頁版的刪減**：Related Work 併入 Introduction；Table 1（datasets）、Figure 3–6、Table 4–5、診斷方法（CKA、separability、H-score、bias、competitor 定義）移到 appendix；t-SNE 與 limitations 小節刪除，limitations 併入 Conclusion 最後一句。需作者確認。

## v2 待決定事項
- 名稱：任務稱 FGIR / UFGIR，資料組標籤為 UFG / FG；策略稱 training strategies（FT / FZ / CAL）。FZ 只訓練 classifier，pretrained weights 為 ImageNet-22k（作者確認）。
- accuracy retention 保留，並註明 LLM 量化文獻稱 accuracy recovery（kurtic2025bf16）。
- BEiTv2 FT 的 bootstrap 下界約為 0，正文改為報告其他 8 個 setting 加上此下界。
- 4 頁版已移除 Fig. 4（ranking agreement）與 swap-by-decile placeholder；長版仍保留，若取得 W3/G128 逐張影像預測可補長版 supplement。
- 選項：appendix 加 difficulty score 的 strip plot（每 setting 一欄，UFG/FG 分色），呈現組間分離穩定、FG 內部順序不穩。

- 不處理：BEiTv2 FZ 在 W3/G128 全面崩潰（FG 0.20、UFG 0.03）不寫入論文；組間順序在該 setting 內仍成立，W3 gap 仍為正（0.17）。
