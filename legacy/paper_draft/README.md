# Paper draft

PTQ、FGIR／UFGIR 的論文草稿、文獻核對與本機引用 PDF。

## 文件

- [完整英文 intro、中文段落說明與引用對照](INTRODUCTION.md)（最新詳細版）
- [Intro 草稿與引用用途](ptq_fgir_intro_draft_20261116.md)
- [研究 gap 與相關研究核對](ptq_fgir_intro_gap_check_20261116.md)
- [PDF 來源、頁數與 SHA-256](references.json)
- [草稿搬移紀錄](relocation_receipt.json)
- [PDF 與 Git 忽略規則核對結果](verification.json)

## 引用 PDF

01–08 對應原 intro 的 [1]–[8]；09–14 為原補充文獻；15–16 為新增 CKA／t-SNE 方法來源。
詳細版的引用編號與 PDF 檔名前綴不同，請以 [INTRODUCTION.md 的 citation map](INTRODUCTION.md#citation-map) 為準。

| ID | 論文 | 用途 | PDF 來源 |
|---|---|---|---|
| 01 | [A Survey of Quantization Methods for Efficient Neural Network Inference](pdfs/01_Gholami_2021_Quantization_Survey.pdf) | intro [1] | 官方公開來源 |
| 02 | [PTQ4ViT: Post-Training Quantization for Vision Transformers with Twin Uniform Quantization](pdfs/02_Yuan_2022_PTQ4ViT.pdf) | intro [2] | Zotero 原始附件副本 |
| 03 | [LampQ: Towards Accurate Layer-wise Mixed Precision Quantization for Vision Transformers](pdfs/03_Kim_LampQ.pdf) | intro [3] | Zotero 原始附件副本 |
| 04 | [Fine-grained Data Distribution Alignment for Post-Training Quantization](pdfs/04_Zhong_2022_FDDA.pdf) | intro [4] | Zotero 原始附件副本 |
| 05 | [Bridging the Reliability Gap: INT8 Quantization Effects on Discrimination and Calibration in Medical Imaging](pdfs/05_Ozdemir_2026_Bridging_Reliability_Gap.pdf) | intro [5] | 官方公開來源 |
| 06 | [Understanding the Unfairness in Network Quantization](pdfs/06_Liu_2025_Unfairness_Network_Quantization.pdf) | intro [6] | 官方公開來源 |
| 07 | [Towards Understanding Model Quantization for Reliable Deep Neural Network Deployment](pdfs/07_Hu_2023_Reliable_Quantization_Deployment.pdf) | intro [7] | Zotero 原始附件副本 |
| 08 | [Benchmark Platform for Ultra-Fine-Grained Visual Categorization Beyond Human Performance](pdfs/08_Yu_2021_Ultra_Fine_Grained_Benchmark.pdf) | intro [8] | Zotero 原始附件副本 |
| 09 | [Learning Contrastive Self-Distillation for Ultra-Fine-Grained Visual Categorization Targeting Limited Samples](pdfs/09_Fang_2024_CSDNet_Ultra_FGVC.pdf) | supplement: UFGIR motivation | Zotero 原始附件副本 |
| 10 | [An Underexplored Dilemma between Confidence and Calibration in Quantized Neural Networks](pdfs/10_Xia_2021_Confidence_Calibration_Quantization.pdf) | supplement: confidence and prediction swaps | Zotero 原始附件副本 |
| 11 | [SynQ: Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning](pdfs/11_Kim_2025_SynQ.pdf) | supplement: synthetic data and fine-tuning | Zotero 原始附件副本 |
| 12 | [Why Do Some Inputs Break Low-Bit LLM Quantization?](pdfs/12_Chang_2025_Inputs_Break_Low_Bit_LLM.pdf) | optional: LLM input-dependent errors | Zotero 原始附件副本 |
| 13 | [How Does Quantization Affect Multilingual LLMs?](pdfs/13_Marchisio_2024_Multilingual_LLM_Quantization.pdf) | optional: multilingual effects | Zotero 原始附件副本 |
| 14 | [Benchmarking the Reliability of Post-training Quantization: a Particular Focus on Worst-case Performance](pdfs/14_Yuan_2023_PTQ_Worst_Case_Reliability.pdf) | supplement: reliability gap memo | 官方公開來源 |
| 15 | [Similarity of Neural Network Representations Revisited](pdfs/15_Kornblith_2019_CKA.pdf) | 詳細版 [9]：CKA 方法 | PMLR 官方 PDF |
| 16 | [Visualizing Data using t-SNE](pdfs/16_van_der_Maaten_2008_tSNE.pdf) | 詳細版 [11]：t-SNE 方法 | JMLR 官方 PDF |

## Git 與備份

- `pdfs/` 及本目錄任何大小寫副檔名的 PDF 均由根 `.gitignore` 排除，PDF 僅保存在本機。
- Markdown 草稿、引用清單與來源核對可由 Git 追蹤；本次沒有執行 commit 或 push。
- Zotero 原始附件未修改；複製後已核對 SHA-256。
- Git 不會備份忽略的 PDF；請以本機備份或 Zotero 保存原始檔。
- 來源與 bibliographic metadata 以附帶論文及官方頁面為準；Zotero 附件可能為 preprint 版本。
