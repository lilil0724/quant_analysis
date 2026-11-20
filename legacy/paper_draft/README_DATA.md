# Quant-dataset robustness：W3/G128 論文資料

來源 run：`20261115_211022_759540_w3_s20004`；核對日期：2026-11-16（Asia/Taipei）。

先讀 [結果核對報告](reports/20261116_140853_554056_runall_s20004_audit/REPORT.md)，再用 [144 列整合主表](data/derived/20261116_140853_554056_runall_s20004_audit/dataset_master.csv)。

| 目錄 | 內容 |
|---|---|
| `data/run_all/20261115_211022_759540_w3_s20004/` | 原始五階段產物；保留全部分析表、圖、weights、coordinates、logs、provenance |
| `data/derived/20261116_140853_554056_runall_s20004_audit/` | Dataset 主表、model/mode summary、matching 主表與 support/NA 摘要 |
| `data/audit/20261116_140853_554056_runall_s20004_audit/` | 數值／hash 核對、輸入清單與 audit 程式 |
| `reports/20261116_140853_554056_runall_s20004_audit/` | 中文核對報告及實際引用的 3 張圖 |

主表 key 是 dataset/model/mode；合併 full grid/static data 前還需比對 checkpoint/seed/precision/head exclusion。
Paired accuracy/drop/rates 是 fractions，轉成百分比或 pp 時乘 100；model_mode_summary 的 accuracy 已是百分比，bias gain 單位為 pp。Margin gap 是 retention ratios 的差值。
所有 model/mode 摘要採 dataset 等權；原始 sample 數差異很大，不直接 pool 全部樣本當作 dataset-level 結論。
本包包含 W3/G128 的分析產物，未含 full weight grid、static attributes 或原始 features/logits。此處 competitor-model AUROC 與舊 sample-margin decile/AUROC 是不同分析。
原始下載檔保留於 quant_analysis；既有 draft.docx 未修改。本資料包的 source manifests/hashes 保留原文，轉移身分見 TRANSFER_MANIFEST.json。

實際複製完成狀態與逐檔核對見 `data/audit/20261116_140853_554056_runall_s20004_audit/COPY_RECEIPT.json`。
