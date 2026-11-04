# Rename: w6a4_analysis → activation_quant

2026-11-04（Asia/Taipei）。依使用者要求在主 checkout 改名整個分析模組。
目錄包含正式程式、唯讀 input、完整 result、歷史程式快照及所有 reports。

`files_manifest.json` 記錄移動前每個檔案的相對路徑、大小與 SHA-256。
原生目錄移動完成後，所有檔案逐一核對，沒有缺檔或 bytes 差異。
核對後僅更新目前的正式程式／README，以及根索引與操作入口；歷史
`data/`、`result/`、`reports/` 內容及其中的 receipts/provenance 保持原 bytes。
`__pycache__` 是可重新產生的 runtime cache，執行後不要求 hash 固定。
模組 `.gitattributes` 對 `reports/**` 設定 `-text`，停用 Git 換行轉換；186 個
保存報告／引用圖片的暫存區 blobs 已與本機檔案逐一核對 SHA-256 完全一致。

歷史路徑的解讀方式：

| 原路徑／命令 | 現行路徑／命令 |
|---|---|
| `C:/project/quant_analysis/w6a4_analysis/<path>` | `C:/project/quant_analysis/activation_quant/<path>` |
| `w6a4_analysis/data/...` | `activation_quant/data/...` |
| `w6a4_analysis/result/...` | `activation_quant/result/...` |
| `w6a4_analysis/reports/...` | `activation_quant/reports/...` |
| `python -m w6a4_analysis.run ...` | `python -m activation_quant.run ...` |
| `python -m w6a4_analysis.acquire ...` | `python -m activation_quant.acquire ...` |

歷史 `provenance/source/w6a4_analysis/` 是原始程式快照內的身分，目錄名保持
不變，不再替換一次。報告的相對圖片連結仍可使用。歷史文件／計畫中的原名
視為當時狀態，並由本 receipt 解析，避免修改已保存的來源證據。

新產物繼續使用新的 input/run ID；改名不重新下載、不重寫舊 run。被 Git 忽略
的資料與完整結果仍需另行備份；Git merge 只整合受追蹤內容。
