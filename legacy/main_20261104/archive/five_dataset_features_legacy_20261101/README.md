# five_dataset_features 舊版原始碼封存

封存日期：2026-11-01（Asia/Taipei）。原始路徑、大小與 SHA-256 見
[manifest.json](manifest.json)。13 個舊版檔案均先複製並核對 hash，再從現行
`five_dataset_features/` 移除；原本帶有 Git 衝突標記的 README 另存為
[README.before_cleanup.md](README.before_cleanup.md)，只供歷史查閱。

本目錄的 `.gitattributes` 停用 Git newline conversion，讓 clone／checkout
仍保留原始 bytes，避免 Windows CRLF 轉換使封存 SHA-256 失效。

## 封存清單

| 原始檔案 | 舊版用途 |
|---|---|
| analyze.py | 21-condition、固定 `[N,13,768]` 分析與 ZIP |
| analysis_snapshot.py | 上述流程的來源／結果 snapshots |
| run_local.ps1 | Windows 舊流程 launcher |
| smoke.py | 舊版 CUB 六條件真實 batch 驗證 |
| geometry_robustness.py | 固定五個 datasets 的 feature geometry／damage 分析 |
| distance_hypothesis.py | FP class distance 與 weight-only loss |
| quantized_class_distance.py | 舊版 final CLS distance 表格 |
| distance_accuracy.py | 舊版 distance drop 與 accuracy 關聯 |
| paired_feature_pca.py | 舊版 FP／W3／W4 paired final CLS PCA |
| verify_paired_feature_pca.py | 上述 PCA 產物驗證 |
| plot_presentation.py | 固定五個 datasets、105 conditions 的簡報圖 |
| plot_cka_robustness.py | 上述舊讀取器的 CKA／accuracy 比較圖 |
| plot_cka_retention_scatter.py | 上述舊讀取器的 CKA／accuracy 散點圖 |

## 封存邊界

這是**原樣保存的來源快照，不是搬移後可直接執行的 Python package**。
程式中的相對 imports、`__file__` 路徑、固定五-dataset 假設與已知限制保持原狀。
部分來源已把 dataset 常數擴成 16，但其他 reader、palette、subplot 和 verifier
仍沿用舊假設；不得把這份快照當成已通過 16-dataset 驗證的現行入口。

現行流程見 [模組 README](../../dataset_features/README.md)。共用計算
`five_dataset_features/metrics.py` 原樣保留；其封存時 hash 列於 manifest 的
`retained_dependencies`。現行 paired／t-SNE 只改從 `common.py` 匯入檔案工具與
dataset 清單，已不依賴此封存區。

舊 data、analysis、results、原始 manifests 和既有報告沒有搬動或改寫。
此封存不包含那些大型輸入，也不取代其獨立備份。

2026-11-04 路徑註記：上段描述封存當日的狀態。現行原始碼與本機資料／結果
已統一至 `dataset_features/`，見 [搬移清單與 SHA-256 核對](../../dataset_features/relocations/20261104_module_consolidation/README.md)。
此封存內的來源快照與 manifest 保持原樣；還原舊程式時仍使用 manifest 的
`original_path`，不要將舊入口覆寫至現行模組。

## 核對與還原

從 repository root 執行下列唯讀核對：

```powershell
$receipt = Get-Content archive/five_dataset_features_legacy_20261101/manifest.json -Raw | ConvertFrom-Json
foreach ($file in $receipt.files) {
    $actual = (Get-FileHash -LiteralPath $file.archive_path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $file.sha256) { throw "Archive checksum mismatch: $($file.archive_path)" }
}
```

若要重現舊流程，先在另一份 checkout 還原 manifest 中 `action=archived` 的
13 個檔案到各自 `original_path`，並核對保留的 `metrics.py` hash；不要直接在
archive 內執行，也不要覆寫現行資料或 manifests。原始 README 僅供參考，無須
還原它的衝突標記。舊命令記錄在 `README.before_cleanup.md`；輸入必須仍是相容
的舊 schema。這次只驗證封存完整性，未重新執行舊分析。
