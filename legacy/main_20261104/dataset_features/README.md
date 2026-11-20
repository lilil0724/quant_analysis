# TGDA feature analysis — 16 datasets

## 現行範圍與目錄狀態

本模組讀取已完成的 TGDA inference exports，分析量化前後的 features 與
predictions；不訓練模型、不載入 checkpoint、不隱含下載或重新 inference。

現行完整 dataset 清單：

```text
aircraft cars cub dogs flowers food inat17 moe nabirds pets
soyageing soygene soyglobal soylocal vegfru cotton
```

Paired 分析與 t-SNE 的預設矩陣為 **16 datasets × 3 models × 3 modes = 144 pairs**。
Models 為 `vit_b16`、`swin_base_patch4_window7_224_in22k`、
`beitv2_base_patch16_224`；modes 為 `ft,fz,cal`。Config-driven 流程預設驗證
16 個 dataset contexts；schema-2 reader 則仍是單一 FP/quant pair 入口。

2026-11-01 的整理只抽出共用檔案工具、解除舊入口依賴並封存舊程式，沒有改變
計算公式、分析階段、抽樣政策、圖形函式或既有輸出位置。舊 21-condition
與五-dataset 專用程式見 [封存說明](../archive/five_dataset_features_legacy_20261101/README.md)
及 [逐檔 SHA-256](../archive/five_dataset_features_legacy_20261101/manifest.json)。

本模組尚未完成根目錄 [AGENTS.md](../AGENTS.md) 的完整資料／結果目錄遷移。
目前繼續使用既有 `data/`、`analysis/`、`results/` 與 server input-root；本次
只統一模組路徑，不改寫既有資料、結果和 manifests，也不宣稱已完成 raw-array 驗證。

### 2026-11-04 模組路徑統一

現行程式與本機資料都集中於 `dataset_features/`。舊目錄
`five_dataset_features/{data,analysis,results,__pycache__}/` 已逐檔核對後移入
此目錄；`analysis/` 與原有歷史摘要合併，沒有覆寫檔案。搬移的 1,418 個檔案
（218,255,389 bytes）及原有 70 個歷史產物均核對 SHA-256 一致。

| 內容 | 現行位置 |
|---|---|
| Serial 20004 compact FP/W3 副本 | `data/w3_g128_results/` |
| 歷史摘要、PCA、t-SNE 圖集 | `analysis/` |
| 既有完整結果 | `results/` |
| 遷移清單與核對紀錄 | [relocations/20261104_module_consolidation/](relocations/20261104_module_consolidation/README.md) |

歷史 manifest、完成的報告、ZIP 與封存來源保留原始 bytes；其中的舊路徑屬於
provenance，可依 relocation receipt 將 `five_dataset_features/` 前綴換成
`dataset_features/` 定位目前檔案。歷史結果仍須配合當時程式與來源重現，不能
因路徑已修正就視為通過現行 code-identity 或 raw-array 驗證。不要在舊目錄新增輸出。

`data/`、`analysis/`、`results/` 均忽略新增產物；整理前已追蹤的歷史摘要維持
追蹤。Git 不備份這次搬移的大型資料，需另行備份 `dataset_features/` 下的資料與結果。
伺服器 raw exports 的來源路徑未更動；同步程式後使用下列新模組命令。

## Codebase 分工

| 檔案 | 職責 |
|---|---|
| `run.py` | Config-driven 逐層分析 CLI、階段協調、續跑與驗證 |
| `run_config.py` | Config、cohort／stage identity、condition／position 定義 |
| `inputs.py` | `tgda_manifest_v1` inventory、manifest／hash／樣本驗證、lazy row alignment |
| `run_snapshot.py` | Config-driven 各階段 receipts、code/environment/input/output hashes |
| `layer_analysis.py` | 逐層 CKA、class distances、norm、accuracy 與 FP/Q 變化 |
| `layer_pca.py` | 每個 dataset／position 擬合 FP PCA，再投影所有 conditions |
| `comparisons.py` | Dataset-level 關聯、group contrasts、matched-N CKA 敏感度 |
| `comparison_values.py` | 比較表與圖共用的數值定義 |
| `plot_run.py`、`report_run.py` | Config-driven 圖集與繁體中文報告 |
| `paired.py`、`run_nano4.sh` | 16-dataset paired FP/W3 指標、final PCA、報告與 compact ZIP |
| `tsne_paired.py`、`run_tsne_nano4.sh` | Paired final features 共同取樣與 joint t-SNE |
| `centered_class_distance.py`、`run_centered_class_distance_nano4.sh` | FP final features 去全域平均前後的 class distances |
| `unified_inputs.py` | 獨立 schema-2 FP/quant 配對驗證與逐層指標；尚無 16-dataset orchestrator |
| `common.py` | Paired／t-SNE 共用的 16-dataset 清單、CSV 與 SHA-256；不含分析邏輯 |
| `metrics.py` | 新舊流程共用的 CKA、class distances、norm、H-score、prediction transitions、PCA、Spearman |

`metrics.py` 與所有現行計算／畫圖函式保持整理前的實作。`centered_class_distance.py`
保持原有獨立 NumPy reader 與來源 hash 邏輯，不為合併工具而新增相依性。

## 環境與 baseline

從 repository root 執行。本機 Python 明確使用：

```powershell
conda run -n opencode_env python -B -m dataset_features.run --help
```

Paired 核心需要 NumPy、Matplotlib；t-SNE 另需 scikit-learn。完整逐層比較使用
NumPy、SciPy、Matplotlib。Server launchers 預設直接使用
`$HOME/.conda/envs/tgda/bin/python`，可透過 `ANALYSIS_PYTHON` 或
`ANALYSIS_CONDA_ENV` 覆寫；不在 job 內安裝依賴。

`fp` 表示相同 checkpoint／test path 下不使用 HQQ 的參考，不可單憑名稱當成純
FP32。舊五-dataset 參考為 AMP-FP16；現行來源以 manifest 的 execution／precision
設定為準。配對須核對 checkpoint、dataset、model、mode、seed、classifier、
precision、preprocessing 與評估設定；schema-2 還要求 checkpoint/data/code/config/HQQ
等來源 hashes。指標定義見 [GLOSSARY.md](GLOSSARY.md)。

## Paired FP/W3：nano4 主分析

輸入格式：

```text
<input-root>/<serial>/<dataset>/<model>/<ft|fz|cal>/
  manifest.json
  samples.json
  fp/<layer>.npy, logits.npy, predictions.csv
  w3_g128/<layer>.npy, logits.npy, predictions.csv
  layer_metrics.csv
  summary.json
```

`paired.py` 驗證選取範圍中每個 pair 的身分、來源 hashes、W3/G128 weight-only
設定、dtype／shape、finite values、sample IDs／labels、logits／prediction 對齊。
不混用不同 serial，不因資料不完整而自動縮小 144-pair 矩陣。

```bash
cd /work/kyle0724/project/quant_analysis
bash dataset_features/run_nano4.sh --serial 20004 --dry-run
sbatch dataset_features/run_nano4.sh --serial 20004 --stage preflight
# 完整矩陣
sbatch dataset_features/run_nano4.sh --serial 20004
# Pilot；後續 report/package 必須使用相同 scope
sbatch dataset_features/run_nano4.sh --serial 20004 --datasets cub --models vit_b16 --modes ft
```

Stages：`all,preflight,analyze,report,package`。`--dry-run` 只列印選取與命令，
不代表來源驗證成功。Input 預設 `<checkout>/../../data/w3_g128_analysis`；可用
`--input-root` 覆寫，launcher 也接受名稱與 `--serial` 相同的 serial 目錄。
Output 預設 `<input-root>/<serial>/analysis`，可用 `--output-root` 覆寫。
在 input parent 內的輸出須留在所選 serial 的 `analysis/` subtree。

逐 pair 輸出 `summary.json`、`layer_metrics.csv`、`fp_pca.npz`、完整
`coords_fp.npy`／`coords_w3_g128.npy` 和 `pca.png`。全體輸出 `all_pairs.csv`、
`coverage.csv`、`REPORT.md`、`analysis_manifest.json` 與 `w3_g128_results.zip`。
ZIP 只包含 compact 指標、PCA basis 與圖片，不包含 full-test features、logits、
完整 PCA coordinates、checkpoint。Per-layer 指標使用所有配對樣本；PCA 只在
FP final features 擬合一次，FP/W3 共用基底。

目前本機 `data/w3_g128_results/` 的保存結果來自 serial 20004，`all_pairs.csv`
有 144 列。此 compact copy 沒有原始 `.npy` arrays；保存的 complete manifest
不等於本機重新驗證了全部 raw sources。

## Paired FP/W3 t-SNE on nano4

從同一 paired schema 讀取 `samples.json`、`fp/final.npy` 與
`w3_g128/final.npy`；需要原始 arrays，compact copy 不足以執行。

```bash
bash dataset_features/run_tsne_nano4.sh --serial 20004 --dry-run
sbatch dataset_features/run_tsne_nano4.sh --serial 20004 --stage preflight
sbatch dataset_features/run_tsne_nano4.sh --serial 20004 --datasets cub --models vit_b16 --modes ft
sbatch dataset_features/run_tsne_nano4.sh --serial 20004
```

預設 `--sampling classes --class-fraction 0.1`，選取
`min(C,20,max(5,ceil(0.1*C)))` 個類別並保留那些類別的全部 test samples。
少於五類時全選。Class IDs 由排序後 labels 與 seed 選出，同 dataset 在不同
models/modes 需使用相同 class IDs；preflight 會檢查。此模式不使用 2,000-point
截斷；舊抽樣政策可指定 `--sampling samples --max-samples 2000`。

先 L2 normalize；必要時共用 PCA-50，再把 FP/W3 拼接後擬合一次 t-SNE。
每個 pair 輸出 `tsne.png`、`coordinates.csv`、`method.json`。Classes mode 預設
寫入 `<serial>/analysis/tsne_classes/`，samples mode 為 `analysis/tsne/`。
相符產物可跳過；`--overwrite` 重算圖。`--seed`、`--perplexity`、`--iterations`
和 `--class-fraction` 控制成本與可重現性。

類別子集在 PCA/t-SNE 之前選取；圖描述該子集的結構，不是完整 dataset 的分離
程度。不同 pair 的獨立 t-SNE 座標不可直接比較；應以不同 sampling seeds 評估
子集依賴。Preflight 不代表已測量完整 runtime／memory。

## Centered FP class distances

`centered_class_distance.py` 支援相同 16 datasets、3 models、3 modes，讀取
paired schema 的 FP final features 與兩條件 predictions。比較
`x/||x||` 與 `(x-mean_FP)/||x-mean_FP||`，保留原有逐塊 class-sum 計算。
來源、公式、CAL pilot／完整矩陣命令與輸出見
[CENTERED_CLASS_DISTANCE.md](CENTERED_CLASS_DISTANCE.md)。此流程不驗證未使用的
layers／logits，也不接受 schema-2 作為相同格式。

## 16-dataset, all-layer workflow (20004)

入口為 `run.py`，與上述 paired schema 分開。讀取 `tgda_manifest_v1`：
`<dataset>/dataset.json`，以及每個 condition 的 `manifest.json`、
`features.npy`、`predictions.csv`。依 canonical sample IDs 對齊 feature/prediction
rows。Feature rows 預設對應 predictions.csv 的實體順序；若 CSV 獨立排序，
manifest 必須明列 `feature_sample_ids`。Reader 使用 lazy mmap row alignment。

Feature shape／position names 必須在 metadata 或 config 中宣告並一致，沒有
自動 layer-map fallback。舊 ViT-B/16 exporter 若缺少宣告，須核對 exporter 後
明列 `[13,768]`、`block1` 到 `block12`、`encoder_norm`。`exemplars/` 和
hidden／incompatible backup directories 不當成 conditions。

先複製 `config/20004.example.json` 為自己的 config，填入實際 paths、primary
condition 和 condition grid。預設 input 為 repository 的 `data/20004/`，output
為 `dataset_features/results/20004/`；`expected_dataset_count=16`。
若要 group contrasts，提供經驗證的 16-dataset mapping，欄位為
`dataset_name,core_group,expanded_group`，labels 為 `FGIR`、`UFGIR`／`Ultra-FGIR`
或 boundary 的空白；不自動沿用舊 15-dataset mapping。

```powershell
conda run -n opencode_env python -B -m dataset_features.run --config dataset_features/config/20004.json --stage inventory
conda run -n opencode_env python -B -m dataset_features.run --config dataset_features/config/20004.json --stage validate
conda run -n opencode_env python -B -m dataset_features.run --config dataset_features/config/20004.json --stage metrics --dataset cub
conda run -n opencode_env python -B -m dataset_features.run --config dataset_features/config/20004.json --stage pca --dataset cub
conda run -n opencode_env python -B -m dataset_features.run --config dataset_features/config/20004.json --stage all
```

Stages：`inventory,validate,metrics,pca,comparisons,plot,report,verify,all`。
Validation 核對 source hashes、sample identities／labels、dimensions、finite values，
並鎖定 cohort 到 `state/`。獨立階段會重驗其 scope 的 validated sources；`all`
重新完整 validation，只重用 code/config/environment/input/receipt/output hashes
仍相符的階段。`verify` 再檢查 raw arrays；更換鎖定 cohort 需檢視 inventory
後以 `--rebuild` 重新 validation，保留舊 lock hash。單 dataset metrics/PCA
不能產生跨 dataset summary。

產物含逐位置 metrics／status、完整維度 PCA displacement、共用 FP basis、
固定條件 dataset comparisons、bootstrap／permutation、leave-one-dataset-out
與 matched-N CKA sensitivity、圖集與報告。Figures 使用 deterministic display
subset，主要 inference-position 指標使用全部配對樣本。Execution precision、
preprocessing、model/head/seed 等決定比較 strata；缺少 metadata 時限於描述性
結果。Condition families 依 manifest 設定區分 weight_only、A8、A4、QKV，
undefined 值保留 coverage／原因，不當成零。

## Schema-2 單 pair 讀取器

`unified_inputs.py` 保持獨立，不是上述兩種格式的自動 adapter。它驗證 complete
schema-2 manifest、來源／execution／preprocessing hashes、sample pairing 與
`features/<layer>.npy`，輸出 `layer_metrics.csv` 和 `sources.json`。

```powershell
conda run -n opencode_env python -B -m dataset_features.unified_inputs --fp <fp-condition> --quant <quant-condition> --output <new-output-directory>
```

Output 必須是來源外的新目錄。此入口不限制 dataset 名稱，但尚未提供完整
16-dataset 批次 orchestrator、coverage check 或 aggregate report。

## Provenance、驗證與限制

- 共用工具抽出後，paired `code_digest` 改記 `paired.py,metrics.py,common.py`。
  舊 manifest 不修改；舊 paired 產物的 report/package 仍會因 code identity
  不符而要求重新 analyze。需要重算時選擇新的 output-root，保留歷史結果。
- t-SNE 原有 code hash 會察覺 import 變更；既有 method.json 不改寫，需要
  新計算時選擇新的 output-root。Config-driven 計算與其 stage code files 未修改。
- CAL features 描述第一次 raw-image backbone call；accuracy/logits 可能保留
  crop/flip aggregation，不能把兩者當成相同 view。
- Dataset 是每個 model/mode 內的比較單位；144 cells 不是 144 個獨立 datasets。
  不同 checkpoints／classes／seed 的關聯是描述性，PCA/CKA 不能建立因果關係。
- Slurm headers 維持 partition `8gpus`、account `MST114495`；資源、Python 與
  logs 見各 launcher。計算使用 CPU，不以本機 checks 宣稱 nano4 已完成或資源足夠。
- 功能測試與 synthetic fixtures 在 repository 外暫存執行並清理。正式驗證仍由
  各入口檢查真實來源／配對／coverage／hash。此次整理的檢查證據見
  [整理紀錄](../docs/superpowers/plans/2026-11-01-feature-code-cleanup.md)。
