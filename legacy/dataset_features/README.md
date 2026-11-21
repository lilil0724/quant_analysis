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
16 個 dataset contexts；獨立 schema-2 CLI 仍為單 pair，paired/t-SNE/centered
入口另透過共用 adapter 支援所選矩陣。

2026-11-01 的整理只抽出共用檔案工具、解除舊入口依賴並封存舊程式，沒有改變
計算公式、分析階段、抽樣政策、圖形函式或既有輸出位置。舊 21-condition
與五-dataset 專用程式見 [封存說明](../archive/five_dataset_features_legacy_20261101/README.md)
及 [逐檔 SHA-256](../archive/five_dataset_features_legacy_20261101/manifest.json)。

本模組尚未完成根目錄 [AGENTS.md](../AGENTS.md) 的完整資料／結果目錄遷移。
目前繼續使用既有 `data/`、`analysis/`、`results/` 與 server input-root；本次
不搬動或改寫既有資料、結果和 manifests，也不宣稱已完成 raw-array 驗證。

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
| `paired_inputs.py` | 既有 W3/G128 分析的唯讀 schema-2 adapter、缺少 margin 欄位的舊公式推導、明確來源選取；不計算新的分析指標 |
| `common.py` | Paired／t-SNE 共用的 16-dataset 清單、CSV 與 SHA-256；不含分析邏輯 |
| `metrics.py` | 新舊流程共用的 CKA、class distances、norm、H-score、prediction transitions、PCA、Spearman |

`metrics.py` 與所有現行計算／畫圖函式保持整理前的實作。`centered_class_distance.py`
保持原有獨立 NumPy reader 與來源 hash 邏輯，不為合併工具而新增相依性。

## 環境與 baseline

從 repository root 執行。本機 Python 明確使用：

```powershell
$python = 'C:\Users\Public\miniconda3\envs\opencode_env\python.exe'
& $python -B -m dataset_features.run --help
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

### DAFB 逐層特徵與 logits

DAFB 的 raw-data 分析實作已移至 [codebase/raw](../../codebase/raw/README.md)。
以下入口會呼叫同一套實作，沿用 legacy 的結果和報告路徑。

從 quant_analysis/legacy 提交 [run_all_nano4.sh](run_all_nano4.sh)，指定 --datasets dafb。
預設執行 ViT-B/16、Swin-B、BEiTv2-B 各自的 FT/FZ/CAL，共九組 matched
unquantized AMP 與 Linear weight-only W3/G128。沿用 TGDA 保存的 schema-2
匯出，唯讀檢查兩組的 manifest、樣本、逐層 .npy、predictions.csv 和
logits.npy。輸入根目錄指到 serial 的上一層。

    sbatch dataset_features/run_all_nano4.sh --datasets dafb --serial 20004 \
      --input-root /work/kyle0724/data/w3_g128_analysis

--preflight-only 只驗證輸入，--models vit_b16 --modes ft 可縮小範圍。
舊的 run_dafb_nano4.sh 也可直接呼叫同一套 DAFB 分析。Nano4 預設使用 tgda 環境，
也可用 ANALYSIS_PYTHON 指定 Python。需要 NumPy、SciPy、pandas、Matplotlib、
scikit-learn 和 threadpoolctl。分析使用 CPU，Slurm 使用 8gpus partition。

每次新建 legacy/dataset_features/result/<run_id>/ 和
legacy/dataset_features/reports/<run_id>/REPORT.md。結果目錄由 .gitignore 的
**/result/ 規則忽略，REPORT.md 可追蹤。all_pairs.csv 匯總 accuracy、
swap、damage／rescue、final H-score 和 normalized margin Q10。各組另有逐層
CKA、class distances、norm、FP 共基底 PCA、matched 樣本的 joint t-SNE，
以及每個樣本的 margins。margin_deciles.csv 與 margin_auroc.csv 統計 FP
margin 與 prediction transitions 的關聯。centered_class_distance.csv 比較
FP final features 全域中心化前後的類別距離。classifier_bias.csv 和
competitor_ranks.csv 紀錄 fold-held-out logit bias 診斷及 W3 winner 的 FP
排名。DAFB 的完整表格和圖放在本次 result 的 dafb/ 子目錄。未指定 --datasets
時照舊跑原本的 16 個 datasets。也可用 --datasets cub,dafb --analyses paired
匯出兩個 datasets 的合併 all_pairs.csv。跨 dataset 的 margin matching
以原有 dataset 選取為準，需要 soy/cotton 和 other 兩組。DAFB 的 margin
分析使用同一 dataset 內的 FP margin 分位。產生的摘要可另行下載分析。

run.json 紀錄完成範圍、input／output hashes 和套件版本。run_all 的 Slurm log
為提交目錄的 weight_only_analysis_<JOBID>.out/.err。CAL 的逐層 features 來自 raw-image
backbone call，logits 可能聚合 crop/flip。

### 一次提交既有分析：run_all_nano4.sh

`run_all_nano4.sh` → `python -m dataset_features.run_all` 是新增的協調入口，
依序呼叫原本的 paired、centered、bias/competitor、margin matching、t-SNE；
不修改公式、抽樣、統計或 plot。條件仍固定為 **W3/G128 weight-only**，不是
10000–10016 accuracy-summary 或任意 bit/group grid 的分析器。

先把目前 `quant_analysis` 程式（含 `dataset_features/run_all.py`、新 launcher 及
先前的雙格式讀取器）同步到 Nano4，再從 repository root 提交：

```bash
cd /work/kyle0724/project/quant_analysis
export ANALYSIS_PYTHON="$HOME/.conda/envs/tgda/bin/python"

# 舊 paired input；新版改為實際 schema-2 serial 與 feature_analysis root。
SERIAL=20004
INPUT_ROOT=/work/kyle0724/data/w3_g128_analysis

# 只列出所有命令；不讀 arrays、不建立輸出、不提交 job。
bash dataset_features/run_all_nano4.sh \
  --serial "$SERIAL" --input-root "$INPUT_ROOT" --dry-run

# 唯讀檢查所有選取階段需要的來源；完成後檢查 Slurm log。
sbatch dataset_features/run_all_nano4.sh \
  --serial "$SERIAL" --input-root "$INPUT_ROOT" --preflight-only

# Preflight 通過後執行；預設自動產生新的 run ID。
sbatch dataset_features/run_all_nano4.sh \
  --serial "$SERIAL" --input-root "$INPUT_ROOT"
```

預設為全部 16 datasets × 3 models × FT/FZ/CAL。每個實際 run 都先完成全部
選取階段的 preflight，再開始分析；任何失敗都停止，已完成／失敗輸出固定保留。
使用 `--run-id NEW_ID` 可自行命名，或用 `--output-root ABSOLUTE_NEW_DIRECTORY`
指定一個全新的目錄；已有目錄一律拒絕，不會自動 overwrite/resume。

只跑部分分析或先做小範圍 pilot：

```bash
# 單 dataset 的 feature pilot，不執行跨 dataset 的 bias/margin 比較。
sbatch dataset_features/run_all_nano4.sh \
  --serial "$SERIAL" --input-root "$INPUT_ROOT" \
  --datasets cub --models vit_b16 --modes cal \
  --analyses paired,centered,tsne

# 兩組 dataset 都存在時，可執行完整流程；仍是 pilot scope，不是全矩陣結果。
sbatch dataset_features/run_all_nano4.sh \
  --serial "$SERIAL" --input-root "$INPUT_ROOT" \
  --datasets cub,soygene --models vit_b16 --modes cal
```

`--datasets`、`--models`、`--modes`、`--analyses` 都接受逗號分隔的名稱。
`--source-map PATH` 可明列舊 FT/FZ + 新 CAL，格式及來源一致性限制見下一節。
Margin matching 沿用原本 soy+cotton／other 分組，所以選取中必須同時有兩組。
其既有 import 會複製完整 FP/Q logits 作為不可變 input snapshot，需預留這部分
額外 server 空間；不會重新下載 W&B 或重新 inference。

產物根目錄為 `dataset_features/result/<run_id>/`：

| 路徑 | 內容 |
|---|---|
| `run.json`、`logs/`、`provenance/` | 本次協調命令、階段／失敗狀態、來源清單、環境、實際程式快照及 hash |
| `paired/` | 原本 paired 表、PCA、REPORT 與 compact ZIP |
| `centered/` | centered distance CSV、means、run_config |
| `bias/` | 原本 bias／competitor 表、報告與完成紀錄 |
| `tsne/` | 各 context 的圖、coordinates.csv、method.json |
| `margin/data/source/` | 原始 logits/predictions 的 input snapshot；通常不需下載回本機 |
| `margin/result/analysis/`、`margin/reports/analysis/` | 原本 margin 統計、圖與報告 |

這是既有分析的協調器，各子流程保留原本輸出／報告契約；不新增第二套統計報告。
每個子流程成功後核對原本 completion receipts，頂層 `run.json` 才標記 complete。
Slurm logs 為提交目錄的 `weight_only_analysis_<JOBID>.out/.err`。預設一個 GPU、
8 CPU、64 GB、24 小時，計算仍以 CPU 為主；這些是 launcher 設定，並非全矩陣
runtime/memory 的實測保證。現有 Python 環境需有 NumPy/pandas/Matplotlib/SciPy/
scikit-learn/threadpoolctl，job 不自動安裝套件。

### 舊 paired 與新版 schema 2 共用原分析

`paired`、`tsne_paired`、`centered_class_distance` 現在會逐 context 自動辨識
舊 paired 格式或新版 `tools.feature_analysis` 的 schema 2。分析條件仍固定為
**weight-only Linear W3/G128**；本次只修改讀取／配對，不變更既有指標、抽樣、
統計與 plot。額外的 CAL classifier-distance CSV 不在這些 backbone 分析中替代使用。

新版 context 的結構是：

```text
<input-root>/<serial>/<dataset>/<model>/<mode>/
  fp/{manifest.json,samples.json,metrics.json,predictions.csv,logits.npy,features/<layer>.npy}
  w3_g128_linear/{manifest.json,samples.json,metrics.json,predictions.csv,logits.npy,features/<layer>.npy}
```

量化 condition 依 manifest 的 W3/G128、activation_bits=null、scope=linear 選取，
不是只依目錄名猜測。多個相容 condition、smoke／failed source、錯誤 checkpoint／
execution／sample identity、checksum 不符均拒絕。FP/Q 必須有相同的 checkpoint、
data/config/code/HQQ hashes、preprocessing、execution、sample IDs 與 labels。
`coverage` 必須是 weight-only 且不含 classifier；calls 會驗證有執行，但不當成
跨 dataset 的量化 protocol，因其隨 test 大小與 CAL views 改變。

- Paired 指標使用所有已匯出 layers、兩條件 logits 與 predictions，且需有 `final`。
- t-SNE 只讀兩條件 final features；不要求未使用的 logits、predictions 或其他 layers。
- Centered distance 只讀 FP final features 與兩條件 predictions；不要求 logits 或 Q features。
- 新版 CSV 缺少的 normalized/true-class margin 從完整 logits 依原 float64 公式補算；
  paired 的 NumPy argmax 與 bias/competitor 的 stored-winner tie policy 各自維持原規則。
- 原始 manifests、arrays 和 CSV 不改寫，也不產生偽裝成原始 legacy export 的 manifest。
  新版未明文保存的 exclusions、quant_zero/scale 不從名稱或預設值補造；coverage
  只能證明所記錄的量化 sites。舊 FT/FZ 與新 CAL 仍需自行核對研究 protocol 的可比性。

新版推論若要支援全部既有分析，使用
`--save_features --save_predictions --save_logits --extract_layers all`。
只存 final 不足以重算未匯出的逐層結果。

同一 serial/root 下可直接改 input root，使用原有命令：

```bash
cd /work/kyle0724/project/quant_analysis
bash dataset_features/run_nano4.sh --serial 20010 \
  --input-root /work/kyle0724/data/feature_analysis --dry-run
sbatch dataset_features/run_nano4.sh --serial 20010 \
  --input-root /work/kyle0724/data/feature_analysis \
  --output-root dataset_features/results/20010/paired_NEW_RUN_ID
```

跨 root／serial 混合來源時，明確傳入 `--source-map`，格式見
[`config/sources.example.json`](config/sources.example.json)。每列指定 dataset/model/mode、
實際 source serial 與 context folder；相對 folder 以 source-map 所在目錄解析。
未列出的選取 contexts 仍使用 `--input-root/<serial>/...`，不自動搜尋其他 serial。
同一 context 重複列出會拒絕。請先把 example 路徑改成實際存在的來源。

```bash
# Pilot：舊 FT + 明確指定的新 CAL；source map 的來源必須真的包含完整必要檔案。
sbatch dataset_features/run_nano4.sh --serial 20004 \
  --input-root /work/kyle0724/data/w3_g128_analysis \
  --source-map dataset_features/config/sources.example.json \
  --datasets cub --models vit_b16 --modes ft,cal \
  --output-root dataset_features/results/dual_input_pilot_NEW_RUN_ID
```

`--serial` 在混合來源時是分析 cohort/output 的標籤；各原始 serial、folder、manifests
與 source-map hash 另記在 provenance，不把新 CAL 偽標為 20004。t-SNE 的 method.json
保存該 pair 的實際 source serial。`--source-map` 同樣可傳給 t-SNE、centered distance、
bias/competitor 與 margin import。Bias/margin 仍拒絕同一 model/mode 內混用不同已記錄的
export／precision／quantization protocols；先完成整個 CAL cohort 再做跨 dataset 比較。

本次沒有搬移歷史 data/result，也沒有改變舊分析的輸出配置。若更新輸入或 adapter，
請選新的 output/run ID；已有分析的 resume identity 會檢查 adapter code。

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
不自動混用不同 serial；只有明列 `--source-map` 才覆寫指定來源，且不因資料不完整而
自動縮小 144-pair 矩陣。

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
layers／logits；schema 2 由 adapter 明確讀取，不視為舊格式 manifest。

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
& $python -B -m dataset_features.run --config dataset_features/config/20004.json --stage inventory
& $python -B -m dataset_features.run --config dataset_features/config/20004.json --stage validate
& $python -B -m dataset_features.run --config dataset_features/config/20004.json --stage metrics --dataset cub
& $python -B -m dataset_features.run --config dataset_features/config/20004.json --stage pca --dataset cub
& $python -B -m dataset_features.run --config dataset_features/config/20004.json --stage all
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
& $python -B -m dataset_features.unified_inputs --fp <fp-condition> --quant <quant-condition> --output <new-output-directory>
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
