# 20004 Coding Plan

日期：2026-10-07。依據 `docs/analysis_20004_plan.md`，此文件定義實作工作、interface、相依順序與驗收。原始版本先記錄設計；目前程式已依此建立 config-driven workflow，以下「實作決策」及「驗收」仍作為範圍與預期行為參考。

## Implementation status

已建立 `run_config.py`、`inputs.py`、`layer_analysis.py`、`layer_pca.py`、`comparisons.py`、`plot_run.py`、`report_run.py`、`run_snapshot.py` 與 `run.py`，並加入 `config/20004.example.json` 和 README 使用說明。現階段包括 inventory/validation、全層 CKA 與 inter/intra 幾何、FP 共用 PCA basis、dataset-level bootstrap/permutation、leave-one-dataset-out 與 CKA sampling sensitivity、圖表、報告及 content-addressed stage receipts。

實際 `data/20004` 尚未提供，因此 exporter layout、真實 16-dataset coverage 與數值仍未驗證。初版只有語法檢查；code-review 修復新增 `tests/test_run_workflow.py` 的 synthetic regression 與端到端 all/resume/verify，涵蓋 sample reorder、NA、execution strata、condition families 與 raw corruption。正式資料 smoke、完整 cohort verify 與真實圖表檢查仍是資料到位後的 gate；未執行 inference。

修復後 reader 不再由 schema/hash 欄位存在推定 layer mapping。Legacy exporter 若沒有 shape/position 宣告，需先核對來源，再於 config 明確指定；feature row 順序預設與 predictions.csv 平行，獨立排序 CSV 必須以 manifest 的 `feature_sample_ids` 記錄 physical feature order。兩者都以逐 chunk 的 lazy row mapping 對齊到 dataset.json。

## 1. 實作決策

1. 延伸既有 `five_dataset_features`，新增 config-driven 入口 `run.py`。保留舊入口與 5-dataset defaults；不以修改全域 `DATASETS` 常數支援 20004。
2. 原始資料讀取 `C:/project/quant_analysis/data/20004`；正式輸出寫入 `C:/project/quant_analysis/five_dataset_features/results/20004`。
3. 新流程重用 `metrics.py` 的純計算函式：`centered_linear_cka`、`class_equal_cosine_distances`、`prediction_metrics`、`pca_fit`、`pca_project`、`norm_quantiles`。優先保持其程式內容不變，避免舊 snapshots 因共用 code hash 改變而失效。
4. 不從舊 `analyze.py`、`distance_accuracy.py`、`quantized_class_distance.py`、`paired_feature_pca.py` 載入 dataset/shape/grid 常數，也不把舊五-dataset CSV 當成 20004 輸入。
5. 現在只有一種已知 exporter 契約，先實作這種格式的 reader；20004 若是其他格式，再根據真實格式新增 adapter。不預先建立 plugin registry 或通用 exporter framework。
6. 新 snapshot 由 `run_snapshot.py` 管理，獨立於舊 `analysis_snapshot.py` 的固定路徑與 outputs。需要的 hash helper 可重用，但必須列入相依 code hashes。
7. Code、metadata、derived tables、figures 各自有明確責任：計算函式回傳數值；stage 負責檔案寫入與 provenance；plot 不重新計算原始指標。
8. 不增加 inference、training、W&B 或遠端操作；ZIP packaging 不列入這次必要範圍。

## 2. 檔案工作清單

以下路徑以 `C:/project/quant_analysis/five_dataset_features` 為基準。

| 檔案 | 動作 | 責任 |
| --- | --- | --- |
| `run.py` | 新增 | CLI、stage dispatch、依賴檢查、resume 與選定 dataset 重跑 |
| `run_config.py` | 新增 | config/schema、context key、condition/position 定義、config lock |
| `inputs.py` | 新增 | inventory、manifest validation、樣本對齊、memory-mapped feature reader |
| `layer_analysis.py` | 新增 | condition accuracy、全層 CKA/distance/norm、FP/Q deltas、layer-drop summaries |
| `comparisons.py` | 新增 | 固定條件 cohort、difficulty/group/accuracy 關聯、resampling、敏感性 |
| `layer_pca.py` | 新增 | 每個 context/position 的 FP basis、所有 Q 投影、paired displacement |
| `plot_run.py` | 新增 | metrics 圖、group 圖、相關 scatter、每層 PCA atlas |
| `run_snapshot.py` | 新增 | stage fingerprints、atomic completion receipts、staleness、完整驗證 |
| `report_run.py` | 新增 | 中文報告、coverage/exclusions、效果量與限制 |
| `config/20004.example.json` | 新增 | config 範例，明確標記尚未確認的 exporter/group 設定 |
| `tests/test_run_inputs.py` | 新增 | config、schema、sample alignment、coverage |
| `tests/test_layer_analysis.py` | 新增 | 每層輸出整合、FP/Q 比率、layer-drop 語義 |
| `tests/test_run_comparisons.py` | 新增 | dataset-level unit、cohort、resampling、undefined cases |
| `tests/test_layer_pca.py` | 新增 | 共用 FP basis、全層 coverage、full/2D displacement |
| `tests/test_run_pipeline.py` | 新增 | 端到端、resume、corruption、stage invalidation、失敗恢復 |
| `README.md` | 更新 | 20004 的輸入格式、CLI、續跑、artifact 說明 |

另更新 repository 根目錄 `README.md` 的分析索引。舊 `run_local.ps1` 先保留既有行為；新的 Python CLI 已足以供 Windows 執行，不先新增第二套 PowerShell orchestration。

## 3. 資料契約

### 3.1 Config

使用 standard-library dataclasses/JSON validation，避免只為設定引入新套件。以下是設計示意，尚非已驗證可執行 config：

```json
{
  "schema_version": 1,
  "run_id": "20004",
  "input_root": "C:/project/quant_analysis/data/20004",
  "output_root": "C:/project/quant_analysis/five_dataset_features/results/20004",
  "expected_dataset_count": 16,
  "source_format": "tgda_manifest_v1",
  "baseline_condition": "fp",
  "primary_condition": "w3_g128",
  "sensitivity_conditions": ["w4_g128"],
  "group_mapping_file": null,
  "chunk_size": 1024,
  "resampling_seed": 100,
  "bootstrap_repeats": 2000,
  "permutation_repeats": 9999,
  "correlation_equivalence_bound": null,
  "division_epsilon": 1e-8
}
```

- Primary/sensitivity conditions 必須存在，不能自動選擇分析結果最有利的條件。W3/G128 是否存在，待 inventory 驗證。
- `group_mapping_file=null` 允許 metrics/PCA 執行；group comparison 回報 `not_configured`，不自動猜 FGIR/UFGIR。
- `division_epsilon` 是數值除法容差，不是「關係很小」的實務門檻。FP gap 非正或低於容差時 relative gap drop 記 NA，absolute change 仍保留。
- Resampling counts 是起始運算預算，不是已驗證的 CI 精度承諾；正式 run 記錄實際有效/無效 repetitions。
- `correlation_equivalence_bound=null` 時只報估計與 CI，不宣稱「實務等價於零」。
- Feature positions 從可信 exporter metadata 讀入；舊 schema 沒有 position names 時，只有在 exporter 身分能確認為既有版本，才套用 block1–12 + encoder_norm 的明確 mapping。未知 tensor 語義需提供 config，不能由 shape 推定。
- 預期 feature shape/width 同樣由可信 metadata 或明確 config 決定，再核對實際 array header；不能只讀 array shape 就宣稱它符合來源契約。舊 exporter 的已知 shape 契約仍是 `[N,13,768]`，新入口能接受其他明確宣告的 feature spec。
- Resolved `run_config.json` 在 inventory 後保存於 output `state/`；`cohort_lock.json` 只在 validation 成功後建立。Inventory 同時記錄觀察到的來源、names、context、condition/position grid；後續不得默默縮減 cohort。

### 3.2 Context 與 table keys

`ContextKey`：`dataset, model, head, checkpoint_id, training_seed, inference_profile`。

`context_id` 由完整的 pairing identity 決定，包含上述欄位與 test-data/preprocessing identity。Checkpoint hash 來自 metadata，標記為 recorded identity；只有實際 checkpoint 可讀並核對後才標記 independently verified。

- Condition summary 唯一鍵：`context_id, condition`。
- Layer metric 唯一鍵：`context_id, condition, position_id`。
- PCA basis 唯一鍵：`context_id, position_id`。
- Cross-dataset strata 使用可比較的 `model, head, training_seed, inference_profile`；**不按 checkpoint hash 分跨 dataset 組**，因為不同 datasets 的 checkpoint 本來就不同。
- 對 absence/unknown metadata 使用明確狀態，不將未記錄的 seed 擅自填成 100。

每張 CSV 另帶可讀的 dataset/model/head/condition/position names。Undefined numeric cells 留空，原因記錄在 metric status 欄或 `metric_status.csv`，不能以 0 代替。

### 3.3 Reader interface

`inputs.py` 對 callers 提供小介面：

```python
discover(config) -> Inventory
validate(config, inventory) -> ValidatedRun
open_pair(validated_run, context_id, condition) -> MatchedPair
```

`MatchedPair` 隱藏 layout、prediction parsing、mmap、sample-ID reorder，提供 FP/Q labels、paired predictions、position metadata 與 chunked feature access。

`open_pair` 只接受有效 validation receipt。順序不同但 IDs/targets 一致時按 FP 順序對齊；逐 chunk 索引，不能因 advanced indexing 把全 feature array 複製進 RAM。

## 4. Stage graph 與 CLI

```text
inventory → validate → metrics → comparisons
                  └→ pca
metrics + comparisons + pca → plot → report → verify
```

`pca` 直接使用 validated FP/Q features 與 predictions；不依賴 metrics CSV。`plot`、`report` 只消費已完成的 derived artifacts。

規劃的 CLI（以下命令要等實作後才能執行）：

```powershell
Set-Location C:/project/quant_analysis

conda run -n opencode_env python -m five_dataset_features.run --config five_dataset_features/config/20004.json --stage inventory
conda run -n opencode_env python -m five_dataset_features.run --config five_dataset_features/config/20004.json --stage validate
conda run -n opencode_env python -m five_dataset_features.run --config five_dataset_features/config/20004.json --stage metrics --dataset cub
conda run -n opencode_env python -m five_dataset_features.run --config five_dataset_features/config/20004.json --stage pca --dataset cub
conda run -n opencode_env python -m five_dataset_features.run --config five_dataset_features/config/20004.json --stage all
conda run -n opencode_env python -m five_dataset_features.run --config five_dataset_features/config/20004.json --stage plot
conda run -n opencode_env python -m five_dataset_features.run --config five_dataset_features/config/20004.json --stage verify
```

- `--stage`：`inventory|validate|metrics|comparisons|pca|plot|report|verify|all`。
- `--dataset` 只適用 metrics/pca，選取該 dataset 的所有已鎖定 contexts；未完成全 cohort 時 comparisons/report 拒絕產生完整報告。
- `--stage all` 依序補足 missing/stale stages，跳過 fingerprint 相符且 artifacts 有效的結果；不把缺失 raw data 當作可跳過的失敗。
- 直接呼叫某 stage 不自動重跑昂貴上游；缺乏依賴時提供精確 rerun command。
- `--rebuild` 忽略該 stage 的可重用計算 cache，仍做正常 provenance/coverage 檢查；不代表跳過 validation，也不刪除原始資料。
- Input root 和 output root 不得相同或使 stage 寫入 raw source；禁止 path traversal 逃出指定 output root。
- 第一個 dataset 的正式檢查應在完整 validation 後執行；資料未到齊前用 synthetic fixture 開發，不假稱 20004 已通過 smoke。

## 5. 實作批次與驗收

### M1：Config、inventory、validation、snapshot 基礎

檔案：`run_config.py`、`inputs.py`、`run_snapshot.py`、`run.py` 的前三個 stages。

工作：

1. 定義資料契約與 config validation，實作既有 manifest layout reader。
2. Inventory 列出全部來源、contexts、conditions、shape、dtype、檔案大小、position mapping、coverage。
3. 依完整 identity/hash 驗證 manifests；chunk scan finite values、zero norms；檢查 sample uniqueness、targets、prediction correctness、shape 與 condition identity。
4. Validation 回報全部可定位 errors，不只停在第一個缺檔；不合格來源不得獲得 complete receipt。
5. 16-dataset count 與 expected context grid 通過後凍結 cohort；missing condition 進 audit，每個分析 scope 使用明確 common cohort。
6. 有 precision probes 就保存 storage-roundtrip sensitivity；沒有則標記 unverified。Probe 不等同 FP32 baseline。
7. 每個 stage 先在 staging directory 寫結果，成功後原子寫入 completion receipt；失敗時保留上一個有效結果並記錄本次失敗。

產物：`inventory.csv`、`coverage.csv`、`validation.json`、`exclusions.csv`、`cohort_lock.json`、`run_config.json`。

驗收：IDs 重排可正確恢復配對；重複 IDs、targets mismatch、raw hash mismatch、unknown positions 不能被悄悄接受；feature width/position count 不硬編碼成 768/13。

### M2：Condition summary 與全層 metrics

檔案：`layer_analysis.py`；重用現有 `metrics.py`。

工作：

1. 每次只載入一個 context 的 FP 與一個 Q condition，按 positions/chunks 計算。
2. Condition summary：FP/Q accuracy、ratio、relative loss、absolute drop pp、swap rate、C→W、W→C。
3. 各層：raw-centered linear CKA、normalized-feature class-equal cosine inter/intra、gap、norm quantiles。
4. Join 同一 context/position 的 FP 值，計算 signed/absolute deltas、gap deterioration、relative gap drop、distance ratio；FP accuracy/gap 分母退化時保留 NA。
5. 產生 `1−CKA` 最大位置與 adjacent-curve-drop 最大位置，保留 ties；block12→encoder_norm 分開標記，沒有正下降則為 none。
6. 每個 context 的完整結果先個別存檔，再以 deterministic key order 合併 run-level tables。選定 dataset 重跑只替換該 context outputs，aggregate receipt 隨之失效。

產物：`condition_summary.csv`、`layer_metrics.csv`、`layer_drop_summary.csv`、`metric_status.csv`。

驗收：非退化 FP self-CKA 接近 1、FP deltas=0；prediction metrics 可獨立從 predictions 重算；Q accuracy 改善時保留 ratio>1/negative loss；distance 使用 class-equal pair definition；動態 inventory 決定應有列數。

### M3：每層 paired PCA

檔案：`layer_pca.py`。

工作：

1. 每個 context/position 用全測試集 FP raw CLS fit mean、components、explained variance；一個 basis 只 fit 一次。
2. FP 和全部 Q 使用 `(X - mean_FP) @ components_FP`，不得重新 center Q 或各自 fit。
3. Coordinates 以 float32、basis 以 float64 保存；保存 position/context/basis identity 和對應 sample-ID hash。
4. 全維度 displacement、relative displacement、2D projected displacement energy fraction 以 chunks 計算；產生全體與四種 prediction outcomes 的 median/P90 summaries。
5. Display indices 用 context/position-independent 的 deterministic sampling 設定固定；跨 conditions/positions 使用同一代表 sample IDs。圖只抽樣，統計使用完整配對測試集。
6. 大型逐樣本結果按 context/condition/position 分檔，必要時 mmap 寫入；不先把全部 datasets 的 sample metrics 拼成一個 RAM DataFrame。

產物：`pca/bases/<context_id>/<position_id>.npz`、`pca/coords/<context_id>/<condition>/<position_id>.npy`、`pca/samples/...`、`pca_summary.csv`、`display_samples.json`。

驗收：所有條件共用 FP basis；零位移 energy fraction 為 NA；2D/全維度位移一致性可核對；degenerate PCA 列出狀態；每個有效 position 都有產物。

### M4：跨 dataset comparisons 與敏感性

檔案：`comparisons.py`。

工作：

1. 依 model/head/seed/inference_profile 建立固定條件的 common cohort，先選 final position；所有 comparisons 記錄實際 dataset IDs。
2. 計算 FP difficulty→gap deterioration、FP gap→accuracy loss、gap deterioration→accuracy loss、CKA→accuracy ratio/absolute drop。
3. Group mapping 核對 dataset aliases/source hash，輸出 core FGIR/UFGIR comparison、boundary sensitivity、各組 N；未配置分組就明確回報 unavailable。
4. Dataset-level bootstrap、dataset-block permutation、leave-one-dataset-out；多個 soybean datasets 增加 family aggregation/deletion sensitivity。
5. 全層結果標記 exploratory；預先固定 FDR test family，保留 undefined comparisons 與原因，不事後只選有利 layer。
6. Common-N、class-balanced matched sampling 的 CKA sensitivity 需回讀 validated arrays；各 repetition 的 FP/Q indices 一致，保存 seed/N/class coverage。不能從既有 CKA scalar 推算抽樣 CKA。
7. 在單一 dataset 內的 cross-condition 相關獨立輸出，按 weight-only/A8/A4/QKV family 分組；與跨 dataset inference 分開。
8. 若另做 paired sample bootstrap，讓同 replicate 的 predictions/features/positions 共用 indices；預設不對每個 layer/condition 都跑昂貴 bootstrap。

產物：`comparisons.csv`、`group_comparisons.csv`、`sensitivity_results.csv`、`comparison_cohorts.json`、`sampling_config.json`。

驗收：不把 layer/condition rows 當獨立 datasets；同 seed 重現結果；constant/tied/insufficient inputs 正確處理；CI 很寬或未設定 practical bound 時不輸出「已證明 CKA 無關」。Formal permutation 結果需附 exchangeability 假設。

### M5：圖表與中文 report

檔案：`plot_run.py`、`report_run.py`。

工作：

1. 主要圖：逐 dataset FP/Q distance curves、final-gap paired plot、difficulty scatter、CKA heatmap/curves、CKA/accuracy scatter、group comparison、layer-drop distribution。
2. 每個 dataset/context 的全層 PCA atlas，先顯示 FP+primary+sensitivity；其他 conditions 分頁。提供 class/outcome views 與固定 samples 的 arrows。
3. 同一 context/position 的 conditions 共用 axis limits；各圖顯示 condition、position names、N、FP baseline precision、PC explained variance、display sampling。
4. 16 datasets 使用 deterministic palette，不能沿用只有十色的 `tab10` 切片造成漏項。
5. 報告列出 coverage、來源與 exclusions；主要和 exploratory 結果分別標示，對假設報 supported/unsupported/inconclusive 與適用條件。
6. `plot` 不載入 `features.npy`，只讀 verified CSV/basis/coordinates；改樣式可單獨重畫，改抽樣規則則需重跑對應 PCA display-selection artifacts。

產物：`figures/*.png`、`figures/*.svg`、PCA atlas、`REPORT.md`、`artifact_index.json`。

驗收：圖完整涵蓋 config cohort；FP/Q 軸與選樣一致；圖數與名稱可預期；代表樣本不被寫成全測試統計；report 中的值能回查到 row keys。必要時逐頁視覺檢查 atlas 的 labels、legend、裁切與閱讀尺度。

### M6：完整 verify、resume、文件

檔案：完成 `run_snapshot.py`、CLI 的 `all/verify`、tests、READMEs。

工作：

1. 每個 stage fingerprint 包含其相關 config 欄位、code dependency hashes、upstream receipt hashes、輸入 identities 與所有 derived output hashes。
2. Final verify 重新核對 raw file hashes、table keys/coverage、FP/Q pairing、PCA basis references、所有 receipts/artifacts；輸出 machine-readable `verification.json`。
3. `verification.json` 另存為 snapshot 的驗證回條，不把它的 hash 寫入被驗證 snapshot 形成循環；同樣不計算 snapshot 自身的 self-hash。
4. `all` 在全部必要 stages 完成後才寫 final run snapshot；部分 cohort 不能標 complete。圖表失敗不得讓 metrics receipt 失效，但 final run 不得顯示完成。
5. 記錄 elapsed time、peak processing scope、artifact sizes，inventory 後才估計正式運算需求。
6. 跑適當現有/new suites，並使用一個實際 dataset 的 smoke 與全部 16 的 verifier；raw data 未到齊時僅報 synthetic/local tests 通過，不報正式 run 成功。

驗收：中斷可續跑；缺檔/corruption/code drift 明確失敗；單一 dataset 重跑能更新 aggregate；plot-only change 不要求重算 metrics；舊 CLI 與既有 tests 維持行為。

## 6. Invalidation 規則

| 改變 | 需失效的結果 |
| --- | --- |
| 某 context 原始 features/predictions/manifest | 該 context validation、metrics、PCA；相關 comparisons、plot、report、final verify |
| metrics 計算規則 | 所有受影響 metrics；下游 comparisons/plot/report；若 PCA 重用同一變更函式，PCA 也失效 |
| PCA fit/projection 規則 | PCA、PCA plots、相關 report；保持 distance/CKA metrics 有效 |
| primary condition、difficulty 定義、group mapping、inference scope | comparisons、相關圖/report；PCA basis 不因分組變更失效 |
| bootstrap/permutation seed/count | 相關 comparisons/CI、圖/report |
| 圖表樣式 | plot、report 的圖檔 identities、final verify |
| cohort/context/condition/position 集合 | 重新 inventory/validation，按新的 coverage 重建適用 stages；不得靜默接續舊集合 |

以 stage 使用的實際檔案與函式依賴決定 code hashes；file-level hash 可保守地多要求重跑，但不可漏掉共用計算依賴。

## 7. 測試策略

以公開 reader/stage interface 測試；不靠 monkeypatch 巨量 internals 驗證 orchestration。

- 延用現有 `test_metrics.py` 的基礎公式測試；新測試重點是新介面、整合、paired identities、動態 shape、stage state。
- Synthetic fixture 使用小型 manifests：多 datasets、非均等 class counts、FP+兩個 Q conditions、block positions+final norm。測試用 D/positions 在 config 中明確宣告，D 故意不等於 768，以測新入口不依賴全域常數；不把此 fixture 說成舊 exporter 原生輸出。另用舊契約 fixture 驗證 `[N,13,768]` 相容性。若需要 sampler/group tests，增加足夠 classes/contexts。
- 必要 cases：IDs 重排/重複、targets mismatch、missing conditions、FP accuracy=0、ratio>1、nonpositive gap、constant CKA、未定義 singleton intra、same IDs 的 shared PCA basis、normalization position 非 block。
- End-to-end fixture：inventory→validate→metrics/PCA→comparisons→plot/report→verify；核對獨立手算/參考值與 artifact coverage。
- Resume tests：完成 stage 可重用；修改 raw bytes 但不改 manifest 時 full verify 拒絕；改 code/config/output bytes 時相應 receipt 失效。
- Failure test：計算中斷不留下 complete receipt，上一個有效輸出保持可驗證；atomic replacement 不產生混合版本結果。
- 不測無關的格式微調，不大量重複已有公式 assertions。

實作後從 repository 根目錄執行：

```powershell
conda run -n opencode_env python -m unittest discover -s five_dataset_features/tests -v
```

## 8. 建議交付順序

1. **第一批：M1 + M2。** 得到可驗證的 16-dataset 全層 CKA/distance/accuracy tables，完成資料與 baseline 契約。
2. **第二批：M3。** 得到每層共同 FP PCA 與全維度 displacement；與第一批 calculations 可各自續跑。
3. **第三批：M4。** 凍結條件、cohort、group mapping 後做研究問題比較與敏感性。
4. **第四批：M5 + M6。** 完成圖表、中文 report、完整 verify 和使用文件。

20004 尚未存在時，先完成前三批的程式與 synthetic integration tests；正式 names/context/layout 驗證及真實資料 smoke 是後續 gate。資料到齊後，先正式 validate、選一個 dataset 跑 metrics/PCA，再全 cohort 執行 `all`。任何缺失、未驗證項與 inference limitations 都保留在最終報告。
