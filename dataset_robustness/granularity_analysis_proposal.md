# 辨識粒度分組與 HQQ 量化退化關聯分析提案

## 1. 研究目的

本分析使用既有 HQQ 實驗，檢查辨識粒度分組是否與量化後的 accuracy
retention 及 bit-width degradation 有系統性關聯。主要比較傳統
fine-grained image recognition（FGIR）與 ultra-fine-grained image
recognition（Ultra-FGIR）資料集。

本分析不建立跨所有條件通用的 dataset robustness 排名，也不將觀察到的
組間差異解讀為辨識粒度的因果效果。Ultra-FGIR 組目前與 soybean、leaf
domain、資料來源及低 FP32 baseline 高度重疊，因此結果只能描述目前
benchmark 內的 association。

## 2. 研究問題與工作假設

### 主要研究問題

在相同 model、checkpoint kind 與 HQQ group size 下，從 4-bit 降到
2-bit 時，Ultra-FGIR datasets 是否比 FGIR datasets 出現更大的 accuracy
retention degradation？

### 次要研究問題

1. 組間差異是否在 3→2-bit 與 4→3-bit 兩段呈現相同方向？
2. 組間差異是否隨 model、checkpoint kind 或 group size 改變方向？
3. 改用 absolute accuracy drop 後，組間差異是否仍存在？
4. 排除低 FP32 baseline contexts 後，效果方向是否維持？
5. 結果是否由單一 dataset 或 dataset 分類邊界主導？

### 工作假設

- H1：Ultra-FGIR 組的 4→2-bit degradation 大於 FGIR 組。
- H2：若關聯具有跨背景一致性，主要組間差值在多數可比較的
  model/checkpoint/group-size strata 中應維持同方向。
- H3：若結果主要來自低 baseline 或特定 dataset，absolute-drop、baseline
  filtering 或 leave-one-dataset-out 分析會明顯削弱或翻轉效果。

以上為方向性工作假設，不以 p-value 作為接受或拒絕假設的門檻。

## 3. 現有資料與可識別範圍

主要輸入為：

```text
results_all/dataset_robustness/legacy/quant_summary.csv
```

目前檔案的 SHA-256 為：

```text
75c8f2fdb5b4510f286f7ddd2d5c81e338f2b5268a9f1c934008115d5efbe1f3
```

目前資料包含：

- 4,320 個 baseline-matched cells，沒有 unmatched cell；
- 15 datasets、3 models、3 checkpoint kinds；
- bits：1、1.58、2、3、4、5、6、8；
- group sizes：8、32、128、512；
- 每個 dataset/model/checkpoint/bit/group cell 只有一個 training seed。

`completeness.csv` 中的 `n_seeds=32` 是跨 32 個量化條件加總的
seed-condition count，不代表 32 個獨立 training seeds。分析不得將量化條件
或 cells 當成獨立訓練重複。

## 4. Dataset 分組規則

Dataset 分組必須由資料集的 label granularity 與原始文獻決定，不得根據
目前 accuracy、baseline 或量化結果反向分類。

### 4.1 核心分組（primary grouping）

| granularity_group | datasets | primary 中的角色 |
| --- | --- | --- |
| FGIR | aircraft、cars、cub、dogs、flowers、nabirds、pets、vegfru | 傳統 subordinate-category FGIR |
| Ultra-FGIR | soyageing、soygene、soyglobal、soylocal | soybean cultivar-level Ultra-FGIR |
| Boundary / excluded from primary | food、inat17、moe | 僅進入分組敏感度分析 |

核心分析共有 12 datasets：8 個 FGIR 與 4 個 Ultra-FGIR。`moe` 是
moeImouto anime face-character recognition dataset，不適合作為一般物件辨識
代表；`inat17` 橫跨多個生物 supercategories；`food` 的 dish-level label
是否應視為傳統 FGIR 存在分類邊界，因此三者不進入核心分組。

### 4.2 分組敏感度

| 版本 | FGIR comparator | Ultra-FGIR | 用途 |
| --- | --- | --- | --- |
| Core | 8 個核心 FGIR datasets | 4 個 soy datasets | Primary |
| Expanded | Core + food + inat17 | 4 個 soy datasets | 檢查合理邊界擴張 |
| Broad non-soy | 全部 11 個 non-soy datasets | 4 個 soy datasets | 檢查既有 11-vs-4 二分結果 |

若不同 grouping 得到不同方向，結果應判為 classification-boundary sensitive，
不得只選擇支持假設的版本。

### 4.3 分組依據

- Ultra-FGIR 的操作定義是區分同一物種內的 cultivar 或更細微類別；四個
  soybean datasets 屬 Ultra-FGVC benchmark family。
- FGIR 的操作定義是區分相同 basic-level category 下的 subordinate classes，
  例如 aircraft variants、car models、bird species 或 dog breeds。
- VegFru 原始工作將其定義為 domain-specific fine-grained recognition。

主要來源：

- Yu et al., *Benchmark Platform for Ultra-Fine-Grained Visual Categorization
  Beyond Human Performance*, ICCV 2021:
  https://openaccess.thecvf.com/content/ICCV2021/html/Yu_Benchmark_Platform_for_Ultra-Fine-Grained_Visual_Categorization_Beyond_Human_Performance_ICCV_2021_paper.html
- Rios et al., *Down-Sampling Inter-Layer Adapter for Parameter and Computation
  Efficient Ultra-Fine-Grained Image Recognition*:
  https://arxiv.org/abs/2409.11051
- Hou et al., *VegFru: A Domain-Specific Dataset for Fine-Grained Visual
  Categorization*, ICCV 2017:
  https://openaccess.thecvf.com/content_ICCV_2017/html/Hou_VegFru_A_Domain-Specific_ICCV_2017_paper.html
- moeImouto dataset provenance:
  https://github.com/arkel23/animesion/tree/main/classification_tagging

正式實作前，`dataset_granularity_mapping.csv` 必須逐列保存 dataset、group、
label level、source URL 與 inclusion status，讓分組可稽核。

## 5. 分析範圍與納入規則

### 5.1 Primary quantization scope

- bits：2、3、4；
- group sizes：8、32、128、512；
- models：BEiTv2、Swin、ViT；
- checkpoint kinds：cal、ft、fz；
- 僅納入 `baseline_matched=True` 且 outcome 為 finite 的 cells；
- 不截斷 `accuracy_ratio > 1`，也不移除 negative accuracy drop。

核心 primary panel 預期包含：

```text
12 datasets × 3 models × 3 checkpoint kinds × 3 bits × 4 group sizes
= 1,296 cells
```

任何 dataset 若缺少某個 primary condition，該 dataset 必須從相應完整 panel
中排除並記錄於 coverage audit，不得以其他條件補值。

### 5.2 Primary outcome

```text
accuracy_ratio = hqq_top1 / fp32_top1
```

### 5.3 Supporting outcomes

```text
absolute_drop_pp = fp32_top1 - hqq_top1
```

另報告各 dataset 的 baseline accuracy 與 HQQ top-1，避免將 ratio ranking
誤解為部署 accuracy ranking。

## 6. Primary estimand

先在每個 `dataset × model × checkpoint kind × group size` 內計算：

```text
D_4to2 = accuracy_ratio_4bit - accuracy_ratio_2bit
D_3to2 = accuracy_ratio_3bit - accuracy_ratio_2bit
D_4to3 = accuracy_ratio_4bit - accuracy_ratio_3bit
```

正值表示降低 bit-width 後 retention 下降。Primary estimand 是每個完全匹配
stratum 內的 dataset-median difference：

```text
Delta_s = median(D_4to2 | Ultra-FGIR) - median(D_4to2 | FGIR)
```

其中：

```text
s = model × checkpoint kind × group size
```

共有 3 × 3 × 4 = 36 個 primary strata。`Delta_s > 0` 表示該背景下
Ultra-FGIR 組從 4-bit 降至 2-bit 時退化更多。

每個 dataset 先產生一個 stratum-level effect，再計算 group median，因此
dataset 等權。不得把 1,296 cells 當成 1,296 個獨立樣本。

## 7. 分析步驟

### Step 1：資料與分類稽核

1. 驗證輸入 SHA-256、欄位 schema 與 primary panel 完整性。
2. 驗證每個 cell 的 baseline matching、finite outcome 與 seed count。
3. 讀入外部 `dataset_granularity_mapping.csv`，拒絕未知或重複 mapping。
4. 產生各 grouping 版本與 baseline sensitivity 的 coverage table。

### Step 2：Dataset-level degradation effects

對每個 dataset/model/checkpoint/group-size 計算 `D_4to2`、`D_3to2`、
`D_4to3`，並保留 `fp32_top1`、各 bit 的 HQQ accuracy 與 ratio。這是後續
組間比較的最小分析表。

### Step 3：Matched-stratum group contrasts

在 36 個 strata 內分別計算：

- FGIR 與 Ultra-FGIR 的 dataset count；
- 各組 median、mean、minimum、maximum；
- primary median difference `Delta_s`；
- mean difference sensitivity；
- group 內 individual dataset effects。

跨 strata 摘要報告正向、負向及不可比較的 strata 數，以及 `Delta_s` 的
median 與完整範圍。不以 cell count 加權。

### Step 4：完整 degradation curves

在每個 model/checkpoint 背景下畫出 2／3／4-bit group curves。圖中必須同時
顯示 group median 與 individual dataset trajectories，避免組平均遮蔽組內
異質性。group size 不可在沒有標示的情況下混合。

### Step 5：Sensitivity analyses

依第 8 節執行 baseline、outcome、dataset membership、aggregation 與 bit
scope 敏感度分析。所有預先指定版本都必須報告，不因結果方向選擇性省略。

### Step 6：Evidence grading

依第 9 節將結果判為 consistent association、context-dependent association
或 insufficient evidence，並逐條列出支持與反對證據。

## 8. 敏感度分析

### 8.1 Low-baseline sensitivity

主分析保留所有有效 contexts。Sensitivity A 排除：

```text
fp32_top1 < 30%
```

目前四個 soy datasets 的 36 個 model/checkpoint contexts 中，17 個低於
30%；其餘 11 個 datasets 的 99 個 contexts 中只有 1 個低於 30%。因此
threshold filtering 會造成不對稱 coverage。

在核心分組及 `baseline >= 30%` 規則下，只有 6/9 model/checkpoint 背景仍有
至少 2 個 Ultra-FGIR datasets；對應 24/36 group-size strata。其餘 strata
標記為不可比較，不補值、不借用其他背景，也不將 absence 解讀為沒有差異。

50% 不作主要門檻，因為它會使 9 個 model/checkpoint 背景中的 5 個完全失去
Ultra-FGIR 組。

### 8.2 Outcome sensitivity

重做 matched-stratum comparison，將主要效果換成：

```text
absolute_D_4to2 = absolute_drop_pp_2bit - absolute_drop_pp_4bit
```

此結果衡量降低 bits 所增加的 accuracy percentage-point loss。Ratio 與
absolute-drop 若方向不同，必須明確報告 metric dependence。

### 8.3 Dataset membership sensitivity

分別使用 Core、Expanded 與 Broad non-soy grouping。三個版本皆使用相同
primary quantization scope 與 estimand。

### 8.4 Leave-one-dataset-out

依序移除每個核心 dataset：

- 4 次 Ultra-FGIR leave-one-out；
- 8 次 FGIR leave-one-out。

每次重算 36 個 `Delta_s` 與方向一致率。若移除單一 dataset 後整體方向
翻轉，結果標記為 dataset-driven。

### 8.5 Aggregation sensitivity

- Primary：group median difference；
- Sensitivity：group mean difference。

### 8.6 Bit-scope sensitivity

- Primary：2／3／4-bit；
- Sensitivity：3／4-bit；
- Context supplement：完整觀測 bit grid。

完整 grid 的極低-bit collapse 與高-bit 微小差異分開解讀，不與 primary
effect 合併成單一平均值。

## 9. 證據分級與允許的結論

### Consistent association

適用於以下證據整體一致時：

- 多數可比較 strata 的 `Delta_s` 方向相同；
- 結果不是只出現在單一 model 或 checkpoint kind；
- individual dataset trajectories 支持 group summary；
- leave-one-dataset-out 不由單一 dataset 造成方向翻轉；
- absolute-drop 與可比較的 baseline sensitivity 沒有推翻主要方向；
- grouping sensitivity 不顯示結論完全依賴分類邊界。

允許的文字：

> 在目前 benchmark 與 HQQ 設定中，Ultra-FGIR 分組與較大的低位元量化退化
> 呈現跨多個背景的一致關聯。

### Context-dependent association

適用於整體存在差異，但效果隨 model、checkpoint、group size、outcome 或
dataset membership 明顯改變或翻轉時。

允許的文字：

> 辨識粒度分組與量化退化的關聯依賴模型與實驗背景，無法形成跨背景一致的
> group ordering。

### Insufficient evidence

適用於效果接近零、方向不一致、由單一 dataset 主導，或 sensitivity 後
Ultra-FGIR coverage 不足時。

允許的文字：

> 目前資料不足以區分辨識粒度關聯與 baseline、domain、資料來源及 checkpoint
> 品質所造成的差異。

無論分級為何，均不得使用「Ultra-FGIR 本質上較不耐量化」等因果或普遍性
敘述。

## 10. 預計圖表與輸出

建議將未來實作隔離於：

```text
results_all/granularity_robustness/
```

### Tables

| 檔案 | 內容 |
| --- | --- |
| `dataset_granularity_mapping.csv` | 分組、來源與 inclusion audit |
| `dataset_degradation_effects.csv` | 每個 dataset/stratum 的相鄰及 4→2-bit effects |
| `stratum_group_contrasts.csv` | 36 個 primary strata 的 group summaries 與 `Delta_s` |
| `granularity_coverage.csv` | 各 primary/sensitivity panel 的 dataset coverage |
| `granularity_sensitivity.csv` | grouping、baseline、outcome、aggregation、bit-scope 比較 |
| `leave_one_dataset_out.csv` | 每次移除 dataset 後的效果與方向一致率 |

### Figures

1. `granularity_retention_curves.png`：2／3／4-bit group median 與 individual
   dataset trajectories；
2. `granularity_stratum_contrasts.png`：36 個 `Delta_s` 的 heatmap 或 dot plot；
3. `granularity_baseline_coverage.png`：baseline distribution 與 filtering 後
   coverage；
4. `granularity_leave_one_out.png`：移除各 dataset 後的效果範圍；
5. `granularity_sensitivity_summary.png`：主要敏感度版本的方向與 effect size。

### Report and provenance

- `report.md`：研究問題、方法、結果、支持與反對證據、限制；
- `run_config.json`：input hash、script hash、group mapping hash、參數與時間；
- `verification.json`：coverage、conservation、hash 與 figure integrity checks。

## 11. 驗收條件

未來實作完成時，必須滿足：

1. 分組來自外部 mapping file，程式中沒有依結果或名稱片段偷偷重分類。
2. Primary input hash、mapping hash 與分析 script hash 已記錄。
3. 核心 1,296 cells 的納入數量可由 dataset × context × condition 重建。
4. 每個 primary stratum 在比較前具有 8 個 FGIR 與 4 個 Ultra-FGIR datasets，
   或在 coverage table 說明缺失。
5. Dataset 是組間比較的權重單位；cells 不被當成獨立 training replicates。
6. 36 個 primary strata 皆有結果或明確的不可比較原因。
7. Core、Expanded、Broad grouping 結果全部輸出。
8. 所有 low-baseline 排除均使用預先指定的 30% 規則並保留 coverage audit。
9. Ratio、absolute drop、median、mean、leave-one-out 與 bit-scope sensitivities
   均完成且沒有選擇性省略。
10. 報告使用 association language，明確揭露 soybean/domain/source、baseline、
    dataset-specific checkpoint 與 one-seed-per-context 限制。
11. 分析模組測試與獨立 verifier 通過，所有 figure source tables 可追溯。

## 12. 主要限制

- Ultra-FGIR 組只有 4 個 datasets，且全部來自 soybean benchmark family。
- Granularity 與 leaf domain、資料來源、class structure 及 baseline accuracy
  高度共線，現有資料無法分離其因果效果。
- 每個 checkpoint context 只有一個 training seed，不能估計跨訓練 seed 的
  reproducibility 或 confidence interval。
- Dataset-specific checkpoints 使 dataset 與 training outcome 無法完全分離。
- 低 baseline filtering 會不成比例地移除 Ultra-FGIR contexts，敏感度分析的
  coverage 本身也是結果的一部分。
- 目前只有 aggregate accuracy，不能分析 per-sample damage/rescue、calibration、
  class imbalance 或 worst-class robustness。

## 13. 本階段範圍

本文件只定義分析提案。此階段不修改既有四個 legacy scripts、不重新下載
W&B runs、不執行新訓練，也不產生新的研究結果。後續實作應建立獨立模組與
輸出目錄，避免改寫 `results_all/quant/` 的既有可重現產物。
