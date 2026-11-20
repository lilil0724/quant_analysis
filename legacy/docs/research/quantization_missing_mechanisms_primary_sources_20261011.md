# Dataset vs quantization robustness：四個尚可補足的機制

核對日期：2026-10-11。只核對原始文獻並提出可反駁的假說；本次沒有執行模型或分析實驗資料。已讀取前一份 `dataset-quantization-mechanisms-sources-20261010.md`，其既有的 activation covariance、head 方向、top-2 margin、damage/rescue 不再當成新發現。

## 1. 任務更新相對於量化格的可保留程度

**新增問題：** 不同 dataset 的 fine-tuning 將任務資訊寫入權重的方式，是否決定這些資訊在低 bit 下的保留程度？同樣的整體 weight MSE，可能遮蔽很不一樣的 task-vector 損傷。

**直接相關的 ViT 證據。** Kim et al., *Task Vector Quantization for Memory-Efficient Model Merging*, ICCV 2025，使用 CLIP ViT-B/32、ViT-L/14。§4.1–4.2 比較量化整個 fine-tuned checkpoint 與量化 `τ = θ_ft − θ_pre`；task vectors 的較窄 range 可減小重建誤差。§5 的 target/cross-task 比較也顯示低精度壓縮會改變任務專屬能力。[原文全文](https://arxiv.org/html/2503.06921)、[正式會議 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Kim_Task_Vector_Quantization_for_Memory-Efficient_Model_Merging_ICCV_2025_paper.pdf)。本次逐節核對以 arXiv HTML 為主。

**外推限制。** 該工作主要研究 checkpoint storage/model merging；保留 FP pretrained base 再加 quantized task vector，與部署 HQQ 整體量化權重的儲存成本、運算方式不同。它沒有證明 TGDA BEiTv2 FT 的問題是「更新被抹去」，也不能解釋完全凍結 backbone 的 FZ。FZ 是這個假說的有用對照。

**相鄰證據。** *Catastrophic Failure of LLM Unlearning via Quantization*, ICLR 2025，§4 討論小權重更新落回同一 quantization interval，並測試 LLM unlearning 的失效。[原文](https://arxiv.org/html/2410.16454v3)。這是 LLM unlearning 證據，不是 ViT transfer-learning 證明。該節「更新需超過一步才改變量化值」的直觀敘述不能當成必要條件：真正條件是是否跨過 rounding boundary；小於一步的更新也可能跨界。

**建議檢驗（本報告推論）。**

1. 驗證 pretrained 初始化與 FT checkpoint 的 lineage、架構與參數對應，只在可對應的 backbone 層計算 `τ`，新 head 另列。
2. 固定相同 grouping、scale、zero 及 rounding 規則，檢查 `Q_shared(W_pre + τ) − Q_shared(W_pre)` 對 `τ` 的方向與幅度保留；同時計算更新後仍在同一 quantization cell 的比例。`|τ|/step` 只能作描述，須保留初始格內位置。
3. HQQ 會調整量化參數，因此共享格是診斷用 counterfactual，還需另報各自實際 HQQ 結果；兩種條件不可混算。
4. 在選定層恢復「任務更新受損方向」的 FP 分量，與相同層、相同參數數量及修正 norm 的隨機方向／一般大誤差修正對照。若沒有較特異的恢復，或更新保留與 damage 無穩定關聯，便降低此假說優先度。

FP base + quantized task vector 可作機制對照，但不得宣稱為相同部署預算的 HQQ 改進。

## 2. Dataset 平均下的 class-specific logit bias

**新增問題：** 量化可能持續抬高某些類別、壓低另一些類別；這種可預測的偏移是否比無方向的 logit noise 更能解釋 dataset 差異？

**weight-only 的數學依據。** Nagel et al., ICCV 2019，§4.2 對固定權重誤差 `ΔW` 推得層輸出平均偏移 `ΔW E[x]`，並以 bias correction 處理。浮點 activation 不代表量化後輸出誤差必然零均值。[原文](https://arxiv.org/html/1906.04721)。它的主要實驗是 CNN，不能直接套用其 BatchNorm 統計估計法到 ViT LayerNorm。

**ViT 原始實驗。** Liu et al., *Post-Training Quantization for Vision Transformer*, NeurIPS 2021，§3.2 Eq.14–16 用 calibration data 估每個 output unit 的量化平均誤差，調整 layer bias。Table 3 中 ViT-B ImageNet 的 similarity-only 75.42% 加入 bias correction 為 75.81%。[原文](https://arxiv.org/html/2106.14156)、[正式論文](https://papers.nips.cc/paper/2021/file/ec8956637a99787bd197eacd77acce5e-Paper.pdf)。這是 W+A 量化與逐層校正，沒有直接驗證 W3 HQQ 的最終 class-logit 校正。

**與既有 centering 的差異（本報告推導）。** 設 `δz_i,c = zQ_i,c − zFP_i,c`，先去掉每筆資料所有類別共同的偏移：

```text
r_i,c = δz_i,c − mean_j(δz_i,j)
μ_c = mean_i∈calibration(r_i,c)
zQ_corrected_i,c = zQ_i,c − μ_c
```

第一行的每筆資料 scalar 不改變 argmax；第二行是跨樣本估每個類別的偏移向量，即使做過第一行仍可非零。此 `c` 指輸出類別，不是以真實標籤分組後得到的 bias；部署時不需要知道樣本標籤。

**最小可反駁試驗。** 在獨立 calibration split 估 `μ`，固定後只在 held-out samples 評估；若只能用現有 test logits，使用預先固定 folds 的 cross-fitting，明確標為事後診斷。比較校正前後 damage/rescue、micro/macro accuracy、class confusion，另以隨機置換 `μ` 的類別索引作對照。可記錄 centered logit error 中穩定均值分量，但不可只以 training/calibration error 下降聲稱 accuracy 改善。

若校正對 held-out accuracy 幾乎沒用，即使 mean error 大，也不能支持它是主要退化原因。若只在相同類別比例有效，須再檢驗 class-prior shift；不要把 calibration 組成造成的校正稱為 dataset intrinsic robustness。

## 3. 接近決策邊界的競爭類別數與完整 margin spectrum

**新增部分：** 既有 top-1/top-2 margin 描述最近一個 logit 競爭者；尚可問「同時有多少個類別接近 winner，以及量化後是哪一個超車」。這是 margin 主線的擴充，不是完全新議題。

**原始幾何支持。** DeepFool（CVPR 2016）§3.1 Eq.8 在 multiclass linear classifier 中對所有對手比較 `gap / ||w_a − w_j||`，最接近的決策超平面未必是 logit 第二名。[會議頁](https://openaccess.thecvf.com/content_cvpr_2016/html/Moosavi-Dezfooli_DeepFool_A_Simple_CVPR_2016_paper.html)、[原始 PDF](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Moosavi-Dezfooli_DeepFool_A_Simple_CVPR_2016_paper.pdf)。這是 input adversarial perturbation 的邊界幾何，不是量化或競爭者數目的實證。

**本報告的精確推導。** FP winner 為 `a`，對各 `j ≠ a` 定義 `m_j = z_a − z_j`。忽略 ties 時，量化後 winner 改變的條件為：

```text
存在 j ≠ a，使 δz_j − δz_a > m_j。
```

因此 top-2 margin 相同的兩個樣本，仍可能有不同的全體 gap 與 differential-error 結構。更多近鄰競爭者「可能」提高翻轉機會，但不能假設各競爭者誤差獨立，也不能宣稱 competitor count 單調決定風險。對固定同一模型與輸入，HQQ 誤差是確定的；機率解釋需指定跨樣本或擾動分布。

**最小檢驗。** 從 FP logits 建立固定尺度的 gap spectrum、近距離競爭者數（threshold 預先決定）、quantized winner 在 FP ranking 中的位置。檢查 FP top-2 以外的類別是否經常造成 damage。在控制 FP correctness、top-2 margin、logit scale、model/mode 與原始 class count 後，測試 spectrum 指標是否增加 held-out dataset/family 的預測能力。若只用事後 `δz` 重建 flip，那是機制描述，不能算 FP-only 預測成果。

未找到直接證明「近鄰競爭者數」可獨立解釋 HQQ ViT 跨 dataset robustness 的原始實驗；以上應標示為具數學動機的待驗證假說。

## 4. Group membership 與座標表示的敏感度

**新增問題：** 我們比較過 group size，但還沒有把「同樣 group size，哪些權重被分在同一組」當成獨立變因。重要的細小權重若與大幅值權重共享量化範圍，可能失去解析度；dataset 學到的通道配置或使用方向，可能與這種 grouping 互相作用。

**目前程式依據。** TGDA 的 `tools/eval_quant.py:54` 建立 `BaseQuantizeConfig`，沒有覆寫 axis；目前 vendored `hqq/hqq/core/quantize.py:1363` 的 config builder 預設 axis=1，而 `Quantizer.quantize` 在約 106–118 行依 axis reshape 成 groups，再沿 axis 取 min/max。這只確認目前程式，不能替代歷史 runs 的版本驗證。

**原始研究支持與界線。** [QuIP, NeurIPS 2023](https://arxiv.org/abs/2307.13304) 研究 weight-only quantization，使用正交變換降低 weight/Hessian 與座標軸的對齊程度，改善量化。它支持量化受表示座標影響，但實驗是 LLM；沒有直接證明 TGDA 的 group membership 造成 dataset 差異。此處的 group permutation 診斷亦非 QuIP 原演算法。

**保持 FP 函數的對照（本報告設計）。** 以 input-channel permutation matrix P 為例，先對 W 的 columns 排序，量化後再還原：

```text
W_quant_permuted = Q(W P^T) P
```

關閉 Q 時 `W P^T P = W`；可先驗證 FP tensor/output 完全還原。選定會改變實際 group membership 的 permutations，保持 bit、group size、其他 quantizer 設定與模型不變，只以不同的分組成員產生量化權重，再以原通道次序評估。需要確認 axis/reshape 與 metadata quantization，不能只改 tensor 儲存順序就假設 group 已改變。

預先固定數個 permutation seeds，比較 dataset retention 排序及各 dataset 的變異。若變動與 dataset 間差異同量級，robustness 應描述為 dataset–checkpoint–quantizer 的互動；若排序幾乎不變，便降低此因素的優先度。這是另一種量化表示的控制實驗，尚不是原部署路徑的等價低 bit 加速方案。

## 一個尚未充分利用的對照：FZ 的共用 backbone

若同一模型的各 dataset FZ checkpoints 真正只訓練 head，先核對所有被量化 backbone tensors 的 hash、buffers、pooling 與量化設定。確認相同後，才可以使用同一份量化 backbone/weight error 配上各 dataset head，研究相同權重擾動如何因輸入與決策函數而產生不同後果。這比只比較不同 FT checkpoints 更能排除「各 dataset 權重本來就不同」這個替代解釋。

FZ 檔名本身不證明上述條件；本次沒有取得實際 checkpoint 作 hash 比對。即使 backbone 相同，dataset input distribution、label space 與 head 仍未被分離，因此它不是純 dataset 因果試驗。

## 優先順序與共同界線

優先做 class-specific bias 的 held-out 校正：如果完整 paired logits 可用，成本低且有明確介入結果。其次做 competitor spectrum，檢查既有 margin 指標漏掉的部分。現有 serial 100004 的原始 paired logits 可供事後診斷；serial 20004 compact 檔案沒有完整 logits，不能直接照做。group membership 與 task-vector 保留則需要 checkpoint 和重新量化，適合第二階段；task-vector 另需精確 pretrained 初始化。FZ 的 backbone identity 檢查可在下一次 checkpoint audit 時一起做。

任何跨 dataset 比較皆先固定 checkpoint identity、pooling、split、preprocessing、precision 及量化 scope。舊 weight grid 與 serial 20004 不同評估版本的結果不可直接串成同一因果鏈。這份文件沒有證明上述任何假說已存在於目前資料。
