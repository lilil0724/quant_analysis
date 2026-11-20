"""Generate a traceable Traditional Chinese results report for run 20004."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .inputs import Context
from .run_config import RunConfig


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (ValueError, TypeError):
        return "NA"
    return "NA" if not __import__("math").isfinite(number) else f"{number:.{digits}f}"


def generate(config: RunConfig, contexts: list[Context], metrics: Path,
             comparisons: Path, pca: Path, plots: Path,
             validation: dict[str, Any], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    primary = config.primary_condition or "(尚未指定)"
    metrics_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []
    for context in contexts:
        folder = metrics / context.context_id
        metrics_rows.extend(_rows(folder / "layer_metrics.csv"))
        summary_rows.extend(_rows(folder / "condition_summary.csv"))
    comparison_rows = _rows(comparisons / "comparisons.csv")
    group_rows = _rows(comparisons / "group_comparisons.csv")
    sampling_rows = _rows(comparisons / "cka_sampling_sensitivity.csv")
    pca_rows: list[dict[str, str]] = []
    for context in contexts:
        pca_rows.extend(_rows(pca / context.context_id / "pca_summary.csv"))

    lines = [
        "# 20004：16-dataset feature 分析報告", "",
        "本報告只涵蓋具有完整 validation 與 stage receipts 的資料。結果是對現有 datasets/checkpoints 的 matched descriptive comparison；重複 quantization conditions、feature positions 不是新增 dataset-level replicas。", "",
        "## Run contract", "",
        f"- Dataset contexts：{len(contexts)}；預期數：{config.expected_dataset_count}",
        f"- Feature dtype / positions：以每個 context manifest 為準；第一個 context shape `{contexts[0].feature_shape if contexts else 'NA'}`",
        f"- FP baseline：`{config.baseline_condition}`；主要 quantization condition：`{primary}`",
        f"- Sensitivity conditions：{', '.join(config.sensitivity_conditions) or '未設定'}",
        f"- Primary group mapping：`{config.group_mapping_file or '未提供'}`",
        f"- Bootstrap/permutation seed：{config.resampling_seed}；repeats：{config.bootstrap_repeats}/{config.permutation_repeats}",
        f"- Practical correlation-equivalence bound：{_fmt(config.correlation_equivalence_bound)} (未設定時不宣稱相關性很小)",
        f"- Validation status：`{validation.get('status', 'unknown')}`；raw files checksummed：{validation.get('n_raw_files', 'unknown')}",
        "- Checkpoint/data hashes 來自 exporter metadata 時屬 recorded identity；只有直接讀檔比對才算 independently verified。", "",
        "## Dataset × primary-condition accuracy and final-position geometry", "",
        "| Dataset | Condition | FP acc | Q acc | Acc ratio | Abs drop (pp) | Final CKA | FP gap | Gap deterioration |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    contexts_sorted = sorted(contexts, key=lambda c: c.dataset)
    for context in contexts_sorted:
        summary = next((r for r in summary_rows if r["context_id"] == context.context_id and
                        r["condition"] == primary), None)
        metric = next((r for r in metrics_rows if r["context_id"] == context.context_id and
                       r["condition"] == primary and
                       int(r["position_id"]) == len(context.position_names) - 1), None)
        fp_metric = next((r for r in metrics_rows if r["context_id"] == context.context_id and
                          r["condition"] == config.baseline_condition and
                          int(r["position_id"]) == len(context.position_names) - 1), None)
        if not summary or not metric:
            continue
        fp_acc = float(summary["fp_accuracy"])
        ratio = float(summary["accuracy"]) / fp_acc if fp_acc else None
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            context.dataset, primary, _fmt(fp_acc), _fmt(summary["accuracy"]),
            _fmt(ratio), _fmt(100 * float(summary["accuracy_drop"])),
            _fmt(metric["cka"]), _fmt(fp_metric["separability"] if fp_metric else None),
            _fmt(metric["gap_deterioration"])))
    lines.extend(["", "## 預先指定的 dataset-level comparisons", "",
                  "每列以 dataset-context 為觀察單位。未知 model/head/training seed metadata 會標成 descriptive；這類 permutation p-value 不輸出。Repeated conditions 的結果另以 within-dataset 描述。", "",
                  "| Stratum | Condition | Comparison | N datasets | Spearman rho | Dataset bootstrap 95% CI | Practical bound | CI within bound | Permutation p | FDR q | Status |",
                  "|---|---|---|---:|---:|---|---:|---|---:|---:|---|"])
    for row in comparison_rows:
        if row.get("condition") != primary or row.get("analysis") is None:
            continue
        ci = f"[{_fmt(row.get('bootstrap_ci95_low'))}, {_fmt(row.get('bootstrap_ci95_high'))}]"
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            row.get("stratum", ""), row.get("condition", ""), row.get("analysis", ""),
            row.get("n_datasets", ""), _fmt(row.get("spearman_rho")), ci,
            _fmt(row.get("practical_equivalence_bound")),
            row.get("bootstrap_ci_within_practical_bound", "NA"),
            _fmt(row.get("dataset_permutation_p_value")), _fmt(row.get("fdr_q_value")),
            row.get("status", "")))
    lines.extend(["", "Leave-one-dataset-out influence range:", "",
                  "| Stratum | Comparison | LOO rho min | LOO rho max | LOO sign stable |",
                  "|---|---|---:|---:|---|"])
    for row in comparison_rows:
        if row.get("condition") == primary and row.get("analysis"):
            lines.append("| {} | {} | {} | {} | {} |".format(
                row.get("stratum", ""), row.get("analysis", ""),
                _fmt(row.get("loo_rho_min")), _fmt(row.get("loo_rho_max")),
                ("NA" if row.get("loo_sign_stable") in (None, "")
                 else row["loo_sign_stable"])))
    lines.extend(["", "## Within-dataset condition families", "",
                  "weight_only、A8、A4 與 QKV 分開描述。每列的 conditions 是同一 dataset/checkpoint 的重複測量，不作 dataset-level permutation；未知 family 不計算混合相關。NA 與不足三個有效條件保留原因。", "",
                  "| Dataset | Family | Valid conditions / total | Conditions | Spearman rho | Undefined reason | Status |",
                  "|---|---|---|---|---:|---|---|"])
    for row in comparison_rows:
        if row.get("analysis") == "within_dataset_gap_deterioration_vs_relative_loss":
            lines.append("| {} | {} | {} / {} | {} | {} | {} | {} |".format(
                row.get("dataset", ""), row.get("condition_family", "unknown"),
                row.get("n_conditions", ""), row.get("n_conditions_total", ""),
                row.get("conditions", ""), _fmt(row.get("spearman_rho")),
                row.get("undefined_reason", ""), row.get("status", "")))
    lines.extend(["", "## FGIR / UFGIR group comparisons", "",
                  "`median_difference = median(CKA_UFGIR) − median(CKA_FGIR)`，正值代表 UFGIR 的 CKA 較大。Permutation 是 dataset-level exchangeability comparison；若 model/head/seed metadata 未驗證，只報描述性差異、不輸出 p-value。若 mapping 未提供或 cohort coverage 不足，group conclusion 不可用；boundary datasets 不會被猜測成 core FGIR/UFGIR。", "",
                  "| Scope | UFGIR datasets (N) | FGIR datasets (N) | Median difference | Bootstrap 95% CI | Permutation p | Status |",
                  "|---|---|---|---:|---|---:|---|"])
    for row in group_rows:
        lines.append("| {} | {} ({}) | {} ({}) | {} | [{}, {}] | {} | {} |".format(
            row.get("scope", ""), row.get("datasets_ufgir", ""), row.get("n_ufgir", ""),
            row.get("datasets_fgir", ""), row.get("n_fgir", ""),
            _fmt(row.get("median_difference")), _fmt(row.get("ci95_low")),
            _fmt(row.get("ci95_high")), _fmt(row.get("permutation_p_value")),
            row.get("status", "")))
    lines.extend(["", "## Matched-N class-balanced CKA sensitivity", "",
                  "| Stratum | Common N | Sampling repeats | Median CKA | P10 | P90 | Status |",
                  "|---|---:|---:|---:|---:|---:|---|"])
    for row in sampling_rows:
        if row.get("status") == "matched_n_class_balanced_sampling_sensitivity":
            lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                row.get("stratum", ""), row.get("common_n", ""),
                row.get("sampling_repeats", ""), _fmt(row.get("median_cka")),
                _fmt(row.get("p10_cka")), _fmt(row.get("p90_cka")), row.get("status", "")))
    pca_primary = [r for r in pca_rows if r["condition"] == primary and r["outcome"] == "all"]
    median_disp = [float(r["relative_displacement_median"]) for r in pca_primary
                   if r.get("relative_displacement_median") not in ("", None)]
    lines.extend(["", "## Layer-wise CKA drop 與 PCA", "",
                  "`layer_drop_summary.csv` 分開列出 `1−CKA` 最大 feature position 與相鄰位置 CKA 曲線最大正下降；兩者都是累積表徵變化描述，不能定位造成 accuracy drop 的層。",
                  f"- 主要條件 `{primary}`：paired PCA summaries {len(pca_primary)} rows；有值 context/position 的相對 full-feature displacement median 中位數：{_fmt(float(__import__('numpy').median(median_disp)) if median_disp else None)}。",
                  f"- PCA uses one FP basis per context/position; explained variance、full displacement、2D projection energy 分別記錄於 `pca_summary.csv`。", "",
                  "## Figures and artifacts", "",
                  "Figures use the complete numeric tables for statistics; PCA plots draw the same deterministic display samples across conditions and positions.", "",
                  "- `cka_heatmap_primary.png/.svg`：16 datasets 的逐層 CKA。",
                  "- `layers_<dataset>.png/.svg`：每個 dataset 的逐層 inter/intra、gap 與 CKA。",
                  "- `scatter_*.png/.svg`：固定條件的 dataset-level scatter。",
                  "- `pca_atlas/`：每個 dataset 的 all-position paired PCA。", "",
                  "## Interpretation limits", "",
                  "- CKA 是 matched samples 的表示結構相似度；單獨不能判斷 accuracy、因果或 quantization robustness ranking。",
                  "- Gap/accuracy 的跨-dataset 關聯受 checkpoint、FP accuracy、class count、sample size 影響；不寫成 quantization 的因果效果。",
                  "- 不顯著不代表沒有關係。若 CI 很寬或未設定實務界線，結論標為 inconclusive。",
                  "- PCA 是視覺投影；2D 距離不取代 full-dimensional displacement，也不能證明分類資訊未變。",
                  "- 本 run 未執行新 inference/training；precision probe 缺失時，feature-storage precision sensitivity 未驗證。", ""])
    report_path = output / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    upstream = {}
    for label, folder in (("metrics", metrics), ("comparisons", comparisons),
                          ("pca", pca), ("plots", plots)):
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                upstream[f"{label}/{path.relative_to(folder).as_posix()}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    index = {"run_id": config.run_id, "config_identity": config.identity,
             "primary_condition": primary, "datasets": [c.dataset for c in contexts_sorted],
             "upstream_sha256": upstream}
    (output / "artifact_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return {"status": "complete", "report": str(report_path),
            "primary_condition": primary, "dataset_contexts": len(contexts_sorted)}
