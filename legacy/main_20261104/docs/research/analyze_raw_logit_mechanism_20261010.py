"""Descriptive paired-logit measurements for the fixed serial-100004 protocol.

No inference, calibration, p-values, metric selection, or model fitting.
Run each dataset and each quantized condition sequentially to bound memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONDITIONS = ("hqq_n3_g8", "hqq_n3_g128", "hqq_n4_g8", "hqq_n4_g128")
BASE = "fp_unquantized"
SOY = {"soyageing", "soygene", "soyglobal", "soylocal"}
BINS = np.array([0, .25, .5, 1, 2, 4, 8, np.inf])
DEFAULT_SOURCE = Path(r"C:\Users\kyle\.codex\visualizations\2026\09\12\01a093a1-c8f1-7fd0-94a2-16db1fa2f2ab\hqq_confidence_serial_100004")
DEFAULT_OUT = Path(r"C:\project\quant_analysis\hqq_confidence\analysis\dataset_relationship_20261010")
DEFAULT_REFERENCE = Path(r"C:\Users\kyle\.codex\visualizations\2026\10\02\01a0fb4d-e996-79a0-8d58-cd96de264aad\hqq_confidence\q10_step1")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_raw(artifact, row, chunk):
    z = artifact["logits"]
    n, c = z.shape
    assert (n, c) == (int(row.n_samples), int(row.num_classes))
    for key in ("target", "sample_id", "correct", "top1_class", "top2_class"):
        assert artifact[key].shape == (n,), key
    assert len(np.unique(artifact["sample_id"])) == n
    for key, val in {"dataset_name": row.dataset_name, "condition": row.condition,
                     "checkpoint_sha256": row.checkpoint_sha256,
                     "git_commit": row.git_commit, "compute_dtype": row.compute_dtype}.items():
        assert str(artifact["metadata_" + key].item()) == str(val), key
    assert bool(artifact["metadata_amp_fp16"].item()) == bool(row.amp_fp16)
    targets = artifact["target"]
    stored_top1 = artifact["top1_class"]
    stored_top2 = artifact["top2_class"]
    stored_correct = artifact["correct"]
    stored_logit1, stored_logit2 = artifact["top1_logit"], artifact["top2_logit"]
    stored_margin = artifact["logit_margin"]
    tie_argmax_disagreement = 0
    tied_maxima_count = 0
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        zz = z[start:stop]
        assert np.isfinite(zz).all()
        top1 = stored_top1[start:stop]
        top2 = stored_top2[start:stop]
        idx = np.arange(stop - start)
        # FP16 ties need not share NumPy argmax's first-index tie rule.
        assert np.all(zz[idx, top1] == zz.max(axis=1))
        tie_argmax_disagreement += int(np.sum(zz.argmax(axis=1) != top1))
        tied_maxima_count += int(np.sum(np.sum(zz == zz.max(axis=1, keepdims=True),axis=1) > 1))
        assert np.all(top1 != top2)
        assert np.all(zz[idx, top2] == np.partition(zz, -2, axis=1)[:, -2])
        assert np.array_equal(zz[idx,top1],stored_logit1[start:stop])
        assert np.array_equal(zz[idx,top2],stored_logit2[start:stop])
        assert np.allclose(zz[idx,top1].astype(np.float64)-zz[idx,top2],stored_margin[start:stop],rtol=1e-6,atol=1e-7)
        assert np.array_equal(top1 == targets[start:stop], stored_correct[start:stop])
    accuracy = np.mean(artifact["correct"].astype(bool))
    assert abs(100 * accuracy - float(row.summary_accuracy)) < 1e-5
    return z, accuracy, tie_argmax_disagreement, tied_maxima_count


def q(values, p):
    return float(np.quantile(values, p))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args()
    assert args.chunk_size > 0
    manifest_path = args.source_root / "data" / "selected_runs.csv"
    manifest = pd.read_csv(manifest_path)
    chosen = manifest[(manifest.serial == 100004) & ~manifest.dataset_name.eq("inat17")
                      & manifest.condition.isin((BASE,) + CONDITIONS)].copy()
    datasets = sorted(chosen.dataset_name.unique())
    assert len(datasets) == 14
    assert len(chosen) == 70
    assert not chosen.duplicated(["dataset_name", "condition"]).any()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_config = json.loads((args.reference_dir / "run_config.json").read_text(encoding="utf-8")) if args.reference_dir.exists() else None
    reference_hashes = {item["path"]:item["sha256"] for item in reference_config["source_hashes"] if item["role"] == "baseline_artifact"} if reference_config else {}
    reference_q10 = pd.read_csv(args.reference_dir / "baseline_q10.csv").set_index("dataset_name") if reference_config else None
    current_hashes = []
    rows, bin_rows, audit_rows, meta_rows = [], [], [], []
    for dataset in datasets:
        part = chosen[chosen.dataset_name.eq(dataset)].set_index("condition", drop=False)
        assert set(part.index) == {BASE, *CONDITIONS}
        for key in ("checkpoint_sha256", "seed", "model_name", "git_commit", "amp_fp16", "compute_dtype", "ckpt_path"):
            assert part[key].nunique(dropna=False) == 1, (dataset, key)
        configs = [json.loads(value) for value in part.config_json]
        for key in ("num_images_train", "num_classes", "num_images_test", "train_trainval", "eval_split", "hqq_exclude"):
            assert len({json.dumps(config.get(key), sort_keys=True) for config in configs}) == 1, (dataset, key)
        config = configs[0]
        assert config["eval_split"] == "test"
        assert int(config["num_classes"]) == int(part.iloc[0].num_classes)
        assert int(config["num_images_test"]) == int(part.iloc[0].n_samples)
        baseline_row = part.loc[BASE]
        meta_rows.append(dict(dataset_name=dataset, num_images_train_config=int(config["num_images_train"]),
                              test_n=int(baseline_row.n_samples), num_classes=int(config["num_classes"]),
                              train_trainval_config=config["train_trainval"], checkpoint_sha256=baseline_row.checkpoint_sha256))
        with np.load(baseline_row.npz_path, allow_pickle=False) as ba:
            baseline, base_acc, baseline_tie_count, baseline_tied_maxima = validate_raw(ba, baseline_row, args.chunk_size)
            baseline_hash = sha256_file(baseline_row.npz_path)
            reference_match = baseline_hash == reference_hashes.get(baseline_row.npz_path) if reference_config else None
            if reference_config:
                assert reference_match, dataset
            current_hashes.append(dict(role="baseline_raw_npz",path=baseline_row.npz_path,sha256=baseline_hash,reference_sha256_match=reference_match))
            ids, target = ba["sample_id"], ba["target"]
            correct = ba["correct"].astype(bool)
            top1, top2 = ba["top1_class"], ba["top2_class"]
            n = len(target)
            std, margin = np.empty(n), np.empty(n)
            for start in range(0, n, args.chunk_size):
                stop = min(start + args.chunk_size, n)
                zz = baseline[start:stop].astype(np.float64)
                ii = np.arange(stop - start)
                std[start:stop] = zz.std(axis=1, ddof=0)
                margin[start:stop] = zz[ii, top1[start:stop]] - zz[ii, top2[start:stop]]
            assert np.all(std > 0)
            norm_margin = margin / std
            if reference_q10 is not None:
                assert reference_q10.loc[dataset,"checkpoint_sha256"] == baseline_row.checkpoint_sha256
                assert abs(q(norm_margin,.1)-reference_q10.loc[dataset,"normalized_margin_q10"]) < 1e-12
            bin_id = np.digitize(norm_margin, BINS[1:-1])
            for condition in CONDITIONS:
                qr = part.loc[condition]
                with np.load(qr.npz_path, allow_pickle=False) as qa:
                    quant, quant_acc, quant_tie_count, quant_tied_maxima = validate_raw(qa, qr, args.chunk_size)
                    current_hashes.append(dict(role="quant_raw_npz",path=qr.npz_path,sha256=sha256_file(qr.npz_path),reference_sha256_match=None))
                    qids = qa["sample_id"]
                    order = pd.Index(qids).get_indexer(ids)
                    assert np.all(order >= 0)
                    assert np.array_equal(qids[order], ids)
                    assert np.array_equal(qa["target"][order], target)
                    qtop1, qcorrect = qa["top1_class"][order], qa["correct"][order].astype(bool)
                    swapped = qtop1 != top1
                    damaged, rescued = correct & ~qcorrect, ~correct & qcorrect
                    rms, linf, gap_noise = np.empty(n), np.empty(n), np.empty(n)
                    for start in range(0, n, args.chunk_size):
                        stop = min(start + args.chunk_size, n)
                        delta = quant[order[start:stop]].astype(np.float64) - baseline[start:stop].astype(np.float64)
                        delta -= delta.mean(axis=1, keepdims=True)
                        delta /= std[start:stop, None]
                        rms[start:stop] = np.sqrt(np.mean(delta * delta, axis=1))
                        linf[start:stop] = np.max(np.abs(delta), axis=1)
                        ii = np.arange(stop - start)
                        gap_noise[start:stop] = delta[ii, top2[start:stop]] - delta[ii, top1[start:stop]]
                    # Strict bound, with a conservative floating-point exclusion zone.
                    tol = 1e-10 * (1 + np.maximum(norm_margin, 2 * linf))
                    guaranteed = norm_margin > 2 * linf + tol
                    violations = int(np.sum(guaranteed & swapped))
                    assert violations == 0
                    assert int(qcorrect.sum()) == int(correct.sum()) - int(damaged.sum()) + int(rescued.sum())
                    ratio = quant_acc / base_acc
                    assert abs(ratio - (1 - (damaged.sum() - rescued.sum()) / correct.sum())) < 1e-12
                    gap_cross = gap_noise > norm_margin + tol
                    # Baseline top2 strictly overtaking baseline top1 guarantees a changed winner.
                    assert np.all(swapped[gap_cross])
                    row = dict(dataset_name=dataset, condition=condition, test_n=n, num_classes=baseline.shape[1],
                               baseline_accuracy=base_acc, quant_accuracy=quant_acc, accuracy_ratio=ratio,
                               swap_rate=float(swapped.mean()), damage_count=int(damaged.sum()), rescue_count=int(rescued.sum()),
                               conditional_damage_rate=float(damaged.sum() / correct.sum()), conditional_rescue_rate=float(rescued.sum() / (~correct).sum()),
                               net_damage_per_swap=float((damaged.sum()-rescued.sum()) / swapped.sum()) if swapped.any() else 0.,
                               baseline_normalized_margin_q10=q(norm_margin, .1), baseline_normalized_margin_median=q(norm_margin,.5),
                               perturbation_rms_median=q(rms,.5), perturbation_rms_p90=q(rms,.9),
                               perturbation_linf_median=q(linf,.5), perturbation_linf_p90=q(linf,.9),
                               directed_gap_noise_median=q(gap_noise,.5), directed_gap_noise_p90=q(gap_noise,.9),
                               absolute_gap_noise_median=q(np.abs(gap_noise),.5), absolute_gap_noise_p90=q(np.abs(gap_noise),.9),
                               guaranteed_count=int(guaranteed.sum()), guaranteed_fraction=float(guaranteed.mean()),
                               guarantee_flip_count=violations, baseline_top2_overtake_count=int(gap_cross.sum()),
                               baseline_top2_overtake_fraction_of_swaps=float(gap_cross.sum()/swapped.sum()) if swapped.any() else 0.,
                               swapped_to_baseline_top2_fraction=float(np.sum(swapped & (qtop1 == top2))/swapped.sum()) if swapped.any() else 0.,
                               soy_group="soy4" if dataset in SOY else "other10")
                    rows.append(row)
                    audit_rows.append(dict(dataset_name=dataset, condition=condition, test_n=n, checkpoint_sha256=qr.checkpoint_sha256,
                                           ids_aligned=True, targets_aligned=True, logits_and_derived_valid=True,
                                           accuracy_algebra_valid=True, guaranteed_count=int(guaranteed.sum()), guarantee_flip_count=violations,
                                           baseline_run_id=baseline_row.run_id, quant_run_id=qr.run_id,
                                           baseline_numpy_argmax_tie_disagreement=baseline_tie_count,
                                           quant_numpy_argmax_tie_disagreement=quant_tie_count,
                                           baseline_tied_maxima_count=baseline_tied_maxima, quant_tied_maxima_count=quant_tied_maxima,
                                           baseline_sha256_matches_reference=reference_match,
                                           baseline_q10_matches_reference=True if reference_q10 is not None else None,
                                           reordered_quant_samples=not np.array_equal(order,np.arange(n))))
                    for index, (low, high) in enumerate(zip(BINS[:-1], BINS[1:])):
                        mask = bin_id == index
                        nb = int(mask.sum())
                        bin_rows.append(dict(dataset_name=dataset, condition=condition, margin_bin_low=low, margin_bin_high=high,
                                             count=nb, swap_count=int(swapped[mask].sum()), swap_rate=float(swapped[mask].mean()) if nb else np.nan,
                                             baseline_correct_count=int(correct[mask].sum()), damage_count=int(damaged[mask].sum()),
                                             rescue_count=int(rescued[mask].sum()), perturbation_rms_median=q(rms[mask],.5) if nb else np.nan,
                                             perturbation_linf_median=q(linf[mask],.5) if nb else np.nan,
                                             absolute_gap_noise_p90=q(np.abs(gap_noise[mask]),.9) if nb else np.nan,
                                             soy_group=row["soy_group"]))
                del quant
        del baseline
        print(dataset, "completed", flush=True)
    cells = pd.DataFrame(rows).merge(pd.DataFrame(meta_rows), on=["dataset_name", "test_n", "num_classes"], validate="many_to_one")
    if reference_config:
        old = pd.read_csv(args.reference_dir / "condition_metrics.csv")
        compared = cells.merge(old,on=["dataset_name","condition"],suffixes=("","_reference"),validate="one_to_one")
        assert len(compared) == 56
        for metric in ("swap_rate","accuracy_ratio","damage_count","rescue_count"):
            assert np.allclose(compared[metric],compared[metric+"_reference"],rtol=0,atol=1e-12),metric
    metrics = ["baseline_normalized_margin_q10", "perturbation_rms_median", "perturbation_rms_p90",
               "perturbation_linf_median", "perturbation_linf_p90", "absolute_gap_noise_p90",
               "num_images_train_config", "num_classes", "test_n"]
    corr = []
    for condition, part in cells.groupby("condition", sort=False):
        for metric in metrics:
            for outcome in ("swap_rate", "accuracy_ratio"):
                rho = float(spearmanr(part[metric], part[outcome]).statistic)
                corr.append(dict(condition=condition, metric=metric, outcome=outcome, dataset_n=14, spearman_rho=rho))
    correlations = pd.DataFrame(corr)
    summary_cols = ["accuracy_ratio", "swap_rate", "baseline_normalized_margin_q10", "perturbation_rms_median",
                    "perturbation_rms_p90", "perturbation_linf_p90", "absolute_gap_noise_p90", "guaranteed_fraction",
                    "baseline_top2_overtake_fraction_of_swaps", "net_damage_per_swap"]
    groups = cells.groupby(["condition", "soy_group"])[summary_cols].mean().reset_index()
    groups["dataset_n"] = groups.soy_group.map({"soy4":4,"other10":10})
    bins = pd.DataFrame(bin_rows)
    # Each dataset gets equal weight within the group; no pooled-image comparison.
    group_bins = bins.groupby(["condition", "soy_group", "margin_bin_low", "margin_bin_high"], dropna=False).agg(
        dataset_n=("swap_rate", "count"), equal_dataset_swap_rate=("swap_rate", "mean"),
        dataset_swap_rate_min=("swap_rate","min"), dataset_swap_rate_max=("swap_rate","max"),
        equal_dataset_perturbation_rms=("perturbation_rms_median", "mean"), total_samples_descriptive=("count", "sum")).reset_index()
    contributors = bins[bins["count"].gt(0)].groupby(["condition","soy_group","margin_bin_low","margin_bin_high"],dropna=False).dataset_name.apply(lambda s:",".join(sorted(s))).rename("contributing_datasets").reset_index()
    group_bins = group_bins.merge(contributors,on=["condition","soy_group","margin_bin_low","margin_bin_high"],how="left")
    baseline_bins = bins[bins.condition.eq(CONDITIONS[0])].copy()
    baseline_bins = baseline_bins.merge(cells[cells.condition.eq(CONDITIONS[0])][["dataset_name","test_n"]],on="dataset_name",validate="many_to_one")
    baseline_bins["baseline_sample_fraction"] = baseline_bins["count"] / baseline_bins.test_n
    baseline_bin_groups = baseline_bins.groupby(["soy_group","margin_bin_low","margin_bin_high"]).agg(
        equal_dataset_fraction=("baseline_sample_fraction","mean"),dataset_n=("dataset_name","nunique")).reset_index()
    outputs = {"raw_logit_dataset_conditions.csv":cells, "raw_logit_correlations.csv":correlations,
               "raw_logit_validation.csv":pd.DataFrame(audit_rows), "raw_logit_margin_bins.csv":bins,
               "raw_logit_equal_dataset_groups.csv":groups, "raw_logit_equal_dataset_margin_bins.csv":group_bins,
               "raw_logit_dataset_metadata.csv":pd.DataFrame(meta_rows),
               "raw_logit_baseline_margin_distribution.csv":baseline_bins[["dataset_name","soy_group","margin_bin_low","margin_bin_high","count","test_n","baseline_sample_fraction"]],
               "raw_logit_equal_dataset_margin_distribution.csv":baseline_bin_groups}
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)
    provenance = dict(source_root=str(args.source_root), manifest=str(manifest_path),
                      manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      serial=100004, datasets=datasets, excluded_datasets=["inat17"], conditions=list(CONDITIONS),
                      baseline="unquantized weights with AMP FP16, not FP32", chunk_size=args.chunk_size,
                      normalized_margin="(baseline top1 logit - top2 logit)/std(baseline class logits,ddof=0)",
                      perturbation="(quant-baseline minus per-sample class mean)/baseline class-logit std",
                      guarantee="normalized margin > 2*centered normalized Linf + conservative float64 tolerance",
                      sample_bins=BINS.tolist(), dataset_weighting="equal within each condition/group",
                      p_values=False, new_inference=False,
                      reference_dir=str(args.reference_dir), baseline_reference_hash_matches=14 if reference_config else None,
                      current_raw_npz_sha256=current_hashes,
                      raw_quant_hash_reference_available=False,
                      reference_condition_metrics_56_pairs_match=True if reference_config else None,
                      wandb_artifact_digest_is_not_file_sha256=True,
                      selected_runs=chosen[["dataset_name","condition","run_id","checkpoint_sha256","artifact_name","artifact_version","artifact_digest","npz_path"]].to_dict("records"))
    (args.output_dir / "raw_logit_run_config.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding="utf-8")
    fig, axes = plt.subplots(1,2,figsize=(12,4.4),layout="constrained")
    colors = {"soy4":"#dc6b42","other10":"#437eaa"}
    x=np.arange(len(BINS)-1)
    for offset,group in ((-.19,"soy4"),(.19,"other10")):
        values=baseline_bin_groups[baseline_bin_groups.soy_group.eq(group)].sort_values("margin_bin_low").equal_dataset_fraction.to_numpy()
        axes[0].bar(x+offset,values,width=.38,label=group,color=colors[group])
    axes[0].set_xticks(x,["[0,.25)","[.25,.5)","[.5,1)","[1,2)","[2,4)","[4,8)","[8,∞)"])
    axes[0].tick_params(axis="x",labelsize=9)
    axes[0].set_ylabel("Mean within-dataset sample fraction")
    axes[0].set_xlabel("Baseline normalized margin bin")
    axes[0].set_title("A. Baseline margin distribution")
    axes[0].legend(frameon=False)
    group_index=groups.set_index(["condition","soy_group"])
    folds={metric:np.array([group_index.loc[(condition,"soy4"),metric]/group_index.loc[(condition,"other10"),metric] for condition in CONDITIONS]) for metric in ("swap_rate","perturbation_rms_median")}
    x=np.arange(4)
    axes[1].bar(x-.19,folds["swap_rate"],width=.38,label="Swap-rate group mean ratio",color="#dc6b42")
    axes[1].bar(x+.19,folds["perturbation_rms_median"],width=.38,label="Normalized RMS median group mean ratio",color="#6e8b59")
    axes[1].axhline(1,color="#888888",linestyle="--",linewidth=1)
    axes[1].set_xticks(x,["W3 g8","W3 g128","W4 g8","W4 g128"])
    axes[1].set_ylabel("Soy4 / other10 (fold difference)")
    axes[1].set_title("B. Swap vs perturbation group contrasts")
    axes[1].legend(frameon=False,fontsize=8,loc="upper center")
    axes[1].set_ylim(0,12.7)
    for offset,metric in ((-.19,"swap_rate"),(.19,"perturbation_rms_median")):
        for xi,yi in zip(x+offset,folds[metric]):
            axes[1].text(xi,yi+.12,f"{yi:.2f}",ha="center",va="bottom",fontsize=9)
    fig.suptitle("Existing paired logits: 14 datasets, serial 100004, unquantized AMP-FP16 baseline",fontsize=12)
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(args.output_dir / "raw_logit_mechanism.png",dpi=180)
    plt.close(fig)
    total_guaranteed = int(cells.guaranteed_count.sum())
    text = ["# serial 100004：原始 paired logits 機制量測", "",
            "此分析只讀既有 predictions.npz，沒有重新推論。14 datasets（排除 inat17）、相同各自 checkpoint 的 unquantized AMP-FP16 baseline，和 W3/W4 × g8/g128 四條件。每 dataset 只有一個 checkpoint/seed；dataset 與其微調模型不可分離。", "",
            "## 定義與驗證", "",
            "baseline normalized margin = (z_top1−z_top2)/std(z)。擾動 = δz 減每張影像的全類別平均，再除 baseline 全類別 logits 的標準差。RMS 代表整體擾動，Linf 為最大單類別擾動；扣共同位移是因為所有 logits 加同一常數不影響 argmax。directed gap noise = (δz_baseline_top2−δz_baseline_top1)/std(z)，只追蹤 baseline runner-up，不能取代全部競爭類別。", "",
            f"70 份 artifact 的 logits/derived predictions/summary accuracy 通過驗證；56 pairs 的 sample IDs、targets、checkpoint、seed、AMP、compute dtype、Git 一致且 aligned。accuracy_ratio = 1−(damage−rescue)/baseline_correct_count 經逐 pair 驗證。margin > 2×centeredLinf 是保留 top1 的充分條件（嚴格不等式並排除 float64 容差區）：共 {total_guaranteed:,} 個 pair-samples 符合，flip 違例 0。這是對 observed perturbation 的事後 deterministic bound，非事前保證。", "",
            "所有14份 baseline 都驗證 stored top1 達最大值、stored top2 達次大值、stored logit_margin 與原始 top1−top2 一致。FP16 ties 使用 artifact stored top1 的 tie rule，不以 NumPy argmax 取代；validation CSV 分別記錄 tied-maxima 與 NumPy tie-rule disagreement 次數。", "",
            "## 全 14 datasets 的逐條件描述相關", "",
            "Spearman 只以 dataset 為單位，每條件 n=14；不報 p-value，全部預先列出的 metrics 同時公開，沒有選最好相關係數。", "",
            "| condition | Q10→swap | Q10→ratio | RMS median→swap | RMS median→ratio | RMS p90→swap | Linf p90→swap | abs gap noise p90→swap |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        cp = correlations[correlations.condition.eq(condition)]
        def rho(metric,outcome="swap_rate"):
            return float(cp[(cp.metric.eq(metric)) & cp.outcome.eq(outcome)].spearman_rho.iloc[0])
        text.append(f"| {condition} | {rho(metrics[0]):.3f} | {rho(metrics[0],'accuracy_ratio'):.3f} | {rho('perturbation_rms_median'):.3f} | {rho('perturbation_rms_median','accuracy_ratio'):.3f} | {rho('perturbation_rms_p90'):.3f} | {rho('perturbation_linf_p90'):.3f} | {rho('absolute_gap_noise_p90'):.3f} |")
    text += ["", "## soy4 與其餘10 datasets（dataset equal）", "",
             "| condition | group | ratio | swap | Q10 | RMS median | RMS p90 | Linf p90 | guaranteed fraction |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in groups.itertuples():
        text.append(f"| {row.condition} | {row.soy_group} | {row.accuracy_ratio:.4f} | {row.swap_rate:.4f} | {row.baseline_normalized_margin_q10:.3f} | {row.perturbation_rms_median:.3f} | {row.perturbation_rms_p90:.3f} | {row.perturbation_linf_p90:.3f} | {row.guaranteed_fraction:.4f} |")
    text += ["", "same-margin bins 使用所有 dataset 共同的 absolute normalized margin 門檻 [0,.25,.5,1,2,4,8,∞]，各 dataset bin swap rate 再等權平均；空 bin 不入平均，CSV 提供 dataset_n 和 sample count。這避免把 test images 當作跨 dataset 獨立重複，但寬 bins 內仍有 margin 分布差異，且各 checkpoint 的擾動方向/相關性不同。", "",
             "soy4 的 normalized RMS median 比 other10 高約 1.27–1.41 倍，swap rate 卻高約 6.66–9.93 倍；Q10 則約 0.029 vs 1.163。資料同時呈現 margin 與擾動的關聯，無法由此分離獨立效應。soy4 的較弱整體 robustness 不能簡化成量化擾動遠較大。", "",
             "baseline normalized margin <0.25 的影像比例先在各 dataset 內計算再等權平均：soy4 為64.3062%，other10為4.1575%。Q10 的dataset中央値為0.030239 vs0.905615，比group mean更不受flowers的高Q10支配。這是baseline分佈的直接差異，非新推論。", "",
             "![既有 paired logits 的 margin 分布與 swap/擾動對照](raw_logit_mechanism.png)", "",
             "全數固定 bins 均保留於 CSV；下表完整列出各 bin 的等dataset平均 swap、實際 contributing dataset 數與 dataset swap rate range。soy4 在 [0.5,1) 的 swap 並沒有一致更高；不能把一個寬 bin 當成完全調整margin後的因果比較。", "",
             "| condition | margin bin | soy4 mean (n; range) | other10 mean (n; range) |", "|---|---|---|---|"]
    for condition in CONDITIONS:
        for low, high in zip(BINS[:-1],BINS[1:]):
            values=[]
            for group in ("soy4","other10"):
                row=group_bins[(group_bins.condition.eq(condition)) & group_bins.soy_group.eq(group) & group_bins.margin_bin_low.eq(low)].iloc[0]
                values.append(f"{row.equal_dataset_swap_rate:.4f} ({int(row.dataset_n)}; {row.dataset_swap_rate_min:.4f}–{row.dataset_swap_rate_max:.4f})")
            text.append(f"| {condition} | [{low:g},{high:g}) | {values[0]} | {values[1]} |")
    text += ["", "[0.5,1) bin 的 contributing datasets 在四條件都相同：soy4 = soyageing, soygene, soyglobal, soylocal；other10 = aircraft, cars, cub, dogs, flowers, food, moe, nabirds, pets, vegfru。全部bins的 contributing_datasets、各dataset sample count 和rate另外完整保留，避免 sparse bins 造成不透明的group比較。", "",
             "## dataset size", "",
             "num_images_train_config、num_classes、num_images_test、train_trainval、eval_split 經 baseline 與四條件 config_json 一致性驗證。train n 是 logged config 的訓練影像數，不是重新掃描實際 server data，也不是 test n；train_trainval 的紀錄另存 metadata CSV。全部逐條件大小相關係數在 raw_logit_correlations.csv，不能解讀為資料量增加的因果效果。", "",
             "## 限制與來源", "",
             "擾動與 margin 皆為相同測試 logits 的 retrospective measurement；不能聲稱 external prediction。跨 dataset baseline accuracy、類別數、資料家族、checkpoint 訓練/難度均混雜。RMS/Linf 不能直接指出哪一層造成錯誤，也沒有把量化擾動當成獨立或均勻噪音。margin 與 perturbation 都可能重要；相關不等於兩者的獨立貢獻或因果。", "",
             f"Primary source: `{manifest_path}`（SHA256 `{provenance['manifest_sha256']}`）及其中明列的70份 artifact；raw_logit_run_config.json 保留 run ID、W&B artifact digest/version、checkpoint SHA、原始 path，以及本次重新計算的70份 NPZ SHA256。W&B artifact digest 不等同 NPZ SHA256。", "",
             f"參照資料：`{args.reference_dir}`。14 baseline NPZ SHA256 與10/02 run_config source_hashes 的 baseline_artifact reference 全一致；本次14 Q10逐dataset與baseline_q10.csv誤差小於1e-12；56pairs的 swap、ratio、damage、rescue 亦與10/02 condition_metrics.csv全一致。56 quant raw NPZ沒有既有SHA256 reference，本次只記錄hash作未來provenance，並未聲稱與舊reference hash吻合。舊 summary 的 Q10→swap rho −0.967033 為四條件 median；本次四係數 median 同為 −0.967033，並無資料變化。", "",
             "流程定義見 [hqq_confidence/README.md](../../README.md)；可重現腳本為 `docs/research/analyze_raw_logit_mechanism_20261010.py`。"]
    (args.output_dir / "raw_logit_mechanism_report.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    print("OUTPUT", args.output_dir, "CELLS",len(cells),"GUARANTEED",total_guaranteed,"VIOLATIONS",cells.guarantee_flip_count.sum(),flush=True)


if __name__ == "__main__":
    main()
