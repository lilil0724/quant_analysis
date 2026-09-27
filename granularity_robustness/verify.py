"""Independently verify saved granularity-robustness artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageStat


KEY = ["dataset_name", "model_name", "ckpt_kind", "hqq_nbits", "hqq_group_size"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path):
    config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    source = Path(config["input_file"])
    mapping_source = Path(config["mapping_file"])
    assert digest(source) == config["input_sha256"]
    assert digest(mapping_source) == config["mapping_sha256"]
    assert digest(Path(__file__).with_name("analyze.py")) == config["script_sha256"]
    mapping = pd.read_csv(root / "dataset_granularity_mapping.csv", keep_default_na=False)
    assert mapping.dataset_name.nunique() == 15
    assert mapping.core_group.value_counts().to_dict() == {"FGIR": 8, "Ultra-FGIR": 4, "": 3}
    raw = pd.read_csv(source)
    assert len(raw) == config["n_input_cells"] == 4320
    assert raw.baseline_matched.astype(str).str.lower().eq("true").all()
    effects = pd.read_csv(root / "dataset_degradation_effects.csv")
    assert len(effects) == 12 * 3 * 3 * 4 == 432
    assert not effects.duplicated(["dataset_name", *KEY[1:3], "hqq_group_size"]).any()
    curves = pd.read_csv(root / "primary_curve_cells.csv")
    assert len(curves) == 1296
    assert set(curves.hqq_nbits) == {2.0, 3.0, 4.0}
    levels = pd.read_csv(root / "retention_level_contrasts.csv")
    assert len(levels) == 3 * 3 * 4 * 3 == 108 and levels.comparable.all()
    for row in levels.itertuples():
        frame = curves[(curves.model_name == row.model_name)
                       & (curves.ckpt_kind == row.ckpt_kind)
                       & (curves.hqq_group_size == row.hqq_group_size)
                       & (curves.hqq_nbits == row.hqq_nbits)]
        medians = frame.groupby("granularity_group").accuracy_ratio.median()
        np.testing.assert_allclose(row.retention_gap,
                                   medians["Ultra-FGIR"] - medians["FGIR"])
    baseline_contexts = pd.read_csv(root / "baseline_contexts.csv", keep_default_na=False)
    assert len(baseline_contexts) == 15 * 3 * 3
    primary = raw[raw.dataset_name.isin(mapping[mapping.core_group.isin(["FGIR", "Ultra-FGIR"])].dataset_name)]
    for row in effects.itertuples():
        frame = primary[(primary.dataset_name == row.dataset_name)
                        & (primary.model_name == row.model_name)
                        & (primary.ckpt_kind == row.ckpt_kind)
                        & (primary.hqq_group_size == row.hqq_group_size)].set_index("hqq_nbits")
        np.testing.assert_allclose(row.ratio_d_4to2,
                                   frame.loc[4.0].accuracy_ratio - frame.loc[2.0].accuracy_ratio)
        np.testing.assert_allclose(row.absolute_d_4to2,
                                   (frame.loc[2.0].fp32_top1 - frame.loc[2.0].hqq_top1)
                                   - (frame.loc[4.0].fp32_top1 - frame.loc[4.0].hqq_top1))
    contrasts = pd.read_csv(root / "stratum_group_contrasts.csv")
    assert len(contrasts) == 36 and contrasts.comparable.all()
    for row in contrasts.itertuples():
        frame = effects[(effects.model_name == row.model_name)
                        & (effects.ckpt_kind == row.ckpt_kind)
                        & (effects.hqq_group_size == row.hqq_group_size)]
        medians = frame.groupby("granularity_group").ratio_d_4to2.median()
        np.testing.assert_allclose(row.group_difference,
                                   medians["Ultra-FGIR"] - medians["FGIR"])
    sensitivity = pd.read_csv(root / "granularity_sensitivity.csv")
    expected = set(config["scenarios"])
    assert set(sensitivity.scenario) == expected
    assert sensitivity.scenario.nunique() == len(expected) == 8
    low = sensitivity[sensitivity.scenario.eq("core_baseline_ge30_ratio_median")].iloc[0]
    assert low.n_comparable_strata == 24
    coverage = pd.read_csv(root / "granularity_coverage.csv")
    assert (coverage.included == coverage.exclusion_reason.fillna("").eq("")).all()
    leave_out = pd.read_csv(root / "leave_one_dataset_out.csv")
    assert len(leave_out) == 12 and leave_out.left_out_dataset.nunique() == 12
    assert set(leave_out.left_out_group) == {"FGIR", "Ultra-FGIR"}
    manifest = pd.read_csv(root / "plots_manifest.csv")
    assert len(manifest) == 6 and not manifest.file.duplicated().any()
    for row in manifest.itertuples():
        assert (root / row.source_table).is_file()
        with Image.open(root / row.file) as image:
            assert image.width >= 800 and image.height >= 400
            assert max(ImageStat.Stat(image.convert("RGB")).var) > 100
    assert (root / "report.md").is_file()
    result = dict(
        status="passed", input_cells=len(raw), primary_dataset_effects=len(effects),
        primary_strata=len(contrasts), sensitivity_scenarios=len(sensitivity),
        leave_one_out_runs=len(leave_out), figure_integrity_checks=len(manifest),
        source_script_and_mapping_hashes_verified=True,
    )
    (root / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "results_all/granularity_robustness")
    print(json.dumps(verify(parser.parse_args().results_dir.resolve()), indent=2))
