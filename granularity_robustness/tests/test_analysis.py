"""Synthetic tests for granularity grouping, contrasts and coverage."""

import unittest

import numpy as np
import pandas as pd

from granularity_robustness.analyze import (
    build_effects,
    contrast_effects,
    leave_one_out,
    load_mapping,
    prepare_cells,
    retention_level_contrasts,
)


def synthetic_cells():
    rows = []
    degradations = {"f1": .10, "f2": .20, "u1": .30, "u2": .50}
    for dataset, degradation in degradations.items():
        baseline = 20.0 if dataset == "u2" else 80.0
        for bit, ratio in [(2.0, .9 - degradation), (3.0, .85), (4.0, .9)]:
            rows.append(dict(
                dataset_name=dataset, model_name="m", ckpt_kind="ft",
                ckpt_path=f"/ckpt/ft/{dataset}.pth", hqq_nbits=bit,
                hqq_group_size=8, hqq_top1=baseline * ratio, fp32_top1=baseline,
                accuracy_ratio=ratio, baseline_matched=True, n_seeds=1,
                matched_seed_count=1,
            ))
    return pd.DataFrame(rows)


def synthetic_mapping():
    return pd.DataFrame({
        "dataset_name": ["f1", "f2", "u1", "u2"],
        "core_group": ["FGIR", "FGIR", "Ultra-FGIR", "Ultra-FGIR"],
    })


class GranularityAnalysisTests(unittest.TestCase):
    def test_exact_dataset_level_effect_and_group_contrast(self):
        cells, excluded = prepare_cells(synthetic_cells())
        self.assertTrue(excluded.empty)
        effects, coverage = build_effects(
            cells, synthetic_mapping(), scenario="test", group_col="core_group",
            bits=[2., 3., 4.], low=2., high=4.)
        self.assertTrue(coverage.included.all())
        values = effects.set_index("dataset_name").ratio_d_4to2
        self.assertAlmostEqual(values["f1"], .1)
        self.assertAlmostEqual(values["u2"], .5)
        contrast = contrast_effects(effects, scenario="test", metric="ratio_d_4to2",
                                    aggregation="median").iloc[0]
        self.assertTrue(contrast.comparable)
        self.assertAlmostEqual(contrast.group_difference, .25)

    def test_low_baseline_is_audited_and_not_silently_replaced(self):
        cells, _ = prepare_cells(synthetic_cells())
        effects, coverage = build_effects(
            cells, synthetic_mapping(), scenario="threshold", group_col="core_group",
            bits=[2., 3., 4.], low=2., high=4., baseline_min=30.)
        excluded = coverage[~coverage.included]
        self.assertEqual(excluded.dataset_name.tolist(), ["u2"])
        self.assertEqual(excluded.exclusion_reason.iloc[0], "baseline_below_threshold")
        contrast = contrast_effects(effects, scenario="threshold", metric="ratio_d_4to2",
                                    aggregation="median").iloc[0]
        self.assertFalse(contrast.comparable)
        self.assertTrue(np.isnan(contrast.group_difference))

    def test_retention_level_contrast_is_matched_by_bit_and_condition(self):
        cells, _ = prepare_cells(synthetic_cells())
        curves = cells.merge(synthetic_mapping(), on="dataset_name").rename(
            columns={"core_group": "granularity_group"})
        levels = retention_level_contrasts(curves)
        two_bit = levels[levels.hqq_nbits.eq(2.0)].iloc[0]
        self.assertTrue(two_bit.comparable)
        self.assertAlmostEqual(two_bit.fgir_median_accuracy_ratio, .75)
        self.assertAlmostEqual(two_bit.ultra_fgir_median_accuracy_ratio, .50)
        self.assertAlmostEqual(two_bit.retention_gap, -.25)

    def test_leave_one_out_keeps_dataset_as_weighting_unit(self):
        cells, _ = prepare_cells(synthetic_cells())
        effects, _ = build_effects(
            cells, synthetic_mapping(), scenario="test", group_col="core_group",
            bits=[2., 3., 4.], low=2., high=4.)
        result = leave_one_out(effects)
        self.assertEqual(set(result.left_out_dataset), {"f1", "f2", "u1", "u2"})
        self.assertTrue(result.n_comparable_strata.eq(0).all())

    def test_prepare_rejects_duplicate_cells_and_bad_ratio(self):
        raw = synthetic_cells()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prepare_cells(pd.concat([raw, raw.iloc[[0]]], ignore_index=True))
        raw.loc[0, "accuracy_ratio"] = .123
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            prepare_cells(raw)

    def test_mapping_requires_exact_dataset_inventory(self):
        import tempfile
        from pathlib import Path

        mapping = pd.DataFrame({
            "dataset_name": ["only"], "core_group": ["FGIR"],
            "expanded_group": ["FGIR"], "broad_group": ["FGIR"],
            "label_level": ["x"], "primary_included": [True],
            "source_url": ["https://example.com"], "notes": ["x"],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.csv"
            mapping.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_mapping(path, ["only", "missing"])


if __name__ == "__main__":
    unittest.main()
