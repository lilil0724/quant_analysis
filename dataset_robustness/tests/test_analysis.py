"""Synthetic tests of inference boundaries, pairing and ranking semantics."""

import unittest

import numpy as np
import pandas as pd

from dataset_robustness.analyze import analyze, prepare_cells, rank_agreement


def fixture():
    rows = []
    # A and B exchange positions at 3 bits; C is always third.
    for model in ['model_one', 'model_two']:
        for bit, values in [(2., [.9, .8, .7]), (3., [.8, .9, .7]), (4., [.9, .8, .7])]:
            for dataset, ratio in zip(['A', 'B', 'C'], values):
                rows.append(dict(dataset_name=dataset, model_name=model, ckpt_kind='ft',
                                 ckpt_path=f'/ckpt/ft/{dataset}_{model}.pth',
                                 hqq_nbits=bit, hqq_group_size=8, baseline_matched=True,
                                 n_seeds=1, matched_seed_count=1, fp32_top1=80.,
                                 hqq_top1=80 * ratio, accuracy_ratio=ratio))
    return pd.DataFrame(rows)


class AnalysisTests(unittest.TestCase):
    def test_known_order_reversal_and_exact_rho(self):
        cells, _ = prepare_cells(fixture())
        tables = analyze(cells)
        orders = tables['pairwise_order_stability'].query('scope == "primary"')
        ab = orders.query('dataset_a == "A" and dataset_b == "B"').iloc[0]
        self.assertEqual((ab.a_higher, ab.b_higher, ab.ties), (2, 1, 0))
        self.assertTrue(ab.order_reverses)
        ac = orders.query('dataset_a == "A" and dataset_b == "C"').iloc[0]
        self.assertFalse(ac.order_reverses)
        correlations = tables['within_context_agreement'].query('scope == "primary"')
        self.assertAlmostEqual(correlations.query('bit_a == 2 and bit_b == 3').iloc[0].rho, .5)

    def test_missing_cell_excludes_dataset_from_entire_scope(self):
        raw = fixture()
        raw = raw[~((raw.dataset_name == 'C') & (raw.hqq_nbits == 2))]
        tables = analyze(prepare_cells(raw)[0])
        coverage = tables['coverage']
        self.assertFalse(coverage.query('scope == "primary" and dataset_name == "C"').included.any())
        self.assertTrue(coverage.query('scope == "without_2bit" and dataset_name == "C"').included.all())
        within = tables['within_context_agreement'].query('scope == "primary"')
        self.assertTrue(within.n_datasets.eq(2).all())
        self.assertTrue(within.rho.isna().all())

    def test_gains_not_clipped_and_negative_drop_retained(self):
        raw = fixture()
        raw.loc[0, ['hqq_top1', 'accuracy_ratio']] = [88., 1.1]
        cells, _ = prepare_cells(raw)
        row = cells.loc[0]
        self.assertAlmostEqual(row.accuracy_ratio, 1.1)
        self.assertAlmostEqual(row.accuracy_drop_points, -8.)

    def test_unmatched_zero_baseline_and_nonfinite_stay_in_audit(self):
        raw = fixture()
        raw.loc[0, 'baseline_matched'] = False
        raw.loc[1, 'fp32_top1'] = 0
        raw.loc[2, 'accuracy_ratio'] = np.inf
        cells, excluded = prepare_cells(raw)
        self.assertEqual(len(excluded), 3)
        self.assertEqual(len(cells), len(raw) - 3)
        self.assertIn('unmatched_baseline', excluded.loc[0, 'exclusion_reason'])
        self.assertIn('invalid_accuracy', excluded.loc[1, 'exclusion_reason'])
        self.assertIn('nonfinite_value', excluded.loc[2, 'exclusion_reason'])

    def test_distinct_checkpoints_must_not_be_averaged(self):
        raw = fixture()
        duplicate = raw.iloc[[0]].copy()
        duplicate['ckpt_path'] = '/ckpt/ft/other.pth'
        with self.assertRaisesRegex(ValueError, 'Multiple rows/checkpoints'):
            prepare_cells(pd.concat([raw, duplicate], ignore_index=True))
        raw.loc[0, 'ckpt_path'] = '/ckpt/ft/other.pth'
        with self.assertRaisesRegex(ValueError, 'identity changes'):
            prepare_cells(raw)

    def test_ratio_integrity_and_seedwise_ratio_semantics(self):
        raw = fixture()
        raw.loc[0, 'accuracy_ratio'] = .5
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            prepare_cells(raw)
        # The upstream summary's mean(seed-wise ratios) is authoritative.
        raw[['n_seeds', 'matched_seed_count']] = 2
        cells, _ = prepare_cells(raw)
        self.assertEqual(cells.loc[0, 'accuracy_ratio'], .5)

    def test_constant_ranking_is_undefined_and_ties_use_average_rank(self):
        n, rho, status = rank_agreement(pd.Series([1., 1., 1.]), pd.Series([1., 2., 3.]))
        self.assertEqual(n, 3)
        self.assertTrue(np.isnan(rho))
        self.assertEqual(status, 'constant_ranking')
        raw = fixture()
        raw.loc[raw.hqq_nbits.eq(2), ['hqq_top1', 'accuracy_ratio']] = [80., 1.]
        cells, _ = prepare_cells(raw)
        self.assertTrue(cells.loc[cells.hqq_nbits.eq(2), 'ratio_rank'].eq(2.).all())


if __name__ == '__main__':
    unittest.main()
