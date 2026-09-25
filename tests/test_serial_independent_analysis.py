import argparse
import math
import unittest

import pandas as pd

from corr_quant import source_serials_match
from summarize_quant import checkpoint_kind, prepare_summary


def summary_args(serials=None):
    return argparse.Namespace(serials=serials, subset_datasets=None, subset_models=None)


def raw_row(**overrides):
    row = {
        'serial': 700,
        'hqq': True,
        'dataset_name': 'demo',
        'model_name': 'vit_b16',
        'ckpt_path': '/results/cal_ckpts/demo_vit_b16_cal.pth',
        'seed': 1,
        'hqq_top1': 70.0,
        'hqq_nbits': 2.0,
        'hqq_group_size': 8,
        'top1': None,
    }
    row.update(overrides)
    return row


class SerialIndependentSummaryTests(unittest.TestCase):
    def fixture(self):
        path = '/results/cal_ckpts/demo_vit_b16_cal.pth'
        return pd.DataFrame([
            raw_row(serial=900, hqq=False, seed=1, ckpt_path=path, top1=80.0),
            raw_row(serial=700, seed=1, ckpt_path=path, hqq_top1=70.0),
            raw_row(serial=701, seed=1, ckpt_path=path, hqq_top1=74.0),
            raw_row(serial=901, hqq=False, seed=2, ckpt_path=path, top1=90.0),
            raw_row(serial=702, seed=2, ckpt_path=path, hqq_top1=81.0),
            raw_row(serial=704, seed=1, ckpt_path=path, hqq_top1=72.0, hqq_group_size=16),
            raw_row(serial=705, seed=3, ckpt_path=path, hqq_top1=78.0, hqq_group_size=16),
            raw_row(
                serial=703,
                ckpt_path='/data/ckpt/ft/demo_vit_b16.pth',
                hqq_top1=75.0,
                hqq_group_size=32,
            ),
        ])

    def test_uses_nonlegacy_serials_as_provenance_not_identity(self):
        _, _, cells = prepare_summary(self.fixture(), summary_args())
        matched = cells[cells['hqq_group_size'].eq(8)].iloc[0]

        self.assertTrue(matched['baseline_matched'])
        self.assertEqual(matched['n_runs'], 3)
        self.assertEqual(matched['n_seeds'], 2)
        self.assertEqual(matched['hqq_source_serials'], '700, 701, 702')
        self.assertEqual(matched['baseline_source_serials'], '900, 901')
        self.assertAlmostEqual(matched['hqq_top1'], 76.5)
        self.assertAlmostEqual(matched['fp32_top1'], 85.0)
        self.assertAlmostEqual(matched['accuracy_ratio'], 0.9)

    def test_partial_seed_baselines_keep_audit_row_without_ratio(self):
        _, _, cells = prepare_summary(self.fixture(), summary_args())
        partial = cells[cells['hqq_group_size'].eq(16)].iloc[0]

        self.assertFalse(partial['baseline_matched'])
        self.assertEqual(partial['n_seeds'], 2)
        self.assertEqual(partial['unmatched_seed_count'], 1)
        self.assertTrue(math.isnan(partial['fp32_top1']))
        self.assertTrue(math.isnan(partial['accuracy_ratio']))

    def test_serial_filter_is_optional_raw_input_scope(self):
        _, _, cells = prepare_summary(self.fixture(), summary_args(serials=[700, 900]))

        self.assertEqual(len(cells), 1)
        cell = cells.iloc[0]
        self.assertTrue(cell['baseline_matched'])
        self.assertEqual(cell['hqq_source_serials'], '700')
        self.assertAlmostEqual(cell['accuracy_ratio'], 70.0 / 80.0)

    def test_summary_does_not_require_a_serial_column(self):
        raw = self.fixture().drop(columns=['serial'])
        _, _, cells = prepare_summary(raw, summary_args())

        matched = cells[cells['hqq_group_size'].eq(8)].iloc[0]
        self.assertTrue(matched['baseline_matched'])
        self.assertEqual(matched['hqq_source_serials'], '')

    def test_checkpoint_classifier_accepts_legacy_and_new_layouts(self):
        kinds = checkpoint_kind(pd.Series([
            '/results/cal_ckpts/demo.pth',
            '../../data/ckpt/fz/demo.pth',
        ]))
        self.assertEqual(kinds.tolist(), ['cal', 'fz'])

    def test_corr_serial_filter_matches_provenance_lists(self):
        values = pd.Series(['700, 701, 702', '703', pd.NA])
        self.assertEqual(source_serials_match(values, [701]).tolist(), [True, False, False])


if __name__ == '__main__':
    unittest.main()
