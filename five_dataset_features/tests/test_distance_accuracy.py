import unittest

import numpy as np

from five_dataset_features.distance_accuracy import build_table, correlation, correlation_tables
from five_dataset_features.quantized_class_distance import build_rows
from five_dataset_features.tests import test_quantized_class_distance


class DistanceAccuracyTests(unittest.TestCase):
    def setUp(self):
        fixture = test_quantized_class_distance.QuantizedClassDistanceTests()
        fixture.setUp()
        self.rows = build_rows(fixture.layers, fixture.summaries)

    def test_relative_drops_and_accuracy_units(self):
        frame = build_table(self.rows)
        self.assertEqual(len(frame), 100)
        row = frame.iloc[0]
        self.assertAlmostEqual(row.gap_drop_fraction, 1 / 3)
        self.assertAlmostEqual(row.inter_drop_fraction, 1 / 9)
        self.assertAlmostEqual(row.intra_drop_fraction, -1 / 3)
        self.assertAlmostEqual(row.distance_ratio_drop_fraction, 1 / 3)
        self.assertAlmostEqual(row.accuracy_ratio, .75)
        self.assertAlmostEqual(row.relative_loss, .25)
        self.assertAlmostEqual(row.accuracy_drop_pp, 20)

    def test_undefined_denominators_are_not_infinite(self):
        self.rows[1]['fp_distance_gap'] = 0
        self.rows[1]['fp_d_intra'] = 0
        frame = build_table(self.rows)
        self.assertTrue(np.isnan(frame.iloc[0].gap_drop_fraction))
        self.assertTrue(np.isnan(frame.iloc[0].intra_drop_fraction))
        self.assertTrue(np.isnan(frame.iloc[0].distance_ratio_drop_fraction))

    def test_duplicate_and_missing_pairs_are_rejected(self):
        for rows in (self.rows[:-1], self.rows + [self.rows[0]]):
            with self.assertRaisesRegex(ValueError, '105 unique'):
                build_table(rows)

    def test_correlation_ties_and_undefined(self):
        n, rho, pearson = correlation([1, 2, 2, 4], [4, 2, 2, 1])
        self.assertEqual(n, 4)
        self.assertAlmostEqual(rho, -1)
        self.assertIsNotNone(pearson)
        self.assertEqual(correlation([1, 1, 1], [2, 3, 4]), (3, None, None))

    def test_fixed_and_within_scopes_preserve_repeated_measurements(self):
        frame = build_table(self.rows)
        fixed, within = correlation_tables(frame)
        self.assertEqual(len(fixed), 20 * 4 * 3)
        self.assertEqual(set(fixed.n_dataset_checkpoint_pairs), {5})
        self.assertEqual(len(within), 5 * 3 * 4 * 2)
        for scope, n in [('all_20', 20), ('exclude_A4_12', 12), ('weight_only_4', 4)]:
            self.assertEqual(set(within[within.scope == scope].n_repeated_conditions), {n})
        self.assertTrue(fixed.spearman_rho.isna().all())


if __name__ == '__main__':
    unittest.main()
