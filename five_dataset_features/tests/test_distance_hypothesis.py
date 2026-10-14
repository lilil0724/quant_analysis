import unittest

import pandas as pd

from five_dataset_features.distance_hypothesis import build_tables, correlations
from five_dataset_features.geometry_robustness import CONDITIONS, DATASETS


class DistanceHypothesisTests(unittest.TestCase):
    def setUp(self):
        self.fp = pd.DataFrame([
            {'dataset': dataset, 'fp_accuracy': .8, 'fp_d_inter': .9,
             'fp_d_intra': .5 - .05 * index, 'fp_separability': .4 + .05 * index}
            for index, dataset in enumerate(DATASETS)
        ])
        self.conditions = pd.DataFrame([
            {'dataset': dataset, 'condition': condition, 'fp_accuracy': .8,
             'accuracy': .8 - .01 * (5 - index) * (number + 1),
             'accuracy_drop': .01 * (5 - index) * (number + 1),
             'relative_loss': .01 * (5 - index) * (number + 1) / .8}
            for index, dataset in enumerate(DATASETS)
            for number, condition in enumerate(CONDITIONS)
        ])

    def test_gap_ratio_and_fixed_condition_loss(self):
        summary, losses = build_tables(self.fp, self.conditions)
        row = summary.iloc[0]
        self.assertAlmostEqual(row.fp_distance_gap, .4)
        self.assertAlmostEqual(row.fp_distance_ratio, 1.8)
        self.assertAlmostEqual(row.mean_accuracy_drop, .125)
        self.assertAlmostEqual(row.mean_relative_loss, .15625)
        self.assertEqual(len(losses), 20)
        corr = correlations(summary, losses)
        selected = corr[(corr.predictor == 'fp_distance_gap') &
                        (corr.condition == 'mean') &
                        (corr.loss_metric == 'relative_loss')].iloc[0]
        self.assertAlmostEqual(selected.spearman_rho, -1)

    def test_rejects_duplicate_condition(self):
        duplicated = pd.concat([self.conditions, self.conditions.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'exactly one row'):
            build_tables(self.fp, duplicated)


if __name__ == '__main__':
    unittest.main()
