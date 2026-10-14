import unittest

import numpy as np

from five_dataset_features.paired_feature_pca import (
    binary_auc, paired_metrics, pick_arrows,
)


class PairedFeaturePcaTests(unittest.TestCase):
    def test_auc_handles_ties_and_undefined_groups(self):
        self.assertAlmostEqual(
            binary_auc([True, False, True, False], [.5, .5, .9, .1]),
            .875,
        )
        self.assertTrue(np.isnan(binary_auc([True, True], [.1, .2])))

    def test_paired_outcomes_and_projected_drift_energy(self):
        fp = np.zeros((4, 768))
        fp[:, 0] = 1
        q = fp.copy()
        q[0, 0] += 1
        q[1, 2] += 1  # Invisible in the FP PCA plane.
        q[2, 1] += 1
        components = np.eye(768, 2)
        fp_rows = [
            {'correct': 1, 'normalized_margin': .5},
            {'correct': 1, 'normalized_margin': .2},
            {'correct': 0, 'normalized_margin': -.1},
            {'correct': 0, 'normalized_margin': -.2},
        ]
        q_rows = [{'correct': x} for x in (1, 0, 1, 0)]

        details, summary = paired_metrics(fp, q, components, fp_rows, q_rows)

        self.assertEqual(details['outcome'].tolist(),
                         ['C_to_C', 'C_to_W', 'W_to_C', 'W_to_W'])
        np.testing.assert_allclose(details['drift'], [1, 1, 1, 0])
        np.testing.assert_allclose(details['drift_2d'], [1, 0, 1, 0])
        self.assertAlmostEqual(summary['drift_energy_captured_2d'], 2 / 3)
        self.assertEqual((summary['n_damage'], summary['n_rescue']), (1, 1))
        self.assertAlmostEqual(summary['drift_damage_auc'], .5)
        self.assertAlmostEqual(summary['fp_margin_damage_auc'], 1)

    def test_arrow_sampling_is_stratified_and_reproducible(self):
        outcomes = np.repeat(
            ['C_to_C', 'C_to_W', 'W_to_C', 'W_to_W'], 10,
        )
        first = pick_arrows(outcomes, 'cub', 'w3_g128', count_per_group=3)
        second = pick_arrows(outcomes, 'cub', 'w3_g128', count_per_group=3)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 12)
        for group in ('C_to_C', 'C_to_W', 'W_to_C', 'W_to_W'):
            self.assertEqual(np.count_nonzero(outcomes[first] == group), 3)


if __name__ == '__main__':
    unittest.main()
