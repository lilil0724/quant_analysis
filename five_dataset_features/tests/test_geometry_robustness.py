import unittest

import numpy as np

from five_dataset_features.geometry_robustness import binary_auc, leave_one_out_gap


class GeometryRobustnessTests(unittest.TestCase):
    def test_leave_one_out_gap_uses_other_same_class_sample(self):
        x = np.array([[1.0, 0.0], [0.8, 0.6], [-1.0, 0.0],
                      [-0.8, -0.6], [0.0, 1.0]])
        labels = np.array([0, 0, 1, 1, 2])
        gap, nearest = leave_one_out_gap(x, labels)
        self.assertTrue(np.all(gap[:4] > 0))
        np.testing.assert_array_equal(nearest[:4], labels[:4])
        self.assertTrue(np.isnan(gap[4]))

    def test_binary_auc_counts_ties_as_half(self):
        self.assertAlmostEqual(binary_auc([1, 0, 1, 0], [0.9, 0.7, 0.7, 0.1]), 0.875)
        self.assertTrue(np.isnan(binary_auc([0, 0], [0.2, 0.4])))


if __name__ == '__main__':
    unittest.main()
