import unittest

import numpy as np

from hqq_confidence.analyze_bias_competitors import (
    assign_folds, crossfit_bias, predict_preserving_ties,
    competitor_features, winner_rank_bounds, train_standardize,
)


class BiasCompetitorTests(unittest.TestCase):
    def test_folds_are_reproducible_and_stratified(self):
        labels = np.repeat(np.arange(3), 20)
        f = assign_folds(labels, 5, 100004)
        np.testing.assert_array_equal(f, assign_folds(labels, 5, 100004))
        for label in range(3):
            np.testing.assert_array_equal(np.bincount(f[labels == label], minlength=5), [4]*5)

    def test_bias_estimate_excludes_heldout_fold(self):
        fp = np.zeros((4, 2))
        q = np.array([[100., 0.], [100., 0.], [2., 0.], [2., 0.]])
        means = crossfit_bias(fp, q, np.array([0, 0, 1, 1]), 2, 2)
        np.testing.assert_allclose(means, [[1., -1.], [50., -50.]])

    def test_constant_class_bias_correction_recovers_predictions(self):
        fp = np.array([[2., 0.], [0., 2.]] * 5)
        q = fp + [3., 0.]
        target = fp.argmax(axis=1)
        folds = assign_folds(target, 5, 1)
        means = crossfit_bias(fp, q, folds, 5, 3)
        pred = predict_preserving_ties(q-means[folds], q.argmax(axis=1))
        np.testing.assert_array_equal(pred, target)

    def test_sample_common_shift_has_no_class_bias(self):
        fp = np.array([[3., 1., 0.]] * 10)
        q = fp + np.arange(10)[:, None]
        means = crossfit_bias(fp, q, np.arange(10) % 5, 5, 3)
        np.testing.assert_allclose(means, 0.)

    def test_zero_correction_preserves_stored_tie_winner(self):
        q = np.array([[2., 2., 0.], [1., 2., 2.]])
        np.testing.assert_array_equal(predict_preserving_ties(q, [1, 2]), [1, 2])
        np.testing.assert_array_equal(predict_preserving_ties(q, [2, 0]), [0, 1])

    def test_competitor_count_excludes_winner_and_is_scale_invariant(self):
        z = np.array([[3., 3., 2., 0.], [4., 1., 0., -1.]])
        m1, counts1 = competitor_features(z, [1, 0], [0, 1])
        m2, counts2 = competitor_features(5*z+10, [1, 0], [0, 1])
        np.testing.assert_allclose(m1, m2)
        np.testing.assert_array_equal(counts1, counts2)
        np.testing.assert_array_equal(counts1[0], [1, 1, 2])

    def test_rank_bounds_do_not_invent_rank_three_for_ties(self):
        z = np.array([[5., 4., 3., 2.], [5., 4., 4., 2.]])
        lo, hi = winner_rank_bounds(z, [2, 2])
        np.testing.assert_array_equal(lo, [3, 2])
        np.testing.assert_array_equal(hi, [3, 3])

    def test_standardization_fits_training_data_only(self):
        train = np.array([[0., 1.], [2., 1.]])
        test = np.array([[100., 1.]])
        a, b, mean, scale = train_standardize(train, test, np.array([.5, .5]))
        np.testing.assert_allclose(mean, [1., 1.])
        np.testing.assert_allclose(scale, [1., 1.])
        np.testing.assert_allclose(a[:, 0], [-1., 1.])
        np.testing.assert_allclose(b[0], [99., 0.])


if __name__ == '__main__':
    unittest.main()
