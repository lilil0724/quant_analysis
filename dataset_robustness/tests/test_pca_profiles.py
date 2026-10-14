"""Check PCA cohort and linear-algebra boundaries with synthetic profiles."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_robustness.pca_profiles import (
    centered_pca, complete_matrix, load_verified_source,
)


class PCAProfilesTest(unittest.TestCase):
    def test_incomplete_dataset_is_audited_and_ratio_above_one_is_retained(self):
        rows = []
        for dataset, ratios in {
                'A': [1.05, .95, .9, .85],
                'B': [.8, .75, .7, .65],
                'C': [.95, .9, .8, .7],
                'D': [.7, .6, .5]}.items():
            for (bit, group), ratio in zip([(2., 8), (2., 32), (3., 8), (3., 32)], ratios):
                rows.append(dict(dataset_name=dataset, hqq_nbits=bit,
                                 hqq_group_size=group, accuracy_ratio=ratio,
                                 accuracy_drop_points=100 * (1 - ratio)))
        wide, coverage = complete_matrix(pd.DataFrame(rows), (2., 3.), (8, 32),
                                         'relative_loss')
        self.assertEqual(wide.shape, (3, 4))
        self.assertEqual(list(wide.index), ['A', 'B', 'C'])
        self.assertAlmostEqual(wide.loc['A', (2., 8)], -.05)
        excluded = coverage.loc[~coverage.included]
        self.assertEqual(excluded.dataset_name.tolist(), ['D'])
        self.assertEqual(excluded.observed_conditions.tolist(), [3])

    def test_svd_pca_matches_covariance_spectrum_and_orthogonality(self):
        matrix = pd.DataFrame([
            [-.05, .05, .1, .15],
            [.2, .25, .3, .35],
            [.05, .1, .2, .3],
            [.3, .2, .1, .05],
        ])
        scores, loadings, explained = centered_pca(matrix)
        eigenvalues = np.linalg.eigvalsh(np.cov(matrix.to_numpy(), rowvar=False))[::-1]
        expected = eigenvalues / eigenvalues.sum()
        np.testing.assert_allclose(explained, expected, atol=1e-12)
        np.testing.assert_allclose(loadings.T @ loadings, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(scores.mean(axis=0), 0, atol=1e-12)

    def test_stale_main_run_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'summary.csv'
            source.write_text('changed source', encoding='utf-8')
            (root / 'run_config.json').write_text(json.dumps({
                'input_file': str(source), 'input_sha256': '0' * 64,
            }), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'source hash changed'):
                load_verified_source(root)


if __name__ == '__main__':
    unittest.main()
