import unittest

import numpy as np

from five_dataset_features.analyze import DATASETS, grid
from five_dataset_features.metrics import class_equal_cosine_distances
from five_dataset_features.quantized_class_distance import build_rows


class QuantizedClassDistanceTests(unittest.TestCase):
    def setUp(self):
        self.layers = [
            {'dataset': dataset, 'condition': condition, 'layer': layer,
             'd_inter': .9 if condition == 'fp' else .8,
             'd_intra': .3 if condition == 'fp' else .4,
             'separability': .6 if condition == 'fp' else .4,
             'n_intra_classes': '2', 'n_inter_class_pairs': '1'}
            for dataset in DATASETS for condition in grid() for layer in range(13)]
        self.summaries = [
            {'dataset': dataset, 'condition': condition, 'n': 10,
             'fp_accuracy': .8, 'accuracy': .8 if condition == 'fp' else .6,
             'accuracy_drop': 0 if condition == 'fp' else .2,
             'relative_loss': 0 if condition == 'fp' else .25}
            for dataset in DATASETS for condition in grid()]

    def test_final_distance_and_matched_fp_changes(self):
        rows = build_rows(self.layers, self.summaries)
        self.assertEqual(len(rows), 105)
        row = rows[1]
        self.assertEqual(row['layer'], 12)
        for key, value in {'distance_gap': .4, 'distance_ratio': 2,
                           'delta_inter': -.1, 'delta_intra': .1, 'delta_gap': -.2,
                           'gap_retention': 2 / 3, 'accuracy_ratio': .75}.items():
            self.assertAlmostEqual(row[key], value)

    def test_rejects_missing_or_duplicate_layers(self):
        for layers in (self.layers[:-1], self.layers + [self.layers[0]]):
            with self.assertRaisesRegex(ValueError, '13 unique layers'):
                build_rows(layers, self.summaries)

    def test_rejects_inconsistent_distance_and_accuracy(self):
        self.layers[12]['separability'] = .7
        with self.assertRaisesRegex(ValueError, 'inconsistent separation'):
            build_rows(self.layers, self.summaries)
        self.layers[12]['separability'] = .6
        self.summaries[1]['fp_accuracy'] = .9
        with self.assertRaisesRegex(ValueError, 'accuracy baseline'):
            build_rows(self.layers, self.summaries)

    def test_class_equal_metric_matches_explicit_unequal_size_pairs(self):
        # Distinct-pair intra means and cross-class pair means, with equal class weights.
        x = np.array([[1., 0], [0, 1], [1, 1], [-1, 0], [0, -1], [2, -1]])
        labels = np.array([0, 0, 1, 1, 1, 2])  # singleton excluded only from intra
        normalized = x / np.linalg.norm(x, axis=1, keepdims=True)
        intra, inter = [], []
        for left in range(3):
            a = normalized[labels == left]
            if len(a) > 1:
                distances = 1 - a @ a.T
                intra.append(distances[np.triu_indices(len(a), k=1)].mean())
            for right in range(left + 1, 3):
                inter.append((1 - a @ normalized[labels == right].T).mean())
        actual = class_equal_cosine_distances(x, labels, chunk_size=2)
        self.assertAlmostEqual(actual['d_intra'], np.mean(intra))
        self.assertAlmostEqual(actual['d_inter'], np.mean(inter))
        self.assertAlmostEqual(actual['separability'], np.mean(inter) - np.mean(intra))
        self.assertEqual(actual['n_intra_classes'], 2)
        self.assertEqual(actual['n_inter_class_pairs'], 3)


if __name__ == '__main__':
    unittest.main()
