import csv
import hashlib
import json
import tempfile
import unittest
from itertools import combinations
from pathlib import Path

import numpy as np

from five_dataset_features import centered_class_distance as analysis


class TestCenteredClassDistance(unittest.TestCase):
    def test_centering_removes_common_mean_without_removing_class_difference(self):
        x = np.array([[1000., -2.], [1000., -1.], [1000., 1.], [1000., 2.]])
        result, mean = analysis.analyze_features(x, np.array([0, 0, 1, 1]), chunk_size=2)
        np.testing.assert_array_equal(mean, [1000., 0.])
        self.assertLess(result['raw_d_inter'], 1e-4)
        self.assertAlmostEqual(result['centered_d_inter'], 2., places=12)
        self.assertAlmostEqual(result['centered_d_intra'], 0., places=12)
        self.assertAlmostEqual(result['centered_normalized_separation'], 1., places=12)
        self.assertGreater(result['mean_energy_fraction'], .999)

    def test_distances_match_explicit_sample_pairs_with_equal_class_pair_weights(self):
        rng = np.random.default_rng(19)
        x = rng.normal(size=(6, 4)) + 5
        labels = np.array([0, 0, 0, 1, 1, 2])
        result, mean = analysis.analyze_features(x, labels, chunk_size=2)
        for prefix, features in [('raw', x), ('centered', x - mean)]:
            unit = features / np.linalg.norm(features, axis=1, keepdims=True)
            inter = np.mean([np.mean([1 - unit[i] @ unit[j]
                                     for i in np.flatnonzero(labels == a)
                                     for j in np.flatnonzero(labels == b)])
                             for a, b in combinations(np.unique(labels), 2)])
            intra = np.mean([np.mean([1 - unit[i] @ unit[j]
                                     for i, j in combinations(np.flatnonzero(labels == c), 2)])
                             for c in np.unique(labels) if np.sum(labels == c) >= 2])
            self.assertAlmostEqual(result[prefix + '_d_inter'], inter, places=12)
            self.assertAlmostEqual(result[prefix + '_d_intra'], intra, places=12)
        self.assertEqual(result['n_intra_classes'], 2)
        self.assertEqual(result['n_inter_class_pairs'], 3)

    def test_zero_centered_vector_is_reported_undefined_without_dropping_samples(self):
        x = np.array([[10., 1.], [10., 0.], [10., -1.]])
        result, _ = analysis.analyze_features(x, np.array([0, 0, 1]), chunk_size=1)
        self.assertEqual(result['n_samples'], 3)
        self.assertEqual(result['centered_zero_norm_count'], 1)
        self.assertEqual(result['centered_status'], 'undefined_zero_norm')
        self.assertIsNone(result['centered_d_inter'])
        self.assertIsNone(result['centered_d_intra'])
        self.assertIsNotNone(result['raw_d_inter'])

    def test_invalid_raw_features_and_labels_are_rejected(self):
        for x in (np.array([[0., 0.], [1., 1.]]), np.array([[np.nan, 0.], [1., 1.]])):
            with self.assertRaises(ValueError):
                analysis.analyze_features(x, np.array([0, 1]))
        with self.assertRaises(ValueError):
            analysis.analyze_features(np.ones((2, 2)), np.array([0]))

    def test_run_checks_source_identity_hashes_and_preserves_inputs_and_old_outputs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            pair = root/'data/20004/cub/vit_b16/cal'
            (pair/'fp').mkdir(parents=True)
            (pair/'w3_g128').mkdir()
            x = np.array([[1000., -2.], [1000., -1.], [1000., 1.], [1000., 2.]], dtype=np.float32)
            np.save(pair/'fp/final.npy', x)
            samples = dict(sample_ids=['a', 'b', 'c', 'd'], targets=[0, 0, 1, 1])
            (pair/'samples.json').write_text(json.dumps(samples), encoding='utf-8')
            for condition, correct in [('fp', [1, 1, 1, 1]), ('w3_g128', [1, 1, 0, 1])]:
                with (pair/condition/'predictions.csv').open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['row', 'sample_id', 'target', 'prediction', 'correct'])
                    for i, (sample, target, flag) in enumerate(zip(samples['sample_ids'], samples['targets'], correct)):
                        writer.writerow([i, sample, target, target if flag else 1-target, flag])
            names = ['fp/final.npy', 'samples.json', 'fp/predictions.csv', 'w3_g128/predictions.csv']
            hashes = {name: hashlib.sha256((pair/name).read_bytes()).hexdigest() for name in names}
            manifest = dict(status='complete', serial=20004, dataset='cub', model='vit_b16',
                            checkpoint_mode='cal', n_samples=4, feature_dtype='float32', layers={'final': [4, 2]},
                            quantization=dict(weight_bits=3, weight_group_size=128, activation_bits=None),
                            samples_sha256=analysis.digest_json(dict(ids=samples['sample_ids'], targets=samples['targets'])),
                            files=hashes)
            (pair/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            args = ['--input-root', str(root/'data'), '--serial', '20004', '--datasets', 'cub',
                    '--models', 'vit_b16', '--modes', 'cal', '--output-root', str(root/'output')]
            analysis.main(args)
            with (root/'output/centered_class_distances.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]['relative_loss_percent']), 25.)
            self.assertAlmostEqual(float(rows[0]['centered_d_inter']), 2., places=12)
            receipt = json.loads((root/'output/run_config.json').read_text())
            self.assertEqual(receipt['status'], 'complete')
            self.assertEqual(receipt['n_pairs'], 1)
            self.assertTrue(all(hashlib.sha256((pair/name).read_bytes()).hexdigest() == value
                                for name, value in hashes.items()))
            with self.assertRaises(FileExistsError):
                analysis.main(args)
            np.save(pair/'fp/final.npy', x + 1)
            args[-1] = str(root/'tampered-output')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                analysis.main(args)
            self.assertFalse((root/'tampered-output').exists())


if __name__ == '__main__':
    unittest.main()
