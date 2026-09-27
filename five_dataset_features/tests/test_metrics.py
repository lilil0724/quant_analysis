import unittest
import json
import shutil
from pathlib import Path

import numpy as np

from five_dataset_features.metrics import (centered_linear_cka,
    class_equal_cosine_distances, pca_fit, pca_project, prediction_metrics,
    spearman_description)
from five_dataset_features.analyze import checked_exemplars, exemplar_indices, sha256


class FeatureMetricTests(unittest.TestCase):
    def test_chunked_cka_matches_centered_sample_gram(self):
        random = np.random.default_rng(16)
        x = random.normal(size=(11, 4)).astype(np.float32)
        y = random.normal(size=(11, 4)).astype(np.float32)
        h = np.eye(11) - np.ones((11, 11)) / 11
        k = h @ (x.astype(np.float64) @ x.astype(np.float64).T) @ h
        l = h @ (y.astype(np.float64) @ y.astype(np.float64).T) @ h
        direct = np.sum(k * l) / (np.linalg.norm(k) * np.linalg.norm(l))
        self.assertAlmostEqual(centered_linear_cka(x, y, chunk_size=3), direct, places=12)
        self.assertAlmostEqual(centered_linear_cka(x, x, chunk_size=2), 1.0, places=12)

    def test_class_equal_distances_match_explicit_pairs(self):
        x = np.array([[1, 0], [0.8, 0.6], [0, 1], [0, 1], [0, 1]], float)
        labels = np.array([0, 0, 1, 1, 1])
        actual = class_equal_cosine_distances(x, labels, chunk_size=2)
        unit = x / np.linalg.norm(x, axis=1, keepdims=True)
        within_0 = 1 - unit[0] @ unit[1]
        within_1 = np.mean([1 - unit[a] @ unit[b] for a, b in ((2, 3), (2, 4), (3, 4))])
        between = np.mean([1 - unit[a] @ unit[b] for a in (0, 1) for b in (2, 3, 4)])
        self.assertAlmostEqual(actual['d_intra'], (within_0 + within_1) / 2)
        self.assertAlmostEqual(actual['d_inter'], between)
        self.assertAlmostEqual(actual['separability'], between - (within_0 + within_1) / 2)

    def test_pairing_and_undefined_endpoints(self):
        fp = [{'sample_id': 'a', 'target': '0', 'prediction': '1', 'correct': '0'},
              {'sample_id': 'b', 'target': '1', 'prediction': '0', 'correct': '0'}]
        result = prediction_metrics(fp, fp)
        self.assertIsNone(result['relative_loss'])
        self.assertIsNone(result['net_damage_r'])
        changed = [dict(fp[0]), dict(fp[1])]
        changed[0]['sample_id'] = 'wrong'
        with self.assertRaises(ValueError):
            prediction_metrics(fp, changed)

    def test_spearman_is_dataset_level_description(self):
        self.assertAlmostEqual(spearman_description([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]), -1)
        self.assertIsNone(spearman_description([1] * 5, [1, 2, 3, 4, 5]))

    def test_damage_rescue_and_relative_loss(self):
        fp = [dict(sample_id=str(i), target=str(t), prediction=str(p), correct=str(int(p == t)))
              for i, (t, p) in enumerate(((0, 0), (1, 1), (2, 0), (3, 0)))]
        quant = [dict(sample_id=str(i), target=str(t), prediction=str(p), correct=str(int(p == t)))
                 for i, (t, p) in enumerate(((0, 1), (1, 1), (2, 2), (3, 0)))]
        result = prediction_metrics(fp, quant)
        self.assertEqual(result['damage_count'], 1)
        self.assertEqual(result['rescue_count'], 1)
        self.assertAlmostEqual(result['swap_rate'], 0.5)
        self.assertAlmostEqual(result['relative_loss'], 0)
        self.assertAlmostEqual(result['net_damage_r'], 0)

    def test_one_fp_pca_basis_projects_new_features(self):
        x = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], float)
        mean, components, explained = pca_fit(x, chunk_size=2)
        fp_coord = pca_project(x, mean, components, chunk_size=2)
        shifted_coord = pca_project(x + [0, 1, 0], mean, components, chunk_size=2)
        self.assertAlmostEqual(explained[0], 1)
        self.assertTrue(np.allclose(fp_coord[:, 0], shifted_coord[:, 0]))

    def test_representatives_are_deterministic_and_class_balanced(self):
        meta = {'targets': [label for label in range(12) for _ in range(3)]}
        first = exemplar_indices(meta)
        self.assertEqual(first, exemplar_indices(meta))
        self.assertEqual(len(first), 20)
        counts = {label: sum(meta['targets'][index] == label for index in first)
                  for label in range(12)}
        self.assertEqual(sorted(counts.values()), [0, 0] + [2] * 10)

    def test_local_package_checks_copied_exemplar_manifest(self):
        meta = {'sample_ids': [str(i) for i in range(36)],
                'targets': [label for label in range(12) for _ in range(3)],
                'data_sha256': 'matching-data'}
        indices = exemplar_indices(meta)
        root = Path(__file__).parent / '.codex_tmp_exemplar_test'
        root.mkdir(exist_ok=False)
        try:
            source = root / 'cub' / 'exemplars'
            source.mkdir(parents=True)
            files = {}
            for index in indices:
                name = f'{index:06d}.jpg'
                (source / name).write_bytes(f'image-{index}'.encode())
                files[name] = sha256(source / name)
            manifest = {'status': 'complete', 'dataset': 'cub',
                        'data_sha256': 'matching-data', 'seed': 100,
                        'indices': indices,
                        'sample_ids': [meta['sample_ids'][i] for i in indices],
                        'files': files}
            (source / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            self.assertEqual(checked_exemplars(root, 'cub', meta), (indices, source))
            (source / f'{indices[0]:06d}.jpg').write_bytes(b'corrupt')
            with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
                checked_exemplars(root, 'cub', meta)
        finally:
            shutil.rmtree(root)


if __name__ == '__main__':
    unittest.main()
