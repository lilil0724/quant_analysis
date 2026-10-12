"""Small paired exports exercise t-SNE without server data or inference."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from five_dataset_features.tsne_paired import run, sample_indices
from five_dataset_features.paired import digest


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PairedTsneTests(unittest.TestCase):
    def setUp(self):
        base = Path('.codex_tmp')
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.root = Path(self.temp.name) / 'w3_g128_analysis'
        self.pair = self.make_pair('cub')

    def tearDown(self):
        self.temp.cleanup()

    def make_pair(self, dataset):
        folder = self.root / '20004' / dataset / 'vit_b16' / 'ft'
        folder.mkdir(parents=True)
        rng = np.random.default_rng(17)
        labels = [i // 8 for i in range(24)]
        (folder / 'samples.json').write_text(json.dumps(dict(
            sample_ids=[f'{dataset}-{i}' for i in range(24)], targets=labels)))
        x = rng.normal(size=(24, 8)).astype('float32')
        for condition, delta in [('fp', 0), ('w3_g128', .15)]:
            (folder / condition).mkdir()
            np.save(folder / condition / 'final.npy', x + delta)
        files = {name: sha256(folder / name) for name in
                 ('samples.json', 'fp/final.npy', 'w3_g128/final.npy')}
        (folder / 'manifest.json').write_text(json.dumps(dict(
            status='complete', serial=20004, dataset=dataset, model='vit_b16',
            checkpoint_mode='ft', n_samples=24, layers={'final': [24, 8]},
            feature_dtype='float32', samples_sha256=digest({'ids':
                [f'{dataset}-{i}' for i in range(24)], 'targets': labels}),
            quantization=dict(weight_bits=3,
            weight_group_size=128, activation_bits=None), files=files)))
        return folder

    def test_class_coverage_and_repeatable_paired_sample(self):
        labels = np.repeat(np.arange(4), 5)
        a = sample_indices(labels, 7, 21)
        b = sample_indices(labels, 7, 21)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(a), 7)
        self.assertEqual(len(np.unique(a)), 7)
        self.assertEqual(set(labels[a]), {0, 1, 2, 3})

    def test_many_classes_keep_repeated_samples_for_visible_neighborhoods(self):
        labels = np.repeat(np.arange(20), 3)
        selected = sample_indices(labels, 10, 21)
        _, counts = np.unique(labels[selected], return_counts=True)
        self.assertEqual(len(selected), 10)
        self.assertEqual(len(counts), 5)
        np.testing.assert_array_equal(counts, np.full(5, 2))

    def test_preflight_requires_every_pair_and_valid_checksums(self):
        before = {p: sha256(p) for p in self.pair.rglob('*') if p.is_file()}
        with self.assertRaisesRegex(ValueError, 'Missing pair'):
            run(self.root, 20004, ['cub', 'pets'], ['vit_b16'], ['ft'],
                stage='preflight')
        self.assertEqual(before, {p: sha256(p) for p in self.pair.rglob('*')
                                  if p.is_file()})
        np.save(self.pair / 'fp/final.npy', np.ones((24, 8), dtype='float32'))
        with self.assertRaisesRegex(ValueError, 'checksum'):
            run(self.root, 20004, ['cub'], ['vit_b16'], ['ft'], stage='preflight')

    def test_joint_embedding_outputs_pair_indices_and_provenance(self):
        before = {p: sha256(p) for p in self.pair.rglob('*') if p.is_file()}
        output = run(self.root, 20004, ['cub'], ['vit_b16'], ['ft'],
                     max_samples=12, perplexity=5, iterations=250, seed=11)
        folder = output / 'cub/vit_b16/ft'
        self.assertTrue((folder / 'tsne.png').is_file())
        rows = np.genfromtxt(folder / 'coordinates.csv', delimiter=',',
                             names=True, dtype=None, encoding='utf-8')
        self.assertEqual(len(rows), 12)
        self.assertEqual(len(np.unique(rows['row'])), 12)
        self.assertEqual(set(rows['target']), {0, 1, 2})
        self.assertTrue(np.isfinite(rows['fp_x']).all())
        self.assertTrue(np.isfinite(rows['w3_y']).all())
        meta = json.loads((folder / 'method.json').read_text())
        self.assertEqual(meta['serial'], 20004)
        self.assertEqual(meta['selected_samples'], 12)
        self.assertEqual(meta['total_samples'], 24)
        self.assertEqual(meta['fit_points'], 24)
        self.assertEqual(meta['source_sha256']['fp/final.npy'],
                         before[self.pair / 'fp/final.npy'])
        self.assertEqual(before, {p: sha256(p) for p in self.pair.rglob('*')
                                  if p.is_file()})
        saved = {p: sha256(p) for p in folder.iterdir() if p.is_file()}
        run(self.root, 20004, ['cub'], ['vit_b16'], ['ft'],
            max_samples=12, perplexity=5, iterations=250, seed=11)
        self.assertEqual(saved, {p: sha256(p) for p in folder.iterdir() if p.is_file()})
        with self.assertRaisesRegex(ValueError, 'Existing t-SNE differs'):
            run(self.root, 20004, ['cub'], ['vit_b16'], ['ft'],
                max_samples=10, perplexity=5, iterations=250, seed=11)


if __name__ == '__main__':
    unittest.main()
