"""Small paired exports exercise t-SNE without server data or inference."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from five_dataset_features import tsne_paired as tsne
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

    def make_pair(self, dataset, mode='ft', labels=None):
        folder = self.root / '20004' / dataset / 'vit_b16' / mode
        folder.mkdir(parents=True)
        rng = np.random.default_rng(17)
        labels = [i // 8 for i in range(24)] if labels is None else labels
        n = len(labels)
        (folder / 'samples.json').write_text(json.dumps(dict(
            sample_ids=[f'{dataset}-{i}' for i in range(n)], targets=labels)))
        x = rng.normal(size=(n, 8)).astype('float32')
        for condition, delta in [('fp', 0), ('w3_g128', .15)]:
            (folder / condition).mkdir()
            np.save(folder / condition / 'final.npy', x + delta)
        files = {name: sha256(folder / name) for name in
                 ('samples.json', 'fp/final.npy', 'w3_g128/final.npy')}
        (folder / 'manifest.json').write_text(json.dumps(dict(
            status='complete', serial=20004, dataset=dataset, model='vit_b16',
            checkpoint_mode=mode, n_samples=n, layers={'final': [n, 8]},
            feature_dtype='float32', samples_sha256=digest({'ids':
                [f'{dataset}-{i}' for i in range(n)], 'targets': labels}),
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

    def test_class_sampling_bounds_rounding_and_all_rows(self):
        for classes, expected in [(3, 3), (37, 5), (80, 8), (101, 11),
                                  (198, 20), (1110, 20), (5089, 20)]:
            with self.subTest(classes=classes):
                # Unequal populations and nonconsecutive class IDs.
                labels = np.repeat(np.arange(classes) * 3 + 7,
                                   np.arange(classes) % 5 + 1)
                chosen = tsne.class_sample_indices(labels, .1, 100)
                selected = np.unique(labels[chosen])
                self.assertEqual(len(selected), expected)
                np.testing.assert_array_equal(
                    chosen, np.flatnonzero(np.isin(labels, selected)))
                np.testing.assert_array_equal(
                    chosen, tsne.class_sample_indices(labels, .1, 100))
                reversed_labels = labels[::-1]
                other = tsne.class_sample_indices(reversed_labels, .1, 100)
                np.testing.assert_array_equal(selected, np.unique(reversed_labels[other]))

    def test_class_sampling_rejects_invalid_fraction(self):
        for fraction in [0, -1, 1.1, float('nan'), float('inf')]:
            with self.subTest(fraction=fraction):
                with self.assertRaises(ValueError):
                    tsne.class_sample_indices(np.arange(10), fraction, 100)

    def test_default_keeps_every_sample_in_selected_classes_and_separate_output(self):
        output = run(self.root, 20004, ['cub'], ['vit_b16'], ['ft'],
                     max_samples=2, perplexity=5, iterations=300)
        self.assertEqual(output.name, 'tsne_classes')
        folder = output / 'cub/vit_b16/ft'
        meta = json.loads((folder / 'method.json').read_text())
        self.assertEqual(meta['selected_samples'], 24)
        self.assertEqual(meta['selected_class_ids'], [0, 1, 2])
        self.assertEqual(meta['class_sample_counts'], {'0': 8, '1': 8, '2': 8})
        self.assertEqual(meta['parameters']['sampling'], 'classes')
        self.assertIsNone(meta['parameters']['max_samples'])
        self.assertEqual(meta['parameters']['class_fraction'], .1)
        self.assertFalse((self.root / '20004/analysis/tsne').exists())
        saved = sha256(folder / 'coordinates.csv')
        run(self.root, 20004, ['cub'], ['vit_b16'], ['ft'],
            max_samples=2000, perplexity=5, iterations=300)
        self.assertEqual(saved, sha256(folder / 'coordinates.csv'))
        with self.assertRaisesRegex(ValueError, 'Existing t-SNE differs'):
            run(self.root, 20004, ['cub'], ['vit_b16'], ['ft'],
                class_fraction=.2, perplexity=5, iterations=300)

    def test_shared_classes_across_modes_with_different_populations(self):
        labels = np.repeat(np.arange(80), np.arange(80) % 4 + 1).tolist()
        self.make_pair('pets', labels=labels)
        self.make_pair('pets', mode='fz', labels=labels[::-1] + labels)
        output = run(self.root, 20004, ['pets'], ['vit_b16'], ['ft', 'fz'],
                     max_samples=2, perplexity=5, iterations=300)
        metas = [json.loads((output / f'pets/vit_b16/{mode}/method.json').read_text())
                 for mode in ['ft', 'fz']]
        self.assertEqual(metas[0]['selected_classes'], 8)
        self.assertEqual(metas[0]['selected_class_ids'], metas[1]['selected_class_ids'])
        self.assertEqual(2 * metas[0]['selected_samples'], metas[1]['selected_samples'])
        for mode, original_labels in [('ft', labels), ('fz', labels[::-1] + labels)]:
            coords = np.genfromtxt(output / f'pets/vit_b16/{mode}/coordinates.csv',
                                   delimiter=',', names=True)
            expected = np.flatnonzero(np.isin(original_labels, metas[0]['selected_class_ids']))
            np.testing.assert_array_equal(coords['row'], expected)
            np.testing.assert_array_equal(coords['target'], np.asarray(original_labels)[expected])

    def test_preflight_rejects_different_classes_before_writing(self):
        self.make_pair('cub', mode='fz', labels=[100 + i // 8 for i in range(24)])
        with self.assertRaisesRegex(ValueError, 'Selected class IDs differ'):
            run(self.root, 20004, ['cub'], ['vit_b16'], ['ft', 'fz'])
        self.assertFalse((self.root / '20004/analysis').exists())

    def test_cli_forwards_sampling_and_fraction_to_preflight(self):
        self.make_pair('pets', labels=np.repeat(np.arange(100), 2).tolist())
        command = [sys.executable, '-B', '-m', 'five_dataset_features.tsne_paired',
                   'preflight', '--input-root', str(self.root), '--serial', '20004',
                   '--datasets', 'pets', '--models', 'vit_b16', '--modes', 'ft',
                   '--class-fraction', '.05']
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('5 classes, 10/200 samples', result.stdout)
        result = subprocess.run(command + ['--sampling', 'samples', '--max-samples', '12'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('12/200 samples', result.stdout)

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
                     sampling='samples', max_samples=12, perplexity=5, iterations=250, seed=11)
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
            sampling='samples', max_samples=12, perplexity=5, iterations=250, seed=11)
        self.assertEqual(saved, {p: sha256(p) for p in folder.iterdir() if p.is_file()})
        with self.assertRaisesRegex(ValueError, 'Existing t-SNE differs'):
            run(self.root, 20004, ['cub'], ['vit_b16'], ['ft'],
                sampling='samples', max_samples=10, perplexity=5, iterations=250, seed=11)


if __name__ == '__main__':
    unittest.main()
