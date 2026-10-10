"""Synthetic TGDA paired exports; no server inputs or inference required."""
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from five_dataset_features.analyze import sha256, write_csv


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


class PairedTests(unittest.TestCase):
    def setUp(self):
        base = Path('.codex_tmp')
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.root = Path(self.temp.name) / 'w3_g128_analysis'
        self.pair = self.export()

    def tearDown(self):
        self.temp.cleanup()

    def export(self, dataset='cub', model='vit_b16', mode='ft', width=4):
        pair = self.root / '20001' / dataset / model / mode
        pair.mkdir(parents=True)
        ids, targets = [f'id{i}' for i in range(6)], [0, 0, 0, 1, 1, 1]
        (pair / 'samples.json').write_text(json.dumps(dict(sample_ids=ids, targets=targets)))
        rng = np.random.default_rng(19)
        features = rng.normal(size=(6, width)).astype('float32')
        for condition in ('fp', 'w3_g128'):
            folder = pair / condition
            folder.mkdir()
            np.save(folder / 'block0.npy', features + (0.2 if condition != 'fp' else 0))
            np.save(folder / 'final.npy', features + (0.1 if condition != 'fp' else 0))
            preds = [0, 0, 1, 1, 0, 1] if condition == 'fp' else [1, 0, 1, 1, 1, 1]
            logits = np.full((6, 2), -1, dtype='float32')
            logits[np.arange(6), preds] = 1
            np.save(folder / 'logits.npy', logits)
            write_csv(folder / 'predictions.csv', [dict(row=i, sample_id=ids[i], target=t,
                prediction=p, correct=int(t == p), margin=2, normalized_margin=0.5,
                true_class_margin=2 if t == p else -2) for i, (t, p) in enumerate(zip(targets, preds))])
        write_csv(pair / 'layer_metrics.csv', [dict(layer='final', condition='fp', cka_vs_fp=1)])
        (pair / 'summary.json').write_text('{}')
        manifest = dict(status='complete', serial=20001, dataset=dataset, model=model,
            checkpoint_mode=mode, n_samples=6, layers={'block0': [6, width], 'final': [6, width]},
            feature_dtype='float32', logit_dtype='float32',
            quantization=dict(weight_bits=3, weight_group_size=128, activation_bits=None),
            samples_sha256=digest({'ids': ids, 'targets': targets}))
        self.seal(pair, manifest)
        return pair

    def seal(self, pair=None, manifest=None):
        pair = pair or self.pair
        manifest = manifest or json.loads((pair / 'manifest.json').read_text())
        manifest['files'] = {p.relative_to(pair).as_posix(): sha256(p)
                             for p in pair.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        (pair / 'manifest.json').write_text(json.dumps(manifest))

    def run_stage(self, stage='all', **kwargs):
        from five_dataset_features.paired import run
        return run(stage, self.root, 20001, kwargs.pop('datasets', ['cub']),
                   kwargs.pop('models', ['vit_b16']), kwargs.pop('modes', ['ft']), **kwargs)

    def test_adapter_exists(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('five_dataset_features.paired'))

    def test_full_analysis_shared_basis_and_source_preservation(self):
        before = {p: sha256(p) for p in self.pair.rglob('*') if p.is_file()}
        out = self.run_stage()
        folder = out / 'cub/vit_b16/ft'
        basis = np.load(folder / 'fp_pca.npz')
        for condition in ('fp', 'w3_g128'):
            features = np.load(self.pair / condition / 'final.npy')
            expected = (features - basis['mean']) @ basis['components']
            np.testing.assert_allclose(np.load(folder / f'coords_{condition}.npy'), expected, atol=1e-6)
        summary = json.loads((folder / 'summary.json').read_text())
        self.assertEqual(summary['damage_count'], 1)
        self.assertEqual(summary['rescue_count'], 1)
        self.assertEqual(summary['net_damage_r'], 0)
        self.assertEqual(before, {p: sha256(p) for p in self.pair.rglob('*') if p.is_file()})
        (out / 'private.txt').write_text('user file')
        self.run_stage('package')
        with zipfile.ZipFile(out / 'w3_g128_results.zip') as archive:
            names = archive.namelist()
        self.assertIn('cub/vit_b16/ft/fp_pca.npz', names)
        self.assertIn('cub/vit_b16/ft/pca.png', names)
        self.assertFalse(any('coords_' in name or 'logits' in name or 'private' in name for name in names))

    def test_variable_layers_and_matrix_metadata(self):
        self.export('pets', mode='cal', width=7)
        self.export('pets', mode='ft', width=5)
        self.export('cub', mode='cal', width=3)
        out = self.run_stage(datasets=['cub', 'pets'], modes=['ft', 'cal'])
        manifest = json.loads((out / 'analysis_manifest.json').read_text())
        self.assertEqual(len(manifest['sources']), 4)
        report = (out / 'REPORT.md').read_text()
        self.assertIn('CAL', report)
        self.assertIn('descriptive', report)
        self.assertIn('vit_b16 / cal', report)
        self.assertIn('pets', report)

    def test_array_shape_dtype_and_finiteness_validation(self):
        path = self.pair / 'fp/final.npy'
        original = np.load(path)
        for changed in [original[:, :3], original.astype('float16'), original * np.nan]:
            np.save(path, changed)
            self.seal()
            with self.assertRaises(ValueError):
                self.run_stage('preflight')
        np.save(path, original)
        np.save(self.pair / 'fp/logits.npy', np.zeros((6, 2), dtype='float64'))
        self.seal()
        with self.assertRaises(ValueError):
            self.run_stage('preflight')

    def test_undefined_metrics_are_json_null_and_failed_zip_preserves_previous(self):
        for condition in ['fp', 'w3_g128']:
            for layer in ['block0', 'final']:
                np.save(self.pair / condition / f'{layer}.npy', np.zeros((6, 4), dtype='float32'))
        self.seal()
        out = self.run_stage()
        summary_text = (out / 'cub/vit_b16/ft/summary.json').read_text()
        self.assertNotIn('NaN', summary_text)
        self.assertIsNone(json.loads(summary_text)['final_cka'])
        prior = sha256(out / 'w3_g128_results.zip')
        with patch('five_dataset_features.paired.zipfile.ZipFile.write', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.run_stage('package')
        self.assertEqual(prior, sha256(out / 'w3_g128_results.zip'))
        self.assertEqual([], list(out.glob('.package-*')))

    def test_missing_pair_prevents_any_output(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.run_stage(datasets=['cub', 'pets'])
        self.assertFalse((self.root / '20001/analysis').exists())

    def test_incomplete_and_identity_rejected(self):
        for key, value in [('status', 'running'), ('model', 'other'), ('serial', 20002)]:
            original = json.loads((self.pair / 'manifest.json').read_text())
            changed = dict(original, **{key: value})
            (self.pair / 'manifest.json').write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                self.run_stage('preflight')
            (self.pair / 'manifest.json').write_text(json.dumps(original))

    def test_checksum_and_required_file_coverage(self):
        (self.pair / 'summary.json').write_text('{"corrupt":true}')
        with self.assertRaises(ValueError):
            self.run_stage('preflight')
        self.seal()
        meta = json.loads((self.pair / 'manifest.json').read_text())
        del meta['files']['fp/final.npy']
        (self.pair / 'manifest.json').write_text(json.dumps(meta))
        with self.assertRaises(ValueError):
            self.run_stage('preflight')

    def test_samples_and_prediction_argmax_rejected_even_with_new_checksums(self):
        from five_dataset_features.analyze import read_csv
        path = self.pair / 'w3_g128/predictions.csv'
        rows = read_csv(path)
        original = [dict(row) for row in rows]
        for key, value in [('sample_id', 'wrong'), ('target', '1'), ('prediction', '0'), ('correct', '1')]:
            rows = [dict(row) for row in original]
            rows[0][key] = value
            write_csv(path, rows)
            self.seal()
            with self.assertRaises(ValueError):
                self.run_stage('preflight')
        write_csv(path, original)
        samples = json.loads((self.pair / 'samples.json').read_text())
        samples['sample_ids'][0] = samples['sample_ids'][1]
        (self.pair / 'samples.json').write_text(json.dumps(samples))
        self.seal()
        with self.assertRaises(ValueError):
            self.run_stage('preflight')

    def test_unsafe_paths_and_output_rejected(self):
        for out in [self.root, self.root / '20001', self.pair, self.pair / 'results', self.root / '20001/cub']:
            with self.assertRaises(ValueError):
                self.run_stage('preflight', output_root=out)
        meta = json.loads((self.pair / 'manifest.json').read_text())
        meta['files']['../escape'] = '0' * 64
        (self.pair / 'manifest.json').write_text(json.dumps(meta))
        with self.assertRaises(ValueError):
            self.run_stage('preflight')

    def test_other_serial_output_rejected_without_changing_any_sources(self):
        import shutil
        other = self.root / '20002/cub/vit_b16/ft'
        shutil.copytree(self.pair, other)
        manifest = json.loads((other / 'manifest.json').read_text())
        manifest['serial'] = 20002
        (other / 'manifest.json').write_text(json.dumps(manifest))
        before = {p: sha256(p) for p in self.root.rglob('*') if p.is_file()}
        for out in [self.root / '20002', self.root / 'results', self.root / '20001/results']:
            for stage in ['preflight', 'analyze']:
                with self.assertRaises(ValueError):
                    self.run_stage(stage, output_root=out)
        self.assertEqual(before, {p: sha256(p) for p in self.root.rglob('*') if p.is_file()})
        external = self.root.parent / 'external_results'
        self.assertEqual(external.resolve(), self.run_stage('preflight', output_root=external))
        self.assertFalse(external.exists())

    def test_integer_feature_dtype_rejected(self):
        manifest = json.loads((self.pair / 'manifest.json').read_text())
        manifest['feature_dtype'] = 'int32'
        for condition in ['fp', 'w3_g128']:
            for layer in ['block0', 'final']:
                path = self.pair / condition / f'{layer}.npy'
                np.save(path, np.load(path).astype('int32'))
        self.seal(manifest=manifest)
        with self.assertRaises(ValueError):
            self.run_stage('preflight')

    def test_report_scope_and_source_provenance_and_package_atomicity(self):
        self.export(mode='fz')
        out = self.run_stage()
        old_zip = sha256(out / 'w3_g128_results.zip')
        with self.assertRaises(ValueError):
            self.run_stage('report', modes=['fz'])
        (self.pair / 'summary.json').write_text('{"changed":1}')
        self.seal()
        for stage in ['report', 'package']:
            with self.assertRaises(ValueError):
                self.run_stage(stage)
        self.assertEqual(old_zip, sha256(out / 'w3_g128_results.zip'))


if __name__ == '__main__':
    unittest.main()
