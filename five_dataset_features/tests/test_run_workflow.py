"""Synthetic regressions for the config-driven all-layer workflow."""

import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from five_dataset_features import comparisons, layer_analysis, layer_pca, plot_run
from five_dataset_features.inputs import discover, validate, open_features, read_csv
from five_dataset_features.run import run
from five_dataset_features.run_config import load_config
from five_dataset_features.run_snapshot import SnapshotStore, file_hash


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class WorkflowFixture:
    def __init__(self, root, names=('a', 'b', 'c'), conditions=None):
        self.root = Path(root)
        self.raw = self.root / 'raw'
        self.specs = conditions or [
            {'name': 'fp', 'weight_bits': None, 'act_bits': None, 'qkv': False},
            {'name': 'w3_g128', 'weight_bits': 3, 'act_bits': None, 'qkv': False}]
        for number, name in enumerate(names):
            self.add_dataset(name, number)
        self.config_path = self.root / 'config.json'
        self.settings = {'schema_version': 1, 'run_id': 'synthetic',
                         'input_root': str(self.raw), 'output_root': str(self.root / 'results'),
                         'expected_dataset_count': len(names), 'primary_condition': 'w3_g128',
                         'chunk_size': 3, 'bootstrap_repeats': 3, 'permutation_repeats': 3,
                         'cka_sampling_repeats': 1, 'cka_sampling_n_max': 6}
        self.save_config()

    def save_config(self):
        self.config_path.write_text(json.dumps(self.settings), encoding='utf-8')
        self.config = load_config(self.config_path)

    def add_dataset(self, name, number):
        folder = self.raw / name
        folder.mkdir(parents=True)
        targets = [0, 0, 1, 1, 2, 2]
        meta = {'schema': 1, 'dataset': name, 'n_samples': 6,
                'sample_ids': [f'{name}/{i}' for i in range(6)], 'targets': targets,
                'checkpoint_sha256': 'c' * 64, 'data_sha256': 'd' * 64,
                'code_sha256': 'e' * 64, 'hqq_commit': 'test', 'feature_dtype': 'float32',
                'feature_shape': [2, 4], 'position_names': ['block1', 'encoder_norm'],
                'model': 'vit_test', 'head': 'ft', 'training_seed': 100,
                'preprocess': {'size': 224}, 'execution': {'amp': False}}
        (folder / 'dataset.json').write_text(json.dumps(meta), encoding='utf-8')
        rng = np.random.default_rng(103 + number)
        fp = rng.normal(size=(6, 2, 4)).astype(np.float32)
        for spec in self.specs:
            condition_dir = folder / spec['name']
            condition_dir.mkdir()
            np.save(condition_dir / 'features.npy', fp if spec['name'] == 'fp' else fp + .03)
            rows = [{'sample_id': sid, 'target': targets[i], 'prediction': targets[i], 'correct': 1}
                    for i, sid in enumerate(meta['sample_ids'])]
            write_csv(condition_dir / 'predictions.csv', rows)
            self.refresh_manifest(name, spec['name'])

    def refresh_manifest(self, dataset, condition):
        folder = self.raw / dataset
        meta = json.loads((folder / 'dataset.json').read_text(encoding='utf-8'))
        spec = next(s for s in self.specs if s['name'] == condition)
        identity = {'schema': meta['schema'], 'checkpoint': meta['checkpoint_sha256'],
                    'data': meta['data_sha256'], 'code': meta['code_sha256'],
                    'hqq': meta['hqq_commit'], 'dtype': meta['feature_dtype'],
                    'execution': meta['execution'], 'condition': spec}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'),
                                          ensure_ascii=False).encode()).hexdigest()
        directory = folder / condition
        files = {p.name: file_hash(p) for p in directory.iterdir()
                 if p.name in ('features.npy', 'predictions.csv', 'precision_probe.npy')}
        (directory / 'manifest.json').write_text(json.dumps({
            'schema': 1, 'dataset': dataset, 'n_samples': meta['n_samples'],
            'condition': spec, 'identity': digest, 'files': files, 'status': 'complete'}),
            encoding='utf-8')

    def inventory(self):
        inventory = discover(self.config)
        validate(self.config, inventory)
        return inventory

    def comparison_tables(self, undefined_cka=False, zero_accuracy=False):
        metrics = self.root / 'manual_metrics'
        contexts = list(self.inventory().contexts)
        for i, context in enumerate(contexts):
            layers, summaries = [], []
            for condition in context.conditions:
                accuracy = 0 if zero_accuracy else .9 - .1 * i
                summaries.append({'condition': condition, 'fp_accuracy': accuracy,
                                  'accuracy': accuracy, 'relative_loss': None if zero_accuracy else 0,
                                  'accuracy_drop': 0})
                layers.append({'condition': condition, 'position_id': 1,
                               'cka': None if undefined_cka else .95 - .02 * i,
                               'separability': .3, 'gap_deterioration': .01 * i,
                               'delta_inter': -.01 * i, 'delta_intra': 0})
            write_csv(metrics / context.context_id / 'condition_summary.csv', summaries)
            write_csv(metrics / context.context_id / 'layer_metrics.csv', layers)
        return contexts, metrics


class RunWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='qa-')
        self.root = Path(self.temp.name)
        self.fixture = WorkflowFixture(self.root)

    def tearDown(self):
        plt.close('all')
        self.temp.cleanup()

    def test_metrics_runs_with_and_without_precision_probes(self):
        for context in self.fixture.inventory().contexts:
            for condition in context.conditions:
                directory = Path(context.conditions[condition]['path'])
                if context.dataset == 'a':
                    np.save(directory / 'precision_probe.npy', np.load(directory / 'features.npy')[:3])
                    self.fixture.refresh_manifest(context.dataset, condition)
        for context in self.fixture.inventory().contexts:
            output = self.root / 'metrics' / context.context_id
            layer_analysis.analyze_context(context, self.fixture.config, output)
            fp_rows = [r for r in read_csv(output / 'layer_metrics.csv') if r['condition'] == 'fp']
            self.assertEqual(len(fp_rows), 2)
            for row in fp_rows:
                self.assertAlmostEqual(float(row['cka']), 1)
            fp_by_position = {r['position_id']: r for r in fp_rows}
            for row in read_csv(output / 'layer_metrics.csv'):
                if row['condition'] != 'fp':
                    fp = fp_by_position[row['position_id']]
                    for name in ('inter', 'intra'):
                        expected = float(row[f'd_{name}']) - float(fp[f'd_{name}'])
                        self.assertAlmostEqual(float(row[f'delta_{name}']), expected)
            if context.dataset == 'a':
                self.assertEqual(len(read_csv(output / 'precision_probe.csv')), 4)

    def test_pca_atlas_renders_from_valid_pca_artifact(self):
        context = self.fixture.inventory().contexts[0]
        output = self.root / 'pca'
        layer_pca.analyze_context(context, self.fixture.config, output)
        names = plot_run._pca_atlases(self.fixture.config, [context],
                                     {context.context_id: output}, self.root / 'plots')
        for name in names:
            self.assertTrue((self.root / 'plots' / (name + '.png')).is_file())

    def test_single_class_geometry_keeps_undefined_inter_and_ratio(self):
        folder = self.fixture.raw / 'a'
        path = folder / 'dataset.json'
        meta = json.loads(path.read_text(encoding='utf-8'))
        meta['targets'] = [0] * meta['n_samples']
        path.write_text(json.dumps(meta), encoding='utf-8')
        for condition in ('fp', 'w3_g128'):
            rows = read_csv(folder / condition / 'predictions.csv')
            for row in rows:
                row.update(target=0, prediction=0, correct=1)
            write_csv(folder / condition / 'predictions.csv', rows)
            self.fixture.refresh_manifest('a', condition)
        context = self.fixture.inventory().contexts[0]
        output = self.root / 'single_class'
        layer_analysis.analyze_context(context, self.fixture.config, output)
        for row in read_csv(output / 'layer_metrics.csv'):
            self.assertEqual(row['d_inter'], '')
            self.assertEqual(row['distance_ratio'], '')
            self.assertEqual(row['d_inter_status'], 'undefined_fewer_than_two_classes')

    def test_permuted_features_and_predictions_remain_paired(self):
        permutation = np.asarray([2, 4, 0, 5, 1, 3])
        folder = self.fixture.raw / 'a'
        fp = np.load(folder / 'fp' / 'features.npy')
        np.save(folder / 'w3_g128' / 'features.npy', fp[permutation])
        rows = read_csv(folder / 'fp' / 'predictions.csv')
        write_csv(folder / 'w3_g128' / 'predictions.csv', [rows[i] for i in permutation])
        self.fixture.refresh_manifest('a', 'w3_g128')
        context = self.fixture.inventory().contexts[0]
        q = open_features(context, 'w3_g128')
        np.testing.assert_array_equal(np.asarray(q[:3, 1, :]), fp[:3, 1, :])
        np.testing.assert_array_equal(q[[5, 1], -1, :], fp[[5, 1], -1, :])
        output = self.root / 'paired_pca'
        layer_pca.analyze_context(context, self.fixture.config, output)
        for row in read_csv(output / 'pca_summary.csv'):
            if row['condition'] == 'w3_g128' and row['outcome'] == 'all':
                self.assertAlmostEqual(float(row['full_displacement_median']), 0)
                self.assertEqual(row['projected_energy_fraction_median'], '')
        metrics = self.root / 'paired_metrics'
        layer_analysis.analyze_context(context, self.fixture.config, metrics)
        for row in read_csv(metrics / 'layer_metrics.csv'):
            self.assertAlmostEqual(float(row['cka']), 1)
            self.assertAlmostEqual(float(row['delta_gap']), 0)

    def test_unknown_exporter_requires_explicit_feature_semantics(self):
        folder = self.fixture.raw / 'a'
        meta = json.loads((folder / 'dataset.json').read_text(encoding='utf-8'))
        meta.pop('feature_shape')
        meta.pop('position_names')
        (folder / 'dataset.json').write_text(json.dumps(meta), encoding='utf-8')
        for condition in ('fp', 'w3_g128'):
            np.save(folder / condition / 'features.npy', np.ones((6, 13, 768), np.float32))
            self.fixture.refresh_manifest('a', condition)
        inventory = discover(self.fixture.config)
        self.assertTrue(any('feature_shape' in error for error in inventory.errors))

    def test_legacy_shape_works_when_explicitly_declared(self):
        fixture = WorkflowFixture(self.root / 'legacy', names=('legacy',))
        folder = fixture.raw / 'legacy'
        meta = json.loads((folder / 'dataset.json').read_text(encoding='utf-8'))
        meta.pop('feature_shape')
        meta.pop('position_names')
        (folder / 'dataset.json').write_text(json.dumps(meta), encoding='utf-8')
        for condition in ('fp', 'w3_g128'):
            np.save(folder / condition / 'features.npy', np.ones((6, 13, 768), np.float32))
            fixture.refresh_manifest('legacy', condition)
        fixture.settings.update(feature_shape=[13, 768],
                                position_names=[*[f'block{i}' for i in range(1, 13)], 'encoder_norm'])
        fixture.save_config()
        self.assertEqual(fixture.inventory().contexts[0].position_names[-1], 'encoder_norm')

    def test_execution_precision_changes_stratum_even_with_explicit_profile(self):
        context = self.fixture.inventory().contexts[0]
        for explicit in (False, True):
            metadata = dict(context.metadata)
            if explicit:
                metadata['inference_profile'] = {'resize': 224}
            left = replace(context, metadata={**metadata, 'execution': {'amp': True}})
            right = replace(context, metadata={**metadata, 'execution': {'amp': False}})
            self.assertNotEqual(comparisons._stratum(left)[0], comparisons._stratum(right)[0])
        unknown = replace(context, metadata={k: v for k, v in context.metadata.items() if k != 'execution'})
        self.assertFalse(comparisons._stratum(unknown)[1])

    def test_undefined_cka_preserves_unrelated_comparisons(self):
        contexts, metrics = self.fixture.comparison_tables(undefined_cka=True)
        output = self.root / 'comparisons'
        comparisons.analyze(self.fixture.config, contexts, metrics, output)
        rows = read_csv(output / 'comparisons.csv')
        cka = next(r for r in rows if r['analysis'] == 'cka_vs_accuracy_ratio')
        self.assertEqual(cka['n_datasets'], '0')
        self.assertEqual(cka['spearman_rho'], '')
        geometry = next(r for r in rows if r['analysis'] == 'fp_difficulty_vs_gap_deterioration')
        self.assertEqual(geometry['n_datasets'], '3')

    def test_zero_fp_accuracy_produces_undefined_ratio_without_crashing(self):
        contexts, metrics = self.fixture.comparison_tables(zero_accuracy=True)
        output = self.root / 'comparisons'
        comparisons.analyze(self.fixture.config, contexts, metrics, output)
        rows = read_csv(output / 'comparisons.csv')
        repeated = [r for r in rows if r['analysis'].startswith('within_dataset')]
        self.assertEqual(len(repeated), 3)
        self.assertTrue(all(r['n_conditions'] == '0' and r['spearman_rho'] == '' for r in repeated))

    def test_condition_families_are_separate_even_for_single_dataset(self):
        specs = [*self.fixture.specs,
                 {'name': 'a8', 'weight_bits': 3, 'act_bits': 8, 'qkv': False},
                 {'name': 'a4', 'weight_bits': 3, 'act_bits': 4, 'qkv': False},
                 {'name': 'qkv', 'weight_bits': 3, 'act_bits': 4, 'qkv': True}]
        fixture = WorkflowFixture(self.root / 'families', names=('one',), conditions=specs)
        contexts, metrics = fixture.comparison_tables()
        output = self.root / 'families_output'
        comparisons.analyze(fixture.config, contexts, metrics, output)
        rows = read_csv(output / 'comparisons.csv')
        repeated = [r for r in rows if r.get('dataset') == 'one']
        self.assertEqual({r.get('condition_family') for r in repeated}, {'weight_only', 'A8', 'A4', 'QKV'})
        self.assertTrue(all(r['n_conditions'] == '1' for r in repeated))

    def test_group_comparison_counts_only_defined_cka(self):
        values = [('a', np.nan), ('b', .8), ('c', .9), ('d', .95)]
        mapping = {'a': 'FGIR', 'b': 'FGIR', 'c': 'UFGIR', 'd': 'UFGIR'}
        rows = comparisons._group_difference(values, mapping, self.fixture.config,
                                             1, 'core', True)
        self.assertEqual(rows[0]['n_fgir'], 1)
        self.assertEqual(rows[0]['status'], 'insufficient_group_coverage')

    def test_explicit_feature_ids_allow_independently_reordered_csv(self):
        folder = self.fixture.raw / 'a' / 'w3_g128'
        rows = read_csv(folder / 'predictions.csv')
        write_csv(folder / 'predictions.csv', list(reversed(rows)))
        self.fixture.refresh_manifest('a', 'w3_g128')
        path = folder / 'manifest.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        manifest['feature_sample_ids'] = [row['sample_id'] for row in rows]
        path.write_text(json.dumps(manifest), encoding='utf-8')
        context = self.fixture.inventory().contexts[0]
        np.testing.assert_array_equal(np.asarray(open_features(context, 'w3_g128')),
                                      np.load(folder / 'features.npy'))
        manifest['feature_sample_ids'][0] = manifest['feature_sample_ids'][1]
        path.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'feature sample IDs'):
            self.fixture.inventory()

    def test_all_resume_and_verify_rejects_raw_corruption(self):
        fixture = WorkflowFixture(self.root / 'e', names=('only',))
        before = {str(p): file_hash(p) for p in fixture.raw.rglob('*') if p.is_file()}
        run(fixture.config, 'all')
        store = SnapshotStore(fixture.config)
        receipts = {str(p): file_hash(p) for p in store.state.rglob('*.json')}
        run(fixture.config, 'all')
        self.assertEqual(receipts, {str(p): file_hash(p) for p in store.state.rglob('*.json')})
        self.assertEqual(before, {str(p): file_hash(p) for p in fixture.raw.rglob('*') if p.is_file()})
        report = (store.current_artifact('report', 'run') / 'REPORT.md').read_text(encoding='utf-8')
        self.assertIn('weight_only', report)
        verification = json.loads((store.state / 'verification.json').read_text(encoding='utf-8'))
        self.assertEqual(verification['status'], 'complete')
        raw = fixture.raw / 'only' / 'fp' / 'features.npy'
        changed = np.load(raw)
        changed[0, 0, 0] += 1
        np.save(raw, changed)
        with self.assertRaisesRegex(ValueError, 'changed|hash|Verification'):
            run(fixture.config, 'verify')


if __name__ == '__main__':
    unittest.main()
