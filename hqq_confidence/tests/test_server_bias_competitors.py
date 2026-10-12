"""Portable server diagnostics on synthetic paired-W3 exports."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from hqq_confidence import server_bias_competitors as server
from hqq_confidence import bias_competitor_core as core


class PortableCoreTests(unittest.TestCase):
    """The independent portable implementation must retain the fixed semantics."""
    def test_portable_core_agrees_with_original_probes(self):
        from hqq_confidence import analyze_bias_competitors as old
        rng = np.random.default_rng(42)
        fp = rng.normal(size=(30, 6)).astype('float32')
        q = fp + rng.normal(size=(30, 6)).astype('float32')*.2
        labels = np.arange(30) % 6
        folds = core.assign_folds(labels)
        np.testing.assert_array_equal(folds, old.assign_folds(labels))
        np.testing.assert_allclose(core.crossfit_bias(fp, q, folds, chunk=7), old.crossfit_bias(fp, q, folds, chunk=7))
        top1 = fp.argmax(axis=1)
        top2 = np.argsort(fp, axis=1)[:, -2]
        m, c = core.competitor_features(fp, top1, top2)
        old_m, old_c = old.competitor_features(fp, top1, top2)
        np.testing.assert_allclose(m, old_m)
        np.testing.assert_array_equal(c, old_c)
        np.testing.assert_allclose(core.model_features(m, c, 6), old.model_features(m, c, 6))
        for a, b in zip(core.winner_rank_bounds(fp, q.argmax(axis=1)), old.winner_rank_bounds(fp, q.argmax(axis=1))):
            np.testing.assert_array_equal(a, b)
        tied = np.array([[2., 2., 0.]])
        np.testing.assert_array_equal(core.predict_preserving_ties(tied, [1]), [1])

    def test_predictor_rejects_overlap_before_fitting(self):
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            core.predict_dataset_split({}, 'damage', ['cub'], ['cub'], False)


class ServerTests(unittest.TestCase):
    def setUp(self):
        base = Path('.codex_tmp')
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='server-bias-', dir=base)
        self.root = Path(self.temp.name)/'source'
        self.output = Path(self.temp.name)/'results'
        for dataset in ('cub', 'pets', 'cars'):
            self.export(dataset)

    def tearDown(self):
        self.temp.cleanup()

    def export(self, dataset, model='vit_b16', mode='ft'):
        pair = self.root/'20004'/dataset/model/mode
        pair.mkdir(parents=True)
        n, c = 30, 3
        target = np.arange(n) % c
        rng = np.random.default_rng(sum(map(ord, dataset)))
        fp = rng.normal(0, .2, size=(n, c)).astype('float32')
        fp[np.arange(n), target] += 1.0
        fp[np.arange(0, n, 7), target[::7]] -= 2.
        quant = fp + np.array([1.2, 0., 0.], dtype='float32')
        ids = [f'{dataset}/{i}' for i in range(n)]
        samples = dict(sample_ids=ids, targets=target.tolist())
        (pair/'samples.json').write_text(json.dumps(samples), encoding='utf-8')
        for name, logits in [('fp', fp), ('w3_g128', quant)]:
            folder = pair/name
            folder.mkdir()
            np.save(folder/'logits.npy', logits)
            pred = logits.argmax(axis=1)
            top = np.sort(logits.astype('float64'), axis=1)
            margin = top[:, -1]-top[:, -2]
            other = logits.astype('float64').copy()
            other[np.arange(n), target] = -np.inf
            pd.DataFrame(dict(row=np.arange(n), sample_id=ids, target=target,
                              prediction=pred, correct=(pred == target).astype(int), margin=margin,
                              normalized_margin=margin/logits.astype('float64').std(axis=1),
                              true_class_margin=logits[np.arange(n), target]-other.max(axis=1))).to_csv(folder/'predictions.csv', index=False)
        meta = dict(status='complete', serial=20004, dataset=dataset, model=model,
                    checkpoint_mode=mode, n_samples=n, logit_dtype='float32',
                    checkpoint=f'/ckpt/{mode}/{dataset}.pth', checkpoint_sha256='a'*64,
                    cfg_sha256='b'*64, test_csv=f'/data/{dataset}/test.csv', test_csv_sha256='c'*64,
                    code_sha256='d'*64, hqq_source_sha256='e'*64, seed=100, amp_fp16=True,
                    pooling='cls', preprocessing=dict(image_size=224),
                    quantization=dict(weight_bits=3, weight_group_size=128, activation_bits=None,
                                      compute_dtype='float16', exclude=['head']),
                    samples_sha256=server.digest_json(dict(ids=ids, targets=target.tolist())))
        self.seal(pair, meta)
        return pair

    def seal(self, pair, meta=None):
        meta = meta or json.loads((pair/'manifest.json').read_text())
        meta['files'] = {p.relative_to(pair).as_posix(): server.sha256_file(p)
                         for p in pair.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        (pair/'manifest.json').write_text(json.dumps(meta), encoding='utf-8')

    def args(self, *extra):
        return server.build_parser().parse_args(['--input-root', str(self.root), '--serial', '20004',
            '--output-dir', str(self.output), '--datasets', 'cub,pets,cars', '--models', 'vit_b16',
            '--modes', 'ft', '--threads', '1', *extra])

    def test_full_pipeline_preserves_sources_and_holds_out_whole_datasets(self):
        before = {p: server.sha256_file(p) for p in self.root.rglob('*') if p.is_file()}
        server.run(self.args())
        self.assertEqual(before, {p: server.sha256_file(p) for p in self.root.rglob('*') if p.is_file()})
        b = pd.read_csv(self.output/'bias_results.csv')
        self.assertEqual(len(b), 3)
        self.assertTrue((b.gain_pp > 0).all())
        fits = json.loads((self.output/'vit_b16/ft/model_fits.json').read_text())
        self.assertEqual(len(fits), 12)
        for fit in fits:
            self.assertFalse(set(fit['train_datasets']) & set(fit['test_datasets']))
        scores = pd.read_csv(self.output/'heldout_prediction_comparisons.csv')
        self.assertEqual(len(scores), 6)
        completion = json.loads((self.output/'completion.json').read_text())
        self.assertEqual(completion['status'], 'complete')
        for name, checksum in completion['output_hashes'].items():
            self.assertEqual(server.sha256_file(self.output/name), checksum)

    def test_preflight_is_read_only(self):
        server.run(self.args('--preflight-only'))
        self.assertFalse(self.output.exists())

    def test_missing_selected_setting_is_an_error(self):
        with self.assertRaisesRegex(ValueError, 'Missing.*cal'):
            server.run(self.args('--modes', 'ft,cal'))
        self.assertFalse(self.output.exists())

    def test_checksum_mismatch_is_rejected(self):
        path = self.root/'20004/cub/vit_b16/ft/fp/logits.npy'
        a = np.load(path)
        a[0, 0] += 1
        np.save(path, a)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            server.run(self.args('--preflight-only'))

    def test_sample_prediction_mismatch_rejected_even_after_resealing(self):
        pair = self.root/'20004/cub/vit_b16/ft'
        path = pair/'fp/predictions.csv'
        rows = pd.read_csv(path)
        rows.loc[0, 'sample_id'] = 'wrong'
        rows.to_csv(path, index=False)
        self.seal(pair)
        with self.assertRaisesRegex(ValueError, 'sample'):
            server.run(self.args('--preflight-only'))

    def test_activation_quantization_is_rejected(self):
        pair = self.root/'20004/cub/vit_b16/ft'
        meta = json.loads((pair/'manifest.json').read_text())
        meta['quantization']['activation_bits'] = 4
        self.seal(pair, meta)
        with self.assertRaisesRegex(ValueError, 'weight-only'):
            server.run(self.args('--preflight-only'))

    def test_tie_preserves_stored_winner(self):
        pair = self.root/'20004/cub/vit_b16/ft'
        logits = np.load(pair/'fp/logits.npy')
        rows = pd.read_csv(pair/'fp/predictions.csv')
        target = rows.target.to_numpy()
        logits[0] = [2., 2., 0.]
        pred = logits.argmax(axis=1)
        pred[0] = 1
        rows['prediction'] = pred
        rows['correct'] = (pred == target).astype(int)
        top = np.sort(logits.astype('float64'), axis=1)
        rows['margin'] = top[:, -1]-top[:, -2]
        rows['normalized_margin'] = rows.margin/logits.astype('float64').std(axis=1)
        other = logits.astype('float64').copy()
        other[np.arange(len(target)), target] = -np.inf
        rows['true_class_margin'] = logits[np.arange(len(target)), target]-other.max(axis=1)
        np.save(pair/'fp/logits.npy', logits)
        rows.to_csv(pair/'fp/predictions.csv', index=False)
        self.seal(pair)
        server.preflight(self.args())
        item = server.check_pair(pair, 20004, 'cub', 'vit_b16', 'ft', keep_predictions=True)
        self.assertEqual(item['fp_pred'][0], 1)

    def test_existing_output_is_preserved(self):
        self.output.mkdir()
        marker = self.output/'user.txt'
        marker.write_text('preserve')
        with self.assertRaises(FileExistsError):
            server.run(self.args())
        self.assertEqual(marker.read_text(), 'preserve')

    def test_single_dataset_marks_prediction_incomplete(self):
        server.run(self.args('--datasets', 'cub'))
        status = pd.read_csv(self.output/'prediction_status.csv')
        lodo = status[status.scope == 'lodo']
        self.assertTrue((lodo.status == 'incomplete').all())
        self.assertIn('too_few_datasets', lodo.reason.iloc[0])

    def test_multiple_models_and_modes_are_kept_separate(self):
        for dataset in ('cub', 'pets', 'cars'):
            self.export(dataset, mode='cal')
            self.export(dataset, model='beitv2_base_patch16_224')
            self.export(dataset, model='beitv2_base_patch16_224', mode='cal')
        server.run(self.args('--models', 'vit_b16,beitv2_base_patch16_224', '--modes', 'ft,cal'))
        b = pd.read_csv(self.output/'bias_results.csv')
        self.assertEqual(len(b), 12)
        summaries = pd.read_csv(self.output/'setting_summary.csv')
        self.assertEqual(len(summaries), 8)
        self.assertTrue((summaries.prediction_status == 'complete').all())
        for model in ('vit_b16', 'beitv2_base_patch16_224'):
            for mode in ('ft', 'cal'):
                fits = json.loads((self.output/model/mode/'model_fits.json').read_text())
                self.assertTrue(all(f['model_name'] == model and f['checkpoint_mode'] == mode for f in fits))

    def test_single_class_training_outcome_does_not_report_partial_average(self):
        for dataset in ('cub', 'pets', 'cars'):
            pair = self.root/'20004'/dataset/'vit_b16/ft'
            np.save(pair/'w3_g128/logits.npy', np.load(pair/'fp/logits.npy'))
            (pair/'w3_g128/predictions.csv').write_bytes((pair/'fp/predictions.csv').read_bytes())
            self.seal(pair)
        server.run(self.args())
        summary = pd.read_csv(self.output/'setting_summary.csv')
        self.assertTrue((summary.prediction_status == 'incomplete').all())
        self.assertTrue(summary.mean_delta_brier.isna().all())
        statuses = pd.read_csv(self.output/'prediction_status.csv')
        lodo = statuses[statuses.scope == 'lodo']
        self.assertTrue(lodo.reason.str.contains('training_outcome_single_class').all())

    def test_family_holdout_excludes_entire_soy_family(self):
        self.export('soyageing')
        self.export('soyglobal')
        server.run(self.args('--datasets', 'cub,pets,soyageing,soyglobal'))
        fits = json.loads((self.output/'vit_b16/ft/model_fits.json').read_text())
        family = [f for f in fits if f['scope'] == 'nonsoy_to_soy']
        self.assertEqual(len(family), 4)
        self.assertTrue(all(set(f['train_datasets']) == {'cub', 'pets'} and
                            set(f['test_datasets']) == {'soyageing', 'soyglobal'} for f in family))

    def test_mixed_export_code_is_rejected(self):
        pair = self.root/'20004/cub/vit_b16/ft'
        meta = json.loads((pair/'manifest.json').read_text())
        meta['code_sha256'] = 'f'*64
        self.seal(pair, meta)
        with self.assertRaisesRegex(ValueError, 'Mixed.*protocol'):
            server.run(self.args('--preflight-only'))


if __name__ == '__main__':
    unittest.main()
