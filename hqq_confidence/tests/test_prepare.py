import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from hqq_confidence.prepare_pairs import (
    EXPECTED_CONDITIONS, condition_config, condition_metrics, load_predictions, prepare,
)


def write_predictions(path, sample_ids, targets, logits, condition):
    logits = np.asarray(logits, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)
    order = np.argsort(-logits, axis=1)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    top1 = order[:, 0]
    top2 = order[:, 1]
    entropy = -(probabilities * np.log(probabilities)).sum(axis=1)
    np.savez_compressed(
        path,
        sample_id=np.asarray(sample_ids), target=targets, logits=logits,
        correct=top1 == targets, top1_class=top1, top2_class=top2,
        top1_logit=logits[np.arange(len(logits)), top1],
        top2_logit=logits[np.arange(len(logits)), top2],
        logit_margin=(logits[np.arange(len(logits)), top1]
                      - logits[np.arange(len(logits)), top2]),
        top1_probability=probabilities[np.arange(len(logits)), top1],
        top2_probability=probabilities[np.arange(len(logits)), top2],
        true_class_probability=probabilities[np.arange(len(logits)), targets],
        entropy=entropy, normalized_entropy=entropy / np.log(logits.shape[1]),
        metadata_schema_version=np.asarray(1),
        metadata_condition=np.asarray(condition),
        metadata_dataset_name=np.asarray('cub'),
        metadata_model_name=np.asarray('vit_b16'),
        metadata_checkpoint_sha256=np.asarray('a' * 64),
        metadata_git_commit=np.asarray('commit'), metadata_split=np.asarray('test'),
        metadata_compute_dtype=np.asarray('float16'),
        metadata_amp_fp16=np.asarray(True),
    )


def write_complete_sweep(directory, shuffled=False):
    sample_ids = np.asarray(['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg'])
    targets = np.asarray([0, 1, 0, 1])
    baseline_logits = np.asarray([[3, 0], [0, 3], [2, 1], [2, 1]], dtype=np.float32)
    rows = []
    for condition in EXPECTED_CONDITIONS:
        logits = baseline_logits.copy()
        ids = sample_ids.copy()
        condition_targets = targets.copy()
        if condition != 'fp_unquantized':
            logits[0] = [0, 3]
            logits[3] = [0, 3]
        if shuffled and condition == 'hqq_n1_g8':
            index = [3, 1, 0, 2]
            ids, condition_targets, logits = ids[index], condition_targets[index], logits[index]
        path = os.path.join(directory, f'{condition}.npz')
        write_predictions(path, ids, condition_targets, logits, condition)
        hqq = condition != 'fp_unquantized'
        rows.append({
            'project': 'entity/project', 'run_id': condition, 'serial': 7,
            'dataset_name': 'cub', 'model_name': 'vit_b16',
            'checkpoint_sha256': 'a' * 64, 'condition': condition,
            'seed': 0, 'npz_path': path, 'git_commit': 'commit',
            'amp_fp16': True, 'compute_dtype': 'float16',
            'summary_accuracy': 75.0, 'quantized_linear_count': 12 if hqq else 0,
        })
    manifest = os.path.join(directory, 'selected_runs.csv')
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


class TestPreparePairs(unittest.TestCase):
    def test_expected_sweep_includes_fractional_and_three_bit_conditions(self):
        self.assertEqual(len(EXPECTED_CONDITIONS), 13)
        self.assertIn('hqq_n1.58_g8', EXPECTED_CONDITIONS)
        self.assertIn('hqq_n3_g128', EXPECTED_CONDITIONS)
        self.assertEqual(condition_config('hqq_n1.58_g8'), (True, 1.58, 8))

    def test_complete_sweep_aligns_samples_by_id_and_decomposes_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_complete_sweep(directory, shuffled=True)
            output = os.path.join(directory, 'prepared')

            prepare(manifest, output)

            transitions = pd.read_csv(os.path.join(output, 'transition_summary.csv'))
            row = transitions[transitions['condition'] == 'hqq_n1_g8'].iloc[0]
            self.assertEqual(row['damage_count'], 1)
            self.assertEqual(row['rescue_count'], 1)
            self.assertEqual(row['accuracy_drop_points'], 0.0)
            pair_path = os.path.join(output, 'pairs', row['sweep_id'], 'hqq_n1_g8.npz')
            with np.load(pair_path) as pair:
                self.assertEqual(
                    pair['sample_id'].tolist(), ['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg'])
                self.assertEqual(pair['damage'].tolist(), [True, False, False, False])
                self.assertEqual(pair['rescue'].tolist(), [False, False, False, True])

    def test_duplicate_condition_is_rejected_as_ambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_complete_sweep(directory)
            frame = pd.read_csv(manifest)
            pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(manifest, index=False)

            with self.assertRaisesRegex(ValueError, 'Ambiguous duplicate runs'):
                prepare(manifest, os.path.join(directory, 'prepared'))

    def test_incomplete_sweep_is_reported_and_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_complete_sweep(directory)
            frame = pd.read_csv(manifest)
            frame[frame['condition'] != 'hqq_n8_g128'].to_csv(manifest, index=False)
            output = os.path.join(directory, 'prepared')

            with self.assertRaisesRegex(ValueError, 'No complete'):
                prepare(manifest, output)

            completeness = pd.read_csv(os.path.join(output, 'completeness.csv'))
            self.assertIn('hqq_n8_g128', completeness.loc[0, 'missing_conditions'])
            self.assertFalse(bool(completeness.loc[0, 'eligible']))

    def test_legacy_schema_without_amp_metadata_is_loaded_by_capability(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_complete_sweep(directory)
            path = os.path.join(directory, 'fp_unquantized.npz')
            with np.load(path) as payload:
                legacy = {key: payload[key] for key in payload.files
                          if key != 'metadata_amp_fp16'}
            np.savez_compressed(path, **legacy)

            prepare(manifest, os.path.join(directory, 'prepared'))

            self.assertTrue(os.path.isfile(os.path.join(
                directory, 'prepared', 'condition_summary.csv')))

    def test_target_mismatch_for_same_sample_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_complete_sweep(directory)
            path = os.path.join(directory, 'hqq_n1_g8.npz')
            write_predictions(
                path, ['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg'], [1, 1, 0, 1],
                [[0, 3], [0, 3], [2, 1], [0, 3]], 'hqq_n1_g8')

            with self.assertRaisesRegex(ValueError, 'Targets differ'):
                prepare(manifest, os.path.join(directory, 'prepared'))

    def test_class_count_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_complete_sweep(directory)
            write_predictions(
                os.path.join(directory, 'hqq_n1_g8.npz'),
                ['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg'], [0, 1, 0, 1],
                [[3, 0, -1], [0, 3, -1], [2, 1, -1], [0, 3, -1]],
                'hqq_n1_g8')

            with self.assertRaisesRegex(ValueError, 'N or C differs'):
                prepare(manifest, os.path.join(directory, 'prepared'))

    def test_checkpoint_mismatch_cannot_form_a_complete_sweep(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_complete_sweep(directory)
            frame = pd.read_csv(manifest)
            frame.loc[frame['condition'] == 'hqq_n1_g8', 'checkpoint_sha256'] = 'b' * 64
            frame.to_csv(manifest, index=False)

            with self.assertRaisesRegex(ValueError, 'No complete'):
                prepare(manifest, os.path.join(directory, 'prepared'))

    def test_debugging_sweep_is_reported_and_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_complete_sweep(directory)
            frame = pd.read_csv(manifest)
            frame['debugging'] = True
            frame.to_csv(manifest, index=False)
            output = os.path.join(directory, 'prepared')

            with self.assertRaisesRegex(ValueError, 'No complete'):
                prepare(manifest, output)

            quality = pd.read_csv(os.path.join(output, 'configuration_quality.csv'))
            self.assertIn('debugging_run', quality.loc[0, 'provenance_issues'])

    def test_uncalibrated_metrics_match_a_uniform_binary_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'uniform.npz')
            write_predictions(path, ['a.jpg'], [0], [[0, 0]], 'fp_unquantized')

            metrics, reliability = condition_metrics(load_predictions(path))

            self.assertAlmostEqual(metrics['nll_uncalibrated'], np.log(2))
            self.assertAlmostEqual(metrics['brier_uncalibrated'], 0.5)
            self.assertAlmostEqual(metrics['ece_uncalibrated'], 0.5)
            self.assertEqual(reliability[0]['mean_confidence'], 0.5)

    def test_loaded_predictions_discard_full_class_matrices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'small.npz')
            write_predictions(
                path, ['a.jpg', 'b.jpg'], [0, 1],
                [[3, 1, 0], [0, 1, 3]], 'fp_unquantized')

            actual = load_predictions(path, chunk_size=1)

            self.assertNotIn('logits', actual)
            self.assertNotIn('_probabilities', actual)
            self.assertNotIn('_log_probabilities', actual)
            self.assertEqual(actual['_shape'], (2, 3))
            self.assertEqual(actual['_num_classes'], 3)
            self.assertIsInstance(actual['_brier_uncalibrated'], float)

    def test_tied_logits_accept_the_topk_class_recorded_by_pytorch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'tied.npz')
            write_predictions(path, ['a.jpg'], [0], [[1, 1]], 'fp_unquantized')
            with np.load(path) as payload:
                arrays = {key: payload[key] for key in payload.files}
            arrays['top1_class'] = np.asarray([1])
            arrays['top2_class'] = np.asarray([0])
            arrays['correct'] = np.asarray([False])
            np.savez_compressed(path, **arrays)

            actual = load_predictions(path)

            self.assertEqual(actual['top1_class'].tolist(), [1])
            self.assertEqual(actual['correct'].tolist(), [False])

    def test_small_probability_rounding_drift_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'rounded.npz')
            write_predictions(path, ['a.jpg'], [0], [[1, 1]], 'fp_unquantized')
            with np.load(path) as payload:
                arrays = {key: payload[key] for key in payload.files}
            arrays['top1_probability'] = arrays['top1_probability'] + 5e-5
            np.savez_compressed(path, **arrays)

            actual = load_predictions(path)

            self.assertEqual(len(actual['target']), 1)

    def test_material_probability_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'invalid.npz')
            write_predictions(path, ['a.jpg'], [0], [[1, 1]], 'fp_unquantized')
            with np.load(path) as payload:
                arrays = {key: payload[key] for key in payload.files}
            arrays['top1_probability'] = arrays['top1_probability'] + 0.01
            np.savez_compressed(path, **arrays)

            with self.assertRaisesRegex(ValueError, 'top1_probability'):
                load_predictions(path)


if __name__ == '__main__':
    unittest.main()
