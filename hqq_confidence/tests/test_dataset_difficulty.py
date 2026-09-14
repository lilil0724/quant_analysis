import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from hqq_confidence.analyze_dataset_difficulty import (
    build_difficulty_table,
    dataset_block_permutation_test,
    extract_baseline_logit_features,
    nested_lodo_comparison,
    normalized_decision_margins,
)


class TestNormalizedDecisionMargin(unittest.TestCase):
    def test_is_invariant_to_positive_logit_scaling(self):
        logits = np.asarray([
            [3.0, 2.0, 1.0],
            [4.0, 1.0, 0.0],
        ], dtype=np.float32)

        original = normalized_decision_margins(logits, chunk_size=1)
        scaled = normalized_decision_margins(logits * 7, chunk_size=2)

        np.testing.assert_allclose(original, scaled, rtol=1e-6)
        np.testing.assert_allclose(
            original,
            [1.224744871391589, 1.7650452162436563],
            rtol=1e-6,
        )


class TestExploratoryDatasetQuantizationDifficulty(unittest.TestCase):
    def test_uses_primary_conditions_and_recovers_concordant_ranking(self):
        rows = []
        losses = {'easy': 0.1, 'middle': 0.2, 'hard': 0.3}
        for dataset, loss in losses.items():
            for bits in (2, 3, 4, 8):
                for group in (8, 128):
                    rows.append({
                        'dataset_name': dataset,
                        'condition': f'hqq_n{bits}_g{group}',
                        'hqq_nbits': bits,
                        'hqq_group_size': group,
                        'accuracy_ratio': 1 - loss,
                    })

        difficulty, primary_rows, concordance = build_difficulty_table(
            pd.DataFrame(rows))

        self.assertEqual(set(primary_rows['hqq_nbits']), {3, 4})
        self.assertAlmostEqual(concordance['kendalls_w'], 1.0)
        scores = difficulty.set_index('dataset_name')['difficulty_score']
        self.assertAlmostEqual(scores['easy'], 1 / 3)
        self.assertAlmostEqual(scores['middle'], 2 / 3)
        self.assertAlmostEqual(scores['hard'], 1.0)
        self.assertAlmostEqual(
            difficulty.set_index('dataset_name').loc[
                'middle', 'median_relative_accuracy_loss'],
            0.2,
        )


class TestBaselineLogitFeatureExtraction(unittest.TestCase):
    def test_reads_only_the_matched_baseline_and_summarizes_vulnerable_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = os.path.join(directory, 'baseline.npz')
            ignored_path = os.path.join(directory, 'quantized.npz')
            logits = np.asarray([
                [3.0, 2.0, 1.0],
                [6.0, 4.0, 2.0],
            ], dtype=np.float32)
            probabilities = np.asarray([0.7, 0.7], dtype=np.float32)
            np.savez_compressed(
                baseline_path,
                logits=logits,
                logit_margin=np.asarray([1.0, 2.0]),
                top1_probability=probabilities,
                top2_probability=np.asarray([0.2, 0.2]),
                normalized_entropy=np.asarray([0.4, 0.4]),
            )
            np.savez_compressed(ignored_path, not_logits=np.asarray([1]))
            manifest = pd.DataFrame([
                {
                    'dataset_name': 'kept', 'serial': 7,
                    'condition': 'fp_unquantized', 'npz_path': baseline_path,
                },
                {
                    'dataset_name': 'kept', 'serial': 7,
                    'condition': 'hqq_n3_g8', 'npz_path': ignored_path,
                },
                {
                    'dataset_name': 'inat17', 'serial': 7,
                    'condition': 'fp_unquantized', 'npz_path': ignored_path,
                },
            ])

            features = extract_baseline_logit_features(
                manifest, serial=7, excluded_datasets=('inat17',), chunk_size=1)

            self.assertEqual(list(features['dataset_name']), ['kept'])
            self.assertAlmostEqual(
                features.loc[0, 'normalized_margin_q10'], 1.224744871391589)
            self.assertAlmostEqual(features.loc[0, 'mean_probability_gap'], 0.5)
            self.assertEqual(features.loc[0, 'num_classes'], 3)


class TestDatasetLevelInference(unittest.TestCase):
    def test_block_permutation_supports_prespecified_negative_alternative(self):
        rows = []
        for dataset_index in range(12):
            for condition in ('hqq_n3_g8', 'hqq_n4_g8'):
                rows.append({
                    'dataset_name': f'd{dataset_index}',
                    'condition': condition,
                    'normalized_margin_q10': float(dataset_index),
                    'relative_accuracy_loss': float(12 - dataset_index),
                })
        frame = pd.DataFrame(rows)

        first, correlations = dataset_block_permutation_test(
            frame, 'normalized_margin_q10', 'relative_accuracy_loss',
            alternative='negative', permutations=999, seed=11)
        second, _ = dataset_block_permutation_test(
            frame, 'normalized_margin_q10', 'relative_accuracy_loss',
            alternative='negative', permutations=999, seed=11)

        self.assertEqual(first, second)
        self.assertEqual(len(correlations), 2)
        self.assertAlmostEqual(first['median_spearman_rho'], -1.0)
        self.assertLess(first['permutation_p_value'], 0.05)

    def test_nested_lodo_rewards_vulnerable_tail_information(self):
        frame = pd.DataFrame({
            'dataset_name': [f'd{index}' for index in range(12)],
            'difficulty_score': np.linspace(1, 0, 12),
            'baseline_accuracy': np.tile([70.0, 80.0, 90.0], 4),
            'num_classes': np.tile([10, 20, 40, 80], 3),
            'normalized_margin_q10': np.linspace(0, 1, 12),
        })

        result, predictions = nested_lodo_comparison(
            frame, permutations=199, seed=12)

        self.assertEqual(len(predictions), 12)
        self.assertGreater(result['mae_improvement_m0_minus_m1'], 0)
        self.assertLess(result['m1_mae'], result['m0_mae'])

    def test_block_permutation_moves_an_entire_condition_profile(self):
        rows = []
        for dataset_index in range(12):
            for condition_index, condition in enumerate(('a', 'b')):
                swap_rate = dataset_index + condition_index / 10
                rows.append({
                    'dataset_name': f'd{dataset_index}',
                    'condition': condition,
                    'swap_rate': swap_rate,
                    'relative_accuracy_loss': 2 * swap_rate,
                })
        result, correlations = dataset_block_permutation_test(
            pd.DataFrame(rows), 'swap_rate', 'relative_accuracy_loss',
            alternative='positive', permutations=999, seed=13)

        self.assertTrue((correlations['spearman_rho'] == 1).all())
        self.assertAlmostEqual(result['median_spearman_rho'], 1.0)
        self.assertLess(result['permutation_p_value'], 0.05)


if __name__ == '__main__':
    unittest.main()
