import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from hqq_confidence.analyze_overconfidence import (
    EXPECTED_PLOTS, analyze_hypothesis, block_permutation_test,
    overconfidence_metrics, transition_metrics,
)


def write_synthetic_prepared(directory):
    prepared = os.path.join(directory, 'prepared')
    os.makedirs(os.path.join(prepared, 'pairs'))
    transition_rows = []
    condition_rows = []
    conditions = [
        (f'hqq_n{bits:g}_g{group}', bits, group)
        for bits in (1, 1.58, 2, 3, 4, 8) for group in (8, 128)
    ]
    for dataset_index in range(4):
        dataset = f'dataset_{dataset_index}'
        sweep_id = f'sweep_{dataset_index}'
        pair_dir = os.path.join(prepared, 'pairs', sweep_id)
        os.makedirs(pair_dir)
        baseline_correct = np.asarray([True] * 12 + [False] * 8)
        confidence = np.linspace(.51, .69, 20) + dataset_index * .08
        margin = np.linspace(.1, 2, 20)
        condition_rows.append({
            'sweep_id': sweep_id, 'dataset_name': dataset, 'model_name': 'vit_b16',
            'serial': 7, 'condition': 'fp_unquantized', 'n_samples': 20,
            'num_classes': 10 + dataset_index, 'accuracy': 60.0,
            'mean_top1_probability': confidence.mean(),
            'mean_logit_margin': margin.mean() + dataset_index * .1,
            'mean_normalized_entropy': .5,
        })
        for condition, bits, group in conditions:
            hqq_correct = baseline_correct.copy()
            damage_count = 4 - dataset_index
            rescue_count = dataset_index // 2
            hqq_correct[:damage_count] = False
            if rescue_count:
                hqq_correct[12:12 + rescue_count] = True
            agreement = np.ones(20, dtype=bool)
            agreement[:damage_count] = False
            agreement[12:12 + rescue_count] = False
            agreement[19] = False
            pair_path = os.path.join(pair_dir, f'{condition}.npz')
            np.savez_compressed(
                pair_path, baseline_correct=baseline_correct,
                hqq_correct=hqq_correct, agreement=agreement,
                baseline_top1_probability=confidence,
                baseline_margin_percentile=pd.Series(margin).rank(pct=True).to_numpy())
            metrics = transition_metrics(baseline_correct, hqq_correct, agreement)
            baseline_accuracy = 60.0
            hqq_accuracy = 100 * hqq_correct.mean()
            transition_rows.append({
                'sweep_id': sweep_id, 'dataset_name': dataset,
                'model_name': 'vit_b16', 'serial': 7, 'condition': condition,
                'hqq_nbits': bits, 'hqq_group_size': group,
                'baseline_accuracy': baseline_accuracy, 'hqq_top1': hqq_accuracy,
                'accuracy_ratio': hqq_accuracy / baseline_accuracy,
                'accuracy_drop_points': baseline_accuracy - hqq_accuracy,
                'damage_rate': damage_count / 12,
                'rescue_rate': rescue_count / 8,
                'prediction_agreement': metrics['prediction_agreement'],
                'pair_file': os.path.relpath(pair_path, prepared),
            })
    pd.DataFrame(transition_rows).to_csv(
        os.path.join(prepared, 'transition_summary.csv'), index=False)
    pd.DataFrame(condition_rows).to_csv(
        os.path.join(prepared, 'condition_summary.csv'), index=False)
    return prepared


class TestOverconfidenceMetrics(unittest.TestCase):
    def test_oece_does_not_count_underconfidence(self):
        metrics, bins = overconfidence_metrics([0.2, 0.4], [True, True], bins=1)
        self.assertEqual(metrics['oece'], 0)
        self.assertAlmostEqual(metrics['ece_uncalibrated'], 0.7)
        self.assertEqual(bins[0]['overconfidence_contribution'], 0)

    def test_transition_decomposition_and_zero_swap(self):
        metrics = transition_metrics(
            [True, True, False, False], [False, True, True, False],
            [False, True, False, False])
        self.assertEqual(metrics['swap_count'], 3)
        self.assertEqual(metrics['damage_count'], 1)
        self.assertEqual(metrics['rescue_count'], 1)
        self.assertEqual(metrics['wrong_to_wrong_swap_count'], 1)
        self.assertEqual(metrics['net_damage_per_swap'], 0)
        no_swap = transition_metrics([True, False], [True, False], [True, True])
        self.assertTrue(np.isnan(no_swap['net_damage_per_swap']))

    def test_damage_only_and_rescue_only_have_opposite_net_damage(self):
        damage = transition_metrics([True], [False], [False])
        rescue = transition_metrics([False], [True], [False])
        self.assertEqual(damage['net_damage_per_swap'], 1)
        self.assertEqual(rescue['net_damage_per_swap'], -1)


class TestHypothesisStatistics(unittest.TestCase):
    def test_block_permutation_is_reproducible_and_detects_negative_direction(self):
        rows = []
        for dataset_index in range(12):
            for condition_index, condition in enumerate(('a', 'b')):
                rows.append({
                    'dataset_name': f'd{dataset_index}',
                    'condition': condition,
                    'hqq_nbits': condition_index + 1,
                    'hqq_group_size': 8,
                    'oece': dataset_index / 20,
                    'net_damage_per_swap': 1 - dataset_index / 20 + condition_index / 100,
                })
        frame = pd.DataFrame(rows)
        first, _ = block_permutation_test(frame, permutations=999, seed=9)
        second, _ = block_permutation_test(frame, permutations=999, seed=9)
        self.assertEqual(first, second)
        self.assertLess(first['median_spearman_rho'], 0)
        self.assertLess(first['permutation_p_value'], .05)

    def test_end_to_end_analysis_writes_tables_report_and_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = write_synthetic_prepared(directory)
            results = os.path.join(directory, 'hypothesis')

            result = analyze_hypothesis(
                prepared, results, serial=7, excluded_datasets=('inat17',),
                expected_datasets=4, permutations=19, model_permutations=3,
                seed=4, dpi=30)

            self.assertIn('decision', result)
            robustness = pd.read_csv(os.path.join(
                results, 'dataset_condition_robustness.csv'))
            self.assertEqual(len(robustness), 48)
            self.assertNotIn('inat17', set(robustness['dataset_name']))
            self.assertEqual(set(EXPECTED_PLOTS), {
                name for name in os.listdir(results) if name.endswith('.png')})
            self.assertTrue(os.path.isfile(os.path.join(results, 'report.md')))


if __name__ == '__main__':
    unittest.main()
