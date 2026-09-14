import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from hqq_confidence.analyze_sample_margin_mechanism import (
    aggregate_auroc_results,
    assign_margin_deciles,
    binary_auroc,
    build_sample_mechanism_tables,
    summarize_decile_curves,
)


PRIMARY_CONDITIONS = (
    'hqq_n3_g8', 'hqq_n3_g128', 'hqq_n4_g8', 'hqq_n4_g128',
)


class TestSampleMarginMetrics(unittest.TestCase):
    def test_binary_auroc_uses_average_ranks_for_ties(self):
        scores = np.asarray([0.0, 1.0, 1.0, 2.0])
        swapped = np.asarray([False, False, True, True])

        auc, status = binary_auroc(scores, swapped)

        self.assertEqual(status, 'ok')
        self.assertAlmostEqual(auc, 0.875)

    def test_binary_auroc_marks_single_class_as_undefined(self):
        auc, status = binary_auroc(
            np.asarray([1.0, 2.0, 3.0]), np.asarray([False, False, False]))

        self.assertTrue(np.isnan(auc))
        self.assertEqual(status, 'undefined_single_class')

    def test_margin_deciles_do_not_split_ties(self):
        values = np.asarray([0.0, 0.0, 1.0, 2.0, 3.0, 4.0])

        deciles = assign_margin_deciles(values)

        self.assertEqual(deciles[0], deciles[1])
        self.assertEqual(deciles.min(), 1)
        self.assertEqual(deciles.max(), 9)


class TestDatasetEqualAggregation(unittest.TestCase):
    def _auroc_rows(self, scores):
        rows = []
        for dataset, score in scores.items():
            for condition in PRIMARY_CONDITIONS:
                rows.append({
                    'dataset_name': dataset,
                    'condition': condition,
                    'population': 'all_matched',
                    'auroc': score,
                    'status': 'ok',
                })
        return pd.DataFrame(rows)

    def test_gate_uses_one_complete_equal_weight_score_per_dataset(self):
        scores = {f'd{index:02d}': 0.70 for index in range(11)}
        scores.update({f'd{index:02d}': 0.45 for index in range(11, 14)})

        datasets, conditions, decision = aggregate_auroc_results(
            self._auroc_rows(scores), expected_datasets=14)

        self.assertEqual(len(datasets), 14)
        self.assertTrue((datasets['identifiable_conditions'] == 4).all())
        self.assertTrue(decision['sign_test_p_value'] < 0.05)
        self.assertEqual(decision['status'], 'PASS')
        self.assertTrue((conditions['median_auroc'] > 0.5).all())

    def test_missing_primary_condition_makes_gate_incomplete(self):
        rows = self._auroc_rows({f'd{index:02d}': 0.70 for index in range(14)})
        mask = (
            (rows['dataset_name'] == 'd00')
            & (rows['condition'] == PRIMARY_CONDITIONS[0])
        )
        rows.loc[mask, ['auroc', 'status']] = [np.nan, 'undefined_single_class']

        datasets, _, decision = aggregate_auroc_results(
            rows, expected_datasets=14)

        self.assertTrue(np.isnan(
            datasets.set_index('dataset_name').loc['d00', 'mean_auroc']))
        self.assertEqual(decision['status'], 'INCOMPLETE')

    def test_decile_summary_gives_datasets_equal_weight(self):
        rows = pd.DataFrame([
            {
                'dataset_name': 'large', 'condition': condition,
                'decile': 1, 'n_samples': 1000, 'swap_rate': 0.2,
            }
            for condition in PRIMARY_CONDITIONS
        ] + [
            {
                'dataset_name': 'small', 'condition': condition,
                'decile': 1, 'n_samples': 10, 'swap_rate': 0.8,
            }
            for condition in PRIMARY_CONDITIONS
        ])

        dataset_curves, aggregate, _ = summarize_decile_curves(
            rows, bootstrap_samples=100, seed=3)

        self.assertEqual(len(dataset_curves), 2)
        self.assertAlmostEqual(aggregate.loc[0, 'mean_swap_rate'], 0.5)
        self.assertEqual(aggregate.loc[0, 'contributing_datasets'], 2)


class TestPreparedPairIntegration(unittest.TestCase):
    def test_aligns_normalized_margins_to_pairs_by_sample_id(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = os.path.join(directory, 'prepared')
            pairs = os.path.join(prepared, 'pairs', 'sweep')
            os.makedirs(pairs)
            baseline_path = os.path.join(directory, 'baseline.npz')
            sample_ids = np.asarray(['a', 'b', 'c', 'd'])
            logits = np.asarray([
                [1.0, 0.9, -1.0],
                [3.0, 1.0, 0.0],
                [4.0, 0.5, 0.0],
                [5.0, 0.2, 0.0],
            ])
            np.savez_compressed(
                baseline_path, sample_id=sample_ids, logits=logits)
            shuffled = np.asarray([2, 0, 3, 1])
            transition_rows = []
            for condition in PRIMARY_CONDITIONS:
                pair_file = os.path.join('pairs', 'sweep', f'{condition}.npz')
                np.savez_compressed(
                    os.path.join(prepared, pair_file),
                    sample_id=sample_ids[shuffled],
                    baseline_correct=np.asarray([True, True, True, True])[
                        shuffled],
                    agreement=np.asarray([True, False, True, False])[shuffled],
                )
                bits = 3.0 if condition.startswith('hqq_n3_') else 4.0
                group = 128 if condition.endswith('g128') else 8
                transition_rows.append({
                    'dataset_name': 'demo',
                    'serial': 7,
                    'condition': condition,
                    'hqq_nbits': bits,
                    'hqq_group_size': group,
                    'pair_file': pair_file,
                })
            pd.DataFrame(transition_rows).to_csv(
                os.path.join(prepared, 'transition_summary.csv'), index=False)
            manifest = pd.DataFrame([{
                'dataset_name': 'demo',
                'serial': 7,
                'condition': 'fp_unquantized',
                'npz_path': baseline_path,
            }])

            auroc_rows, decile_rows = build_sample_mechanism_tables(
                prepared, manifest, serial=7, excluded_datasets=(),
                expected_datasets=1, chunk_size=2)

            self.assertEqual(len(auroc_rows), 8)
            self.assertEqual(len(decile_rows), 16)
            self.assertTrue((auroc_rows['status'] == 'ok').all())
            self.assertTrue((auroc_rows['swap_count'] == 2).all())


if __name__ == '__main__':
    unittest.main()
