import os
import tempfile
import unittest

import pandas as pd

from hqq_confidence.analyze_fragility import analyze
from hqq_confidence.prepare_pairs import prepare
from hqq_confidence.tests.test_prepare import write_complete_sweep


class TestAnalyzeFragility(unittest.TestCase):
    def test_confidence_bins_use_baseline_correct_samples_and_are_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_complete_sweep(directory)
            prepared = os.path.join(directory, 'prepared')
            first = os.path.join(directory, 'stats-first')
            second = os.path.join(directory, 'stats-second')
            prepare(manifest, prepared)

            analyze(prepared, first, bootstrap_samples=50, seed=17)
            analyze(prepared, second, bootstrap_samples=50, seed=17)

            actual = pd.read_csv(os.path.join(first, 'confidence_bins.csv'))
            comparison = pd.read_csv(os.path.join(second, 'confidence_bins.csv'))
            pd.testing.assert_frame_equal(actual, comparison)
            selected = actual[
                (actual['condition'] == 'hqq_n1_g8') & (actual['confidence_decile'] == 9)
            ].iloc[0]
            self.assertEqual(selected['n_samples'], 2)
            self.assertEqual(selected['damage_rate'], 0.5)
            self.assertEqual(selected['ci95_low'], 0.5)
            self.assertEqual(selected['ci95_high'], 0.5)
            self.assertTrue(os.path.isfile(os.path.join(first, 'report.md')))
            self.assertTrue(os.path.isfile(os.path.join(first, 'bootstrap_metadata.json')))
            contexts = pd.read_csv(os.path.join(
                first, 'condition_statistics_by_context.csv'))
            self.assertEqual(
                set(contexts['context_type']),
                {'dataset_name', 'model_name', 'ckpt_kind'},
            )
            self.assertEqual(set(contexts['context_value']), {'cub', 'vit_b16', 'unknown'})


if __name__ == '__main__':
    unittest.main()
