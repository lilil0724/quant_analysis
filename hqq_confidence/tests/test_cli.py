import os
import subprocess
import sys
import tempfile
import unittest

from hqq_confidence.plot_fragility import EXPECTED_PLOTS
from hqq_confidence.tests.test_prepare import write_complete_sweep


class TestOfflineWorkflow(unittest.TestCase):
    def test_cached_artifacts_run_through_prepare_analyze_and_plot_clis(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_complete_sweep(directory)
            prepared = os.path.join(directory, 'prepared')
            stats = os.path.join(directory, 'stats')
            plots = os.path.join(directory, 'plots')
            environment = os.environ.copy()
            environment['MPLCONFIGDIR'] = os.path.join(directory, 'matplotlib')
            commands = [
                [sys.executable, '-m', 'hqq_confidence.prepare_pairs',
                 '--manifest-file', manifest, '--output-dir', prepared],
                [sys.executable, '-m', 'hqq_confidence.analyze_fragility',
                 '--input-dir', prepared, '--results-dir', stats,
                 '--bootstrap-samples', '10', '--seed', '4'],
                [sys.executable, '-m', 'hqq_confidence.plot_fragility',
                 '--prepared-dir', prepared, '--stats-dir', stats,
                 '--results-dir', plots, '--dpi', '40'],
            ]

            for command in commands:
                subprocess.run(command, check=True, cwd=os.getcwd(), env=environment,
                               capture_output=True, text=True)

            self.assertTrue(os.path.isfile(os.path.join(prepared, 'run_config.json')))
            self.assertTrue(os.path.isfile(os.path.join(stats, 'run_config.json')))
            self.assertTrue(os.path.isfile(os.path.join(plots, 'run_config.json')))
            self.assertEqual(
                {name for name in os.listdir(plots) if name.endswith('.png')},
                set(EXPECTED_PLOTS))


if __name__ == '__main__':
    unittest.main()
