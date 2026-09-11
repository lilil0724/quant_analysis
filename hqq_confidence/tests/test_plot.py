import os
import tempfile
import unittest

from hqq_confidence.analyze_fragility import analyze
from hqq_confidence.plot_fragility import EXPECTED_PLOTS, plot_all
from hqq_confidence.prepare_pairs import prepare
from hqq_confidence.tests.test_prepare import write_complete_sweep


class TestPlotFragility(unittest.TestCase):
    def test_plot_stage_writes_every_documented_figure(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_complete_sweep(directory)
            prepared = os.path.join(directory, 'prepared')
            stats = os.path.join(directory, 'stats')
            plots = os.path.join(directory, 'plots')
            prepare(manifest, prepared)
            analyze(prepared, stats, bootstrap_samples=10, seed=0)

            written = plot_all(prepared, stats, plots, dpi=40)

            self.assertEqual({os.path.basename(path) for path in written}, set(EXPECTED_PLOTS))
            self.assertTrue(all(os.path.getsize(path) > 0 for path in written))
            self.assertTrue(os.path.isfile(os.path.join(plots, 'plots_manifest.csv')))


if __name__ == '__main__':
    unittest.main()
