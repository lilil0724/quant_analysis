import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import five_dataset_features.analysis_snapshot as snapshot
from five_dataset_features.analyze import DATASETS, grid, package
from five_dataset_features.analysis_snapshot import (
    DATASET_OUTPUTS, REPORT_OUTPUTS, SNAPSHOT_FILE, SnapshotError,
    record_dataset, record_report, source_identity, verify_dataset,
    verify_report,
)


class AnalysisSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, path, content=b'initial'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def dataset(self, name, conditions):
        self.write(self.root / name / 'dataset.json')
        for condition in conditions:
            self.write(self.root / name / condition / 'manifest.json')
        output = self.root / 'analysis' / name
        for filename in (*DATASET_OUTPUTS,
                         *(f'coords_{condition}.npy' for condition in conditions)):
            self.write(output / filename)
        initial = source_identity(self.root, name, conditions)
        record_dataset(self.root, name, conditions, initial)

    def report(self, datasets, conditions):
        for filename in REPORT_OUTPUTS:
            self.write(self.root / 'analysis' / filename)
        record_report(self.root, datasets, conditions)

    def test_dataset_snapshot_rejects_changed_source_or_derived_file(self):
        conditions = ('fp', 'w3_g8')
        self.dataset('cub', conditions)
        verify_dataset(self.root, 'cub', conditions)

        manifest = self.root / 'cub' / 'w3_g8' / 'manifest.json'
        self.write(manifest, b'updated source')
        with self.assertRaisesRegex(SnapshotError, 'source manifests changed: w3_g8/manifest.json'):
            verify_dataset(self.root, 'cub', conditions)

        self.write(manifest)
        self.write(self.root / 'analysis' / 'cub' / 'summary.csv', b'updated analysis')
        with self.assertRaisesRegex(SnapshotError, 'derived files changed: summary.csv'):
            verify_dataset(self.root, 'cub', conditions)

    def test_old_output_without_snapshot_requires_reanalysis(self):
        with self.assertRaisesRegex(SnapshotError, 'Rerun dataset cub'):
            verify_dataset(self.root, 'cub', ('fp',))

    def test_analysis_code_change_requires_all_datasets(self):
        conditions = ('fp',)
        self.dataset('cub', conditions)
        changed = dict(snapshot._code_hashes(''))
        changed['analyze.py'] = '0' * 64
        with patch.object(snapshot, '_code_hashes', return_value=changed):
            with self.assertRaisesRegex(SnapshotError, 'Rerun all five datasets'):
                verify_dataset(self.root, 'cub', conditions)

    def test_one_dataset_rerun_requires_new_report(self):
        conditions = ('fp', 'w3_g8')
        for dataset in DATASETS:
            self.dataset(dataset, conditions)
        self.report(DATASETS, conditions)
        verify_report(self.root, DATASETS, conditions)

        changed = DATASETS[0]
        self.write(self.root / changed / 'fp' / 'manifest.json', b'updated source')
        self.write(self.root / 'analysis' / changed / 'summary.csv', b'updated analysis')
        initial = source_identity(self.root, changed, conditions)
        record_dataset(self.root, changed, conditions, initial)
        with self.assertRaisesRegex(SnapshotError, 'Rerun report, then package'):
            verify_report(self.root, DATASETS, conditions)

        record_report(self.root, DATASETS, conditions)
        verify_report(self.root, DATASETS, conditions)
        self.write(self.root / 'analysis' / 'all_conditions.csv', b'changed report')
        with self.assertRaisesRegex(SnapshotError, 'report files changed: all_conditions.csv'):
            verify_report(self.root, DATASETS, conditions)

    def test_package_stops_before_replacing_zip_when_source_is_stale(self):
        conditions = grid()
        for dataset in DATASETS:
            self.dataset(dataset, conditions)
        self.report(DATASETS, conditions)
        archive = self.root / 'analysis' / 'five_dataset_hqq_results.zip'
        self.write(archive, b'existing archive')
        self.write(self.root / DATASETS[0] / 'fp' / 'manifest.json', b'updated source')

        with self.assertRaisesRegex(SnapshotError, 'Rerun dataset cotton'):
            package(self.root)
        self.assertEqual(archive.read_bytes(), b'existing archive')
        self.assertTrue((self.root / 'analysis' / DATASETS[0] / SNAPSHOT_FILE).is_file())


if __name__ == '__main__':
    unittest.main()
