import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.baseline_recovery.replot_20007 import load_records


class RecoverySourceTests(unittest.TestCase):
    def test_verified_records_load_and_tampered_baseline_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = root / 'selected_runs.jsonl'
            target = root / 'serial_20007_runs.json'
            selected.write_text('{"id":"fp"}\n', encoding='utf-8')
            target.write_text('[{"id":"quant"}]', encoding='utf-8')
            (root / 'manifest.json').write_text(json.dumps(dict(status='complete', artifacts={
                selected.name:hashlib.sha256(selected.read_bytes()).hexdigest()})), encoding='utf-8')
            (root / 'source_manifest.json').write_text(json.dumps(dict(files={target.name:{
                'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}})), encoding='utf-8')
            self.assertEqual(load_records(selected, target), ([{'id':'fp'}],[{'id':'quant'}]))
            selected.write_text('{"id":"wrong_fp"}\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_records(selected, target)

    def test_partial_cohort_is_rejected_before_generating_main_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = root / 'selected_runs.jsonl'
            target = root / 'serial_20007_runs.json'
            (root / 'manifest.json').write_text(json.dumps(dict(status='partial_pending_server_rerun')), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'complete recovered cohort'):
                load_records(selected, target)
