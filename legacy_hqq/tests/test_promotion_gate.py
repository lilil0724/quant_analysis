import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.baseline_recovery.verify_main import verify_raw_source, verify_downstream, verify_artifacts


class PromotionGateTests(unittest.TestCase):
    def test_same_shape_old_raw_is_rejected_beside_corrected_source_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, curated = root/'raw.csv', root/'curated.csv'
            raw.write_text('dataset,top1\npets,67.95\n', encoding='utf-8')
            curated.write_text('dataset,top1\npets,77.57\n', encoding='utf-8')
            receipt = root/'source_manifest.json'
            receipt.write_text(json.dumps(dict(source=str(curated), destination=str(raw),
                sha256=hashlib.sha256(curated.read_bytes()).hexdigest())), encoding='utf-8')
            with self.assertRaises(AssertionError):
                verify_raw_source(raw, curated, receipt)

    def test_old_passed_receipt_does_not_hide_replaced_downstream_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = root/'dataset_robustness/results/main'
            stage.mkdir(parents=True)
            source = root/'summary.csv'
            source.write_text('accuracy_ratio\n0.5\n', encoding='utf-8')
            (stage/'run_config.json').write_text(json.dumps(dict(input_file=str(source),
                input_sha256=hashlib.sha256(source.read_bytes()).hexdigest())), encoding='utf-8')
            (stage/'verification.json').write_text('{"status":"passed"}', encoding='utf-8')
            source.write_text('accuracy_ratio\n0.9\n', encoding='utf-8')
            with self.assertRaises(AssertionError):
                verify_downstream(root)

    def test_changed_plot_is_rejected_even_when_pair_table_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plot = root/'heatmap.png'
            plot.write_bytes(b'original plot')
            artifacts = {plot.name:hashlib.sha256(plot.read_bytes()).hexdigest()}
            plot.write_bytes(b'old plot copied over new plot')
            with self.assertRaises(AssertionError):
                verify_artifacts(root, artifacts)
