import os
import shutil
import tempfile
import unittest

from hqq_confidence.download_artifacts import download


class FakeArtifact:
    type = 'prediction-logits'
    name = 'hqq-logits-cub-vit_b16-fp_unquantized-aaaaaaaaaaaa'
    version = 'v3'
    digest = 'artifact-digest'

    def __init__(self, source):
        self.source = source
        self.downloaded_to = None

    def download(self, root):
        self.downloaded_to = root
        os.makedirs(root, exist_ok=True)
        shutil.copyfile(self.source, os.path.join(root, 'predictions.npz'))
        return root


class FakeRun:
    id = 'run-123'
    name = 'baseline'
    group = 'group'
    state = 'finished'
    job_type = 'hqq-confidence-eval'
    created_at = '2026-09-11T00:00:00Z'
    config = {
        'serial': 7, 'dataset_name': 'cub', 'model_name': 'vit_b16',
        'ckpt_path': 'checkpoints/cub.pth', 'checkpoint_sha256': 'a' * 64,
        'condition': 'fp_unquantized', 'seed': 0, 'debugging': False,
        'fp16': True, 'hqq_compute_dtype': 'float16', 'git_commit': 'commit',
    }
    summary = {
        'top1': 75.0, 'n_samples': 4, 'num_classes': 2,
        'quantized_linear_count': 0,
    }

    def __init__(self, artifact):
        self.artifact = artifact
        self.logged_artifacts_called = False

    def logged_artifacts(self):
        self.logged_artifacts_called = True
        return [self.artifact]


class FakeApi:
    def __init__(self, run):
        self.selected_run = run
        self.artifact_called = False

    def runs(self, path, filters):
        return [self.selected_run]

    def artifact(self, *args, **kwargs):
        self.artifact_called = True
        raise AssertionError('Global artifact lookup must not be used')


class DuplicateApi(FakeApi):
    def runs(self, path, filters):
        duplicate = FakeRun(self.selected_run.artifact)
        duplicate.id = 'run-duplicate'
        return [self.selected_run, duplicate]


class TestDownloadArtifacts(unittest.TestCase):
    def test_download_uses_artifact_logged_by_selected_run(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'source.npz')
            with open(source, 'wb') as handle:
                handle.write(b'fixture')
            artifact = FakeArtifact(source)
            run = FakeRun(artifact)
            api = FakeApi(run)
            output = os.path.join(directory, 'selected_runs.csv')

            manifest = download(
                api, 'entity/project', output, os.path.join(directory, 'cache'))

            self.assertTrue(run.logged_artifacts_called)
            self.assertFalse(api.artifact_called)
            self.assertEqual(manifest.loc[0, 'run_id'], 'run-123')
            self.assertEqual(manifest.loc[0, 'artifact_version'], 'v3')
            self.assertEqual(
                os.path.basename(manifest.loc[0, 'npz_path']), 'predictions.npz')
            self.assertTrue(os.path.isfile(manifest.loc[0, 'npz_path']))

    def test_legacy_hqq_top1_summary_is_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'source.npz')
            with open(source, 'wb') as handle:
                handle.write(b'fixture')
            run = FakeRun(FakeArtifact(source))
            run.config = dict(run.config, condition='hqq_n1_g8')
            run.summary = dict(run.summary, top1=62.5, quantized_linear_count=12)

            manifest = download(
                FakeApi(run), 'entity/project',
                os.path.join(directory, 'selected_runs.csv'),
                os.path.join(directory, 'cache'))

            self.assertEqual(manifest.loc[0, 'summary_accuracy_key'], 'top1')
            self.assertEqual(manifest.loc[0, 'summary_accuracy'], 62.5)

    def test_hqq_top1_is_the_canonical_hqq_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'source.npz')
            with open(source, 'wb') as handle:
                handle.write(b'fixture')
            run = FakeRun(FakeArtifact(source))
            run.config = dict(run.config, condition='hqq_n1_g8')
            run.summary = dict(
                run.summary, hqq_top1=61.25, top1=62.5,
                quantized_linear_count=12)

            manifest = download(
                FakeApi(run), 'entity/project',
                os.path.join(directory, 'selected_runs.csv'),
                os.path.join(directory, 'cache'))

            self.assertEqual(manifest.loc[0, 'summary_accuracy_key'], 'hqq_top1')
            self.assertEqual(manifest.loc[0, 'summary_accuracy'], 61.25)

    def test_duplicate_sweep_condition_requires_explicit_run_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'source.npz')
            with open(source, 'wb') as handle:
                handle.write(b'fixture')
            api = DuplicateApi(FakeRun(FakeArtifact(source)))

            with self.assertRaisesRegex(ValueError, 'Ambiguous duplicate runs'):
                download(api, 'entity/project',
                         os.path.join(directory, 'selected_runs.csv'),
                         os.path.join(directory, 'cache'))


if __name__ == '__main__':
    unittest.main()
