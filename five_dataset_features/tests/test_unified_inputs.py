import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from five_dataset_features.unified_inputs import read_condition, read_pair


class TestUnifiedInputs(unittest.TestCase):
    def test_reader_validates_variable_width_arrays_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'fp'
            (root / 'features').mkdir(parents=True)
            np.save(root / 'features/final.npy', np.ones((3, 4), dtype=np.float32))
            np.save(root / 'features/block_00.npy', np.ones((3, 6), dtype=np.float32))
            (root / 'samples.json').write_text(json.dumps({'sample_ids': ['a', 'b', 'c'], 'targets': [0, 1, 0]}))
            (root / 'metrics.json').write_text(json.dumps({'eval/top1_pct': 100.}))
            files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in root.rglob('*') if p.is_file()}
            manifest = {'schema': 2, 'status': 'complete', 'n_samples': 3, 'condition': {'name': 'fp'},
                        'feature_dtype': 'float32', 'layers': {'features/final.npy': [3, 4], 'features/block_00.npy': [3, 6]},
                        'files': files, 'metadata': {
                            'checkpoint_sha256': 'a' * 64, 'data_sha256': 'b' * 64,
                            'code_sha256': 'c' * 64, 'cfg_sha256': 'd' * 64, 'hqq_source_sha256': 'e' * 64,
                            'preprocessing': {'image_size': 224, 'test_resize_size': 300,
                                'test_resize_directly': False, 'test_square_resize_center_crop': True,
                                'custom_mean_std': False, 'mean': None, 'std': None,
                                'pre_resize_factor': None, 'resize_size': 300, 'dataset_in_memory': False,
                                'cal_ap_only': False, 'aug_type': 'crop_mask'}},
                        'execution': {'seed': 100, 'fp16': True, 'hqq_compute_dtype': 'float16',
                            'batch_size': 2, 'cal_ap_only': False, 'dataset_in_memory': False,
                            'reduce_pattern': None, 'reduce_op': 'mean'}}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            actual = read_condition(root)
            self.assertEqual(actual['features']['final'].shape, (3, 4))
            self.assertEqual(actual['features']['block_00'].shape, (3, 6))
            for array in actual['features'].values():
                array._mmap.close()
            (root / 'metrics.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                read_condition(root)

    def test_smoke_and_path_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'manifest.json').write_text(json.dumps({'schema': 2, 'status': 'smoke'}))
            with self.assertRaisesRegex(ValueError, 'complete'):
                read_condition(root)
            (root / 'manifest.json').write_text(json.dumps({'schema': 2, 'status': 'complete',
                                                          'files': {'../outside': 'hash'}}))
            with self.assertRaisesRegex(ValueError, 'path'):
                read_condition(root)

    def test_pairing_rejects_different_source_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            roots = []
            for name, checkpoint in [('fp', 'a' * 64), ('a4_linear', 'b' * 64)]:
                root = Path(folder) / name
                (root / 'features').mkdir(parents=True)
                np.save(root / 'features/final.npy', np.ones((2, 4), dtype=np.float32))
                (root / 'samples.json').write_text(json.dumps({'sample_ids': ['a', 'b'], 'targets': [0, 1]}))
                (root / 'metrics.json').write_text('{}')
                files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in root.rglob('*') if p.is_file()}
                (root / 'manifest.json').write_text(json.dumps({
                    'schema': 2, 'status': 'complete', 'n_samples': 2, 'condition': {'name': name},
                    'feature_dtype': 'float32', 'layers': {'features/final.npy': [2, 4]},
                    'files': files, 'metadata': {
                        'checkpoint_sha256': checkpoint, 'data_sha256': 'b' * 64,
                        'code_sha256': 'c' * 64, 'cfg_sha256': 'd' * 64, 'hqq_source_sha256': 'e' * 64,
                        'preprocessing': {'image_size': 224, 'test_resize_size': 300,
                            'test_resize_directly': False, 'test_square_resize_center_crop': True,
                            'custom_mean_std': False, 'mean': None, 'std': None,
                            'pre_resize_factor': None, 'resize_size': 300, 'dataset_in_memory': False,
                            'cal_ap_only': False, 'aug_type': 'crop_mask'}},
                    'execution': {'seed': 100, 'fp16': True, 'hqq_compute_dtype': 'float16',
                        'batch_size': 2, 'cal_ap_only': False, 'dataset_in_memory': False,
                        'reduce_pattern': None, 'reduce_op': 'mean'}}))
                roots.append(root)
            with self.assertRaisesRegex(ValueError, 'source'):
                read_pair(*roots)

    def test_pair_rejects_shared_missing_or_invalid_provenance(self):
        import copy
        with tempfile.TemporaryDirectory() as folder:
            roots = []
            manifests = []
            for name in ('fp', 'a4_linear'):
                root = Path(folder) / name
                (root / 'features').mkdir(parents=True)
                np.save(root / 'features/final.npy', np.ones((2, 4), dtype=np.float32))
                (root / 'samples.json').write_text(json.dumps({'sample_ids': ['a', 'b'], 'targets': [0, 1]}))
                (root / 'metrics.json').write_text('{}')
                files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in root.rglob('*') if p.is_file()}
                manifest = {'schema': 2, 'status': 'complete', 'n_samples': 2, 'condition': {'name': name},
                    'feature_dtype': 'float32', 'layers': {'features/final.npy': [2, 4]}, 'files': files,
                    'metadata': {'checkpoint_sha256': 'a' * 64, 'data_sha256': 'b' * 64,
                        'code_sha256': 'c' * 64, 'cfg_sha256': 'd' * 64, 'hqq_source_sha256': 'e' * 64,
                        'preprocessing': {'image_size': 224, 'test_resize_size': 300,
                            'test_resize_directly': False, 'test_square_resize_center_crop': True,
                            'custom_mean_std': False, 'mean': None, 'std': None,
                            'pre_resize_factor': None, 'resize_size': 300, 'dataset_in_memory': False,
                            'cal_ap_only': False, 'aug_type': 'crop_mask'}},
                    'execution': {'seed': 100, 'fp16': True, 'hqq_compute_dtype': 'float16', 'batch_size': 2,
                        'cal_ap_only': False, 'dataset_in_memory': False, 'reduce_pattern': None, 'reduce_op': 'mean'}}
                roots.append(root)
                manifests.append(manifest)
            cases = [('metadata', key, None) for key in
                     ('checkpoint_sha256', 'data_sha256', 'code_sha256', 'cfg_sha256', 'hqq_source_sha256')]
            cases += [('metadata', 'checkpoint_sha256', value) for value in ('', 'not-a-sha256', 'z' * 64, 123)]
            cases += [('execution', key, None) for key in
                      ('seed', 'fp16', 'hqq_compute_dtype', 'batch_size', 'cal_ap_only',
                       'dataset_in_memory', 'reduce_pattern', 'reduce_op')]
            cases += [('execution', 'batch_size', 0), ('execution', 'seed', True),
                      ('metadata', 'preprocessing', {}), ('execution', None, {}), ('metadata', None, {})]
            for section, key, value in cases:
                with self.subTest(section=section, key=key, value=value):
                    for root, original in zip(roots, manifests):
                        modified = copy.deepcopy(original)
                        if key is None:
                            modified[section] = value
                        elif value is None:
                            modified[section].pop(key)
                        else:
                            modified[section][key] = value
                        (root / 'manifest.json').write_text(json.dumps(modified))
                    with self.assertRaisesRegex(ValueError, 'provenance'):
                        read_pair(*roots)

    def test_pair_accepts_complete_matching_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            roots = []
            for name in ('fp', 'a4_linear'):
                root = Path(folder) / name
                (root / 'features').mkdir(parents=True)
                np.save(root / 'features/final.npy', np.ones((2, 4), dtype=np.float32))
                (root / 'samples.json').write_text(json.dumps({'sample_ids': ['a', 'b'], 'targets': [0, 1]}))
                (root / 'metrics.json').write_text('{}')
                files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in root.rglob('*') if p.is_file()}
                (root / 'manifest.json').write_text(json.dumps({
                    'schema': 2, 'status': 'complete', 'n_samples': 2, 'condition': {'name': name},
                    'feature_dtype': 'float32', 'layers': {'features/final.npy': [2, 4]}, 'files': files,
                    'metadata': {'checkpoint_sha256': 'a' * 64, 'data_sha256': 'b' * 64,
                        'code_sha256': 'c' * 64, 'cfg_sha256': 'd' * 64, 'hqq_source_sha256': 'e' * 64,
                        'preprocessing': {'image_size': 224, 'test_resize_size': 300,
                            'test_resize_directly': False, 'test_square_resize_center_crop': True,
                            'custom_mean_std': False, 'mean': None, 'std': None,
                            'pre_resize_factor': None, 'resize_size': 300, 'dataset_in_memory': False,
                            'cal_ap_only': False, 'aug_type': 'crop_mask'}},
                    'execution': {'seed': 100, 'fp16': True, 'hqq_compute_dtype': 'float16', 'batch_size': 2,
                        'cal_ap_only': False, 'dataset_in_memory': False, 'reduce_pattern': None, 'reduce_op': 'mean'}}))
                roots.append(root)
            fp, quant = read_pair(*roots)
            try:
                np.testing.assert_array_equal(fp['features']['final'], quant['features']['final'])
                self.assertEqual(fp['sample_ids'], ['a', 'b'])
            finally:
                for condition in (fp, quant):
                    for array in condition['features'].values():
                        array._mmap.close()


if __name__ == '__main__':
    unittest.main()
