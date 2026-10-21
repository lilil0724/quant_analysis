import unittest

from legacy_hqq.recover_baselines import select_context, canonical, differences


class BaselineRecoveryTests(unittest.TestCase):
    def test_corrected_readout_replaces_both_baseline_and_quantized_runs(self):
        old = dict(id='old', state='finished', config=dict(classifier='pool', hqq=False), summary=dict(top1=50))
        new = dict(id='new', state='finished', config=dict(classifier='cls', hqq=False), summary=dict(top1=80))
        quant = dict(id='quant', state='finished', config=dict(classifier='cls', hqq=True), summary=dict(hqq_top1=70))
        selected = select_context([old], [new, quant], affected=True)
        self.assertEqual([r['id'] for r in selected], ['new', 'quant'])

    def test_incomplete_rerun_does_not_fall_back_to_historical_cal(self):
        old = dict(id='old_cal', state='finished', config=dict(classifier='pool', hqq=False), summary=dict(top1=90))
        self.assertEqual(select_context([old], [], affected=True), [])

    def test_archived_unaffected_run_is_preserved_when_removed_from_wandb(self):
        old = dict(id='saved', state='finished', config=dict(classifier='cls', hqq=False), summary=dict(top1=80))
        self.assertEqual(select_context([old], [], affected=False), [old])

    def test_current_record_updates_same_run_id_without_duplication(self):
        old = dict(id='same', state='running', config={}, summary={})
        new = dict(id='same', state='finished', config=dict(hqq=False), summary=dict(top1=80))
        self.assertEqual(select_context([old], [new], affected=False), [new])

    def test_running_and_nonfinite_runs_are_excluded(self):
        running = dict(id='running', state='running', config=dict(hqq=True), summary=dict(hqq_top1=70))
        bad = dict(id='bad', state='finished', config=dict(hqq=True), summary=dict(hqq_top1=float('nan')))
        self.assertEqual(select_context([], [running, bad], affected=False), [])

    def test_data_root_alias_keeps_full_checkpoint_path(self):
        self.assertEqual(canonical('../../data/ckpt/ft/pets.pth'), '/work/kyle0724/data/ckpt/ft/pets.pth')
        self.assertNotEqual(canonical('/other/ckpt/ft/pets.pth'), canonical('../../data/ckpt/ft/pets.pth'))

    def test_classifier_mismatch_is_rejected_even_with_same_checkpoint(self):
        quant = dict(dataset_name='pets', seed=100, fp16=True, image_size=224, classifier='cls',
                     test_resize_size=300, test_multiple=0, test_square_resize_center_crop=True,
                     num_images_test=100, ckpt_path='/work/kyle0724/data/ckpt/ft/pets.pth',
                     dataset_root_path='/work/kyle0724/data/dataset/pets')
        baseline = dict(quant, classifier='pool')
        mismatch, _ = differences(quant, baseline)
        self.assertEqual(mismatch, ['classifier'])

    def test_checkpoint_hash_conflict_is_rejected_and_missing_fields_remain_unknown(self):
        quant = dict(dataset_name='pets', seed=100, fp16=True, image_size=224, classifier='cls',
                     test_resize_size=300, test_multiple=0, test_square_resize_center_crop=True,
                     num_images_test=100, ckpt_path='/work/kyle0724/data/ckpt/ft/pets.pth',
                     dataset_root_path='/work/kyle0724/data/dataset/pets', checkpoint_sha256='a')
        baseline = dict(quant, checkpoint_sha256='b')
        mismatch, unrecorded = differences(quant, baseline)
        self.assertEqual(mismatch, ['checkpoint_sha256'])
        self.assertIn('cal_single_crop', unrecorded)
