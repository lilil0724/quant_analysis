"""Descriptive W+A minus verified historical weight-only comparisons."""
import json

import numpy as np
import pandas as pd

from .quant import UNKNOWN, encoded, prepare_quant_pairs

WEIGHT_FIELDS = ('hqq_nbits', 'hqq_group_size', 'hqq_compute_dtype',
                 'hqq_quant_zero', 'hqq_quant_scale', 'fp16')


def comparison_key(row):
    identity = json.loads(row['identity'])
    identity.pop('condition')
    identity.pop('checkpoint_sha256', None)
    settings = json.loads(row['condition_json'])
    return encoded(dict(evaluation=identity, baseline_run_id=row['baseline_run_id'],
                        weight_settings={field: settings[field] for field in WEIGHT_FIELDS}))


def prepare_weight_delta(records, frame):
    """Use the verified recover_baselines cohort's legacy W-only source contract.

    That source explicitly selects weight-only linear runs with excluded heads;
    missing legacy activation/scope fields are accepted ONLY under this contract.
    Scope/head differences to 20008 are retained and are not causal A-only effects.
    """
    weights = {(float(row.hqq_nbits), int(row.hqq_group_size))
               for row in frame.loc[frame.accepted].itertuples()}
    candidates = [r for r in records if r['config'].get('hqq') is True
                  and r['config'].get('hqq_act_bits') is None
                  and r['config'].get('hqq_scope') in (None, 'linear')
                  and r['config'].get('hqq_exclude') == 'head'
                  and (float(r['config']['hqq_nbits']), int(r['config']['hqq_group_size'])) in weights]
    if not candidates:
        raise ValueError('Verified legacy weight-only cohort has no matching W/group settings')
    weight_frame = prepare_quant_pairs(records, candidates)
    by_key = {}
    for row in weight_frame.loc[weight_frame.accepted].to_dict('records'):
        by_key.setdefault(comparison_key(row), []).append(row)
    rows = []
    for row in frame.to_dict('records'):
        matches = by_key.get(comparison_key(row), []) if row['accepted'] else []
        reason = '' if len(matches) == 1 else 'ambiguous_weight_only' if len(matches) > 1 else 'missing_compatible_weight_only'
        if not row['accepted']:
            reason = 'invalid_wa_pair:' + row['mismatch_fields']
        weight = matches[0] if len(matches) == 1 else None
        if weight:
            left_hash = json.loads(row['identity']).get('checkpoint_sha256')
            right_hash = json.loads(weight['identity']).get('checkpoint_sha256')
            if left_hash and right_hash and left_hash != right_hash:
                reason = 'checkpoint_sha256_mismatch'
        accepted = not reason
        w_top1 = weight['quant_top1_pct'] if weight else np.nan
        wa_settings = json.loads(row['condition_json'])
        w_settings = json.loads(weight['condition_json']) if weight else {}
        different = {field: {'wa': wa_settings.get(field, UNKNOWN), 'weight_only': w_settings.get(field, UNKNOWN)}
                     for field in sorted(set(wa_settings) | set(w_settings))
                     if wa_settings.get(field, UNKNOWN) != w_settings.get(field, UNKNOWN)}
        delta_identity = json.loads(row['identity'])
        delta_identity['weight_only_run_id'] = weight['quant_run_id'] if weight else ''
        rows.append(dict(**row, weight_only_run_id=weight['quant_run_id'] if weight else '',
                         weight_only_run_url=weight['quant_run_url'] if weight else '',
                         weight_only_serial=weight['serial'] if weight else None,
                         weight_only_top1_pct=w_top1,
                         weight_only_condition_json=weight['condition_json'] if weight else '',
                         weight_delta_accepted=accepted, weight_delta_reason=reason,
                         weight_only_candidate_ids=encoded([w['quant_run_id'] for w in matches]),
                         quant_setting_differences=encoded(different), delta_identity=encoded(delta_identity),
                         delta_top1_pp=row['quant_top1_pct']-w_top1 if accepted else np.nan,
                         delta_accuracy_ratio=(row['quant_top1_pct']-w_top1)/row['baseline_top1_pct'] if accepted else np.nan))
    result = pd.DataFrame(rows)
    raw = {r['id']: r for r in records}
    # The original pair table reports each side's baseline batch match; retain
    # the actual W-only batch size here for comparison provenance.
    result['weight_only_batch_size'] = result.weight_only_run_id.map(
        lambda run_id: raw[run_id]['config'].get('batch_size') if run_id in raw else None)
    return result, weight_frame


def verify_weight_delta(frame):
    accepted = frame.loc[frame.weight_delta_accepted]
    np.testing.assert_allclose(accepted.delta_top1_pp,
                               accepted.quant_top1_pct-accepted.weight_only_top1_pct, rtol=0, atol=1e-12)
    np.testing.assert_allclose(accepted.delta_accuracy_ratio,
                               accepted.delta_top1_pp/accepted.baseline_top1_pct, rtol=0, atol=1e-12)
    if frame.loc[~frame.weight_delta_accepted, ['delta_top1_pp','delta_accuracy_ratio']].notna().any().any():
        raise ValueError('Unpaired weight comparison contains a numeric delta')
    return dict(target_runs=len(frame), matched_weight_deltas=len(accepted),
                weight_only_runs=int(accepted.weight_only_run_id.nunique()),
                delta_formula_verified=True, delta_units='percentage points',
                interpretation='descriptive cross-setting difference; scope/head settings differ')


def add_weight_quotient(frame):
    """Add per-pair acc_ratio(W+A)/acc_ratio(W-only), preserving old schemas."""
    result = frame.copy()
    result['weight_only_acc_ratio'] = result.weight_only_top1_pct / result.baseline_top1_pct
    valid = (result.weight_delta_accepted & result.weight_only_acc_ratio.gt(0)
             & np.isfinite(result.weight_only_acc_ratio))
    quotient = result.accuracy_ratio / result.weight_only_acc_ratio.where(valid)
    valid &= np.isfinite(quotient)
    result['weight_ratio_accepted'] = valid
    result['weight_ratio_reason'] = np.where(valid, '', np.where(
        ~result.weight_delta_accepted, result.weight_delta_reason, 'invalid_or_zero_weight_only_denominator'))
    result['wa_over_weight_acc_ratio'] = quotient.where(valid)
    return result


def verify_weight_quotient(frame):
    accepted = frame.loc[frame.weight_ratio_accepted]
    np.testing.assert_allclose(accepted.wa_over_weight_acc_ratio,
                               accepted.accuracy_ratio/accepted.weight_only_acc_ratio, rtol=0, atol=1e-12)
    np.testing.assert_allclose(accepted.wa_over_weight_acc_ratio,
                               accepted.quant_top1_pct/accepted.weight_only_top1_pct, rtol=0, atol=1e-12)
    if (frame.loc[~frame.weight_ratio_accepted, 'wa_over_weight_acc_ratio'].notna().any()
            or not np.isfinite(accepted.wa_over_weight_acc_ratio).all()
            or not accepted.weight_only_top1_pct.gt(0).all()):
        raise ValueError('Invalid W+A / weight-only quotient')
    return dict(defined_quotients=len(accepted), undefined_quotients=len(frame)-len(accepted),
                quotient_formula_verified=True, units='dimensionless', neutral_value=1)
