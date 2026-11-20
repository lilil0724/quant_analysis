"""General quantization conditions, compatible AMP pairs and real-data audits."""
import collections
import hashlib
import itertools
import json
import math

import numpy as np
import pandas as pd

from legacy_hqq.recover_baselines import (
    DATASETS, MODELS, MODES, EVAL_FIELDS, canonical, context, differences,
)

CONTEXT = ['dataset', 'model', 'mode']
UNKNOWN = 'unrecorded'
CONDITION_FIELDS = ('hqq_nbits', 'hqq_act_bits', 'hqq_group_size', 'hqq_act_group_size',
                    'hqq_scope', 'hqq_exclude', 'hqq_compute_dtype', 'hqq_quant_zero',
                    'hqq_quant_scale', 'fp16', 'quant_cfg_path')
# Operational paths/logging do not define numerical quantization conditions.
OPERATIONAL = {'hqq_verbose', 'hqq_report_path', 'hqq_compare_fp32'}
FILTER_FIELDS = {'weight_bits': 'hqq_nbits', 'activation_bits': 'hqq_act_bits',
                 'weight_group_sizes': 'hqq_group_size',
                 'activation_group_sizes': 'hqq_act_group_size', 'scopes': 'hqq_scope'}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def condition(config):
    settings = {name: config.get(name, UNKNOWN) for name in CONDITION_FIELDS}
    settings.update({name: value for name, value in config.items()
                     if name.startswith('hqq_') and name not in OPERATIONAL})
    # Normalize numeric identity without rounding fractional bit widths.
    for name in ('hqq_nbits', 'hqq_act_bits', 'hqq_group_size', 'hqq_act_group_size'):
        value = settings[name]
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            settings[name] = int(value) if float(value).is_integer() else float(value)
        elif isinstance(value, float) and not math.isfinite(value):
            settings[name] = 'invalid_nonfinite'
    identity = encoded(settings)
    return hashlib.sha256(identity.encode()).hexdigest()[:16], settings


def label(settings):
    def show(value, group=False):
        return ('full' if group else 'off') if value is None else str(value)
    return (f"W{show(settings['hqq_nbits'])} / A{show(settings['hqq_act_bits'])}; "
            f"WG{show(settings['hqq_group_size'], True)} / AG{show(settings['hqq_act_group_size'], True)}; "
            f"{settings['hqq_scope']}; exclude={settings['hqq_exclude'] or '(none)'}")


def select_targets(targets, filters=None):
    filters = filters or {}
    selected = []
    for run in targets:
        config = run['config']
        _, settings = condition(config)
        if filters.get('serials') and str(config.get('serial')) not in {str(s) for s in filters['serials']}:
            continue
        if all(not filters.get(option) or settings[field] in filters[option]
               for option, field in FILTER_FIELDS.items()):
            selected.append(run)
    if not selected:
        raise ValueError('No target runs match the requested filters')
    return selected


def metric(value, baseline=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (0 < value <= 100 if baseline else 0 <= value <= 100))


def quant_issues(config):
    issues = []
    if config.get('hqq') is not True:
        issues.append('not_hqq')
    if config.get('fp16') is not True:
        issues.append('not_AMP_fp16')
    if config.get('debugging', False) is not False:
        issues.append('debugging')
    for name in ('hqq_nbits', 'hqq_act_bits'):
        value = config.get(name, UNKNOWN)
        if name == 'hqq_act_bits' and (value is None or value == UNKNOWN):
            continue
        if not (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value > 0):
            issues.append(f'invalid_{name}')
    for name in ('hqq_group_size', 'hqq_act_group_size'):
        value = config.get(name, UNKNOWN)
        if name == 'hqq_act_group_size' and (value is None or config.get('hqq_act_bits') is None):
            continue
        if not (isinstance(value, int) and not isinstance(value, bool) and value > 0):
            issues.append(f'invalid_{name}')
    return issues


def prepare_quant_pairs(records, targets):
    """Retain each target once; ambiguous/invalid pairs receive N/A and reasons."""
    if len({r['id'] for r in targets}) != len(targets):
        raise ValueError('Duplicate target run IDs')
    baselines = collections.defaultdict(list)
    for record in records:
        if record['config'].get('hqq') is False:
            baselines[context(record)].append(record)
    rows = []
    for record in targets:
        config = record['config']
        key = context(record)
        condition_id, settings = condition(config)
        issues = quant_issues(config)
        if record['state'] != 'finished':
            issues.append('target_not_finished')
        quant = record['summary'].get('hqq_top1')
        if not metric(quant):
            issues.append('invalid_quant_top1')
        candidates = []
        rejected = {}
        for baseline in baselines[key]:
            b = baseline['config']
            mismatch, unrecorded = differences(config, b)
            if (baseline['state'] != 'finished' or b.get('fp16') is not True
                    or b.get('debugging', False) is not False
                    or not metric(baseline['summary'].get('top1'), baseline=True)):
                mismatch.append('invalid_baseline')
            rejected[baseline['id']] = mismatch
            if not mismatch:
                candidates.append((baseline, unrecorded))
        baseline = None
        shared = []
        if len(candidates) == 1:
            baseline, shared = candidates[0]
        elif len(candidates) > 1:
            issues.append('ambiguous_baseline')
        else:
            issues.append('incompatible_baseline' if baselines[key] else 'missing_baseline')
        b = baseline['config'] if baseline else {}
        fp = baseline['summary']['top1'] if baseline else np.nan
        # Quant duplicates ignore serial/run ID but retain full evaluation identity.
        identity = dict(context=key, condition=condition_id,
                        evaluation={f: config.get(f, UNKNOWN) for f in EVAL_FIELDS},
                        ckpt_path=canonical(config.get('ckpt_path')),
                        dataset_root_path=canonical(config.get('dataset_root_path')),
                        checkpoint_sha256=config.get('checkpoint_sha256'))
        rows.append(dict(dataset=key[0], model=key[1], mode=key[2],
                         condition_id=condition_id, condition_label=label(settings),
                         condition_json=encoded(settings),
                         **settings, serial=config.get('serial'), seed=config.get('seed'),
                         quant_run_id=record['id'], quant_run_url=record.get('url', ''),
                         state=record['state'], quant_top1_pct=quant if metric(quant) else np.nan,
                         baseline_run_id=baseline['id'] if baseline else '',
                         baseline_run_url=baseline.get('url', '') if baseline else '',
                         baseline_serial=b.get('serial'), baseline_top1_pct=fp,
                         quant_classifier=config.get('classifier'), baseline_classifier=b.get('classifier'),
                         ckpt_path=canonical(config.get('ckpt_path')),
                         baseline_precision='unquantized_AMP',
                         checkpoint_bytes_verified=bool(config.get('checkpoint_sha256')
                             and config.get('checkpoint_sha256') == b.get('checkpoint_sha256')),
                         batch_size_matches=bool(baseline and config.get('batch_size') == b.get('batch_size')),
                         shared_unrecorded_fields=';'.join(shared),
                         unrecorded_quant_fields=';'.join(k for k,v in settings.items() if v == UNKNOWN),
                         baseline_candidate_audit=encoded(rejected),
                         identity=encoded(identity), issues=issues))
    frame = pd.DataFrame(rows)
    duplicates = frame.identity.duplicated(keep=False)
    for index in frame.index[duplicates]:
        frame.at[index, 'issues'] = frame.at[index, 'issues'] + ['duplicate_quant_identity']
    frame['accepted'] = frame.issues.map(lambda values: not values)
    frame['mismatch_fields'] = frame.issues.map(';'.join)
    frame['accuracy_ratio'] = (frame.quant_top1_pct / frame.baseline_top1_pct).where(frame.accepted)
    frame['accuracy_drop_pp'] = (frame.baseline_top1_pct - frame.quant_top1_pct).where(frame.accepted)
    frame = frame.drop(columns=['issues']).sort_values(['condition_id', *CONTEXT, 'seed', 'quant_run_id'])
    return frame.reset_index(drop=True)


def coverage(frame, expected_contexts=None):
    expected = set(map(tuple, expected_contexts or itertools.product(DATASETS, MODELS, MODES)))
    rows = []
    missing = []
    for condition_id, group in frame.groupby('condition_id', sort=True):
        observed = set(map(tuple, group[CONTEXT].itertuples(index=False, name=None)))
        accepted = set(map(tuple, group.loc[group.accepted, CONTEXT].itertuples(index=False, name=None)))
        for key in sorted(expected - accepted):
            missing.append(dict(condition_id=condition_id, dataset=key[0], model=key[1], mode=key[2],
                                reason='missing_target' if key not in observed else 'no_accepted_pair'))
        rows.append(dict(condition_id=condition_id, condition_label=group.condition_label.iloc[0],
                         expected_contexts=len(expected), observed_contexts=len(observed & expected),
                         accepted_contexts=len(accepted & expected), target_runs=len(group),
                         accepted_runs=int(group.accepted.sum()), rejected_runs=int((~group.accepted).sum()),
                         unexpected_contexts=len(observed - expected), missing_contexts=len(expected - accepted),
                         complete=expected == accepted and not (~group.accepted).any()))
    return pd.DataFrame(rows), pd.DataFrame(missing, columns=['condition_id', *CONTEXT, 'reason'])


def verify_quant_pairs(frame):
    if frame.quant_run_id.duplicated().any():
        raise ValueError('Target appears more than once')
    accepted = frame.loc[frame.accepted]
    if accepted.identity.duplicated().any():
        raise ValueError('Accepted quantization identities are ambiguous')
    if not accepted.baseline_run_id.ne('').all():
        raise ValueError('Accepted pair has no baseline')
    for field, actual in [('accuracy_ratio', accepted.quant_top1_pct / accepted.baseline_top1_pct),
                          ('accuracy_drop_pp', accepted.baseline_top1_pct - accepted.quant_top1_pct)]:
        np.testing.assert_allclose(accepted[field], actual, rtol=0, atol=1e-12)
        if frame.loc[~frame.accepted, field].notna().any():
            raise ValueError('Rejected pair has a numeric derived metric')
    if not (np.isfinite(accepted.accuracy_ratio).all()
            and accepted.quant_top1_pct.between(0, 100).all()
            and accepted.baseline_top1_pct.gt(0).all()):
        raise ValueError('Invalid accepted metrics')
    return dict(target_runs=len(frame), usable_ratios=len(accepted),
                unpaired=len(frame)-len(accepted), conditions=int(frame.condition_id.nunique()),
                formula_verified=True, tolerance=1e-12)
