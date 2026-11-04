"""Immutable runs and read-only verification for schema-2 quantization inputs."""
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import pandas as pd

from .acquire import BASE, read_manifest, sha256, validate_id
from .plot import plot_conditions, plot_weight_activation, plot_dataset_activation, plot_weight_delta
from .quant import coverage, prepare_quant_pairs, select_targets, verify_quant_pairs


def load_inputs(input_id):
    directory = BASE / 'data' / validate_id(input_id)
    manifest = read_manifest(directory)
    if manifest.get('schema_version') != 2:
        raise ValueError('Expected a schema-2 input')
    cohort = json.loads((directory / 'baseline_cohort_receipt.json').read_text(encoding='utf-8-sig'))
    original = json.loads((directory / 'baseline_input_manifest.json').read_text(encoding='utf-8-sig'))
    baseline = directory / 'baseline_selected_runs.jsonl'
    if (cohort['status'] != 'complete' or original['status'] != 'complete'
            or sha256(baseline) != cohort['artifacts']['selected_runs.jsonl']
            or sha256(baseline) != original['files']['selected_runs.jsonl']['sha256']
            or sha256(directory / 'baseline_input_manifest.json') != manifest['baseline_manifest_sha256']):
        raise ValueError('Copied baseline disagrees with its historical receipts')
    def jsonl(path):
        return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s]
    records, targets = jsonl(baseline), jsonl(directory / 'target_runs.jsonl')
    if (len(targets) != manifest['total_runs']
            or len({r['id'] for r in targets}) != len(targets)
            or [r['id'] for r in targets] != manifest['run_ids']
            or any(int(r['config']['serial']) not in manifest['serials'] for r in targets)):
        raise ValueError('Target records disagree with their acquisition receipt')
    return records, targets, manifest


def write_report(receipt, report, frame=None, counts=None):
    run_id, input_id = receipt['run_id'], receipt['input_id']
    lines = ['# Weight / activation quantization accuracy analysis', '',
             f"- Run: `{run_id}`; status: `{receipt['status']}`.",
             f"- Started: {receipt['started_at']}; finished: {receipt.get('finished_at', 'pending')} (Asia/Taipei).",
             f"- Duration: {receipt.get('duration_seconds', 0):.3f} seconds.",
             f"- Input: `../../data/{input_id}/`; manifest SHA-256: `{receipt.get('input_manifest_sha256', 'unavailable')}`.",
             f"- Full output and machine receipt: `../../result/{run_id}/run.json`.", '',
             '## Source and method', '',
             f"W&B project: `{receipt.get('project', 'unavailable')}`. Analysis reads a saved config/summary snapshot; no GPU inference.",
             f"Filters: `{json.dumps(receipt.get('filters', {}), sort_keys=True)}`.",
             'Each condition retains weight/activation bits, group sizes, scope, exclusions, compute precision, quant flags and other recorded numerical HQQ settings.',
             'Only explicit activation null means off; absent settings remain unrecorded. Serial is provenance, not a quantization condition.',
             'Baseline: matched unquantized AMP fp16, using the verified copied local cohort. Match dataset, model, mode, full checkpoint path, seed, classifier, precision and recorded evaluation settings.',
             'accuracy_ratio = quant_top1_pct / baseline_top1_pct; accuracy_drop_pp = baseline_top1_pct - quant_top1_pct. Ratios are not clipped.',
             'Ambiguous, unfinished, invalid, duplicate or incompatible pairs remain N/A with reasons in tables/acc_ratio_pairs.csv. Missing expected contexts are in audit/missing_contexts.csv.', '',
             '| Input file | SHA-256 |', '|---|---|']
    for name, entry in receipt.get('input_files', {}).items():
        lines.append(f"| `{name}` | `{entry['sha256']}` |")
    if frame is not None:
        lines += ['', '## Coverage and results', '',
                  f"Accepted {int(frame.accepted.sum())}/{len(frame)} target runs across {frame.condition_id.nunique()} conditions.",
                  f"Coverage: {receipt['coverage_status']}. Complete conditions: {int(counts.complete.sum())}/{len(counts)}.",
                  'Per-condition completeness uses the expected dataset/model/mode matrix declared by the input, not the number of downloaded rows. Multiple seeds do not increase context counts.', '',
                  '| Condition | Settings | Accepted runs | Accepted / expected contexts | Missing |',
                  '|---|---|---:|---:|---:|']
        for row in counts.itertuples():
            lines.append(f'| `{row.condition_id}` | {row.condition_label} | {row.accepted_runs}/{row.target_runs} | '
                         f'{row.accepted_contexts}/{row.expected_contexts} | {row.missing_contexts} |')
        lines += ['', 'Accepted ratio summaries (descriptive only; conditions are kept separate):', '',
                  '| Condition | Min | Median | Max |', '|---|---:|---:|---:|']
        for cid, group in frame.loc[frame.accepted].groupby('condition_id', sort=True):
            values = group.accuracy_ratio
            lines.append(f'| `{cid}` | {values.min():.6f} | {values.median():.6f} | {values.max():.6f} |')
        issues = frame.loc[~frame.accepted, 'mismatch_fields'].value_counts()
        if len(issues):
            lines += ['', 'Rejected run reasons:', '']
            lines += [f'- `{reason}`: {count}.' for reason, count in issues.items()]
        lines += ['', f"Accepted pairs with different batch sizes: {int((~frame.loc[frame.accepted, 'batch_size_matches']).sum())}.",
                  'Heatmaps show context means across accepted seeds/evaluations. W/A comparisons hold every recorded non-bit condition setting fixed and use the same full evaluation identities and baseline IDs in every included condition.',
                  'W/A panels average per context first, then weight datasets equally. The exact shared population and excluded population sizes are recorded in audit/comparison_populations.csv.', '']
        if any(ref['kind'] == 'dataset_activation' for ref in receipt.get('figures', [])):
            lines += ['Dataset-by-activation plots are separate for each fixed weight setting. Each cell is the mean of matched model/mode ratios after seed/evaluation averaging, using identical full evaluation identities across activation settings.',
                      'Row labels show available / expected model-mode contexts, not the reference image\'s historical counts. FGIR/UFGIR display labels and ordering follow the supplied reference image (including food/inat17/moe in FGIR and cotton in UFGIR); this does not change the granularity module\'s research grouping.',
                      'These plots share the fixed 0–1.6 color scale centered at 1. Full values remain in CSV and annotations; color saturation does not clip numerical ratios. Their populations are in audit/dataset_activation_populations.csv.', '']
        if receipt.get('plot_mode') == 'weight_delta':
            delta = receipt['weight_delta_verification']
            metric = receipt.get('delta_metric', 'top1_pp')
            limit = receipt['figures'][0].get('color_limit', 100)
            quotient = metric == 'acc_ratio_quotient'
            lines += ['', '## W+A / weight-only acc_ratio' if quotient else '## W+A minus weight-only delta', '',
                      'delta_top1_pp = W+A Top-1 (%) - weight-only Top-1 (%). Negative means the W+A evaluation scored lower. delta_accuracy_ratio is also saved as delta_top1_pp / matched AMP Top-1.',
                      'delta_accuracy_ratio = (W+A Top-1 / matched AMP Top-1) - (weight-only Top-1 / the same matched AMP Top-1). Compute each paired ratio difference before model/mode averaging; this is not a ratio of dataset means.',
                      f"Plotted metric: `{metric}`; units: {'dimensionless quotient' if quotient else 'dimensionless acc_ratio difference' if metric == 'acc_ratio' else 'Top-1 percentage points'}.",
                      ('Plotted quotient = acc_ratio(W+A) / acc_ratio(W-only) = W+A Top-1 / W-only Top-1, because each pair uses the same AMP baseline. First compute each per-evaluation quotient, then context means; this is not a quotient of dataset means. Neutral value is 1; below 1 means W+A scored lower. Zero or invalid denominators are N/A.'
                       if quotient else 'The selected delta is computed before averaging paired contexts.'),
                      f"Matched comparisons: {delta['matched_weight_deltas']}/{delta['target_runs']}; distinct weight-only runs: {delta['weight_only_runs']}.",
                      'IMPORTANT: W+A serial 20008 is extended scope with all heads included. The verified historical weight-only cohort is legacy linear scope with head excluded. This is a descriptive difference between those settings, NOT an isolated activation-quantization effect.',
                      'Only W bits/group size, compute dtype, quant-zero/scale, precision and full evaluation identity plus the same AMP baseline are matched. Other quantization differences are retained in quant_setting_differences in tables/weight_delta_pairs.csv.',
                      'The weight-only source is the preserved recover_baselines selected cohort. Its documented source contract allows historical missing activation/scope fields as legacy weight-only linear; this rule is not applied to arbitrary new exports.',
                      ('Each cell first computes per-evaluation quotients' if quotient else 'Each cell first computes per-evaluation differences') + ', then averages seeds/evaluations within model/mode, then weights model/mode contexts equally. Across A, use the same evaluation identities AND the same weight-only run IDs. Missing/ambiguous comparisons remain N/A.',
                      (f'All three plots share the scale 0 to {limit:g}, centered at 1.' if quotient else
                       f'All three plots share the zero-centered scale {-limit:g} to {limit:g}.') + ' Limits expand if needed to contain all plotted values; numerical values are not clipped. FGIR/UFGIR display order follows the supplied reference image and does not change canonical research groups.',
                      'Raw paired values, IDs, source serials, settings and reasons are in tables/weight_delta_pairs.csv; comparator records are in tables/weight_only_pairs.csv; exact shared populations are in audit/weight_delta_populations.csv.', '']
        for ref in receipt.get('figures', []):
            lines += [f"![{ref['caption']}](figures/{ref['stem']}.png)", '',
                      f"[Editable SVG](figures/{ref['stem']}.svg)", '']
    if receipt.get('error'):
        lines += ['', '## Failure', '', receipt['error']]
    code = receipt.get('code', {})
    command = f'conda run -n opencode_env python -m activation_quant.run --input-id {input_id}'
    if receipt.get('plot_mode') == 'weight_delta':
        command += ' --weight-delta --delta-metric ' + receipt.get('delta_metric', 'top1_pp')
    option_names = {'weight_bits':'--weight-bits', 'activation_bits':'--activation-bits',
                    'weight_group_sizes':'--weight-group-sizes', 'activation_group_sizes':'--activation-group-sizes',
                    'scopes':'--scopes', 'serials':'--serials'}
    for option, values in receipt.get('filters', {}).items():
        null_label = 'full' if option == 'activation_group_sizes' else 'off'
        command += ' ' + option_names[option] + ' ' + ' '.join(null_label if v is None else str(v) for v in values)
    lines += ['', '## Limitations and reproduction', '',
              'This is a descriptive comparison of saved evaluation summaries. Scope/head/group changes must not be interpreted as isolated bit-width effects.',
              'Config/checkpoint-path equality does not prove identical checkpoint bytes. Matching recorded checkpoint SHA-256 values are reported separately; otherwise weights remain unverified.',
              'Shared unrecorded evaluation fields, unknown quant settings and batch-size differences are retained in the pair audit. Remote hardware, model implementation and inference correctness were not independently verified.',
              'Observed conditions are analyzed; an unobserved entire condition is not claimed to be covered. Missing cells within the declared expected context matrix remain N/A.',
              f"Git: `{code.get('commit', 'unavailable')}`; dirty: `{code.get('dirty', 'unknown')}`. Code/environment/source hashes are in run.json and provenance/source/.", '',
              '```powershell', command + ' --run-id NEW_RUN_ID',
              f'conda run -n opencode_env python -m activation_quant.run --verify-run {run_id}', '```', '',
              'Every rerun uses a new ID. Original inputs/runs are immutable. Report PNG/SVG copies match the full output by SHA-256.', '']
    (report / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def execute(input_id, run_id=None, filters=None, weight_delta=False, delta_metric='top1_pp'):
    from .run import TZ, reserve_run, source_identity, write_receipt

    filters = {key: values for key, values in (filters or {}).items() if values}
    run_id = run_id or datetime.now(TZ).strftime('%Y%m%d_%H%M%S_%f_quant_accuracy')
    output, report = reserve_run(run_id)
    started = time.perf_counter()
    receipt = dict(schema_version=2, run_id=run_id, analysis='quant_accuracy', input_id=input_id,
                   status='running', started_at=datetime.now(TZ).isoformat(), timezone='Asia/Taipei',
                   command=[sys.executable, '-m', 'activation_quant.run', *sys.argv[1:]], filters=filters,
                   plot_mode='weight_delta' if weight_delta else 'standard',
                   delta_metric=delta_metric,
                   report=f'reports/{run_id}/REPORT.md')
    path = output / 'run.json'
    try:
        write_receipt(path, receipt)
        receipt['code'] = source_identity(output)
        receipt['environment'] = dict(python=platform.python_version(), conda_environment='opencode_env',
                                     packages={name: version(name) for name in ('numpy','pandas','matplotlib','wandb')})
        receipt['input_manifest_sha256'] = sha256(BASE / 'data' / validate_id(input_id) / 'source_manifest.json')
        records, targets, manifest = load_inputs(input_id)
        receipt.update(project=manifest['project'], input_files=manifest['files'],
                       expected_contexts=manifest['expected_contexts'])
        selected = select_targets(targets, filters)
        receipt.update(acquired_target_runs=len(targets), selected_target_runs=len(selected),
                       excluded_by_filters=len(targets)-len(selected))
        for folder in ('tables', 'figures', 'audit'):
            (output / folder).mkdir()
        frame = prepare_quant_pairs(records, selected)
        frame.to_csv(output / 'tables/acc_ratio_pairs.csv', index=False, lineterminator='\n')
        counts, missing = coverage(frame, manifest['expected_contexts'])
        counts.to_csv(output / 'audit/condition_coverage.csv', index=False, lineterminator='\n')
        missing.to_csv(output / 'audit/missing_contexts.csv', index=False, lineterminator='\n')
        verification = verify_quant_pairs(frame)
        receipt.update(verification=verification,
                       coverage_status='complete' if counts.complete.all() else 'partial')
        (output / 'audit/verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
        if weight_delta:
            from .delta import prepare_weight_delta, verify_weight_delta
            delta_frame, weight_frame = prepare_weight_delta(records, frame)
            if delta_metric == 'acc_ratio_quotient':
                from .delta import add_weight_quotient, verify_weight_quotient
                delta_frame = add_weight_quotient(delta_frame)
                receipt['weight_quotient_verification'] = verify_weight_quotient(delta_frame)
                if not delta_frame.weight_ratio_accepted.all():
                    receipt['coverage_status'] = 'partial'
            delta_frame.to_csv(output/'tables/weight_delta_pairs.csv', index=False, lineterminator='\n')
            weight_frame.to_csv(output/'tables/weight_only_pairs.csv', index=False, lineterminator='\n')
            receipt['weight_delta_verification'] = verify_weight_delta(delta_frame)
            if not delta_frame.weight_delta_accepted.all():
                receipt['coverage_status'] = 'partial'
            refs = plot_weight_delta(delta_frame, manifest['expected_contexts'], output, delta_metric)
            if not refs:
                raise ValueError('No shared weight-only/W+A population is available for plotting')
        else:
            refs, norm, cmap = plot_conditions(frame, manifest['expected_contexts'], output)
            refs += plot_weight_activation(frame, output, norm, cmap)
            refs += plot_dataset_activation(frame, manifest['expected_contexts'], output)
        receipt['figures'] = refs
        (report / 'figures').mkdir()
        for ref in refs:
            for extension in ('png','svg'):
                name = f"{ref['stem']}.{extension}"
                shutil.copy2(output / 'figures' / name, report / 'figures' / name)
                if sha256(output / 'figures' / name) != sha256(report / 'figures' / name):
                    raise ValueError(f'Report image mismatch: {name}')
        receipt.update(status='complete', finished_at=datetime.now(TZ).isoformat(),
                       duration_seconds=time.perf_counter()-started)
        write_report(receipt, report, frame, counts)
        receipt['artifacts'] = {p.relative_to(BASE).as_posix(): sha256(p)
                                for root in (output,report) for p in sorted(root.rglob('*'))
                                if p.is_file() and p != path}
        write_receipt(path, receipt)
    except Exception as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}',
                       finished_at=datetime.now(TZ).isoformat(), duration_seconds=time.perf_counter()-started)
        receipt.pop('artifacts', None)
        (output / 'logs').mkdir(exist_ok=True)
        (output / 'logs/error.log').write_text(traceback.format_exc(), encoding='utf-8')
        write_report(receipt, report)
        write_receipt(path, receipt)
        raise
    print(json.dumps(dict(run_id=run_id, status=receipt['status'], coverage=receipt['coverage_status'],
                         **verification, report=str(report / 'REPORT.md'))))
    return output


def verify_run(run_id):
    validate_id(run_id)
    output, report = BASE / 'result' / run_id, BASE / 'reports' / run_id
    receipt = json.loads((output / 'run.json').read_text(encoding='utf-8'))
    if receipt.get('schema_version') != 2 or receipt['status'] != 'complete' or receipt['run_id'] != run_id:
        raise ValueError('Run is not a completed schema-2 run')
    if sha256(BASE / 'data' / validate_id(receipt['input_id']) / 'source_manifest.json') != receipt['input_manifest_sha256']:
        raise ValueError('Input manifest changed')
    records, targets, manifest = load_inputs(receipt['input_id'])
    for name, expected in receipt['artifacts'].items():
        artifact = (BASE / name).resolve()
        if not artifact.is_relative_to(BASE.resolve()) or sha256(artifact) != expected:
            raise ValueError(f'Output checksum mismatch: {name}')
    frame = prepare_quant_pairs(records, select_targets(targets, receipt['filters']))
    if (output / 'tables/acc_ratio_pairs.csv').read_text(encoding='utf-8') != frame.to_csv(index=False, lineterminator='\n'):
        raise ValueError('Saved pairs disagree with a fresh input-derived computation')
    counts, missing = coverage(frame, manifest['expected_contexts'])
    delta_verification = {}
    if receipt.get('plot_mode') == 'weight_delta':
        from .delta import prepare_weight_delta, verify_weight_delta
        delta_frame, weight_frame = prepare_weight_delta(records, frame)
        if receipt.get('delta_metric') == 'acc_ratio_quotient':
            from .delta import add_weight_quotient, verify_weight_quotient
            delta_frame = add_weight_quotient(delta_frame)
            if verify_weight_quotient(delta_frame) != receipt['weight_quotient_verification']:
                raise ValueError('Saved quotient verification disagrees with fresh computation')
        for name, value in [('weight_delta_pairs', delta_frame), ('weight_only_pairs', weight_frame)]:
            if (output/'tables'/f'{name}.csv').read_text(encoding='utf-8') != value.to_csv(index=False, lineterminator='\n'):
                raise ValueError(f'Saved {name} disagrees with fresh pair computation')
        delta_verification = verify_weight_delta(delta_frame)
        if delta_verification != receipt['weight_delta_verification']:
            raise ValueError('Saved weight delta verification disagrees with fresh computation')
    for name, value in [('condition_coverage', counts), ('missing_contexts', missing)]:
        if (output / 'audit' / f'{name}.csv').read_text(encoding='utf-8') != value.to_csv(index=False, lineterminator='\n'):
            raise ValueError(f'Saved {name} disagrees with a fresh computation')
    if list(report.rglob('*.md')) != [report / 'REPORT.md'] or list(output.rglob('*.md')):
        raise ValueError('A run must have exactly one Markdown report')
    text = (report / 'REPORT.md').read_text(encoding='utf-8')
    expected_images = set()
    for ref in receipt['figures']:
        for ext in ('png','svg'):
            name = f"{ref['stem']}.{ext}"
            expected_images.add(name)
            if f'figures/{name}' not in text or sha256(output/'figures'/name) != sha256(report/'figures'/name):
                raise ValueError(f'Unreferenced or changed report image: {name}')
    if {p.name for p in (report/'figures').iterdir()} != expected_images:
        raise ValueError('Unexpected images in report')
    verification = verify_quant_pairs(frame)
    if verification != receipt['verification']:
        raise ValueError('Saved verification disagrees with fresh verification')
    return dict(status='passed', run_id=run_id, input_files=len(manifest['files']),
                output_hashes=len(receipt['artifacts']), **verification, weight_delta=delta_verification,
                weight_quotient=receipt.get('weight_quotient_verification'))
