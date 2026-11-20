"""Recompute the legacy HQQ analysis from an immutable serial-grid snapshot.

CAL uses only serial 10016, including its own unquantized AMP baselines.
Other modes use the newest compatible finished result per condition.
The existing summary, factor-model and plotting stages retain their formulas.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .recover_baselines import BITS, DATASETS, GROUPS, MODELS, MODES, canonical, context, differences

BASE = Path(__file__).resolve().parent
TZ = timezone(timedelta(hours=8))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp():
    return datetime.now(TZ).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def metric_valid(run):
    value = run['summary'].get('hqq_top1' if run['config'].get('hqq') else 'top1')
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 100


def select(records, output):
    groups, excluded = {}, []
    for run in records:
        config = run['config']
        key = context(run)
        reason = None
        if not 10000 <= int(config['serial']) <= 10016:
            reason = 'outside_serial_range'
        elif run['state'] != 'finished' or not metric_valid(run):
            reason = 'unfinished_or_invalid_metric'
        elif key[2] == 'cal' and int(config['serial']) != 10016:
            reason = 'CAL_replaced_by_10016'
        elif config.get('debugging') or config.get('dummy_loader') or config.get('model_mode') != 'eval':
            reason = 'not_full_test_evaluation'
        if reason:
            excluded.append(dict(run_id=run['id'], reason=reason))
        else:
            groups.setdefault(key, []).append(run)
    selected, audits, coverage, missing = [], [], [], []
    expected = list(itertools.product(BITS, GROUPS))
    for key in itertools.product(DATASETS, MODELS, MODES):
        runs = groups.get(key, [])
        baseline = sorted((r for r in runs if r['config'].get('hqq') is False and r['summary']['top1'] > 0), key=lambda r: (r['created_at'], r['id']))
        base = baseline[-1] if baseline else None
        if base:
            if base['config'].get('fp16') is not True:
                raise ValueError(f'Expected unquantized AMP baseline: {key}')
            selected.append(base)
        excluded.extend(dict(run_id=r['id'], reason='older_baseline') for r in baseline[:-1])
        accepted = 0
        for bits, group in expected:
            candidates = [r for r in runs if r['config'].get('hqq') is True and float(r['config']['hqq_nbits']) == bits and int(r['config']['hqq_group_size']) == group]
            candidates.sort(key=lambda r: (r['created_at'], r['id']), reverse=True)
            chosen = None
            for run in candidates:
                c = run['config']
                mismatch, unrecorded = differences(c, base['config']) if base else (['missing_baseline'], [])
                for field in ('model_mode', 'cam_aug', 'test_only', 'debugging', 'dummy_loader', 'max_dataset_iters'):
                    if base and c.get(field) != base['config'].get(field):
                        mismatch.append(field)
                if c.get('hqq_scope') not in (None, 'linear') or c.get('hqq_act_bits') is not None:
                    mismatch.append('not_weight_only_linear')
                if c.get('hqq_compute_dtype') != 'float16':
                    mismatch.append('not_float16_compute')
                if key[2] == 'cal' and (c.get('hqq_exclude') != 'head,model.dfsm.fc,model.dfsm.cam_head' or c.get('hqq_quant_zero') is not False or c.get('hqq_quant_scale') is not False):
                    mismatch.append('unexpected_CAL_10016_quantization_settings')
                keep = not mismatch and chosen is None
                reason = ';'.join(mismatch) if mismatch else ('selected' if keep else 'older_compatible_condition')
                audits.append(dict(dataset=key[0], model=key[1], mode=key[2], nbits=bits, group_size=group, quant_run_id=run['id'], baseline_run_id=base['id'] if base else '', accepted=keep, reason=reason, shared_unrecorded_fields=';'.join(unrecorded), batch_size_matches=c.get('batch_size') == base['config'].get('batch_size') if base else False, checkpoint_bytes_verified=bool(base and c.get('checkpoint_sha256') and c.get('checkpoint_sha256') == base['config'].get('checkpoint_sha256'))))
                if keep:
                    chosen = run
            if chosen:
                selected.append(chosen)
                accepted += 1
            else:
                missing.append(dict(dataset=key[0], model=key[1], mode=key[2], nbits=bits, group_size=group, reason='no_compatible_finished_pair'))
        coverage.append(dict(dataset=key[0], model=key[1], mode=key[2], baseline_run_id=base['id'] if base else '', baseline_serial=base['config']['serial'] if base else None, classifier=base['config'].get('classifier') if base else None, accepted_conditions=accepted, expected_conditions=32, complete=accepted == 32 and base is not None))
    rows = []
    for r in selected:
        row = dict(r['config'])
        row.update(r['summary'])
        row.update(run_id=r['id'], run_url=r['url'], created_at=r['created_at'], state=r['state'], host=None, ckpt_path=canonical(r['config']['ckpt_path']))
        rows.append({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    pd.DataFrame(rows).to_csv(output / 'curated_16_datasets.csv', index=False)
    pd.DataFrame(audits).to_csv(output / 'pair_audit.csv', index=False)
    pd.DataFrame(coverage).to_csv(output / 'context_coverage.csv', index=False)
    pd.DataFrame(missing, columns=['dataset', 'model', 'mode', 'nbits', 'group_size', 'reason']).to_csv(output / 'missing_conditions.csv', index=False)
    pd.DataFrame(excluded).to_csv(output / 'excluded_runs.csv', index=False)
    (output / 'selected_runs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected), encoding='utf-8')
    return selected, coverage, audits, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--run-id')
    args = parser.parse_args()
    source = args.input_dir.resolve()
    if not source.is_relative_to(BASE / 'data'):
        parser.error('--input-dir must be inside quant_acc/data')
    manifest = json.loads((source / 'source_manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete':
        raise ValueError('Input download is incomplete')
    for name, record in manifest['files'].items():
        path = source / name
        if sha(path) != record['sha256'] or path.stat().st_size != record['size_bytes']:
            raise ValueError(f'Input integrity mismatch: {name}')
    run_id = args.run_id or datetime.now(TZ).strftime('%Y%m%d_%H%M%S') + '_legacy_hqq_s10000_10016_cal10016'
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in run_id) or run_id in ('.', '..'):
        parser.error('Invalid --run-id')
    result, report = BASE / 'result' / run_id, BASE / 'reports' / run_id
    if result.exists() or report.exists():
        raise FileExistsError(f'Refusing to overwrite run {run_id}')
    result.mkdir(parents=True)
    report.mkdir(parents=True)
    for folder in ('prepared', 'summary', 'corr', 'figures', 'audit', 'logs', 'provenance/source'):
        (result / folder).mkdir(parents=True)
    run = dict(run_id=run_id, status='running', started_at=stamp(), timezone='Asia/Taipei', command=['conda', 'run', '-n', 'opencode_env', 'python', '-m', 'quant_acc.rerun_serial_grid', '--input-dir', str(source), '--run-id', run_id], input_id=manifest['input_id'], input_dir=str(source), input_manifest_sha256=sha(source / 'source_manifest.json'), inputs=manifest['files'], python=sys.version, platform=platform.platform(), package_versions=dict(pandas=pd.__version__, numpy=np.__version__), git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), git_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()), selection_policy='16 datasets; CAL FP and HQQ strictly serial 10016; newest compatible finished baseline and per-condition HQQ for FT/FZ; no historical fallback; no duplicate averaging', checkpoint_bytes_identity='unknown unless matching checkpoint_sha256 fields are present', stages=[])
    start = time.monotonic()
    write_json(result / 'run.json', run)
    try:
        for p in BASE.glob('*.py'):
            shutil.copy2(p, result / 'provenance/source' / p.name)
        shutil.copy2(source / 'acquire_snapshot.py', result / 'provenance/source/acquire_snapshot.py')
        records = [json.loads(line) for line in (source / 'runs.jsonl').read_text(encoding='utf-8').splitlines()]
        if len({r['id'] for r in records}) != len(records):
            raise ValueError('Duplicate run IDs in input')
        selected, coverage, audits, missing = select(records, result / 'audit')
        print(f'Selected {len(selected)} runs; {sum(c["complete"] for c in coverage)}/144 complete contexts; missing {len(missing)}', flush=True)
        summary = result / 'prepared/quant_summary.csv'
        prepared = result / 'prepared/prepared_accuracy_cells.csv'
        ablation = result / 'corr/factor_ablation_accuracy.csv'
        commands = [
            ('prepare', 'summarize_quant', ['--input-file', result / 'audit/curated_16_datasets.csv', '--output-file', summary, '--results-dir', result / 'summary']),
            ('analyze', 'corr_quant', ['--input-file', summary, '--output-file', ablation, '--results-dir', result / 'corr', '--prepared-output', prepared, '--no-markdown']),
            ('plot', 'plot_quant_acc_bit_group', ['--input-file', prepared, '--ablation-file', ablation, '--corr-dir', result / 'corr', '--results-dir', result / 'figures', '--output-file', result / 'figures/quant_accuracy_overview.png']),
        ]
        for stage, module, options in commands:
            command = [sys.executable, '-m', 'quant_acc.' + module, *map(str, options)]
            print('Running', stage, flush=True)
            with (result / f'logs/{stage}.log').open('w', encoding='utf-8') as stream:
                subprocess.run(command, cwd=BASE.parent, stdout=stream, stderr=subprocess.STDOUT, check=True)
            run['stages'].append(dict(stage=stage, command=command, status='complete'))
            write_json(result / 'run.json', run)
        cells = pd.read_csv(summary)
        plotted = pd.read_csv(prepared)
        raw = pd.read_csv(result / 'audit/curated_16_datasets.csv')
        cal = raw[raw['run_id'].isin([r['id'] for r in selected if context(r)[2] == 'cal'])]
        assert cal['serial'].eq(10016).all()
        assert len(cells) == sum(c['accepted_conditions'] for c in coverage)
        assert cells['baseline_matched'].all() and len(plotted) == len(cells)
        assert np.allclose(cells['accuracy_ratio'], cells['hqq_top1'] / cells['fp32_top1'], rtol=1e-12, atol=1e-12)
        assert len(list((result / 'summary').glob('quant_*.xlsx'))) == 16
        assert not raw['run_id'].duplicated().any()
        for r in selected:
            if r['config']['hqq']:
                matches = [a for a in audits if a['quant_run_id'] == r['id'] and a['accepted']]
                assert len(matches) == 1
        figures = sorted((result / 'figures').glob('*.png'))
        run['validation'] = dict(downloaded_runs=len(records), selected_runs=len(selected), baselines=int((~raw.hqq).sum()), hqq_pairs=len(cells), matched_cells=len(plotted), contexts=len(coverage), complete_contexts=sum(c['complete'] for c in coverage), missing_conditions=len(missing), datasets=16, CAL_10016_runs=len(cal), CAL_10016_only=True, duplicate_selected_run_ids=0, formula_check=True, workbooks=16, figures=len(figures), checkpoint_bytes_verified_pairs=sum(a['accepted'] and a['checkpoint_bytes_verified'] for a in audits))
        report_figures = report / 'figures'
        report_figures.mkdir()
        cited = ['quant_accuracy_overview.png', 'accuracy_heatmap_cal.png', 'factor_contribution_accuracy.png']
        for name in cited:
            shutil.copy2(result / 'figures' / name, report_figures / name)
            assert sha(result / 'figures' / name) == sha(report_figures / name)
        modes = cells.groupby('ckpt_kind').agg(cells=('accuracy_ratio', 'size'), median_ratio=('accuracy_ratio', 'median')).reset_index()
        terms = pd.read_csv(ablation).sort_values('delta_r2', ascending=False)
        run.update(status='complete', finished_at=stamp(), elapsed_seconds=round(time.monotonic() - start, 3))
        lines = ['# Legacy HQQ accuracy analysis — 10000–10016', '', f'- Run: `{run_id}`; status: complete.', f'- Time: {run["started_at"]} → {run["finished_at"]} (Asia/Taipei).', f'- Input: `{source}`; W&B `nycu_pcs/TiT`; {len(records):,} downloaded runs.', f'- Source manifest SHA-256: `{run["input_manifest_sha256"]}`.', f'- Git: `{run["git_commit"]}`; dirty={run["git_dirty"]}.', '', '## Method and pairing', '', 'All 16 datasets, including cotton; 3 models × FT/FZ/CAL × 8 bit widths × 4 group sizes. CAL baseline and HQQ results use serial 10016 only. FT/FZ use the newest finished positive AMP baseline per dataset/model/mode, then the newest compatible HQQ run for each bit/group condition. Older duplicates are not averaged. Checkpoint path aliases use the existing explicit ../../data/ → /work/kyle0724/data/ rule.', '', 'Pairing checks complete checkpoint path, model, dataset, mode, seed, classifier, AMP precision, resize/crop/evaluation settings and dataset root; all decisions and missing optional fields are saved in audit/pair_audit.csv. The baseline is unquantized AMP; fp32_top1 is a legacy column name. Accuracy ratio = HQQ top-1 / matched unquantized AMP top-1. Factor models and plots reuse the existing quant_acc stages and formulas.', '', '## Results and validation', '', f'- {len(cells):,} matched HQQ cells; {run["validation"]["baselines"]} baselines; {sum(c["complete"] for c in coverage)}/144 complete contexts; missing conditions: {len(missing)}.', f'- {len(cal):,} selected CAL runs exclusively from 10016.', f'- Formula and selected run-ID uniqueness checks passed; 16 dataset XLSX workbooks; {len(figures)} PNG figures; report figure hashes match the full outputs.', '', '| Mode | Cells | Median accuracy ratio |', '|---|---:|---:|']
        lines.extend(f'| {r.ckpt_kind} | {r.cells} | {r.median_ratio:.6f} |' for r in modes.itertuples())
        lines += ['', '![Overview](figures/quant_accuracy_overview.png)', '', '![CAL](figures/accuracy_heatmap_cal.png)', '', '## Descriptive factor contributions', '', '| Term | Ablation delta R² |', '|---|---:|']
        lines.extend(f'| {r.term_group} | {r.delta_r2:.6f} |' for r in terms.itertuples())
        lines += ['', '![Factor contributions](figures/factor_contribution_accuracy.png)', '', '## Limitations', '', 'Checkpoint bytes were not downloaded or independently hashed; matching paths/configuration do not prove byte-identical weights. A single recorded seed does not establish uncertainty across retraining. CAL 10016 excludes DFSM classifiers and leaves quantization scale/zero metadata unquantized. FT/FZ retain their recorded per-context settings, including heterogeneous metadata-quantization settings; CAL versus FT/FZ and dataset/model effects are descriptive and confounded. Batch-size mismatches and optional unrecorded evaluation fields remain visible in the pair audit. No server inference or causal comparison was performed.', '', '## Reproduce', '', '```powershell', subprocess.list2cmdline(run['command']), '```', '', f'Complete outputs: `{result}`. Input hashes, commands, source snapshots, output hashes and validation are in `run.json`. Existing input and result directories are never overwritten. The input snapshot includes acquisition code and its manifest; a new download requires a new input ID.', '']
        (report / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
        assert len(list(report.rglob('*.md'))) == 1
        run['report'] = str(report / 'REPORT.md')
        run['report_sha256'] = sha(report / 'REPORT.md')
        run['source_sha256'] = {p.name: sha(p) for p in (result / 'provenance/source').glob('*.py')}
        run['outputs'] = {str(p.relative_to(result)): dict(size_bytes=p.stat().st_size, sha256=sha(p)) for p in result.rglob('*') if p.is_file() and p.name != 'run.json'}
        write_json(result / 'run.json', run)
        print(json.dumps(run['validation'], indent=2), flush=True)
        print('REPORT', report / 'REPORT.md', flush=True)
    except Exception as error:
        run.update(status='failed', finished_at=stamp(), error=repr(error), elapsed_seconds=round(time.monotonic() - start, 3))
        write_json(result / 'run.json', run)
        (report / 'REPORT.md').write_text(f'# Legacy HQQ run failed\n\nRun: {run_id}\n\nError: {error!r}\n\nSee `{result}/run.json` and logs.\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
