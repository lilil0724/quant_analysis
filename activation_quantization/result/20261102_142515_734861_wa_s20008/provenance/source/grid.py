"""Offline summary of the complete serial-20008 W/A grid and optional baselines."""
import collections
import hashlib
import itertools
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd

from .contracts import DATASETS, MODELS, MODES, context, differences
from .storage import validate_id, load_manifest, contained

ROOT = Path(__file__).resolve().parent
TZ = timezone(timedelta(hours=8))
WEIGHTS = (3, 4, 5)
ACTIVATIONS = (1, 1.58, 2, 3, 4, 5, 6)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def table(frame):
    frame = frame.reset_index()
    def cell(value):
        if isinstance(value, float):
            return f'{value:.3f}' if math.isfinite(value) else 'N/A'
        return str(value).replace('|', '\\|')
    lines = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
             '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    lines.extend('| ' + ' | '.join(cell(x) for x in row) + ' |'
                 for row in frame.itertuples(index=False, name=None))
    return '\n'.join(lines)


def execute(input_id, run_id=None):
    validate_id(input_id)
    run_id = validate_id(run_id or datetime.now(TZ).strftime('%Y%m%d_%H%M%S_%f_wa_s20008'))
    inp = ROOT / 'data' / input_id
    out = ROOT / 'result' / run_id
    report_dir = ROOT / 'reports' / run_id
    if out.exists() or report_dir.exists():
        raise FileExistsError('Run/report already exists; choose a new run ID')
    manifest = load_manifest(inp)
    if manifest.get('serial') != 20008:
        raise ValueError('Expected serial 20008 input')
    out.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    receipt = dict(run_id=run_id, input_id=input_id, serial=20008, analysis='activation_quantization', status='running',
                   started_at=datetime.now(TZ).isoformat(), timezone='Asia/Taipei',
                   command=subprocess.list2cmdline([sys.executable, '-B', '-m',
                       'activation_quantization.run', '--serial', '20008', '--input-id', input_id, '--run-id', run_id]),
                   python=sys.version, pandas=pd.__version__,
                   input_manifest_sha256=sha(inp / 'source_manifest.json'),
                   inputs=manifest['files'])
    write_json(out / 'run.json', receipt)
    report = report_dir / 'REPORT.md'
    try:
        targets = json.loads((inp / 'target.json').read_text(encoding='utf-8'))
        baselines = json.loads((inp / 'baselines.json').read_text(encoding='utf-8'))
        groups = collections.defaultdict(list)
        for base in baselines:
            value = base['summary'].get('top1')
            if base['state'] == 'finished' and isinstance(value, (int, float)) and math.isfinite(value) and 0 < value <= 100:
                groups[context(base)].append(base)
        rows = []
        for run in targets:
            c, s = run['config'], run['summary']
            key = context(run)
            fixed = dict(serial=20008, hqq=True, hqq_group_size=32,
                         hqq_act_group_size=32, hqq_scope='extended', fp16=True,
                         hqq_compute_dtype='float16', hqq_exclude='', debugging=False,
                         hqq_quant_zero=True, hqq_quant_scale=True)
            if run['state'] != 'finished' or any(c.get(k) != v for k, v in fixed.items()):
                raise ValueError(f'Unexpected settings/state: {run["id"]}')
            value = s.get('hqq_top1')
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError(f'Invalid top1: {run["id"]}')
            candidates = []
            audit = []
            for base in groups[key]:
                mismatch, unrecorded = differences(c, base['config'])
                audit.append(dict(id=base['id'], mismatches=mismatch))
                if not mismatch:
                    candidates.append((base, unrecorded))
            base, unrecorded = candidates[0] if len(candidates) == 1 else (None, [])
            reason = 'matched' if base else 'ambiguous' if candidates else 'missing' if not groups[key] else 'config_mismatch'
            fp = base['summary']['top1'] if base else float('nan')
            rows.append(dict(dataset=key[0], model=key[1], mode=key[2],
                w=c['hqq_nbits'], a=c['hqq_act_bits'], top1=value, run_id=run['id'], url=run['url'],
                seed=c['seed'], classifier=c['classifier'], checkpoint=c['ckpt_path'],
                baseline_status=reason, baseline_run_id=base['id'] if base else '',
                baseline_top1=fp, drop_pp=fp-value if base else float('nan'),
                accuracy_ratio=value/fp if base else float('nan'),
                batch_size_matches=c.get('batch_size') == base['config'].get('batch_size') if base else None,
                checkpoint_bytes_verified=bool(base and c.get('checkpoint_sha256') and
                    c.get('checkpoint_sha256') == base['config'].get('checkpoint_sha256')),
                shared_unrecorded_fields=';'.join(unrecorded), candidate_audit=json.dumps(audit)))
        df = pd.DataFrame(rows)
        keys = ['dataset', 'model', 'mode', 'w', 'a']
        expected = set(itertools.product(DATASETS, MODELS, MODES, WEIGHTS, ACTIVATIONS))
        if df.duplicated(keys).any() or set(df[keys].itertuples(index=False, name=None)) != expected:
            raise ValueError('Grid is incomplete, duplicated, or contains unexpected contexts')
        # Within each checkpoint, all evaluation settings and precision must remain fixed.
        reference = {}
        for run in targets:
            key = context(run)
            if key in reference:
                mismatch, _ = differences(run['config'], reference[key])
                if mismatch:
                    raise ValueError(f'W/A grid evaluation settings differ: {key}, {mismatch}')
            else:
                reference[key] = run['config']
        tables = out / 'tables'
        tables.mkdir()
        df.sort_values(keys).to_csv(tables / 'runs_and_baseline_audit.csv', index=False)
        wa = df.pivot_table(index='a', columns='w', values='top1').reindex(ACTIVATIONS)
        model_a = df.pivot_table(index='a', columns='model', values='top1').reindex(ACTIVATIONS)
        mode_a = df.pivot_table(index='a', columns='mode', values='top1').reindex(ACTIVATIONS)
        paired = df.pivot(index=['dataset', 'model', 'mode', 'w'], columns='a', values='top1')
        transitions = []
        for left, right in zip(ACTIVATIONS[:-1], ACTIVATIONS[1:]):
            delta = paired[right] - paired[left]
            transitions.append(dict(from_a=left, to_a=right, n=len(delta), mean_gain_pp=delta.mean(),
                median_gain_pp=delta.median(), improved=int((delta > 0).sum()),
                worsened=int((delta < 0).sum()), tied=int((delta == 0).sum())))
        baseline_agg = df[df.baseline_status.eq('matched')].groupby(['a', 'w']).agg(
            n=('top1', 'size'), baseline_mean=('baseline_top1', 'mean'),
            quant_mean=('top1', 'mean'), mean_drop_pp=('drop_pp', 'mean'),
            mean_accuracy_ratio=('accuracy_ratio', 'mean'))
        for name, frame in [('wa_mean_top1', wa), ('model_a_mean_top1', model_a),
                            ('mode_a_mean_top1', mode_a), ('baseline_summary', baseline_agg),
                            ('paired_a_transitions', pd.DataFrame(transitions))]:
            frame.to_csv(tables / f'{name}.csv')
        from .plot import render
        plot_validation = render(df, out, report_dir)
        coverage = df[['dataset', 'model', 'mode', 'baseline_status']].drop_duplicates()
        counts = coverage.baseline_status.value_counts().to_dict()
        validation = dict(target_runs=len(df), grid_contexts=len(reference), expected_runs=len(expected),
            duplicate_cells=0, invalid_top1=0, consistent_within_checkpoint_eval_settings=True,
            baseline_contexts=counts, checkpoint_bytes_verified=int(df.checkpoint_bytes_verified.sum()),
            input_hashes_verified=True, plots=plot_validation)
        write_json(out / 'validation.json', validation)
        source = out / 'provenance' / 'source'
        source.mkdir(parents=True)
        for path in sorted(ROOT.glob('*.py')):
            shutil.copy2(path, source / path.name)
        receipt['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT.parent, text=True).strip()
        receipt['git_status'] = subprocess.check_output(['git', 'status', '--short'], cwd=ROOT.parent, text=True)
        receipt['code_sha256'] = {p.name: sha(p) for p in source.iterdir()}
        text = '\n'.join([
            '# W&B serial 20008 簡要分析', '',
            f'分析時間：{datetime.now(TZ).isoformat()}（Asia/Taipei）。', '',
            '來源：W&B `nycu_pcs/TiT`，Public API 全量 config/summary；target filter `config.serial=20008`。',
            f'輸入：`data/{input_id}/source_manifest.json`；指標 `hqq_top1`，單位為 %。', '',
            '## 範圍與方法', '',
            '3,024/3,024 finished；16 datasets × 3 models × FT/FZ/CAL × W3/4/5 × A1/1.58/2/3/4/5/6。',
            'Weight/activation G32、extended、head 納入、AMP fp16。平均值為每個 context 等權，不依測試樣本數加權。',
            '逐 checkpoint 核對完整路徑、dataset、model/mode、seed、classifier、precision 與既有 evaluation fields；W/A 比較保持這些設定一致。', '',
            '## Activation 觀察圖', '',
            '### 1. Accuracy–activation bits', '',
            '九個面板分開 model × FT/FZ/CAL；W3/W4/W5 各一色。淡線是個別 dataset，粗線是同一組 16 datasets 的等權平均；不是信賴區間。', '',
            '![Accuracy versus activation bits](figures/accuracy_vs_activation.png)', '',
            '### 2. 各 dataset 相對 A6 的損傷', '',
            '`loss_vs_a6_pp = Acc(相同 checkpoint, W, A6) - Acc(相同 checkpoint, W, A)`。',
            '正值（紅）代表比 A6 差，負值（藍）代表比 A6 好；不截斷、不強制單調。三張圖共用 dataset 順序及對稱色階。',
            '**A6 仍有 activation quantization；此圖不是相對未量化 activation 的完整損傷，也不使用缺失的 FP baseline 補值。**', '',
            '![W3 activation loss relative to A6](figures/activation_loss_vs_a6_w3.png)', '',
            '![W4 activation loss relative to A6](figures/activation_loss_vs_a6_w4.png)', '',
            '![W5 activation loss relative to A6](figures/activation_loss_vs_a6_w5.png)', '',
            '完整 PNG／可編輯 SVG 與四頁 PDF `activation_observation_atlas.pdf` 位於本次 result 的 figures/。',
            f'Plot coverage：{plot_validation["paired_rows"]} 個配對點，{plot_validation["negative_loss_cells"]} 個負值，189 個平均點，每點 16 datasets。', '',
            '## W/A 平均 Top-1 (%)', '', table(wa), '',
            '## 模型 × activation 平均 Top-1 (%)', '', table(model_a), '',
            '## FT/FZ/CAL × activation 平均 Top-1 (%)', '', table(mode_a), '',
            '## 同 checkpoint、同 weight 的 activation 成對增量', '', table(pd.DataFrame(transitions)), '',
            '## 未量化 AMP baseline', '',
            f'144 contexts 的配對 coverage：{counts}。僅唯一且 config 相容者計算 drop/ratio；缺失與多個相容候選保留 N/A，不任選 baseline。',
            '完整候選與 mismatch、checkpoint、run URL 皆保存於 `runs_and_baseline_audit.csv`。', '',
            table(baseline_agg), '',
            '## 解讀與限制', '',
            '- A≤2 的平均 accuracy 僅約 0.7–1.3%；A3 才明顯恢復，A4→A6 的額外收益較小。',
            '- A3 時 BEiTv2 的絕對 accuracy 明顯低於另外兩種模型；跨模型 robustness 仍須看完整 matched baseline，不能只靠絕對 accuracy 排名。',
            '- W3 在 A5→A6 的平均 accuracy 略降；bit 數提高不保證每個條件單調改善。',
            '- 現行 TGDA launcher 文件說明 post_softmax_no_patch_v1，但 W&B config/summary 沒有 coverage.protocol；未讀取 server evaluation.json，故本分析無法逐 run 驗證插點或 patch 排除。',
            '- 不以 serial 20007 當控制組：其 W/A 設定與歷史 protocol 不同。',
            '- 未取得 checkpoint bytes hashes、逐樣本預測或 features；不作因果定位、統計顯著性或 integer-kernel 效能宣稱。',
            '- Baseline 是未量化 AMP，並非純 FP32；coverage 不完整，部分 batch size 不同。',
            '- 每個 context 只有一組 checkpoint/seed；跨 datasets 平均僅為描述性摘要。', '',
            '## 重現', '', '```powershell',
            f'& "{sys.executable}" -B -m activation_quantization.run --input-id {input_id} --run-id NEW_RUN_ID',
            f'& "{sys.executable}" -B -m activation_quantization.run --verify-run {run_id}', '```', '',
        ])
        report.write_text(text, encoding='utf-8')
        # Verify serialized tables and formulas before finalizing.
        saved = pd.read_csv(tables / 'runs_and_baseline_audit.csv')
        assert len(saved) == 3024 and not saved.duplicated(keys).any()
        matched = saved[saved.baseline_status.eq('matched')]
        assert ((matched.baseline_top1 - matched.top1 - matched.drop_pp).abs() < 1e-10).all()
        assert ((matched.top1 / matched.baseline_top1 - matched.accuracy_ratio).abs() < 1e-10).all()
        assert saved.loc[~saved.baseline_status.eq('matched'), 'accuracy_ratio'].isna().all()
        receipt.update(status='complete', finished_at=datetime.now(TZ).isoformat(), validation=validation,
            report=str(report.relative_to(ROOT)), report_sha256=sha(report),
            outputs={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file() and p.name != 'run.json'})
        write_json(out / 'run.json', receipt)
        print(json.dumps(validation, ensure_ascii=False))
        print('W/A means\n' + wa.round(3).to_string())
        print('Paired transitions\n' + pd.DataFrame(transitions).round(3).to_string(index=False))
        print('Baseline summary\n' + baseline_agg.round(3).to_string())
        print('REPORT', report)
        return out
    except Exception as exc:
        receipt.update(status='failed', error=repr(exc), finished_at=datetime.now(TZ).isoformat())
        write_json(out / 'run.json', receipt)
        report.write_text(f'# Serial 20008 analysis failed\n\n{exc!r}\n', encoding='utf-8')
        raise


def verify_run(run_id):
    """Verify both the original serial-20008 receipt schema and consolidated runs."""
    validate_id(run_id)
    out = ROOT / 'result' / run_id
    receipt = json.loads((out / 'run.json').read_text(encoding='utf-8'))
    if receipt['status'] != 'complete' or receipt['run_id'] != run_id:
        raise ValueError('Run is not complete')
    inp = ROOT / 'data' / validate_id(receipt['input_id'])
    manifest = load_manifest(inp)
    if manifest.get('serial') != 20008 or sha(inp / 'source_manifest.json') != receipt['input_manifest_sha256']:
        raise ValueError('Input manifest changed or has a different serial')
    if not receipt.get('outputs'):
        raise ValueError('Run has no output manifest')
    for name, expected in receipt['outputs'].items():
        if sha(contained(out, name)) != expected:
            raise ValueError(f'Output checksum mismatch: {name}')
    report_dir = ROOT / 'reports' / run_id
    report = contained(ROOT, receipt['report'])
    if report != (report_dir / 'REPORT.md').resolve() or sha(report) != receipt['report_sha256']:
        raise ValueError('Report path/hash mismatch')
    if list(report_dir.rglob('*.md')) != [report_dir / 'REPORT.md'] or list(out.rglob('*.md')):
        raise ValueError('A run must have exactly one Markdown report')
    saved = pd.read_csv(out / 'tables/runs_and_baseline_audit.csv')
    keys = ['dataset', 'model', 'mode', 'w', 'a']
    expected = set(itertools.product(DATASETS, MODELS, MODES, WEIGHTS, ACTIVATIONS))
    if saved.duplicated(keys).any() or set(saved[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError('Saved grid is incomplete or duplicated')
    matched = saved[saved.baseline_status.eq('matched')]
    if (not ((matched.baseline_top1 - matched.top1 - matched.drop_pp).abs() < 1e-10).all()
            or not ((matched.top1 / matched.baseline_top1 - matched.accuracy_ratio).abs() < 1e-10).all()
            or not saved.loc[~saved.baseline_status.eq('matched'), 'accuracy_ratio'].isna().all()):
        raise ValueError('Saved baseline formulas or missing values are invalid')
    if 'plots' in receipt.get('validation', {}):
        from .plot import verify_tables
        verify_tables(saved, out / 'tables')
        for name in receipt['validation']['plots']['figure_stems']:
            if sha(out / 'figures' / f'{name}.png') != sha(report_dir / 'figures' / f'{name}.png'):
                raise ValueError('Report figure does not match the result figure')
    return dict(status='passed', serial=20008, run_id=run_id, target_runs=len(saved),
                input_files=len(manifest['files']), output_hashes=len(receipt['outputs']) + 1)
