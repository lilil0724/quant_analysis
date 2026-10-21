"""Create or verify one immutable W6A4 analysis run."""
import argparse
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import pandas as pd

from .acquire import sha256, validate_id
from .analyze import prepare_pairs, verify_pairs, plot_heatmap, KEYS, SETTINGS

BASE = Path(__file__).resolve().parent
REPO = BASE.parent
TZ = timezone(timedelta(hours=8))


def load_inputs(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'source_manifest.json').read_text(encoding='utf-8-sig'))
    if manifest['status'] != 'complete':
        raise ValueError('Input snapshot is not complete')
    for name, entry in manifest['files'].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError('Input manifest path escapes its snapshot')
        if sha256(path) != entry['sha256'] or path.stat().st_size != entry['bytes']:
            raise ValueError(f'Input checksum mismatch: {name}')
    cohort = json.loads((directory / 'cohort_receipt.json').read_text(encoding='utf-8-sig'))
    target = json.loads((directory / 'target_receipt.json').read_text(encoding='utf-8-sig'))
    reference = json.loads((directory / 'reference_receipt.json').read_text(encoding='utf-8-sig'))
    if (cohort['status'] != 'complete' or reference['status'] != 'complete'
            or sha256(directory / 'selected_runs.jsonl') != cohort['artifacts']['selected_runs.jsonl']
            or sha256(directory / 'serial_20007_runs.json') != target['files']['serial_20007_runs.json']['sha256']
            or sha256(directory / 'reference_pairs.csv') != reference['artifacts']['acc_ratio_pairs.csv']):
        raise ValueError('Imported files disagree with their historical receipts')
    records = [json.loads(line) for line in (directory / 'selected_runs.jsonl').read_text(encoding='utf-8').splitlines()]
    targets = json.loads((directory / 'serial_20007_runs.json').read_text(encoding='utf-8'))
    return records, targets, manifest


def reserve_run(run_id):
    validate_id(run_id)
    output, report = BASE / 'result' / run_id, BASE / 'reports' / run_id
    if output.exists() or report.exists():
        raise FileExistsError(f'Run already exists: {run_id}; choose a new ID')
    output.mkdir(parents=True)
    report.mkdir(parents=True)
    return output, report


def write_receipt(path, receipt):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.run-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(receipt, stream, indent=2)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def source_identity(output):
    sources = sorted(Path(__file__).parent.glob('*.py')) + [REPO / 'legacy_hqq/recover_baselines.py', REPO / 'requirements.txt']
    files = {}
    for source in sources:
        relative = source.relative_to(REPO).as_posix()
        copy = output / 'provenance/source' / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, copy)
        files[relative] = sha256(source)
        if sha256(copy) != files[relative]:
            raise ValueError(f'Code snapshot mismatch: {source}')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()
    return dict(commit=git('rev-parse','HEAD'), dirty=bool(git('status','--porcelain')),
                status=git('status','--short').splitlines(), files=files)


def write_report(receipt, report, frame=None):
    run_id = receipt['run_id']
    lines = ['# Serial 20007 W6A4 分析', '',
             f"- Run ID：`{run_id}`；狀態：`{receipt['status']}`。",
             f"- 輸入版本：`{receipt['input_id']}`。",
             f"- 輸入目錄：`../../data/{receipt['input_id']}/`；來源路徑與取得時間見其中 source_manifest.json。",
             f"- 開始：{receipt['started_at']}；結束：{receipt.get('finished_at', '尚未結束')}（Asia/Taipei）。",
             f"- 分析耗時：{receipt.get('duration_seconds', 0):.3f} 秒。",
             f"- 完整輸出：`../../result/{run_id}/`；完整機器紀錄：`../../result/{run_id}/run.json`。", '',
             '## 輸入與方法', '',
             'W&B nycu_pcs/TiT 的既有 config/summary 快照；本次未下載資料、訓練或執行 GPU 推論。',
             '16 datasets × 3 models × FT/FZ/CAL。量化為 W6A4、weight G32、activation G32、extended scope。',
             'Baseline 是相容的未量化 AMP；accuracy_ratio = W6A4 top1 / baseline top1，accuracy 使用百分比，ratio 無單位。',
             '核對完整 checkpoint 路徑、seed、classifier、precision 與已記錄的評估設定；ratio 不截斷。', '',
             '| 輸入檔案 | SHA-256 |', '|---|---|']
    for name, entry in receipt.get('input_files', {}).items():
        lines.append(f"| `{name}` | `{entry['sha256']}` |")
    if frame is not None:
        lines += ['', '## 結果與核對', '',
                  f'- 可配對 {int(frame.accepted.sum())}/144 格；與遷移前參考表逐格比較 run IDs 與四個數值欄位。',
                  '- 公式與參考數值容差為 1e-12；對照細節見完整輸出 audit/verification.json。',
                  '- dogs、flowers、moe、pets、vegfru 的 BEiTv2 使用修正版 baseline；FT/FZ 為 cls，CAL 為 pool。',
                  '- 其餘 baseline 來自已驗證的歷史快照；來源與原 manifest 均保存在本分析 data/。',
                  f'- Batch size 不同的配對：{int((~frame.batch_size_matches).sum())} 格，保留在配對表。', '',
                  '![W6A4 accuracy ratio](figures/acc_ratio_heatmap.png)', '',
                  '[可編輯 SVG](figures/acc_ratio_heatmap.svg)', '',
                  '各模型的 ratio 摘要（合併本次 16 datasets × FT/FZ/CAL，僅為描述統計）：', '',
                  '| Model | 最低 ratio | 中位 ratio | 最高 ratio |', '|---|---:|---:|---:|']
        for model, group in frame.groupby('model', sort=True):
            ratio = group.accuracy_ratio
            lines.append(f'| {model} | {ratio.min():.6f} | {ratio.median():.6f} | {ratio.max():.6f} |')
    if receipt.get('error'):
        lines += ['', '## 失敗原因', '', receipt['error']]
    code = receipt.get('code', {})
    lines += ['', '## 限制與重現', '',
              '這是既有評估的描述性比較。Config 與路徑匹配不等於 checkpoint bytes 相同；未提供共同權重 hash 時維持未驗證。',
              '本分析未獨立驗證伺服器硬體、環境或新推論正確性；重算一致僅證明此快照的分析遷移。',
              f"Git commit：`{code.get('commit', 'unavailable')}`；dirty：`{code.get('dirty', 'unknown')}`。",
              '本次相關程式快照與 SHA-256 位於完整輸出的 provenance/source/ 與 run.json。', '',
              '```powershell', f'conda run -n opencode_env python -m w6a4_analysis.run --input-id {receipt["input_id"]} --run-id NEW_RUN_ID',
              f'conda run -n opencode_env python -m w6a4_analysis.run --verify-run {run_id}', '```', '',
              '重跑使用新的 run ID；原 run 固定保存。報告用圖與完整輸出圖片逐檔核對 SHA-256。', '']
    (report / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def execute(input_id, run_id=None):
    validate_id(input_id)
    run_id = run_id or datetime.now(TZ).strftime('%Y%m%d_%H%M%S_%f_w6a4_s20007')
    output, report = reserve_run(run_id)
    started = time.perf_counter()
    receipt = dict(run_id=run_id, analysis='w6a4_analysis', status='running', input_id=input_id,
                   started_at=datetime.now(TZ).isoformat(), timezone='Asia/Taipei',
                   command=[sys.executable, '-m', 'w6a4_analysis.run', *sys.argv[1:]],
                   environment=dict(python=platform.python_version(), conda_environment='opencode_env',
                                    packages={name: version(name) for name in ('numpy','pandas','matplotlib')}),
                   settings=SETTINGS, report=f'reports/{run_id}/REPORT.md')
    receipt_path = output / 'run.json'
    try:
        write_receipt(receipt_path, receipt)
        receipt['code'] = source_identity(output)
        directory = BASE / 'data' / input_id
        receipt['input_manifest_sha256'] = sha256(directory / 'source_manifest.json')
        records, targets, manifest = load_inputs(directory)
        receipt['input_files'] = manifest['files']
        for folder in ('tables','figures','audit'):
            (output / folder).mkdir()
        frame = prepare_pairs(records, targets)
        frame.to_csv(output / 'tables/acc_ratio_pairs.csv', index=False)
        verification = verify_pairs(frame, pd.read_csv(directory / 'reference_pairs.csv'))
        plot_heatmap(frame, output / 'figures')
        (output / 'audit/verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
        (report / 'figures').mkdir()
        for extension in ('png','svg'):
            source = output / f'figures/acc_ratio_heatmap.{extension}'
            copy = report / 'figures' / source.name
            shutil.copy2(source, copy)
            if sha256(source) != sha256(copy):
                raise ValueError('Report image mismatch')
        receipt.update(status='complete', verification=verification,
                       finished_at=datetime.now(TZ).isoformat(), duration_seconds=time.perf_counter()-started)
        write_report(receipt, report, frame)
        receipt['artifacts'] = {p.relative_to(BASE).as_posix(): sha256(p)
                                for root in (output,report) for p in sorted(root.rglob('*'))
                                if p.is_file() and p != receipt_path}
        write_receipt(receipt_path, receipt)
    except Exception as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}',
                       finished_at=datetime.now(TZ).isoformat(), duration_seconds=time.perf_counter()-started)
        receipt.pop('artifacts', None)
        try:
            (output / 'logs').mkdir(exist_ok=True)
            (output / 'logs/error.log').write_text(traceback.format_exc(), encoding='utf-8')
        except Exception as logging_error:
            print(f'Could not save failure log: {logging_error}', file=sys.stderr)
        for label, save in (('failure report', lambda: write_report(receipt, report)),
                            ('failed receipt', lambda: write_receipt(receipt_path, receipt))):
            try:
                save()
            except Exception as saving_error:
                print(f'Could not save {label}: {saving_error}', file=sys.stderr)
        raise
    print(json.dumps(dict(run_id=run_id, status='complete', usable_ratios=144, report=str(report/'REPORT.md'))))
    return output


def verify_run(run_id):
    validate_id(run_id)
    output, report = BASE / 'result' / run_id, BASE / 'reports' / run_id
    receipt = json.loads((output / 'run.json').read_text(encoding='utf-8'))
    if receipt['status'] != 'complete' or receipt['run_id'] != run_id:
        raise ValueError('Run is not complete')
    directory = BASE / 'data' / validate_id(receipt['input_id'])
    if sha256(directory / 'source_manifest.json') != receipt['input_manifest_sha256']:
        raise ValueError('Input manifest changed')
    records, targets, manifest = load_inputs(directory)
    for name, expected in receipt['artifacts'].items():
        path = (BASE / name).resolve()
        if not path.is_relative_to(BASE.resolve()) or sha256(path) != expected:
            raise ValueError(f'Output checksum mismatch: {name}')
    fresh = prepare_pairs(records, targets)
    saved = pd.read_csv(output / 'tables/acc_ratio_pairs.csv')
    verification = verify_pairs(saved, fresh)
    verify_pairs(saved, pd.read_csv(directory / 'reference_pairs.csv'))
    if list(report.rglob('*.md')) != [report / 'REPORT.md'] or list(output.rglob('*.md')):
        raise ValueError('A run must have exactly one Markdown report')
    for extension in ('png','svg'):
        name = f'acc_ratio_heatmap.{extension}'
        if sha256(output/'figures'/name) != sha256(report/'figures'/name):
            raise ValueError('Report image does not match full output')
    return dict(status='passed', run_id=run_id, input_files=len(manifest['files']),
                output_hashes=len(receipt['artifacts']), **verification)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-id', default='baseline_20261021')
    parser.add_argument('--run-id')
    parser.add_argument('--verify-run')
    args = parser.parse_args()
    if args.verify_run:
        print(json.dumps(verify_run(args.verify_run), indent=2))
    else:
        execute(args.input_id, args.run_id)


if __name__ == '__main__':
    main()
