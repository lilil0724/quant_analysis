import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'legacy_hqq/summarize_quant.py').is_file())
OUT = ROOT / 'legacy_hqq/result/20261115_weight_only_s10000_10016_audit'
receipt_path = OUT / 'run.json'
receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
os.environ.update(PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4')

def stage(label, module, *args):
    command = [sys.executable, '-B', '-u', '-m', module, *map(str, args)]
    log_path = OUT / 'logs' / (label + '.log')
    print('START', label, flush=True)
    started = datetime.now(ZoneInfo('Asia/Taipei')).isoformat()
    with log_path.open('x', encoding='utf-8') as log:
        result = subprocess.run(command, cwd=ROOT, env=os.environ, stdout=log, stderr=subprocess.STDOUT, text=True)
    receipt['stages'][label] = dict(command=command, started_at=started, exit_code=result.returncode,
                                  status='passed' if result.returncode == 0 else 'failed', log=str(log_path))
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('END', label, result.returncode, flush=True)
    if result.returncode:
        print(log_path.read_text(encoding='utf-8')[-7000:], flush=True)
        raise RuntimeError('Stage failed: ' + label)

try:
    legacy = OUT / 'legacy'
    summary = legacy / 'quant_summary.csv'
    stage('summary', 'legacy_hqq.summarize_quant',
          '--input-file', OUT / 'prepared/curated_historical_runs.csv',
          '--output-file', summary, '--results-dir', legacy)
    stage('factors', 'legacy_hqq.corr_quant', '--input-file', summary,
          '--output-file', legacy / 'corr/factor_ablation_accuracy.csv', '--results-dir', legacy / 'corr',
          '--prepared-output', OUT / 'prepared/prepared_accuracy_cells.csv')
    stage('plots', 'legacy_hqq.plot_quant_acc_bit_group',
          '--input-file', OUT / 'prepared/prepared_accuracy_cells.csv',
          '--output-file', legacy / 'plots/overview.png', '--results-dir', legacy / 'plots',
          '--corr-dir', legacy / 'corr')
    stage('dataset_ranking', 'dataset_robustness.analyze', '--input-file', summary,
          '--results-dir', OUT / 'dataset_robustness')
    stage('verify_dataset_ranking', 'dataset_robustness.verify', '--results-dir', OUT / 'dataset_robustness')
    stage('dataset_loss_pca', 'dataset_robustness.pca_profiles',
          '--main-results', OUT / 'dataset_robustness', '--output-dir', OUT / 'dataset_loss_pca')
    stage('verify_dataset_loss_pca', 'dataset_robustness.verify_pca', '--output-dir', OUT / 'dataset_loss_pca')
    # Recover the existing 15-dataset grouping from Git; do not invent a new label for cotton.
    mapping_source = '3738244:granularity_robustness/dataset_granularity_mapping.csv'
    mapping = OUT / 'provenance/legacy_dataset_granularity_mapping.csv'
    mapping.write_bytes(subprocess.check_output(['git', 'show', mapping_source], cwd=ROOT))
    data = pd.read_csv(summary)
    legacy15 = OUT / 'prepared/quant_summary_legacy15.csv'
    data[data.dataset_name.ne('cotton')].to_csv(legacy15, index=False)
    receipt['mapping_source'] = dict(git_object=mapping_source, sha256=hashlib.sha256(mapping.read_bytes()).hexdigest(),
                                     scope='Existing 15-dataset core/expanded/broad mapping; cotton excluded explicitly')
    stage('legacy15_granularity', 'granularity_robustness.analyze',
          '--input-file', legacy15, '--mapping-file', mapping, '--results-dir', OUT / 'legacy15_granularity')
    stage('verify_legacy15_granularity', 'granularity_robustness.verify', '--results-dir', OUT / 'legacy15_granularity')
    receipt['status'] = 'partial_missing_corrected_cal_10016'
except BaseException as error:
    receipt.update(status='failed', error=str(error))
    raise
finally:
    receipt['finished_at'] = datetime.now(ZoneInfo('Asia/Taipei')).isoformat()
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print('ANALYSIS_FINISHED', receipt['status'], flush=True)
