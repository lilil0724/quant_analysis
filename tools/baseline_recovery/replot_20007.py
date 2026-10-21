"""Recompute serial 20007 ratios using the recovered baseline cohort."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legacy_hqq.recover_baselines import context, differences, DATASETS, MODELS, MODES


def load_records(selected, target):
    receipt = json.loads((selected.parent / 'manifest.json').read_text(encoding='utf-8'))
    if receipt['status'] != 'complete':
        raise ValueError('A complete recovered cohort is required for the main plot')
    expected = receipt['artifacts'][selected.name]
    if hashlib.sha256(selected.read_bytes()).hexdigest() != expected:
        raise ValueError('Selected baseline source hash mismatch')
    target_receipt = json.loads((target.parent / 'source_manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256(target.read_bytes()).hexdigest() != target_receipt['files'][target.name]['sha256']:
        raise ValueError('Serial 20007 source hash mismatch')
    records = [json.loads(line) for line in selected.read_text(encoding='utf-8').splitlines()]
    return records, json.loads(target.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected-file', type=Path, default=ROOT / 'data/baseline_recovery/selected_20261021/selected_runs.jsonl')
    parser.add_argument('--target-file', type=Path, default=ROOT / 'outputs/serial_20007/data/snapshot_20261021/serial_20007_runs.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/serial_20007/results/main')
    args = parser.parse_args()
    selected, target, out = args.selected_file.resolve(), args.target_file.resolve(), args.output_dir.resolve()
    records, targets = load_records(selected, target)
    out.mkdir(parents=True, exist_ok=True)
    baselines = {context(r):r for r in records if r['config'].get('hqq') is False}
    rows = []
    for r in targets:
        c = r['config']
        key = context(r)
        b = baselines.get(key)
        mismatch, unrecorded = differences(c, b['config']) if b else (['missing_corrected_baseline'], [])
        assert r['state'] == 'finished' and c['serial'] == 20007
        assert (c['hqq_nbits'],c['hqq_act_bits'],c['hqq_group_size'],c['hqq_act_group_size'],c['hqq_scope']) == (6,4,32,32,'extended')
        quant = r['summary']['hqq_top1']
        fp = b['summary']['top1'] if b else np.nan
        accepted = not mismatch
        rows.append(dict(dataset=key[0], model=key[1], mode=key[2], w6a4_run_id=r['id'],
            w6a4_run_url=r['url'], baseline_run_id=b['id'] if b else '',
            baseline_run_url=b['url'] if b else '', baseline_serial=b['config']['serial'] if b else np.nan,
            w6a4_top1_pct=quant, baseline_top1_pct=fp,
            accuracy_ratio=quant/fp if accepted else np.nan, accuracy_drop_pp=fp-quant if accepted else np.nan,
            accepted=accepted, mismatch_fields=';'.join(mismatch), shared_unrecorded_fields=';'.join(unrecorded),
            w6a4_classifier=c.get('classifier'), baseline_classifier=b['config'].get('classifier') if b else '',
            batch_size_matches=c.get('batch_size')==b['config'].get('batch_size') if b else False,
            checkpoint_bytes_verified=bool(b and c.get('checkpoint_sha256') and c.get('checkpoint_sha256')==b['config'].get('checkpoint_sha256')),
            baseline_precision='unquantized_AMP'))
    df = pd.DataFrame(rows).sort_values(['dataset','model','mode'])
    assert len(df)==144 and not df.duplicated(['dataset','model','mode']).any()
    np.testing.assert_allclose(df.loc[df.accepted,'accuracy_ratio'],
        df.loc[df.accepted,'w6a4_top1_pct']/df.loc[df.accepted,'baseline_top1_pct'],rtol=0,atol=0)
    assert df.loc[~df.accepted,'accuracy_ratio'].isna().all()
    df.to_csv(out/'acc_ratio_pairs.csv',index=False)
    wide = df.pivot(index='dataset',columns=['model','mode'],values='accuracy_ratio').reindex(
        index=DATASETS,columns=pd.MultiIndex.from_product([MODELS,MODES]))
    wide.to_csv(out/'heatmap_values.csv')
    norm = TwoSlopeNorm(vmin=0,vcenter=1,vmax=max(1.05,float(df.accuracy_ratio.max())))
    cmap=plt.get_cmap('RdBu').copy()
    cmap.set_bad('#e5e7eb')
    fig,axes=plt.subplots(1,3,figsize=(16,9),sharey=True,layout='constrained')
    for axis,model,label in zip(axes,MODELS,['ViT-B/16','Swin-B','BEiTv2-B']):
        matrix=wide[model].to_numpy()
        im=axis.imshow(np.ma.masked_invalid(matrix),cmap=cmap,norm=norm,aspect='auto')
        axis.set_title(label)
        axis.set_xticks(range(3),[m.upper() for m in MODES])
        axis.set_yticks(range(len(DATASETS)),DATASETS)
        for i,j in np.ndindex(matrix.shape):
            value=matrix[i,j]
            rgb=cmap(norm(value))[:3] if np.isfinite(value) else (1,1,1)
            light=.2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2]
            axis.text(j,i,f'{value:.3f}' if np.isfinite(value) else 'N/A',ha='center',va='center',fontsize=8,
                      color='white' if light<.5 else '#374151')
    fig.colorbar(im,ax=axes,label='W6A4 accuracy / matched unquantized AMP accuracy',shrink=.75)
    fig.suptitle('Serial 20007: recovered baseline (W6A4, weight G32 / activation G32, extended)')
    caption = (f'{df.accepted.sum()}/144 usable; all corrected baselines available' if df.accepted.all()
               else f'{df.accepted.sum()}/144 usable; gray cells await a compatible corrected baseline')
    fig.supxlabel(caption)
    for extension in ('png','svg','pdf'):
        fig.savefig(out/f'acc_ratio_heatmap.{extension}',dpi=180)
    plt.close(fig)
    text=['# Serial 20007：修正版 baseline 重算','',
          f'- 本次 {df.accepted.sum()}/144 格可配對；{(~df.accepted).sum()} 格保留 N/A。',
          '- dogs、flowers、moe、pets、vegfru 的 BEiTv2 改用本次重新下載的 eval_quant 未量化 AMP baseline。',
          '- FT/FZ 必須同為 cls，CAL 必須同為 pool；沒有將新分母配上舊 weight-only 量化結果。',
          ('- 新版 FT/FZ/CAL baseline 已全部到齊，無缺格。' if df.accepted.all()
           else '- 新 CAL baseline 尚未完成時不沿用舊 CAL，缺格列於 acc_ratio_pairs.csv。'),
          '- 其他 contexts 使用經雜湊驗證的歷史 baseline，完整路徑 alias 明確記錄於資料 manifest。',
          '- batch size 差異保留在逐格表；未提供 checkpoint byte hashes 的格子不能證明跨時間權重相同。',
          '- ratio 不截斷，這是既有結果的描述性比較。','',
          '![Recovered accuracy ratio](acc_ratio_heatmap.png)','']
    (out/'report.md').write_text('\n'.join(text),encoding='utf-8')
    verification=dict(status='complete' if df.accepted.all() else 'partial_pending_server_rerun',
        generated_at=datetime.now(timezone.utc).isoformat(),cells=144,usable_ratios=int(df.accepted.sum()),
        unpaired=int((~df.accepted).sum()),formula_verified=True,
        rejected_reasons=df.loc[~df.accepted,'mismatch_fields'].value_counts().to_dict(),
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (selected,target)},
        artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.name!='verification.json'})
    (out/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in verification.items() if k not in ('inputs','artifacts')},indent=2))
    print(df.loc[~df.accepted,['dataset','model','mode','mismatch_fields']].to_string(index=False))


if __name__=='__main__':
    main()
