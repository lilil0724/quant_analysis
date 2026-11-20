import json, hashlib, sys, shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import wandb
from wandb.apis.public.runs import _create_runs_query

root = Path(__file__).resolve().parents[1]
now = datetime.now(timezone(timedelta(hours=8)))
input_id = now.strftime('%Y%m%d_%H%M%S') + '_wandb_s10000_10016'
out = root / 'quant_acc/data' / input_id
out.mkdir(parents=True, exist_ok=False)
query = {'$or': [{'config.serial': {'$in': list(range(10000,10017))}}, {'config.serial': {'$in': list(map(str,range(10000,10017)))}}]}
manifest = {'input_id':input_id,'status':'running','started_at':now.isoformat(),'timezone':'Asia/Taipei','source_project':'nycu_pcs/TiT','filters':query,'command':sys.argv,'checkpoint_bytes_identity':'unknown'}
def write():
    (out/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
write()
try:
    api = wandb.Api(timeout=120)
    runs = api.runs('nycu_pcs/TiT',filters=query,per_page=200)
    runs.QUERY = _create_runs_query(lazy=False)
    total = len(runs)
    print('Expected remote runs:',total,flush=True)
    records=[]
    with (out/'runs.jsonl').open('w',encoding='utf-8') as stream:
        for run in runs:
            attrs=run._attrs
            record={'id':run.id,'name':run.name,'url':run.url,'state':run.state,'created_at':run.created_at,'updated_at':attrs.get('updatedAt'),'config':dict(run.config),'summary':dict(run.summary)}
            stream.write(json.dumps(record,ensure_ascii=False,allow_nan=True)+'\n')
            records.append(record)
            if len(records)%1000==0: print('Downloaded',len(records),'/',total,flush=True)
    rows=[]
    for r in records:
        row=dict(r['config']); row.update(r['summary'])
        row.update(run_id=r['id'],run_url=r['url'],run_name=r['name'],state=r['state'],created_at=r['created_at'],updated_at=r['updated_at'])
        rows.append({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'all_runs.csv',index=False)
    counts=frame.groupby(['serial','state'],dropna=False).size().rename('runs').reset_index()
    counts.to_csv(out/'download_counts.csv',index=False)
    shutil.copy2(__file__,out/'acquire_snapshot.py')
    manifest.update(status='complete',finished_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),expected_remote_runs=total,downloaded_runs=len(records),files={p.name:{'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in out.iterdir() if p.name!='source_manifest.json'})
    assert len(records)==total and len({r['id'] for r in records})==total
    write()
    print(counts.to_string(index=False),flush=True)
    print('INPUT_DIR',out,flush=True)
except Exception as e:
    manifest.update(status='failed',error=repr(e)); write(); raise
