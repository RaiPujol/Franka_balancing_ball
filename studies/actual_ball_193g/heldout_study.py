"""Held-out initial-position and synthetic-noise study, simulation only."""
import os, sys, json, csv, itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'studies'/'actual_ball_193g'))
import ball_study as study
ROOT=repo/'output'/'heldout_robustness_20261120'
ROOT.mkdir(parents=True, exist_ok=True)
study.CASES_BY_NAME.update({
 'held_20ms':(.02,.0001),'held_50ms':(.05,.0005),'held_80ms':(.08,.001),
 'high_50ms':(.05,.006),'high_80ms':(.08,.006)})
# Fix random draws before looking at controller outcomes.
rng=np.random.default_rng(20261120)
starts=[]
for i in range(6):
    radius=rng.uniform(.02,.055)
    angle=rng.uniform(0,2*np.pi)
    starts.append((2201+i,tuple(np.round(radius*np.array([np.cos(angle),np.sin(angle)]),6))))
jobs=[('mpc',task,condition,seed,start,.3,.4) for task,condition,(seed,start) in
      itertools.product(('circle','centering'),('held_20ms', 'held_50ms', 'held_80ms', 'high_50ms', 'high_80ms'),starts)]
def work(job):
    try:
        return study.trial(job,suite='gravity',out=ROOT/'details')
    except Exception as e:
        return dict(method=job[0],task=job[1],case=job[2],seed=job[3],start_xy_m=job[4],
                    performance_pass=False, exception=repr(e))
if __name__=='__main__':
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[name]='1'
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(work,jobs))
    (ROOT/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    fields=sorted({k for row in rows for k in row})
    with (ROOT/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
    print(len(rows),'trials',sum(x['performance_pass'] for x in rows),'passes',flush=True)
    for key in sorted({(x['case'],x['task']) for x in rows}):
        same=[r for r in rows if (r['case'],r['task'])==key]
        print(key,sum(r['performance_pass'] for r in same),'/',len(same),flush=True)
