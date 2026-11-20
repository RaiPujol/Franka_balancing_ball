"""Exploratory filter tuning for the synthetic high-noise scenario."""
import os,sys,json,csv,itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'studies'/'actual_ball_193g'))
import ball_study as study
out=root/'output'/'heldout_robustness_20261120'
out.mkdir(parents=True, exist_ok=True)
study.CASES_BY_NAME['stress_50ms']=(.05,.006)
seeded_starts=((2201,(.026957,.037022)),(2202,(-.0415,-.026371)))
tunings=((.005,.1),(.005,.3),(.01,.1),(.01,.3),(.02,.1),(.02,.3))
factory=study.controller_factory
def work(job):
 measurement_std,accel_std,task,(seed,start)=job
 def tuned_factory(method):
  make=factory(method)
  def create(*args,**kwargs):
   control=make(*args,**kwargs)
   control.ball_filter.measurement_std=measurement_std
   control.ball_filter.accel_std=accel_std
   return control
  return create
 trial=('mpc',task,'stress_50ms',seed,start,.3,.4)
 try:
  with patch.object(study,'controller_factory',tuned_factory):
   row=study.trial(trial,suite='gravity',out=out/f'tune_{measurement_std}_{accel_std}')
 except Exception as e:
  row={'task':task,'seed':seed,'performance_pass':False,'exception':repr(e)}
 row.update(measurement_std_m=measurement_std,accel_std=accel_std)
 return row
if __name__=='__main__':
 for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[n]='1'
 jobs=list(itertools.product((.005,.01,.02),(.1,.3),('circle','centering'),seeded_starts))
 with ProcessPoolExecutor(max_workers=4) as p:rows=list(p.map(work,jobs))
 (out/'tuning.json').write_text(json.dumps(rows,indent=2)+'\n')
 fields=sorted({k for r in rows for k in r})
 with (out/'tuning.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
 for tune in tunings:
  for task in ('circle','centering'):
   a=[r for r in rows if (r['measurement_std_m'],r['accel_std'])==tune and r['task']==task]
   field='shape_tracking_rms_m' if task=='circle' else 'final_second_max_center_error_m'
   print(tune,task,sum(r['performance_pass'] for r in a),'/',len(a),'metrics mm',[round(r.get(field,0)*1000,2) for r in a],flush=True)
