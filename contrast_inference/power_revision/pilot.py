from pathlib import Path
import json,time
import numpy as np,pandas as pd
from methods import *
from run_simulations import sample
from concurrent.futures import ProcessPoolExecutor
def job(rep):
 seed=61021000+rep;x,corr=sample('gaussian_star',1008,12,np.random.default_rng(seed));out=[]
 start=time.perf_counter();full=full_orders(x,seed=seed+500000);tf=time.perf_counter()-start
 start=time.perf_counter();split,diag=split_orders(x,seed=seed+600000);ts=time.perf_counter()-start
 for name,raw in {**full,**split}.items():
  ans=project_order(raw,12)
  out.append(dict(rep=rep,method=name,hub=int(ans['rank_upper'][0]==1),excluded=int((ans['rank_lower']>1).sum()),
      width=float(np.mean(ans['degree_upper']-ans['degree_lower'])),orders=raw['raw_order_count'],
      seconds=tf if name in full else ts,tree_p=diag['diagnostics'][name.split('-')[-1]]['global_tree_p']))
 return out
if __name__=='__main__':
 root=Path(__file__).parent/'results';root.mkdir(exist_ok=True)
 with ProcessPoolExecutor(max_workers=6) as ex:rows=[v for rr in ex.map(job,range(20)) for v in rr]
 df=pd.DataFrame(rows);df.to_csv(root/'pilot.csv',index=False)
 print(df.groupby('method').mean().round(4).to_string())
