"""Frozen validation, separate from pilot. All cells are retained."""
from pathlib import Path
import json,time,os,argparse,hashlib
import numpy as np,pandas as pd
from concurrent.futures import ProcessPoolExecutor,as_completed
from methods import *
from run_simulations import sample
DEST=Path(__file__).parent/'results'/'validation'
def design():
 cells=[]
 for model in ['gaussian_star','t5_star']:
  for n in [252,1008,4096]:cells.append(dict(model=model,n=n,p=12,repetitions=300))
 for model,n in [('near_tie',1008),('near_tie',4096),('chain',1008),('two_hubs',1008)]:
  cells.append(dict(model=model,n=n,p=12,repetitions=200))
 for model in ['independent','discrete_independent','equicorrelated','tied_hubs']:
  cells.append(dict(model=model,n=1008,p=12,repetitions=1000))
 return dict(alpha=.05,seed=81279001,matchings=64,training_fraction=.2,cells=cells,
  primary='FS-T64',secondary='FS-S64 and IUT-S64',
  stake='S64=.25; A64=clip(previous sum/(4+previous squared sum),0,.5); T64=training plug-in clipped [.01,.5]',
  note='Pilot seeds 61021000+r; validation seeds 81279001+100000*cell+rep. Choice fixed before validation.',
  main_projection='same cut-cycle minor and degree-conservation projection for every method; no LP',
  method_multiplicity='Each method is evaluated separately at alpha .05; their reported conclusions are not pooled.')

def make_sample(cell,seed):
 rng=np.random.default_rng(seed);n,p,model=cell['n'],cell['p'],cell['model']
 if model=='discrete_independent':return rng.integers(1,5,size=(n,p)).astype(float),np.eye(p)
 if model=='equicorrelated':
  return (rng.normal(size=(n,1))+rng.normal(size=(n,p)))/np.sqrt(2),np.full((p,p),.5)+.5*np.eye(p)
 if model=='tied_hubs':
  a=np.r_[.99,.99,np.linspace(.45,.55,p-2)]
  return rng.normal(size=(n,1))*a+rng.normal(size=(n,p))*np.sqrt(1-a*a),np.outer(a,a)+np.diag(1-a*a)
 return sample(model,n,p,rng)

def worker(job):
 ci,cell,rep,config=job;seed=config['seed']+ci*100000+rep;p=cell['p']
 x,corr=make_sample(cell,seed);edges=edge_index(p);truth=2/np.pi*np.arcsin(corr[edges[:,0],edges[:,1]])
 _,true_degree=max_tree(truth,edges,p)
 unique_tree=cell['model'] not in ['independent','discrete_independent','equicorrelated','tied_hubs']
 unique_hub=unique_tree and (true_degree==true_degree.max()).sum()==1
 t=time.perf_counter();full=full_orders(x,seed=seed+20000000);fulltime=time.perf_counter()-t
 t=time.perf_counter();split,diag=split_orders(x,seed=seed+30000000);splittime=time.perf_counter()-t
 rows=[]
 for name,raw in {**full,**split}.items():
  t=time.perf_counter();ans=project_order(raw,p);projecttime=time.perf_counter()-t
  false=bool(np.any(raw['relation'] & (truth[:,None]<=truth[None,:]+1e-12)))
  rows.append(dict(cell=ci,rep=rep,seed=seed,p=p,n=cell['n'],model=cell['model'],method=name,
     false_order=int(false),hub=int(ans['rank_upper'][np.argmax(true_degree)]==1) if unique_hub else np.nan,
     certified_hubs=int((ans['rank_upper']==1).sum()),excluded=int((ans['rank_lower']>1).sum()),
     exact_degrees=int((ans['degree_lower']==ans['degree_upper']).sum()),
     degree_width=float(np.mean(ans['degree_upper']-ans['degree_lower'])),
     rank_width=float(np.mean(ans['rank_upper']-ans['rank_lower'])),orders=int(raw['raw_order_count']),
     combined_calibration_seconds=fulltime if name in full else splittime,projection_seconds=projecttime,
     ntrain=diag['ntrain'],ntest=diag['ntest'],tree_comparisons=diag['comparisons']))
 return ci,rep,rows

def run(workers=6):
 DEST.mkdir(parents=True,exist_ok=True);config=design();path=DEST/'design.json'
 if path.exists():assert json.loads(path.read_text())==config
 else:path.write_text(json.dumps(config,indent=2))
 jobs=[(ci,c,r,config) for ci,c in enumerate(config['cells']) for r in range(c['repetitions'])
       if not (DEST/f'cell{ci:02d}_{r:04d}.json').exists()]
 start=time.time()
 with ProcessPoolExecutor(max_workers=workers) as pool:
  futures=[pool.submit(worker,j) for j in jobs]
  for k,f in enumerate(as_completed(futures),1):
   ci,r,rows=f.result();(DEST/f'cell{ci:02d}_{r:04d}.json').write_text(json.dumps(rows))
   if k%100==0:print(f'{k}/{len(jobs)} datasets; {time.time()-start:.1f} s',flush=True)
 rows=[r for f in sorted(DEST.glob('cell*.json')) for r in json.loads(f.read_text())]
 df=pd.DataFrame(rows);df.to_csv(DEST/'replications.csv.gz',index=False)
 groups=['p','model','n','method'];s=df.groupby(groups).mean(numeric_only=True).reset_index()
 s=s.merge(df.groupby(groups).size().rename('repetitions').reset_index());s.to_csv(DEST/'summary.csv',index=False)
 print('DONE',len(df),'method-dataset records',flush=True)
 print(s[s.method.isin(['P','E64','A64','FS-T64','FS-S64','IUT-S64'])][groups+['repetitions','hub','false_order','excluded']].round(4).to_string(index=False))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=6);run(a.parse_args().workers)
