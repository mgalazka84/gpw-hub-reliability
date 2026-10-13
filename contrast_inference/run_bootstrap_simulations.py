"""Measure calibration at small and GPW-sized dimensions, including nulls."""
from pathlib import Path
import sys,argparse,time,json
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'statistical_papers'))
from hub_inference import bootstrap_covariances,correlation_edges,calibrated_band,degree_envelopes,rank_envelopes,max_tree
from run_simulations import sample
from order_inference import edge_index,bootstrap_order,project_order

ROOT=Path(__file__).parent;DEST=ROOT/'results'
def run(reps=100,B=499):
 cells=[(p,model,n) for p in [12,52] for model in ['independent','gaussian_star','near_tie','ar_star','ar_independent','t5_star'] for n in [252,1008]]
 config=dict(repetitions=reps,bootstrap_repetitions=B,cells=cells,alpha=.05,seed=681203,
  design='held-out validation; all calibration failures retained',
  studentizer='fixed bootstrap standard deviation of each local Fisher-z difference',
  serial_block='round(n**(1/3)), minimum 2',
  baseline='centered Fisher-z simultaneous rectangular band')
 DEST.mkdir(exist_ok=True);(DEST/'bootstrap_design.json').write_text(json.dumps(config,indent=2))
 rows=[];start=time.time()
 for ci,(p,model,n) in enumerate(cells):
  edges=edge_index(p);block=max(2,round(n**(1/3))) if model.startswith('ar_') else 1
  for rep in range(reps):
   seed=681203+ci*100000+rep;rng=np.random.default_rng(seed)
   if model=='ar_independent':
    xx,_=sample('independent',n+200,p,rng)
    for t in range(1,len(xx)):xx[t]=.4*xx[t-1]+np.sqrt(1-.4**2)*xx[t]
    x=xx[200:];corr=np.eye(p)
   else:x,corr=sample(model,n,p,rng)
   observed,boot=bootstrap_covariances(x,B,block,rng)
   r=correlation_edges(observed,edges);br=correlation_edges(boot,edges)
   ans=project_order(bootstrap_order(r,br,p),p)
   lower,upper,q,se=calibrated_band(r,br);oldlo,oldhi=degree_envelopes(lower,upper,edges,p);oldrl,oldrh=rank_envelopes(oldlo,oldhi)
   truth=np.arctanh(corr[edges[:,0],edges[:,1]]);centered=truth-truth.mean()
   a,b=ans['comparison_a'],ans['comparison_b'];contrasts=truth[a]-truth[b]
   contrast_cover=np.all(ans['contrast_lower']<=contrasts)&np.all(ans['contrast_upper']>=contrasts)
   false=np.any(ans['relation'] & (truth[:,None]<=truth[None,:]+1e-12))
   null=model in ['independent','ar_independent']
   if null:tdlo=np.ones(p,int);tdhi=np.full(p,p-1,int);trlo=np.ones(p,int);trhi=np.full(p,p,int)
   else:
    _,tdlo=max_tree(truth,edges,p);tdhi=tdlo.copy();trlo=1+(tdlo[None,:]>tdlo[:,None]).sum(axis=1);trhi=(tdlo[None,:]>=tdlo[:,None]).sum(axis=1)
   for method,dlo,dhi,rl,rh in [('bootstrap_order',ans['degree_lower'],ans['degree_upper'],ans['rank_lower'],ans['rank_upper']),('bootstrap_box',oldlo,oldhi,oldrl,oldrh)]:
    rows.append(dict(p=p,model=model,n=n,rep=rep,seed=seed,block=block,method=method,
     input_coverage=float(contrast_cover) if method=='bootstrap_order' else float(np.all(lower<=centered)&np.all(upper>=centered)),
     order_fwer=float(false) if method=='bootstrap_order' else np.nan,
     degree_coverage=float(np.all(dlo<=tdlo)&np.all(dhi>=tdhi)),rank_coverage=float(np.all(rl<=trlo)&np.all(rh>=trhi)),
     mean_degree_width=float(np.mean(dhi-dlo)),mean_rank_width=float(np.mean(rh-rl)),
     top1_candidates=int((rl==1).sum()),true_hub_certified=float(rh[0]==1) if not null else np.nan,
     forced_edges=int(ans['forced'].sum()) if method=='bootstrap_order' else np.nan,
     allowed_edges=int(ans['allowed'].sum()) if method=='bootstrap_order' else np.nan,
     order_count=ans['raw_order_count'] if method=='bootstrap_order' else np.nan))
   if (rep+1)%20==0:
    pd.DataFrame(rows).to_csv(DEST/'bootstrap_replications.csv',index=False)
    print(f'bootstrap p={p} {model} n={n}: {rep+1}/{reps}; {time.time()-start:.1f}s',flush=True)
  df=pd.DataFrame(rows);summary=df.groupby(['p','model','n','block','method']).mean(numeric_only=True).reset_index()
  summary['repetitions']=reps;summary.to_csv(DEST/'bootstrap_summary.csv',index=False)
 print(summary.to_string(index=False),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--repetitions',type=int,default=100);a.add_argument('--bootstrap',type=int,default=499);args=a.parse_args();run(args.repetitions,args.bootstrap)
