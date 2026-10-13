"""Held-out evaluation of exact comparison inference against Hoeffding boxes."""
from pathlib import Path
import argparse,sys,json,time
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'statistical_papers'))
from hub_inference import kendall_finite_band,degree_envelopes,rank_envelopes,max_tree
from run_simulations import sample
from order_inference import kendall_order,project_order,edge_index,fast_kendall_band

ROOT=Path(__file__).parent;DEST=ROOT/'results'

def run(reps=200):
 cells=[]
 for p in [12,52]:
  for model in ['gaussian_star','t5_star','near_tie','independent']:
   for n in ([252,1008,4096] if p==12 else [252,4096]):cells.append((p,model,n))
 cells.extend([(12,'discrete_independent',252),(12,'discrete_independent',1008)])
 config=dict(repetitions=reps,cells=cells,alpha=.05,seed=937410,
  design='held-out development validation; pilot seeds excluded',
  comparisons='both directions of all pairs of edges sharing a vertex',
  exact_scope='IID rows, predetermined disjoint observation pairs, independent uniform tie breakers',
  baseline='full-sample Kendall tau-a Hoeffding simultaneous rectangular band')
 DEST.mkdir(exist_ok=True);(DEST/'exact_design.json').write_text(json.dumps(config,indent=2))
 rows=pd.read_csv(DEST/'exact_replications.csv').to_dict('records') if (DEST/'exact_replications.csv').exists() else []
 done={(r['p'],r['model'],r['n'],r['rep']) for r in rows if r['method']=='exact_order'}
 start=time.time()
 for ci,(p,model,n) in enumerate(cells):
  edges=edge_index(p)
  for rep in range(reps):
   if (p,model,n,rep) in done:continue
   seed=937410+ci*100000+rep;rng=np.random.default_rng(seed)
   if model=='discrete_independent':x=rng.integers(-1,2,size=(n,p)).astype(float);corr=np.eye(p)
   else:x,corr=sample(model,n,p,rng)
   truth=(2/np.pi)*np.arcsin(corr[edges[:,0],edges[:,1]])
   ans=project_order(kendall_order(x,rng=np.random.default_rng(seed+40000000)),p)
   tau,L,U,epsilon=fast_kendall_band(x)
   oldlo,oldhi=degree_envelopes(L,U,edges,p);oldrl,oldrh=rank_envelopes(oldlo,oldhi)
   null=model in ['independent','discrete_independent']
   if null:tdlo=np.ones(p,int);tdhi=np.full(p,p-1,int);trlo=np.ones(p,int);trhi=np.full(p,p,int)
   else:
    _,tdlo=max_tree(truth,edges,p);tdhi=tdlo.copy()
    trlo=1+(tdlo[None,:]>tdlo[:,None]).sum(axis=1);trhi=(tdlo[None,:]>=tdlo[:,None]).sum(axis=1)
   false=np.any(ans['relation'] & (truth[:,None]<=truth[None,:]+1e-12))
   for method,dlo,dhi,rl,rh in [('exact_order',ans['degree_lower'],ans['degree_upper'],ans['rank_lower'],ans['rank_upper']),('hoeffding_box',oldlo,oldhi,oldrl,oldrh)]:
    rows.append(dict(p=p,model=model,n=n,rep=rep,seed=seed,method=method,
     degree_coverage=float(np.all(dlo<=tdlo)&np.all(dhi>=tdhi)),
     rank_coverage=float(np.all(rl<=trlo)&np.all(rh>=trhi)),
     order_fwer=float(false) if method=='exact_order' else np.nan,
     mean_degree_width=float(np.mean(dhi-dlo)),mean_rank_width=float(np.mean(rh-rl)),
     top1_candidates=int((rl==1).sum()),top1_certified=int((rh==1).sum()),
     true_hub_certified=float(rh[0]==1) if not null else np.nan,
     forced_edges=int(ans['forced'].sum()) if method=='exact_order' else np.nan,
     order_count=ans['raw_order_count'] if method=='exact_order' else np.nan,
     epsilon=epsilon if method=='hoeffding_box' else np.nan))
   if (rep+1)%25==0:
    pd.DataFrame(rows).to_csv(DEST/'exact_replications.csv',index=False)
    print(f'exact p={p} {model} n={n}: {rep+1}/{reps}; {time.time()-start:.1f}s',flush=True)
  df=pd.DataFrame(rows);summary=df.groupby(['p','model','n','method']).mean(numeric_only=True).reset_index()
  summary['repetitions']=reps;summary.to_csv(DEST/'exact_summary.csv',index=False)
 print(summary.to_string(index=False),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--repetitions',type=int,default=200);args=a.parse_args();run(args.repetitions)
