"""Practical Pearson row-bootstrap edge/hub frequencies on matched MC samples.

These Gaussian models have the same population Kendall and Pearson optimal
trees. Frequency >= .95 is evaluated as a heuristic, not a confidence claim.
"""
from pathlib import Path
import sys,json,time,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'statistical_papers'))
from hub_inference import max_tree
from run_simulations import sample
from comparators import resampled_pearson,edge_index
from run_comparative_simulations import design

DEST=Path(__file__).parent/'results'/'frequency_reference'

def worker(job):
    ci,cell,rep,config=job;p,n,model=cell['p'],cell['n'],cell['model'];seed=config['seed']+ci*100000+rep
    x,corr=sample(model,n,p,np.random.default_rng(seed));edges=edge_index(p)
    weights=np.corrcoef(x,rowvar=False)[edges[:,0],edges[:,1]]
    point,degree=max_tree(weights,edges,p);anchor=int(np.argmax(degree));freq=np.zeros(len(edges));unique=np.zeros(p)
    draws=resampled_pearson(x,B=499,seed=seed+70000000)
    for row in draws:
        chosen,d=max_tree(row,edges,p);freq[chosen]+=1;top=d==d.max()
        if top.sum()==1:unique+=top
    freq/=499;unique/=499;null=model=='independent'
    if null:necessary=np.zeros(len(edges),bool)
    else:
        chosen,_=max_tree(corr[edges[:,0],edges[:,1]],edges,p);necessary=np.zeros(len(edges),bool);necessary[chosen]=True
    declared=unique[anchor]>=.95
    return ci,rep,dict(p=p,n=n,model=model,rep=rep,seed=seed,method='bootstrap_frequency_095',
        B=499,anchor=anchor,observed_unique_hub=int((degree==degree.max()).sum()==1),
        anchor_unique_hub_frequency=float(unique[anchor]),true_hub_certified=int(declared and anchor==0) if not null else np.nan,
        false_hub_declaration=int(declared and (null or anchor!=0)),
        false_forced_edge=int(np.any((freq>=.95)&~necessary)),
        high_frequency_edges=int((freq>=.95).sum()),mean_population_edge_frequency=float(np.mean(freq[necessary])) if not null else np.nan)


def run(workers=2):
    DEST.mkdir(parents=True,exist_ok=True);config=design();config['reference']='Pearson IID row-bootstrap; B499; threshold .95, heuristic'
    (DEST/'design.json').write_text(json.dumps(config,indent=2));jobs=[]
    for ci,c in enumerate(config['cells']):
        if c['p']==12 and c['model'] in ['gaussian_star','near_tie','independent'] and c['n'] in [1008,4096]:
            jobs.extend((ci,c,r,config) for r in range(c['repetitions']) if not (DEST/f'cell{ci:02d}_{r:04d}.json').exists())
    start=time.time();done=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(worker,j) for j in jobs]):
            ci,rep,row=future.result();(DEST/f'cell{ci:02d}_{rep:04d}.json').write_text(json.dumps(row));done+=1
            if done%100==0:print('frequency reference',done,len(jobs),time.time()-start,flush=True)
    df=pd.DataFrame([json.loads(p.read_text()) for p in sorted(DEST.glob('cell*.json'))]);df.to_csv(DEST/'replications.csv',index=False)
    group=['p','n','model','method'];summary=df.groupby(group).mean(numeric_only=True).reset_index()
    summary=summary.merge(df.groupby(group).size().rename('repetitions').reset_index(),on=group);summary.to_csv(DEST/'summary.csv',index=False)
    print('COMPLETED frequency',len(df),time.time()-start,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=2);args=a.parse_args();run(args.workers)
