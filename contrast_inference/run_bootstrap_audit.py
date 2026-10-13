"""Paired IID null audit: resampling size, studentizer and bootstrap mechanism.

1000 independent data sets in every cell; B=499 is nested in B=1999.
Original fixed bootstrap-SD contrasts are compared with influence-SD row
bootstrap and Gaussian influence multipliers. The audit records false orders
without attributing failures to bootstrap methods as a class.
"""
from pathlib import Path
import json,time,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from comparators import *

ROOT=Path(__file__).parent;DEST=ROOT/'results'/'bootstrap_audit'

def design():
    return dict(version='0.5',alpha=.05,seed=38192101,repetitions=1000,B=[499,1999],
        cells=[dict(p=12,n=252),dict(p=52,n=252),dict(p=52,n=1008)],
        methods=['legacy_row_499','legacy_row_1999','influence_row_1999','influence_multiplier_1999'],
        family='local; two-sided maximum over all unique contrasts',
        estimand='Fisher-z Pearson edge differences',
        truth='IID Gaussian independence: all population contrasts are zero',
        status='review-driven paired audit; fixed common datasets and nested row resamples')


def worker(job):
    ci,cell,rep,config=job;p,n=cell['p'],cell['n'];seed=config['seed']+ci*100000+rep
    x=np.random.default_rng(seed).normal(size=(n,p));t=time.perf_counter()
    score,phi=pearson_projection(x);cov=covariance_of_mean(phi)
    boots=resampled_pearson(x,B=1999,seed=seed+10000000)
    multipliers=gaussian_draws(phi,B=1999,seed=seed+20000000)
    rows=[]
    for method,B,covariance,draws,centered in [
        ('legacy_row_499',499,None,boots[:499],False),
        ('legacy_row_1999',1999,None,boots,False),
        ('influence_row_1999',1999,cov,boots,False),
        ('influence_multiplier_1999',1999,cov,multipliers,True)]:
        ans=order_from_draws(score,draws,p,covariance=covariance,centered=centered)
        inputs=bool(np.all(ans['contrast_lower']<=0)&np.all(ans['contrast_upper']>=0))
        false=bool(ans['relation'].any())
        assert false==(not inputs)
        rows.append(dict(p=p,n=n,model='gaussian_independence',rep=rep,seed=seed,method=method,B=B,
            order_fwer=int(false),input_coverage=int(inputs),critical=ans['critical'],
            raw_orders=ans['raw_order_count'],seconds=time.perf_counter()-t))
    return ci,rep,rows


def run(workers=4,pilot=False):
    DEST.mkdir(parents=True,exist_ok=True);config=design()
    if pilot:config['repetitions']=1;config['seed']=938192101
    dest=DEST/('pilot' if pilot else 'production');dest.mkdir(exist_ok=True)
    (dest/'design.json').write_text(json.dumps(config,indent=2));jobs=[]
    for ci,cell in enumerate(config['cells']):
        for rep in range(config['repetitions']):
            if not (dest/f'cell{ci:02d}_{rep:04d}.json').exists():jobs.append((ci,cell,rep,config))
    start=time.time();completed=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(worker,job) for job in jobs]
        for future in as_completed(futures):
            ci,rep,rows=future.result();(dest/f'cell{ci:02d}_{rep:04d}.json').write_text(json.dumps(rows))
            completed+=1
            if completed%50==0:print(f'audit {completed}/{len(jobs)}, {time.time()-start:.1f}s',flush=True)
    rows=[r for f in sorted(dest.glob('cell*.json')) for r in json.loads(f.read_text())];df=pd.DataFrame(rows)
    df.to_csv(DEST/('pilot_replications.csv' if pilot else 'replications.csv'),index=False)
    group=['p','n','model','method','B'];summary=df.groupby(group).mean(numeric_only=True).reset_index()
    summary=summary.merge(df.groupby(group).size().rename('repetitions').reset_index(),on=group)
    summary.to_csv(DEST/('pilot_summary.csv' if pilot else 'summary.csv'),index=False)
    print('COMPLETED audit',len(rows),time.time()-start,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=4);a.add_argument('--pilot',action='store_true')
    args=a.parse_args();run(args.workers,args.pilot)
