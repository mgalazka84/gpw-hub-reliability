"""Evaluate E8/E64 on every matched production dataset and both family grids."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,time,argparse
import numpy as np
import pandas as pd
from run_comparative_simulations import design,sample,max_tree,certify_lp,edge_index,project_order
from order_inference import fast_kendall_band
from evalue_order import evalue_orders,LAMBDAS

ROOT=Path(__file__).parent;DEST=ROOT/'results'/'evalue'

def worker(job):
    ci,cell,rep,config=job;p,n,model=cell['p'],cell['n'],cell['model'];seed=config['seed']+ci*100000+rep
    rng=np.random.default_rng(seed)
    if model=='discrete_independent':x=rng.integers(-1,2,size=(n,p)).astype(float);corr=np.eye(p)
    else:x,corr=sample(model,n,p,rng)
    edges=edge_index(p);truth=2/np.pi*np.arcsin(corr[edges[:,0],edges[:,1]])
    null=model in ['independent','discrete_independent']
    if null:tdlo=np.ones(p,int);tdhi=np.full(p,p-1,int);trlo=np.ones(p,int);trhi=np.full(p,p,int)
    else:
        _,tdlo=max_tree(truth,edges,p);tdhi=tdlo.copy()
        trlo=1+(tdlo[None,:]>tdlo[:,None]).sum(axis=1);trhi=(tdlo[None,:]>=tdlo[:,None]).sum(axis=1)
    unique=(not null) and np.sum(tdlo==tdlo.max())==1
    tau=fast_kendall_band(x)[0];rows=[]
    families=['local','all'] if p==12 and (rep<100 or null) else ['local']
    for family in families:
        t=time.perf_counter();orders=evalue_orders(x,seed=seed+41000000,family=family,tau=tau)
        test_time=time.perf_counter()-t
        for R,raw in orders.items():
            method=f'evalue_{R}';t=time.perf_counter();ans=project_order(raw,p);project_time=time.perf_counter()-t
            false=float(np.any(raw['relation'] & (truth[:,None]<=truth[None,:]+1e-12)))
            stages=[('marginal',ans['marginal_rank_lower'],ans['marginal_rank_upper']),
                ('pair',ans['pair_rank_lower'],ans['pair_rank_upper']),
                ('conservation',ans['conservation_rank_lower'],ans['conservation_rank_upper']),
                ('joint',ans['rank_lower'],ans['rank_upper'])]
            if p==12 and rep<100:
                t=time.perf_counter();lp=certify_lp(ans,p);lp_time=time.perf_counter()-t
                stages.append(('path',lp['rank_lower'],lp['rank_upper']))
            for stage,rl,rh in stages:
                target=lp if stage=='path' else ans;lo,hi=target['degree_lower'],target['degree_upper']
                rows.append(dict(p=p,n=n,model=model,rep=rep,seed=seed,method=method,family=family,stage=stage,
                    order_fwer=false,input_coverage=np.nan,degree_coverage=float(np.all(lo<=tdlo)&np.all(hi>=tdhi)),
                    rank_coverage=float(np.all(rl<=trlo)&np.all(rh>=trhi)),mean_degree_width=float(np.mean(hi-lo)),
                    mean_rank_width=float(np.mean(rh-rl)),true_hub_certified=float(rh[np.argmax(tdlo)]==1) if unique else np.nan,
                    top1_certified=int((rh==1).sum()),top1_candidates=int((rl==1).sum()),exact_degrees=int((hi==lo).sum()),
                    forced_edges=int(target['forced'].sum()),raw_orders=raw['raw_order_count'],closed_orders=raw['closed_order_count'],
                    test_seconds=test_time,project_seconds=project_time,lp_seconds=lp_time if stage=='path' else 0,
                    lp_solves=target.get('lp_solves',0) if stage=='path' else 0,
                    verified_certificates=target.get('python_verified_certificates',0) if stage=='path' else 0,
                    acyclic_filter_deletions=raw.get('acyclic_filter_deletions',0)))
    return ci,rep,rows

def run(workers=8):
    config=design();config['evalue_extension']=dict(matchings=[8,64],fixed_lambdas=LAMBDAS.tolist(),
        additional_lambda='min(0.5,1/sqrt(floor(n/2))); fixed by sample size, ensures consistency',
        matching_seed_offset=41000000,nesting='E8 and E64 use the same first eight matchings as the M8 comparator',
        scope='Retrospective matched development comparison; every original dataset; all null family cells 1000 repetitions')
    dest=DEST/'production';dest.mkdir(parents=True,exist_ok=True);(dest/'design.json').write_text(json.dumps(config,indent=2));jobs=[]
    for ci,cell in enumerate(config['cells']):
        jobs.extend((ci,cell,r,config) for r in range(cell['repetitions']) if not (dest/f'cell{ci:02d}_{r:04d}.json').exists())
    start=time.time();completed=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(worker,j) for j in jobs]):
            ci,rep,rows=future.result();(dest/f'cell{ci:02d}_{rep:04d}.json').write_text(json.dumps(rows));completed+=1
            if completed%100==0:print('evalue',completed,len(jobs),time.time()-start,flush=True)
    rows=[r for p in sorted(dest.glob('cell*.json')) for r in json.loads(p.read_text())]
    df=pd.DataFrame(rows);df.to_csv(DEST/'replications.csv.gz',index=False)
    group=['p','model','n','method','family','stage'];s=df.groupby(group).mean(numeric_only=True).reset_index()
    s=s.merge(df.groupby(group).size().rename('repetitions').reset_index(),on=group);s.to_csv(DEST/'summary.csv',index=False)
    print('COMPLETED evalue',len(rows),time.time()-start,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=8);args=a.parse_args();run(args.workers)
