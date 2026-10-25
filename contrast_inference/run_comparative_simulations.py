"""Review-driven matched test/projection/family ablations; resumable by dataset.

All new null cells have 1000 datasets. Non-null cells have 100 or 200.
LP is evaluated for the first 100 p=12 datasets per cell with all degree-pair
objectives. Empty order relations and a forced unique tree have known sharp
endpoints and skip redundant LP calls. No population information chooses tests.
"""
from pathlib import Path
import sys,json,time,argparse,os,warnings
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'statistical_papers'))
from hub_inference import max_tree
from run_simulations import sample
from order_inference import kendall_order,project_order,edge_index,_closure
from comparators import *
from path_projection import refine_degrees,_problem,path_constraints

ROOT=Path(__file__).parent;DEST=ROOT/'results'/'comparative'

def design():
    cells=[]
    for p in [12,52]:
        for model in ['gaussian_star','t5_star','near_tie']:
            for n in ([252,1008,4096] if p==12 else [252,4096]):
                cells.append(dict(p=p,model=model,n=n,repetitions=200 if p==12 else 100))
    for n in [252,1008,4096]:cells.append(dict(p=12,model='independent',n=n,repetitions=1000))
    cells.append(dict(p=12,model='discrete_independent',n=1008,repetitions=1000))
    cells.append(dict(p=52,model='independent',n=1008,repetitions=1000))
    for model in ['chain','two_hubs']:
        for n in [1008,4096]:cells.append(dict(p=12,model=model,n=n,repetitions=100))
    return dict(version='0.5',alpha=.05,seed=1592801,multiplier_repetitions=999,
        matchings=8,lp_repetitions=100,cells=cells,
        methods=['paired_exact','merged_8','u_hoeffding','hoeffding_box','multiplier_contrast','multiplier_box'],
        families='local everywhere; all-edge family ablation p=12 first 100 datasets per cell',
        projections=['marginal','pair','conservation','joint','path'],
        status='review-driven retrospective development; all specified cells retained')


def verify_dual_python(cert,c,A,b,E,eb,lower,upper):
    """Independent Python integers; no sparse int64 products or solver status."""
    Q=cert['denominator'];res=[int(v)*Q for v in c];value=0
    for idx,y in zip(cert['y_indices'],cert['y_numerators']):
        assert y<=0;value+=int(b[idx])*int(y)
        for k in range(A.indptr[idx],A.indptr[idx+1]):res[int(A.indices[k])]-=int(A.data[k])*int(y)
    for idx,z in zip(cert['z_indices'],cert['z_numerators']):
        value+=int(eb[idx])*int(z)
        for k in range(E.indptr[idx],E.indptr[idx+1]):res[int(E.indices[k])]-=int(E.data[k])*int(z)
    value+=sum(v*int(lower[i] if v>=0 else upper[i]) for i,v in enumerate(res))
    assert value==cert['numerator']


def certify_lp(ans,p,degree_pairs=True):
    """Verify path-LP bounds; omit degree-pair objectives for the lean stage.

    The default preserves the original full projection. Both variants use the
    same constraints and degree objectives. Omitting pair objectives therefore
    preserves degree bounds and can only weaken the rank information.
    """
    if not ans['relation'].any() or int(ans['forced'].sum())==p-1:
        return {**ans,'path_constraints':0,'lp_diagnostics':[],'lp_solves':0}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');out=refine_degrees(ans,p,pairs=[(i,j) for i in range(p) for j in range(p) if i!=j] if degree_pairs else None)
    paths=path_constraints(ans['relation'],ans['edges'],p,ans['allowed'],ans['forced'])
    A,b,E,eb,L,U=_problem(ans['edges'],p,ans['allowed'],ans['forced'],paths);m=len(ans['edges'])
    count=0
    for record in out['lp_diagnostics']:
        cert=record['certificate']
        if cert is None:continue
        c=np.zeros(3*m,dtype=int);i=record['vertex'];c[:m]=np.any(ans['edges']==i,axis=1)
        if 'other' in record:c[:m]-=np.any(ans['edges']==record['other'],axis=1).astype(int)
        else:c*=record['direction']
        verify_dual_python(cert,c,A,b,E,eb,L,U);count+=1
    return {**out,'lp_solves':len(out['lp_diagnostics']),'python_verified_certificates':count}


def worker(job):
    ci,cell,rep,config=job;p,n,model=cell['p'],cell['n'],cell['model']
    seed=config['seed']+ci*100000+rep;rng=np.random.default_rng(seed)
    if model=='discrete_independent':x=rng.integers(-1,2,size=(n,p)).astype(float);corr=np.eye(p)
    else:x,corr=sample(model,n,p,rng)
    edges=edge_index(p);truth=2/np.pi*np.arcsin(corr[edges[:,0],edges[:,1]])
    null=model in ['independent','discrete_independent']
    if null:dlo=np.ones(p,int);dhi=np.full(p,p-1,int);rlo=np.ones(p,int);rhi=np.full(p,p,int)
    else:
        _,dlo=max_tree(truth,edges,p);dhi=dlo.copy()
        rlo=1+(dlo[None,:]>dlo[:,None]).sum(axis=1)
        rhi=(dlo[None,:]>=dlo[:,None]).sum(axis=1)
    unique_hub=(not null) and np.sum(dlo==dlo.max())==1
    t=time.perf_counter();tau,phi=kendall_projection(x);cov=covariance_of_mean(phi)
    draws=gaussian_draws(phi,config['multiplier_repetitions'],seed+31000000)
    precompute=time.perf_counter()-t;rows=[]
    families=config.get('family_override', ['local','all'] if p==12 and rep<100 else ['local'])
    for family in families:
        orders=[]
        t=time.perf_counter();ans=kendall_order(x,rng=np.random.default_rng(seed+40000000),family=family)
        orders.append(('paired_exact',ans,time.perf_counter()-t))
        t=time.perf_counter();ans=merged_pair_order(x,seed=seed+41000000,family=family)
        orders.append(('merged_8',ans,time.perf_counter()-t))
        t=time.perf_counter();ans=hoeffding_difference_order(x,tau,family=family)
        orders.append(('u_hoeffding',ans,time.perf_counter()-t+precompute))
        a,b=comparisons(edges,p,family);epsilon=np.sqrt(2*np.log(2*len(edges)/.05)/(n//2))
        L=np.maximum(-1,tau-epsilon);U=np.minimum(1,tau+epsilon)
        relation=np.zeros((len(edges),len(edges)),bool)
        relation[a[L[a]>U[b]],b[L[a]>U[b]]]=True;relation[b[L[b]>U[a]],a[L[b]>U[a]]]=True
        raw=int(relation.sum());relation=_closure(relation,tau)
        orders.append(('hoeffding_box',dict(score=tau,edges=edges,relation=relation,
            raw_order_count=raw,closed_order_count=int(relation.sum())),precompute))
        t=time.perf_counter();direct,box=multiplier_orders(tau,phi,p,draws=draws,covariance=cov,family=family)
        elapsed=time.perf_counter()-t+precompute
        orders.extend([('multiplier_contrast',direct,elapsed),('multiplier_box',box,elapsed)])
        for method,raw,test_time in orders:
            t=time.perf_counter();ans=project_order(raw,p);project_time=time.perf_counter()-t
            false=float(np.any(raw['relation'] & (truth[:,None]<=truth[None,:]+1e-12)))
            inputs=np.nan
            if 'contrast_lower' in raw:
                aa,bb=raw['comparison_a'],raw['comparison_b'];delta=truth[aa]-truth[bb]
                inputs=float(np.all(raw['contrast_lower']<=delta)&np.all(raw['contrast_upper']>=delta))
            if 'edge_lower' in raw:inputs=float(np.all(raw['edge_lower']<=truth)&np.all(raw['edge_upper']>=truth))
            stages=[('marginal',ans['marginal_rank_lower'],ans['marginal_rank_upper']),
                ('pair',ans['pair_rank_lower'],ans['pair_rank_upper']),
                ('conservation',ans['conservation_rank_lower'],ans['conservation_rank_upper']),
                ('joint',ans['rank_lower'],ans['rank_upper'])]
            if p==12 and rep<config['lp_repetitions']:
                t=time.perf_counter();lp=certify_lp(ans,p);lp_time=time.perf_counter()-t
                stages.append(('path',lp['rank_lower'],lp['rank_upper']))
            for stage,rl,rh in stages:
                target=lp if stage=='path' else ans;lo,hi=target['degree_lower'],target['degree_upper']
                rows.append(dict(p=p,n=n,model=model,rep=rep,seed=seed,method=method,family=family,stage=stage,
                    order_fwer=false,input_coverage=inputs,
                    degree_coverage=float(np.all(lo<=dlo)&np.all(hi>=dhi)),
                    rank_coverage=float(np.all(rl<=rlo)&np.all(rh>=rhi)),
                    mean_degree_width=float(np.mean(hi-lo)),mean_rank_width=float(np.mean(rh-rl)),
                    true_hub_certified=float(rh[np.argmax(dlo)]==1) if unique_hub else np.nan,
                    top1_certified=int((rh==1).sum()),top1_candidates=int((rl==1).sum()),
                    exact_degrees=int((hi==lo).sum()),forced_edges=int(target['forced'].sum()),
                    raw_orders=raw['raw_order_count'],closed_orders=raw['closed_order_count'],
                    test_seconds=test_time,project_seconds=project_time,
                    lp_seconds=lp_time if stage=='path' else 0,
                    lp_solves=target.get('lp_solves',0) if stage=='path' else 0,
                    verified_certificates=target.get('python_verified_certificates',0) if stage=='path' else 0,
                    acyclic_filter_deletions=raw.get('acyclic_filter_deletions',0)))
    return ci,rep,rows


def run(workers=6,pilot=False):
    DEST.mkdir(parents=True,exist_ok=True);config=design()
    if pilot:
        config['seed']=991582901
        config['cells']=[dict(p=12,n=1008,model='gaussian_star',repetitions=2),dict(p=52,n=1008,model='independent',repetitions=2)]
    dest=DEST/('pilot' if pilot else 'production');dest.mkdir(exist_ok=True)
    (dest/'design.json').write_text(json.dumps(config,indent=2));jobs=[]
    for ci,cell in enumerate(config['cells']):
        for rep in range(cell['repetitions']):
            if not (dest/f'cell{ci:02d}_{rep:04d}.json').exists():jobs.append((ci,cell,rep,config))
    start=time.time();completed=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(worker,job) for job in jobs]
        for future in as_completed(futures):
            ci,rep,rows=future.result();(dest/f'cell{ci:02d}_{rep:04d}.json').write_text(json.dumps(rows))
            completed+=1
            if completed%50==0:print(f'comparative {completed}/{len(jobs)}, {time.time()-start:.1f}s',flush=True)
    rows=[r for f in sorted(dest.glob('cell*.json')) for r in json.loads(f.read_text())]
    df=pd.DataFrame(rows);df.to_csv(DEST/('pilot_replications.csv' if pilot else 'replications.csv'),index=False)
    group=['p','model','n','method','family','stage'];summary=df.groupby(group).mean(numeric_only=True).reset_index()
    counts=df.groupby(group).size().rename('repetitions').reset_index();summary=summary.merge(counts,on=group)
    summary.to_csv(DEST/('pilot_summary.csv' if pilot else 'summary.csv'),index=False)
    print('COMPLETED',len(rows),'records',time.time()-start,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=6);parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args();run(args.workers,args.pilot)
