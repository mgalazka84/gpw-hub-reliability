"""All pooled/stratified UCI morphometric applications, with exact tie handling.

Primary illustration: Abalone, all eight numeric recorded measurements.
Secondary: all three sex groups, all seven Dry Bean varieties, and pooled
Dry Bean as a heterogeneity diagnostic. IID within each population is an
explicit working assumption, not established by a repository description.
"""
from pathlib import Path
import sys,json,time,argparse,hashlib,gzip
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'statistical_papers'))
from hub_inference import max_tree
from comparators import *
from order_inference import project_order,fast_kendall_band,_closure
from run_comparative_simulations import certify_lp

ROOT=Path(__file__).parent;DATA=ROOT/'data'/'cross_section';DEST=ROOT/'results'/'cross_section'
ABALONE_NAMES=['Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']

def datasets():
    a=pd.read_csv(DATA/'abalone.data',header=None);b=pd.read_excel(DATA/'Dry_Bean_Dataset.xlsx')
    out=[('Abalone','pooled',a.iloc[:,1:].to_numpy(float),ABALONE_NAMES)]
    out.extend(('Abalone',s,a.loc[a.iloc[:,0]==s].iloc[:,1:].to_numpy(float),ABALONE_NAMES) for s in ['F','I','M'])
    out.extend(('Dry Bean',s,b.loc[b.Class==s].iloc[:,:-1].to_numpy(float),list(b.columns[:-1])) for s in sorted(b.Class.unique()))
    out.append(('Dry Bean','pooled',b.iloc[:,:-1].to_numpy(float),list(b.columns[:-1])))
    return out


def worker(job):
    ci,entry,config=job;dataset,stratum,x,names=entry;seed=config['seed']+ci*100000
    n,p=x.shape;x=x[np.random.default_rng(seed).permutation(n)];edges=edge_index(p)
    tau,phi=kendall_projection(x);cov=covariance_of_mean(phi)
    draws=gaussian_draws(phi,config['multiplier_repetitions'],seed+30000000)
    chosen,degree=max_tree(tau,edges,p);pearson=np.corrcoef(x,rowvar=False)[edges[:,0],edges[:,1]]
    pearson_tree,_=max_tree(pearson,edges,p);overlap=len(set(chosen)&set(pearson_tree))
    B=config['row_bootstrap_repetitions'];rng=np.random.default_rng(seed+50000000)
    frequencies=np.zeros(len(edges));top_membership=np.zeros(p);unique_top=np.zeros(p)
    for b in range(B):
        bx=x[rng.integers(n,size=n)];btau=fast_kendall_band(bx)[0];tree,bd=max_tree(btau,edges,p)
        frequencies[tree]+=1;top=bd==bd.max();top_membership+=top
        if top.sum()==1:unique_top+=top
    frequencies/=B;top_membership/=B;unique_top/=B
    rows=[];nodes=[];certificate_records=[]
    for family in ['local','all']:
        exact=kendall_order(x,family=family,rng=np.random.default_rng(seed+10000000))
        merged=merged_pair_order(x,seed=seed+20000000,family=family)
        direct,box=multiplier_orders(tau,phi,p,draws=draws,covariance=cov,family=family)
        hoeff=hoeffding_difference_order(x,tau,family=family)
        a,b=comparisons(edges,p,family);epsilon=np.sqrt(2*np.log(2*len(edges)/.05)/(n//2))
        L=np.maximum(-1,tau-epsilon);U=np.minimum(1,tau+epsilon)
        rel=np.zeros((len(edges),len(edges)),bool)
        rel[a[L[a]>U[b]],b[L[a]>U[b]]]=True;rel[b[L[b]>U[a]],a[L[b]>U[a]]]=True
        hbox=dict(score=tau,edges=edges,relation=_closure(rel,tau),raw_order_count=int(rel.sum()))
        for method,raw in [('paired_exact',exact),('merged_8',merged),('u_hoeffding',hoeff),
                           ('hoeffding_box',hbox),('multiplier_contrast',direct),('multiplier_box',box)]:
            t=time.perf_counter();minor=project_order(raw,p);ans=certify_lp(minor,p)
            rows.append(dict(dataset=dataset,stratum=stratum,n=n,p=p,seed=seed,method=method,family=family,
                raw_orders=raw['raw_order_count'],closed_orders=int(raw['relation'].sum()),
                forced_edges=int(ans['forced'].sum()),allowed_edges=int(ans['allowed'].sum()),
                minor_degree_width=float(np.mean(minor['degree_upper']-minor['degree_lower'])),
                lp_degree_width=float(np.mean(ans['degree_upper']-ans['degree_lower'])),
                mean_rank_width=float(np.mean(ans['rank_upper']-ans['rank_lower'])),
                exact_degrees=int((ans['degree_lower']==ans['degree_upper']).sum()),
                certified_hubs=int((ans['rank_upper']==1).sum()),
                excluded_hubs=int((ans['rank_lower']>1).sum()),
                path_constraints=ans.get('path_constraints',0),lp_solves=ans['lp_solves'],
                verified_certificates=ans.get('python_verified_certificates',0),
                projection_seconds=time.perf_counter()-t,pearson_kendall_shared_edges=overlap,
                alpha=.05,omitted_degenerate_contrasts=raw.get('omitted_degenerate_contrasts',0)))
            for i,name in enumerate(names):
                nodes.append(dict(dataset=dataset,stratum=stratum,n=n,p=p,method=method,family=family,vertex=i,name=name,
                    sample_degree=int(degree[i]),degree_lower=int(ans['degree_lower'][i]),degree_upper=int(ans['degree_upper'][i]),
                    rank_lower=int(ans['rank_lower'][i]),rank_upper=int(ans['rank_upper'][i]),
                    bootstrap_top_membership=float(top_membership[i]),bootstrap_unique_hub_frequency=float(unique_top[i])))
            improvements=[r for r in ans['lp_diagnostics'] if r['certificate'] is not None]
            if improvements:certificate_records.append(dict(dataset=dataset,stratum=stratum,method=method,family=family,p=p,
                relation_rows=[np.flatnonzero(r).tolist() for r in raw['relation']],diagnostics=improvements))
    edge_rows=[dict(dataset=dataset,stratum=stratum,edge=e,a=names[i],b=names[j],kendall_tau_a=float(tau[e]),
        pearson=float(pearson[e]),kendall_tree=int(e in chosen),pearson_tree=int(e in pearson_tree),
        bootstrap_frequency=float(frequencies[e])) for e,(i,j) in enumerate(edges)]
    return ci,rows,nodes,edge_rows,certificate_records


def run(workers=2):
    DEST.mkdir(parents=True,exist_ok=True)
    config=dict(version='0.5',alpha=.05,seed=71005019,multiplier_repetitions=999,row_bootstrap_repetitions=499,
        primary='Abalone pooled: all numeric measurements, including Rings',
        secondary='All Abalone sex strata and all Dry Bean variety strata; pooled Dry Bean heterogeneity diagnostic',
        scope='Separate nominal analysis per dataset/method/family; no joint coverage across analyses',
        row_order='One independent random permutation before predetermined pairing; no outcome-based selection',
        assumptions='Independent measurements sampled from one distribution within each analysed population; not verified by repository metadata',
        interpretation='Morphometric association; no causal or conditional-independence interpretation',
        preprocessing='Original values; no outlier removal, deduplication, imputation or fitted residualization',
        data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob('*') if p.is_file()})
    (DEST/'design.json').write_text(json.dumps(config,indent=2));start=time.time()
    entries=datasets();jobs=[(ci,e,config) for ci,e in enumerate(entries) if not (DEST/f'dataset{ci:02d}.json').exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(worker,job) for job in jobs]):
            ci,rows,nodes,edges,certificates=future.result()
            (DEST/f'dataset{ci:02d}.json').write_text(json.dumps(dict(rows=rows,nodes=nodes,edges=edges,certificates=certificates)))
            print('cross-section',ci,entries[ci][0:2],time.time()-start,flush=True)
    results=[json.loads(p.read_text()) for p in sorted(DEST.glob('dataset*.json'))]
    for name,key in [('summary','rows'),('nodes','nodes'),('edges','edges')]:
        pd.DataFrame([r for result in results for r in result[key]]).to_csv(DEST/(name+'.csv'),index=False)
    with gzip.open(DEST/'certificates.jsonl.gz','wt') as f:
        for result in results:
            for record in result['certificates']:f.write(json.dumps(record)+'\n')
    print('COMPLETED cross-section',len(results),'datasets',time.time()-start,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=2);args=a.parse_args();run(args.workers)
