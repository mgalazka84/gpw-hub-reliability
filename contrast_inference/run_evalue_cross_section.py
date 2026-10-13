"""E8/E64 applications on the identical primary/secondary measurement grid."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,time,argparse
import numpy as np
import pandas as pd
from run_cross_section import datasets,DEST as CROSS
from evalue_order import evalue_orders
from order_inference import project_order,fast_kendall_band,edge_index
from run_comparative_simulations import certify_lp

DEST=Path(__file__).parent/'results'/'evalue'/'cross_section'

def worker(job):
    ci,entry=job;dataset,stratum,x,names=entry;seed=71005019+ci*100000;n,p=x.shape
    x=x[np.random.default_rng(seed).permutation(n)];tau=fast_kendall_band(x)[0]
    base=json.loads((CROSS/f'dataset{ci:02d}.json').read_text())
    old_nodes={r['vertex']:r for r in base['nodes'] if r['method']=='paired_exact' and r['family']=='local'}
    rows=[];nodes=[];certificates=[]
    for family in ['local','all']:
        for R,raw in evalue_orders(x,seed=seed+20000000,family=family,tau=tau).items():
            method=f'evalue_{R}';t=time.perf_counter();minor=project_order(raw,p);ans=certify_lp(minor,p)
            rows.append(dict(dataset=dataset,stratum=stratum,n=n,p=p,seed=seed,method=method,family=family,
                raw_orders=raw['raw_order_count'],closed_orders=int(raw['relation'].sum()),forced_edges=int(ans['forced'].sum()),
                allowed_edges=int(ans['allowed'].sum()),minor_degree_width=float(np.mean(minor['degree_upper']-minor['degree_lower'])),
                lp_degree_width=float(np.mean(ans['degree_upper']-ans['degree_lower'])),mean_rank_width=float(np.mean(ans['rank_upper']-ans['rank_lower'])),
                exact_degrees=int((ans['degree_upper']==ans['degree_lower']).sum()),certified_hubs=int((ans['rank_upper']==1).sum()),
                excluded_hubs=int((ans['rank_lower']>1).sum()),path_constraints=ans.get('path_constraints',0),lp_solves=ans['lp_solves'],
                verified_certificates=ans.get('python_verified_certificates',0),projection_seconds=time.perf_counter()-t,
                pearson_kendall_shared_edges=base['rows'][0]['pearson_kendall_shared_edges'],alpha=.05,omitted_degenerate_contrasts=0))
            for i,name in enumerate(names):
                nodes.append({**old_nodes[i],'method':method,'family':family,
                    'degree_lower':int(ans['degree_lower'][i]),'degree_upper':int(ans['degree_upper'][i]),
                    'rank_lower':int(ans['rank_lower'][i]),'rank_upper':int(ans['rank_upper'][i])})
            improvements=[r for r in ans['lp_diagnostics'] if r['certificate'] is not None]
            if improvements:certificates.append(dict(dataset=dataset,stratum=stratum,method=method,family=family,p=p,
                relation_rows=[np.flatnonzero(r).tolist() for r in raw['relation']],diagnostics=improvements))
    return ci,dict(rows=rows,nodes=nodes,certificates=certificates)

def run(workers=2):
    DEST.mkdir(parents=True,exist_ok=True);entries=datasets();start=time.time()
    jobs=[(ci,e) for ci,e in enumerate(entries) if not (DEST/f'dataset{ci:02d}.json').exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(worker,j) for j in jobs]):
            ci,out=future.result();(DEST/f'dataset{ci:02d}.json').write_text(json.dumps(out));print('E cross',ci,time.time()-start,flush=True)
    print('COMPLETED E cross',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=2);args=a.parse_args();run(args.workers)
