"""Finite-sample Kendall band: IID validation and conservatism assessment."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from run_simulations import sample
from hub_inference import *

def main():
    root=Path(__file__).parent/'results';rows=[];p=12;start=time.time()
    designs=[('independent',252),('near_tie',252),('gaussian_star',1008),('gaussian_star',4096),('t5_star',1008),('t5_star',4096)]
    for mi,(model,n) in enumerate(designs):
        for rep in range(500):
            seed=301774+mi*100000+rep
            x,rho=sample(model,n,p,np.random.default_rng(seed))
            tau,lower,upper,eps=kendall_finite_band(x)
            edges=edge_index(p);truth=2/np.pi*np.arcsin(rho[edges[:,0],edges[:,1]])
            lo,hi=degree_envelopes(lower,upper,edges,p);rl,ru=rank_envelopes(lo,hi)
            tl,tu=degree_envelopes(truth,truth,edges,p)
            rows.append(dict(model=model,n=n,rep=rep,seed=seed,epsilon=eps,
                band_coverage=float(np.all(lower<=truth)&np.all(upper>=truth)),
                degree_coverage=float(np.all(lo<=tl)&np.all(hi>=tu)),
                top1_candidate_size=int((rl<=1).sum()),top1_certified=int((ru<=1).sum()),
                mean_degree_width=float((hi-lo).mean())))
        pd.DataFrame(rows).to_csv(root/'kendall_replications.csv',index=False)
        print(f'Kendall: {model}, n={n}; {time.time()-start:.1f}s',flush=True)
    df=pd.DataFrame(rows)
    df.groupby(['model','n'])[['epsilon','band_coverage','degree_coverage','top1_candidate_size','top1_certified','mean_degree_width']].mean().reset_index().to_csv(root/'kendall_summary.csv',index=False)
    (root/'kendall_design.json').write_text(json.dumps(dict(repetitions=500,p=p,alpha=.05,designs=designs,iid_only=True),indent=2))

if __name__=='__main__':main()
