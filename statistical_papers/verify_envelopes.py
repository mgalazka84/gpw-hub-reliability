"""Independent exhaustive checks against every labelled tree on small graphs."""
from pathlib import Path
import itertools, json, time
import numpy as np
from hub_inference import edge_index, degree_envelopes, rank_envelopes, bootstrap_covariances, transform_cov

def enumerate_trees(p):
    edges=edge_index(p); lookup={tuple(e):j for j,e in enumerate(edges)}
    masks=[]; degrees=[]
    for code in itertools.product(range(p), repeat=p-2):
        d=np.bincount(code,minlength=p)+1; original=d.copy(); selected=[]
        for v in code:
            u=int(np.flatnonzero(d==1)[0]); selected.append(lookup[tuple(sorted((u,v)))])
            d[u]-=1;d[v]-=1
        u,v=np.flatnonzero(d==1);selected.append(lookup[(u,v)])
        mask=np.zeros(len(edges),int);mask[selected]=1
        masks.append(mask);degrees.append(original)
    return edges,np.array(masks),np.array(degrees)

def main():
    start=time.time();rng=np.random.default_rng(701217); count=0; witness_count=0
    for p,reps in [(3,200),(4,400),(5,400),(6,40)]:
        edges,trees,degrees=enumerate_trees(p)
        for rep in range(reps):
            lower=rng.integers(-3,4,len(edges));upper=lower+rng.integers(0,5,len(edges))
            # A tree is possible iff optimal at its own favourable corner.
            scenarios=np.where(trees,upper,lower)
            values=trees@scenarios.T
            own=np.sum(trees*scenarios,axis=1)
            possible=own==values.max(axis=0)
            lo,hi=degree_envelopes(lower,upper,edges,p)
            assert np.array_equal(lo,degrees[possible].min(axis=0))
            assert np.array_equal(hi,degrees[possible].max(axis=0))
            rl,ru=rank_envelopes(lo,hi)
            for d in degrees[possible]:
                exact_lo=1+(d[None,:]>d[:,None]).sum(axis=1)
                exact_hi=(d[None,:]>=d[:,None]).sum(axis=1)
                assert np.all(rl<=exact_lo) and np.all(ru>=exact_hi)
            count+=1;witness_count+=int(possible.sum())
    # Independently compare weighted moments with directly resampled/refitted data.
    x=rng.normal(size=(31,7));seed=173
    _,boot=bootstrap_covariances(x,20,4,np.random.default_rng(seed))
    rr=np.random.default_rng(seed);starts=rr.integers(31,size=(20,8))
    ix=((starts[...,None]+np.arange(4))%31).reshape(20,-1)[:,:31]
    errors=[]
    for b in range(20):
        z=x[ix[b]]
        errors.append(np.max(np.abs(boot[b]-np.cov(z,rowvar=False,bias=True))))
        design=np.column_stack([np.ones(31),z[:,-1]])
        residual=z[:,:-1]-design@np.linalg.lstsq(design,z[:,:-1],rcond=None)[0]
        assert np.allclose(transform_cov(boot[b],6,'factor_residual'),np.cov(residual,rowvar=False,bias=True))
    report=dict(interval_boxes=count,possible_tree_witnesses=witness_count,
        exhaustive_sizes=[3,4,5,6],bootstrap_covariance_max_error=float(max(errors)),
        status='passed',elapsed_seconds=time.time()-start)
    path=Path(__file__).parent/'results';path.mkdir(exist_ok=True)
    (path/'exhaustive_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
