from pathlib import Path
import json
import numpy as np
from scipy.stats import binom
from scipy.special import logsumexp
from evalue_order import matching_loge,LAMBDAS,evalue_order,raw_contrast_counts,lambda_grid
from order_inference import paired_counts

def run():
    checks=0
    for D in [0,1,2,5,20,100,1000]:
        A=np.arange(D+1);B=D-A;loge=matching_loge(A,B,lambda_grid(max(D,1)))[:D+1]
        for q in [0,.1,.3,.49,.5]:
            mean=float(np.exp(logsumexp(binom.logpmf(A,D,q)+loge)))
            assert mean<=1+2e-12
            pv=np.exp(-np.maximum(loge,0))
            for level in [.0001,.001,.01,.05,.1,.5,1]:
                assert binom.pmf(A,D,q)[pv<=level].sum()<=level+2e-12
            checks+=1
    rng=np.random.default_rng(92811901);count_cases=0
    for kind in ['continuous','discrete','constant']:
        for family in ['local','all']:
            for rep in range(10):
                x=rng.normal(size=(20,6)) if kind=='continuous' else rng.integers(-2,3,size=(20,6)).astype(float)
                if kind=='constant':x[:,0]=0
                seed=991831+rep;r1=np.random.default_rng(seed);r2=np.random.default_rng(seed)
                stats=paired_counts(x,r1,family);sign=np.sign(x[::2]-x[1::2]);ties=sign==0
                if ties.any():aux=r2.random(x.shape);aux=np.sign(aux[::2]-aux[1::2]);sign[ties]=aux[ties]
                e=stats['edges'];h=sign[:,e[:,0]]*sign[:,e[:,1]];a,b=stats['comparison_a'],stats['comparison_b']
                A=((h[:,a]==1)&(h[:,b]==-1)).sum(axis=0);B=((h[:,a]==-1)&(h[:,b]==1)).sum(axis=0)
                np.testing.assert_array_equal(stats['positive_counts'],A);np.testing.assert_array_equal(stats['negative_counts'],B)
                np.testing.assert_array_equal(stats['score'],h.mean(axis=0));count_cases+=1
                raw=raw_contrast_counts(x,np.random.default_rng(seed),family)
                rawsign=np.sign(x[::2]-x[1::2]);rawh=rawsign[:,e[:,0]]*rawsign[:,e[:,1]]
                contrast=(rawh[:,a]-rawh[:,b])/2
                for name,value in [('positive_counts',1),('negative_counts',-1),('half_positive',.5),('half_negative',-.5)]:
                    np.testing.assert_array_equal(raw[name],np.sum(contrast==value,axis=0))
    # Independent integer fractions check the log implementation on small D.
    from fractions import Fraction
    for A in range(8):
        for B in range(8):
            exact=sum((1+Fraction(float(l)))**A*(1-Fraction(float(l)))**B for l in LAMBDAS)/len(LAMBDAS)
            observed=np.exp(matching_loge(np.array([A]),np.array([B]))[0])
            np.testing.assert_allclose(observed,float(exact),rtol=3e-15,atol=3e-15)
    report=dict(status='PASS',conditional_expectation_cases=checks,markov_pvalue_checks=checks*7,
        direct_kernel_count_cases=count_cases,exact_fraction_cases=64,lambdas=LAMBDAS.tolist(),
        seed_scope='Verification/pilot seeds are excluded from production datasets')
    # IID five-point contrasts: direct exact integration, including asymmetric
    # distributions whose mean is zero but discordant signs are not balanced.
    import itertools
    five_cases=0;support=np.array([-1,-.5,0,.5,1])
    for N in range(1,6):
        labels=np.array(list(itertools.product(range(5),repeat=N)));z=support[labels];grid=lambda_grid(N)
        evidence=np.mean(np.prod(1+z[:,:,None]*grid,axis=1),axis=1)
        for probs in [np.full(5,.2),np.array([0,.4,.4,0,.2]),np.array([.3,.2,.2,.2,.1])]:
            mass=np.prod(probs[labels],axis=1);assert (probs@support)<=1e-15
            assert mass@evidence<=1+2e-12
            for level in [.01,.05,.1,.5]:assert mass[evidence>=1/level].sum()<=level+2e-12
            five_cases+=1
    report['five_point_exact_expectation_cases']=five_cases
    (Path(__file__).parent/'results'/'evalue_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':run()
