"""Independent small-instance checks of kernels, projections and resampling."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import kendalltau, binom
from comparators import *
from order_inference import fast_kendall_band,project_order


def run():
    rng=np.random.default_rng(827910);cases=0
    for n in [3,4,9,20,51]:
        for kind in ['continuous','discrete','constant']:
            for rep in range(8):
                x=rng.normal(size=(n,6)) if kind=='continuous' else rng.integers(-2,3,size=(n,6)).astype(float)
                if kind=='constant':x[:,1]=0
                tau,phi=kendall_projection(x);edges=edge_index(6)
                signs=np.sign(x[:,None,:]-x[None,:,:])
                h=signs[:,:,edges[:,0]]*signs[:,:,edges[:,1]]
                row=h.sum(axis=1)/(n-1);expected=row.mean(axis=0)
                np.testing.assert_allclose(tau,expected,atol=2e-15)
                np.testing.assert_allclose(phi,2*(row-expected),atol=2e-15)
                np.testing.assert_allclose(tau,fast_kendall_band(x)[0],atol=2e-15)
                cases+=1
    x=rng.normal(size=(29,5));score,phi=pearson_projection(x);delta=1e-5
    z=x-x.mean(axis=0);edges=edge_index(5)
    for i in [0,3,19]:
        weight=np.full(len(x),1/len(x));weight[i]+=delta
        weight=(weight/weight.sum());mu=weight@x
        covariance=(x-mu).T@((x-mu)*weight[:,None])
        a,b=edges.T;r=covariance[a,b]/np.sqrt(covariance[a,a]*covariance[b,b])
        np.testing.assert_allclose((np.arctanh(r)-score)/delta,phi[i],rtol=5e-4,atol=8e-5)
    draws=resampled_pearson(x,31,821);check_rng=np.random.default_rng(821)
    for b in range(31):
        rows=check_rng.integers(len(x),size=len(x));r=np.corrcoef(x[rows],rowvar=False)
        np.testing.assert_allclose(draws[b],np.arctanh(r[edges[:,0],edges[:,1]]),atol=2e-15)
    for family in ['local','all']:
        ans=merged_pair_order(x,matchings=8,seed=981,family=family)
        a,b=ans['comparison_a'],ans['comparison_b'];rr=np.random.default_rng(981);pv=[]
        for k in range(8):
            xx=x if k==0 else x[rr.permutation(len(x))]
            raw=kendall_order(xx,rng=rr,close=False,family=family);pv.append(raw['pvalues'])
        np.testing.assert_allclose(ans['pvalues'],np.minimum(1,2*np.mean(pv,axis=0)))
        expected=len(edges)*(len(edges)-1) if family=='all' else 5*4*3
        assert ans['hypotheses']==expected
        aa,bb=comparisons(edges,5,family);scales=rng.uniform(.3,2,size=len(aa));draws=rng.normal(size=(31,len(edges)))
        for centered in [False,True]:
            maximum,_,_=max_contrast(draws,score,aa,bb,scales,centered)
            errors=draws[:,aa]-draws[:,bb]
            if not centered:errors-=score[aa]-score[bb]
            np.testing.assert_allclose(maximum,np.max(np.abs(errors/scales),axis=1),atol=2e-15)
    # Unrestricted tree projection must retain all tied degree/rank targets.
    blank=dict(edges=edges,relation=np.zeros((len(edges),len(edges)),bool))
    ans=project_order(blank,5)
    assert np.all(ans['degree_lower']==1) and np.all(ans['degree_upper']==4)
    assert np.all(ans['rank_lower']==1) and np.all(ans['rank_upper']==5)
    chain_edges=np.array([[0,1],[1,2],[2,3]])
    chain=project_order(dict(edges=chain_edges,relation=np.zeros((3,3),bool)),4)
    np.testing.assert_array_equal(chain['degree_lower'],[1,2,2,1])
    np.testing.assert_array_equal(chain['degree_upper'],[1,2,2,1])
    for x in [np.ones((12,5)),np.tile(np.arange(12.)[:,None],(1,5))]:
        score,phi=kendall_projection(x)
        direct,box=multiplier_orders(score,phi,5,B=31)
        assert not direct['relation'].any() and not box['relation'].any()
        assert not np.isnan(direct['contrast_lower']).any()
        assert np.isneginf(direct['contrast_lower']).all()
        assert np.isposinf(direct['contrast_upper']).all()
    report=dict(status='PASS',kendall_kernel_cases=cases,pearson_finite_difference_rows=3,
        direct_row_bootstrap_replicates=31,merged_families=2,
        paired_full_variance_ratio=2/(2*(2/9)),
        null_binomial_tail=float(binom.sf(13,200,.05)))
    (Path(__file__).parent/'results'/'comparator_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':run()
