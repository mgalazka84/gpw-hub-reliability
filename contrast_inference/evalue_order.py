"""Finite-sample evidence aggregation across shared-observation matchings.

For fixed lambda in [0,1), the product of 1+lambda*Z has null expectation
(1+lambda*E[Z])**N <= 1, with raw tau-a contrast Z in {-1,-.5,0,.5,1}.
Uniform averaging over deterministic lambda choices and matchings preserves
this e-value property under arbitrary dependence. A size-dependent lambda
ensures consistency for every fixed positive gap. Markov reciprocals + Holm
control FWER; ties need no auxiliary randomization in the test statistic.
"""
import numpy as np
from scipy.special import logsumexp
from order_inference import paired_counts,fast_kendall_band,edge_index
from comparators import from_pvalues

LAMBDAS=np.array([1/32,1/16,1/8,1/4,1/2,3/4])

def lambda_grid(N):
    return np.r_[LAMBDAS,min(.5,1/np.sqrt(N))]


def raw_contrast_counts(x,rng=None,family='local'):
    # Consume the same permutation/tie RNG stream as the M8 comparator. These
    # auxiliary draws align matchings; the evidence uses original zero ties.
    stats=paired_counts(x,rng,family);N=stats['paired_observations'];p=x.shape[1]
    sign=np.sign(x[:2*N:2]-x[1:2*N:2])
    if not (sign==0).any():
        return {**stats,'half_positive':np.zeros_like(stats['positive_counts']),
            'half_negative':np.zeros_like(stats['negative_counts'])}
    edges=stats['edges'];h=sign[:,edges[:,0]]*sign[:,edges[:,1]];h2=h*h
    sums=h.sum(axis=0);squares=h2.sum(axis=0);a,b=stats['comparison_a'],stats['comparison_b']
    cross=[np.empty(len(a)) for _ in range(3)];position=0
    groups=range(p) if family=='local' else [-1]
    for v in groups:
        incident=np.flatnonzero(np.any(edges==v,axis=1)) if v>=0 else np.arange(len(edges))
        u,w=np.triu_indices(len(incident),1);H=h[:,incident];H2=h2[:,incident]
        ST=H.T@H;PQ=H2.T@H2;PT=H2.T@H
        end=position+len(u)
        # ST, PQ, PT; SQ is the transpose of PT.
        for array,matrix in zip(cross,[ST,PQ,PT]):array[position:end]=matrix[u,w]
        if position==0:SQ=np.empty(len(a))
        SQ[position:end]=PT[w,u];position=end
    ST,PQ,PT=cross
    positive=np.rint((PQ-PT+SQ-ST)/4).astype(int)
    negative=np.rint((PQ+PT-SQ-ST)/4).astype(int)
    half_positive=np.rint((squares[a]+squares[b]+sums[a]-sums[b]-2*PQ-SQ+PT)/2).astype(int)
    half_negative=np.rint((squares[a]+squares[b]-sums[a]+sums[b]-2*PQ+SQ-PT)/2).astype(int)
    assert all(np.all(v>=0) for v in [positive,negative,half_positive,half_negative])
    assert np.all(positive+negative+half_positive+half_negative<=N)
    return {**stats,'score':sums/N,'positive_counts':positive,'negative_counts':negative,
        'half_positive':half_positive,'half_negative':half_negative}


def matching_loge(A,B,lambdas=LAMBDAS,half_positive=None,half_negative=None):
    A=np.asarray(A)[:,None];B=np.asarray(B)[:,None]
    plus=np.log1p(lambdas);minus=np.log1p(-lambdas)
    pos=A*plus+B*minus;neg=B*plus+A*minus
    if half_positive is not None:
        C=np.asarray(half_positive)[:,None];D=np.asarray(half_negative)[:,None]
        pos+=C*np.log1p(lambdas/2)+D*np.log1p(-lambdas/2)
        neg+=D*np.log1p(lambdas/2)+C*np.log1p(-lambdas/2)
    pos=logsumexp(pos,axis=1)-np.log(len(lambdas))
    neg=logsumexp(neg,axis=1)-np.log(len(lambdas))
    return np.r_[pos,neg]


def evalue_orders(x,alpha=.05,matchings=(8,64),seed=71829011,family='local',tau=None):
    matchings=tuple(sorted(set(matchings)))
    if not matchings or min(matchings)<1:raise ValueError('At least one fixed matching required')
    rng=np.random.default_rng(seed);accumulator=None
    lambdas=lambda_grid(len(x)//2)
    if tau is None:tau=fast_kendall_band(x)[0]
    out={}
    for repetition in range(max(matchings)):
        xx=x if repetition==0 else x[rng.permutation(len(x))]
        stats=raw_contrast_counts(xx,rng,family)
        loge=matching_loge(stats['positive_counts'],stats['negative_counts'],lambdas,
            stats['half_positive'],stats['half_negative'])
        accumulator=loge if accumulator is None else np.logaddexp(accumulator,loge)
        R=repetition+1
        if R in matchings:
            total=accumulator-np.log(R);pv=np.exp(-np.maximum(total,0))
            result=from_pvalues(tau,stats['edges'],stats['comparison_a'],stats['comparison_b'],pv,alpha,family)
            out[R]={**result,'log_evalues':total,'matchings':R,'lambdas':lambdas.tolist()}
    return out


def evalue_order(x,alpha=.05,matchings=64,seed=71829011,family='local',tau=None):
    return evalue_orders(x,alpha,(matchings,),seed,family,tau)[matchings]
