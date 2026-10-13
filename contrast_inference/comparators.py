"""Matched edge-order comparators for Kendall tau-a and Pearson correlation.

Exact methods: disjoint-pair McNemar/Holm, dependence-safe p-value averaging,
and Hoeffding/Holm for full order-two U-statistic contrasts. Multiplier methods
are asymptotic, require IID rows and nondegenerate first-order projections.
Every method can use the same fixed local or complete comparison family.
"""
import ctypes
import numpy as np
from order_inference import edge_index, local_comparisons, kendall_order, _closure, _minor_library


def comparisons(edges, p, family='local'):
    if family=='local':return local_comparisons(edges,p)
    if family=='all':return np.triu_indices(len(edges),1)
    raise ValueError('Unknown comparison family')


def holm(pvalues, alpha=.05):
    pvalues=np.asarray(pvalues,float)
    if np.any(~np.isfinite(pvalues)) or np.any((pvalues<0)|(pvalues>1)):
        raise ValueError('Invalid p-values')
    rejected=np.zeros(len(pvalues),bool)
    for k,i in enumerate(np.argsort(pvalues,kind='stable')):
        if pvalues[i]>alpha/(len(pvalues)-k):break
        rejected[i]=True
    return rejected


def from_pvalues(score, edges, a, b, pv, alpha=.05, family='local'):
    reject=holm(pv,alpha);M=len(a)
    relation=np.zeros((len(edges),len(edges)),bool)
    relation[a[reject[:M]],b[reject[:M]]]=True
    relation[b[reject[M:]],a[reject[M:]]]=True
    # A data-dependent deletion cannot enlarge the false-rejection event.
    # Needed for p-value merging: its rejections need not follow mean scores.
    keep=score[:,None]>score[None,:]
    removed=int(np.sum(relation & ~keep));relation &= keep
    raw=int(relation.sum());relation=_closure(relation,score)
    return dict(score=score,edges=edges,relation=relation,pvalues=pv,
        comparison_a=a,comparison_b=b,family=family,hypotheses=len(pv),
        raw_order_count=raw,closed_order_count=int(relation.sum()),
        acyclic_filter_deletions=removed)


def merged_pair_order(x, alpha=.05, matchings=8, seed=83071, family='local'):
    if matchings<1:raise ValueError('At least one matching required')
    rng=np.random.default_rng(seed);values=[];scores=[]
    for repetition in range(matchings):
        xx=x if repetition==0 else x[rng.permutation(len(x))]
        ans=kendall_order(xx,alpha,rng,close=False,family=family)
        values.append(ans['pvalues']);scores.append(ans['score'])
    if matchings==1:combined=values[0]
    else:combined=np.minimum(1,2*np.mean(values,axis=0))
    out=from_pvalues(np.mean(scores,axis=0),ans['edges'],ans['comparison_a'],
        ans['comparison_b'],combined,alpha,family)
    out['matchings']=matchings
    return out


def kendall_projection(x):
    """Exact full tau-a and empirical first-order projection via tie-safe sweeps."""
    x=np.ascontiguousarray(x,dtype=np.float64);n,p=x.shape
    if n<3 or not np.isfinite(x).all():raise ValueError('Complete sample, n>=3 required')
    rows=np.empty((n,p*(p-1)//2),dtype=np.float64)
    lib=_minor_library();ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    lib.kendall_row_means.argtypes=[ctypes.c_int,ctypes.c_int,ptr,ptr]
    lib.kendall_row_means.restype=ctypes.c_int
    if lib.kendall_row_means(n,p,x,rows):raise ValueError('Kernel projection failed')
    tau=rows.mean(axis=0)
    influence=2*(rows-tau)
    return tau,influence


def hoeffding_difference_order(x, tau=None, alpha=.05, family='local'):
    n,p=x.shape;edges=edge_index(p);a,b=comparisons(edges,p,family)
    if tau is None:tau,_=kendall_projection(x)
    delta=tau[a]-tau[b];N=n//2
    positive=np.exp(-N*np.maximum(delta,0)**2/8)
    negative=np.exp(-N*np.maximum(-delta,0)**2/8)
    return from_pvalues(tau,edges,a,b,np.r_[positive,negative],alpha,family)


def gaussian_draws(influence, B=999, seed=31589):
    n=len(influence);rng=np.random.default_rng(seed)
    return rng.standard_normal((B,n)) @ influence/n


def covariance_of_mean(influence):
    n=len(influence)
    return influence.T @ influence/(n*n)


def contrast_scales(cov, a, b):
    variance=cov[a,a]+cov[b,b]-2*cov[a,b]
    return np.sqrt(np.maximum(variance,0))


def max_contrast(draws, score, a, b, scales, centered=False):
    maximum=np.zeros(len(draws))
    degenerate=scales<=1e-12
    safe=np.where(degenerate,np.inf,scales)
    lib=_minor_library();double=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    integer=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    lib.max_edge_contrast.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,double,double,integer,integer,double,ctypes.c_int,double]
    lib.max_edge_contrast.restype=ctypes.c_int
    status=lib.max_edge_contrast(len(draws),len(score),len(a),
        np.ascontiguousarray(draws,dtype=np.float64),np.ascontiguousarray(score,dtype=np.float64),
        np.ascontiguousarray(a,dtype=np.int32),np.ascontiguousarray(b,dtype=np.int32),
        np.ascontiguousarray(safe,dtype=np.float64),int(centered),maximum)
    if status:raise ValueError('Invalid contrast maximum input')
    return maximum,safe,int(degenerate.sum())


def order_from_draws(score, draws, p, scales=None, covariance=None,
                     alpha=.05, family='local', centered=True):
    edges=edge_index(p);a,b=comparisons(edges,p,family)
    if scales is None:
        if covariance is not None:scales=contrast_scales(covariance,a,b)
        else:
            deviations=draws-draws.mean(axis=0)
            covariance=deviations.T@deviations/(len(draws)-1)
            scales=contrast_scales(covariance,a,b)
    maximum,safe,omitted=max_contrast(draws,score,a,b,scales,centered)
    critical=float(np.quantile(maximum,1-alpha,method='higher'))
    delta=score[a]-score[b];width=np.full_like(safe,np.inf);active=np.isfinite(safe)
    width[active]=critical*safe[active];lower=delta-width;upper=delta+width
    relation=np.zeros((len(edges),len(edges)),bool)
    relation[a[lower>0],b[lower>0]]=True;relation[b[upper<0],a[upper<0]]=True
    raw=int(relation.sum());relation=_closure(relation,score)
    return dict(score=score,edges=edges,relation=relation,critical=critical,
        comparison_a=a,comparison_b=b,family=family,contrast_lower=lower,
        contrast_upper=upper,raw_order_count=raw,closed_order_count=int(relation.sum()),
        omitted_degenerate_contrasts=omitted,bootstrap_repetitions=len(draws))


def multiplier_orders(score, influence, p, B=999, seed=31589,
                      alpha=.05, family='local', draws=None, covariance=None):
    if covariance is None:covariance=covariance_of_mean(influence)
    if draws is None:draws=gaussian_draws(influence,B,seed)
    direct=order_from_draws(score,draws,p,covariance=covariance,
        alpha=alpha,family=family,centered=True)
    edges=direct['edges'];a,b=direct['comparison_a'],direct['comparison_b']
    se=np.sqrt(np.maximum(np.diag(covariance),0));safe=np.where(se<=1e-12,np.inf,se)
    critical=float(np.quantile(np.max(np.abs(draws/safe),axis=1),1-alpha,method='higher'))
    width=np.full_like(safe,np.inf);active=np.isfinite(safe);width[active]=critical*safe[active]
    lower=score-width;upper=score+width
    relation=np.zeros((len(edges),len(edges)),bool)
    relation[a[lower[a]>upper[b]],b[lower[a]>upper[b]]]=True
    relation[b[lower[b]>upper[a]],a[lower[b]>upper[a]]]=True
    raw=int(relation.sum());relation=_closure(relation,score)
    box=dict(score=score,edges=edges,relation=relation,critical=critical,
        comparison_a=a,comparison_b=b,family=family,raw_order_count=raw,
        closed_order_count=int(relation.sum()),edge_lower=lower,edge_upper=upper,
        bootstrap_repetitions=len(draws))
    return direct,box


def pearson_projection(x):
    """Empirical influence of Fisher-z Pearson correlations, centered exactly."""
    x=np.asarray(x,float);n,p=x.shape;edges=edge_index(p)
    z=x-x.mean(axis=0);sd=np.sqrt(np.mean(z*z,axis=0))
    if np.any(sd<=0):raise ValueError('Constant Pearson margin')
    z=z/sd;a,b=edges.T;r=np.mean(z[:,a]*z[:,b],axis=0)
    if np.any(np.abs(r)>=1-1e-12):raise ValueError('Degenerate Fisher-z correlation')
    influence=(z[:,a]*z[:,b]-.5*r*(z[:,a]**2+z[:,b]**2))/(1-r*r)
    influence-=influence.mean(axis=0)
    return np.arctanh(r),influence


def resampled_pearson(x, B=1999, seed=83015):
    """IID row bootstrap using unique second moments; first B draws are nested."""
    x=np.asarray(x,float);n,p=x.shape;edges=edge_index(p)
    z=x-x.mean(axis=0);a,b=edges.T
    products=np.column_stack([z[:,a]*z[:,b],z*z])
    rng=np.random.default_rng(seed);out=[]
    for start in range(0,B,128):
        size=min(128,B-start);indices=rng.integers(n,size=(size,n))
        counts=np.zeros((size,n));np.add.at(counts,(np.arange(size)[:,None],indices),1)
        means=counts@z/n;moments=counts@products/n
        variances=moments[:,len(edges):]-means*means
        covariance=moments[:,:len(edges)]-means[:,a]*means[:,b]
        r=covariance/np.sqrt(variances[:,a]*variances[:,b])
        out.append(np.arctanh(np.clip(r,-1+1e-12,1-1e-12)))
    return np.concatenate(out)
