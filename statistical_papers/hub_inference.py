"""Simultaneous degree/rank envelopes for maximum-correlation spanning trees.

The combinatorial envelopes are exact over the rectangular weight box, not
over its intersection with the positive-semidefinite correlation cone.
Statistical coverage is inherited from the input simultaneous band.
"""
from __future__ import annotations
import numpy as np


def edge_index(p):
    return np.column_stack(np.triu_indices(p, 1))


def max_tree(weights, edges, p, secondary=None):
    """Kruskal: descending primary then descending secondary, with exact ties.

    No finite epsilon is added to primary weights. Secondary scores implement
    lexicographic optimization among maximum-weight trees.
    """
    if secondary is None:
        secondary = np.zeros(len(edges), dtype=int)
    order = np.lexsort((np.arange(len(edges)), -np.asarray(secondary), -weights))
    parent = list(range(p)); size = [1]*p
    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    chosen = []
    for e in order:
        a,b = edges[e]; u,v = root(a),root(b)
        if u == v: continue
        if size[u] < size[v]: u,v = v,u
        parent[v] = u; size[u] += size[v]; chosen.append(e)
        if len(chosen) == p-1: break
    if len(chosen) != p-1: raise ValueError('Input graph is disconnected')
    chosen = np.asarray(chosen, dtype=int)
    degree = np.bincount(edges[chosen].ravel(), minlength=p)
    return chosen, degree


def degree_envelopes(lower, upper, edges, p):
    """Exact coordinatewise degree extrema over interval-optimal trees."""
    lower,upper = np.asarray(lower),np.asarray(upper)
    if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise ValueError('Weights must be finite')
    if np.any(lower > upper): raise ValueError('Empty intervals')
    lo = np.zeros(p, dtype=int); hi = np.zeros(p, dtype=int)
    for i in range(p):
        incident = np.any(edges == i, axis=1)
        adverse = np.where(incident, lower, upper)
        favourable = np.where(incident, upper, lower)
        _,d = max_tree(adverse, edges, p, -incident.astype(int)); lo[i]=d[i]
        _,d = max_tree(favourable, edges, p, incident.astype(int)); hi[i]=d[i]
    return lo,hi


def rank_envelopes(lo, hi):
    """Outer bounds covering every descending ranking/tie resolution."""
    p=len(lo)
    lower=np.array([1+np.sum(np.delete(lo,i)>hi[i]) for i in range(p)])
    upper=np.array([p-np.sum(np.delete(hi,i)<lo[i]) for i in range(p)])
    return lower,upper


def correlation_edges(cov, edges):
    var=np.diagonal(cov,axis1=-2,axis2=-1)
    if np.any(var<=1e-15): raise ValueError('Degenerate variance')
    a,b=edges.T
    return np.clip(cov[...,a,b]/np.sqrt(var[...,a]*var[...,b]),-1+1e-12,1-1e-12)


def transform_cov(cov, p, kind):
    c=cov[...,:p,:p].copy()
    if kind=='demeaned':
        return c-c.mean(axis=-1,keepdims=True)-c.mean(axis=-2,keepdims=True)+c.mean(axis=(-1,-2),keepdims=True)
    if kind=='factor_residual':
        f=cov[...,:p,p]
        return c-f[..., :,None]*f[...,None,:]/cov[...,p,p][...,None,None]
    if kind!='raw': raise ValueError(kind)
    return c


def bootstrap_covariances(x, repetitions, block, rng):
    """Circular common-row resampling via count-weighted first/second moments."""
    x=np.asarray(x,float)
    if not np.isfinite(x).all(): raise ValueError('Complete data required')
    x=x-x.mean(axis=0); n,q=x.shape
    starts=rng.integers(n,size=(repetitions,int(np.ceil(n/block))))
    ix=((starts[...,None]+np.arange(block))%n).reshape(repetitions,-1)[:,:n]
    counts=np.zeros((repetitions,n),float)
    np.add.at(counts,(np.arange(repetitions)[:,None],ix),1)
    means=(counts@x)/n
    products=(x[:,:,None]*x[:,None,:]).reshape(n,q*q)
    boot=((counts@products)/n).reshape(repetitions,q,q)-means[:,:,None]*means[:,None,:]
    observed=x.T@x/n
    return observed,boot


def calibrated_band(r, boot_r, alpha=.05, centered=True):
    """Bootstrap max standardized error band on the (centered) Fisher-z scale.

    This is an asymptotic procedure. Finite-sample exactness is not claimed.
    Standard errors are fixed across bootstrap draws, not nested studentized.
    """
    z=np.arctanh(r); bz=np.arctanh(boot_r)
    if centered:
        z=z-z.mean();bz=bz-bz.mean(axis=1,keepdims=True)
    se=bz.std(axis=0,ddof=1)
    if np.any(se<1e-12): raise ValueError('Degenerate bootstrap standard error')
    maxerr=np.max(np.abs((bz-z)/se),axis=1)
    critical=float(np.quantile(maxerr,1-alpha,method='higher'))
    return z-critical*se,z+critical*se,critical,se


def kendall_finite_band(x,alpha=.05):
    """Finite-sample simultaneous Kendall tau-a band for IID observations.

    Hoeffding's order-two U-statistic inequality plus a union bound. This
    guarantee does not apply to serially dependent returns. Ties use sign(0)=0.
    """
    from scipy.stats import kendalltau
    x=np.asarray(x,float);n,p=x.shape;edges=edge_index(p)
    if n<2 or not np.isfinite(x).all(): raise ValueError('Complete IID sample required')
    tau=[]
    total=n*(n-1)/2
    for i,j in edges:
        # Convert scipy's tau-b to tau-a when tied observations are present.
        value=kendalltau(x[:,i],x[:,j]).statistic
        _,ci=np.unique(x[:,i],return_counts=True);_,cj=np.unique(x[:,j],return_counts=True)
        pairs_i=np.sum(ci*(ci-1)/2);pairs_j=np.sum(cj*(cj-1)/2)
        denom=np.sqrt((total-pairs_i)*(total-pairs_j))
        tau.append(0.0 if denom==0 else value*denom/total)
    tau=np.asarray(tau)
    epsilon=np.sqrt(2*np.log(2*len(edges)/alpha)/(n//2))
    return tau,np.maximum(-1,tau-epsilon),np.minimum(1,tau+epsilon),float(epsilon)


def fractional_top(degree,k):
    cutoff=np.sort(degree)[-k]
    out=(degree>cutoff).astype(float); tie=degree==cutoff
    out[tie]=(k-out.sum())/tie.sum()
    return out


def analyse_covariances(observed,boot,p,kind='raw',alpha=.05,selection=True,centered=True):
    edges=edge_index(p)
    r=correlation_edges(transform_cov(observed,p,kind),edges)
    br=correlation_edges(transform_cov(boot,p,kind),edges)
    lower,upper,critical,se=calibrated_band(r,br,alpha,centered)
    dlo,dhi=degree_envelopes(lower,upper,edges,p)
    rlo,rhi=rank_envelopes(dlo,dhi)
    _,degree=max_tree(r,edges,p)
    k=max(1,int(np.ceil(p*.1)))
    out=dict(r=r,lower=lower,upper=upper,degree_lower=dlo,degree_upper=dhi,
             rank_lower=rlo,rank_upper=rhi,degree=degree,k=k,critical=critical,
             candidate=rlo<=k,certified=rhi<=k)
    if selection:
        bdeg=np.array([max_tree(w,edges,p)[1] for w in br])
        out['selection']=np.mean([fractional_top(d,k) for d in bdeg],axis=0)
        out['percentile_lower']=np.quantile(bdeg,alpha/2,axis=0,method='lower')
        out['percentile_upper']=np.quantile(bdeg,1-alpha/2,axis=0,method='higher')
    return out
