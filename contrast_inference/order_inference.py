"""Infer tree degrees from simultaneous edge-order statements.

The finite-sample Kendall procedure uses independent disjoint observation pairs
and exact conditional binomial tests. The Pearson bootstrap is nominal and fails calibration in several validation designs.
Classical cut/cycle screening produces an outer family, not the exact family of
all weight-compatible optimal trees. Pairwise degree extrema are exact on that
outer graphic-matroid family; reported rank bounds are conservative.
"""
from __future__ import annotations
from pathlib import Path
import ctypes
import subprocess
import os
import uuid
from functools import lru_cache
import numpy as np
from scipy.stats import binom


def edge_index(p):
    return np.column_stack(np.triu_indices(p, 1))


def local_comparisons(edges, p):
    """A fixed, data-independent family of pairs sharing a vertex."""
    aa,bb=[],[]
    for v in range(p):
        incident=np.flatnonzero(np.any(edges==v,axis=1))
        u,w=np.triu_indices(len(incident),1)
        aa.extend(incident[u]);bb.extend(incident[w])
    return np.asarray(aa,int),np.asarray(bb,int)


def _closure(relation, score):
    """Transitive closure of a strict relation consistent with sample scores."""
    m=len(score)
    if np.any(relation & ~(score[:,None]>score[None,:])):
        raise ValueError('Order certificates are inconsistent with sample scores')
    reach=[0]*m
    for e in np.argsort(score):
        bits=0
        for f in np.flatnonzero(relation[e]):
            bits|=(1<<int(f))|reach[f]
        reach[e]=bits
    nbyte=(m+7)//8
    packed=np.frombuffer(b''.join(v.to_bytes(nbyte,'little') for v in reach),dtype=np.uint8).reshape(m,nbyte)
    return np.unpackbits(packed,axis=1,bitorder='little')[:,:m].astype(bool)


@lru_cache(maxsize=32)
def fixed_comparison_indices(p,family):
    edges=edge_index(p)
    if family=='local':
        a,b=local_comparisons(edges,p)
        common=np.where(edges[a,0]==edges[b,0],edges[a,0],
            np.where(edges[a,0]==edges[b,1],edges[a,0],edges[a,1]))
        left=edges[a].sum(axis=1)-common;right=edges[b].sum(axis=1)-common
    elif family=='all':a,b=np.triu_indices(len(edges),1);left=right=None
    else:raise ValueError('family must be local or all')
    for v in [edges,a,b,left,right]:
        if v is not None:v.flags.writeable=False
    return edges,a,b,left,right


def paired_counts(x,rng=None,family='local'):
    """Exact discordance counts; local kernels cancel the shared central sign.

    For e=(v,a), f=(v,b), H_e H_f=S_a S_b. One p-by-p sign
    Gram matrix supplies every local comparison, including randomized ties.
    """
    x=np.asarray(x,float)
    if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all():
        raise ValueError('A complete IID sample with at least two rows is required')
    if rng is None:rng=np.random.default_rng(271828)
    n,p=x.shape;N=n//2;edges,a,b,left,right=fixed_comparison_indices(p,family)
    sign=np.sign(x[:2*N:2]-x[1:2*N:2]).astype(np.int64);ties=sign==0
    if ties.any():
        auxiliary=rng.random((2*N,p));auxiliary_sign=np.sign(auxiliary[::2]-auxiliary[1::2]).astype(np.int64)
        # Continuous tie breakers have no equal outcomes; ensure binary signs
        # also in the exceptionally rare finite-precision RNG collision.
        zero=auxiliary_sign==0
        if zero.any():auxiliary_sign[zero]=2*rng.integers(0,2,size=zero.sum())-1
        sign[ties]=auxiliary_sign[ties]
    gram=sign.astype(float).T@sign.astype(float)
    sums=gram[edges[:,0],edges[:,1]];score=sums/N
    if family=='local':
        products=gram[left,right]
    else:
        kernels=(sign[:,edges[:,0]]*sign[:,edges[:,1]]).astype(float)
        products=(kernels.T@kernels)[a,b]
    positive=np.rint((N+sums[a]-sums[b]-products)/4).astype(int)
    negative=np.rint((N-sums[a]+sums[b]-products)/4).astype(int)
    assert np.all(positive>=0) and np.all(negative>=0) and np.all(positive+negative<=N)
    return dict(score=score,edges=edges,positive_counts=positive,negative_counts=negative,
        comparison_a=a,comparison_b=b,family=family,paired_observations=N)


def kendall_order(x,alpha=.05,rng=None,close=True,family='local'):
    """Finite-sample McNemar--Holm under IID rows, including tau-a ties."""
    stats=paired_counts(x,rng,family);positive=stats['positive_counts'];negative=stats['negative_counts']
    a,b=stats['comparison_a'],stats['comparison_b'];discordant=positive+negative
    pv=np.concatenate([binom.sf(positive-1,discordant,.5),binom.sf(negative-1,discordant,.5)])
    order=np.argsort(pv,kind='stable');K=len(pv);rejected=np.zeros(K,bool)
    for k,h in enumerate(order):
        if pv[h]>alpha/(K-k):break
        rejected[h]=True
    relation=np.zeros((len(stats['edges']),len(stats['edges'])),bool)
    relation[a[rejected[:len(a)]],b[rejected[:len(a)]]]=True
    relation[b[rejected[len(a):]],a[rejected[len(a):]]]=True
    raw_count=int(relation.sum())
    if close:relation=_closure(relation,stats['score'])
    return {**stats,'relation':relation,'raw_order_count':raw_count,
        'closed_order_count':int(relation.sum()),'hypotheses':K,'pvalues':pv}


def bootstrap_order(r,bootstrap_r,p,alpha=.05,close=True):
    """Joint max standardized Fisher-z contrast bootstrap over local pairs.

    Shared common shifts cancel identically. Studentizers are fixed across
    replicates. This is not a finite-sample guarantee for dependent returns.
    """
    edges=edge_index(p);a,b=local_comparisons(edges,p)
    score=np.arctanh(np.clip(r,-1+1e-12,1-1e-12))
    draws=np.arctanh(np.clip(bootstrap_r,-1+1e-12,1-1e-12))
    B=len(draws);maximum=np.zeros(B);scales=np.empty(len(a));estimates=score[a]-score[b]
    for start in range(0,len(a),1024):
        stop=min(start+1024,len(a))
        delta=draws[:,a[start:stop]]-draws[:,b[start:stop]]
        sd=delta.std(axis=0,ddof=1)
        if np.any(sd<1e-12):raise ValueError('Degenerate contrast studentizer')
        scales[start:stop]=sd
        maximum=np.maximum(maximum,np.max(np.abs((delta-estimates[start:stop])/sd),axis=1))
    critical=float(np.quantile(maximum,1-alpha,method='higher'))
    lower=estimates-critical*scales;upper=estimates+critical*scales
    relation=np.zeros((len(edges),len(edges)),bool)
    relation[a[lower>0],b[lower>0]]=True
    relation[b[upper<0],a[upper<0]]=True
    raw_count=int(relation.sum())
    if close:relation=_closure(relation,score)
    return dict(relation=relation,score=score,edges=edges,critical=critical,
                raw_order_count=raw_count,closed_order_count=int(relation.sum()),
                contrast_lower=lower,contrast_upper=upper,
                comparison_a=a,comparison_b=b)


def _connected(edges,keep,p,a,b):
    parent=list(range(p))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for u,v in edges[keep]:
        ru,rv=root(int(u)),root(int(v))
        if ru!=rv:parent[ru]=rv
        if root(a)==root(b):return True
    return False


def screen_edges(relation,edges,p):
    """Classical strict cycle exclusions and strict cut inclusions."""
    m=len(edges);allowed=np.ones(m,bool);forced=np.zeros(m,bool)
    for e,(a,b) in enumerate(edges):
        if _connected(edges,relation[:,e],p,int(a),int(b)):allowed[e]=False
        alternatives=~relation[e].copy();alternatives[e]=False
        if not _connected(edges,alternatives,p,int(a),int(b)):forced[e]=True
    return allowed,forced


_LIB=None
def _minor_library():
    global _LIB
    if _LIB is None:
        root=Path(__file__).parent;source=root/'graphic_minor.cpp';target=root/'_graphic_minor.so'
        if not target.exists() or target.stat().st_mtime<source.stat().st_mtime:
            temporary=root/f'_graphic_minor.{os.getpid()}.{uuid.uuid4().hex}.so.tmp'
            subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(temporary)],check=True)
            temporary.replace(target)
        _LIB=ctypes.CDLL(str(target))
        pointer=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
        _LIB.envelopes.argtypes=[ctypes.c_int,ctypes.c_int,pointer,pointer,pointer,pointer,pointer,pointer]
        _LIB.envelopes.restype=ctypes.c_int
        dpointer=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        _LIB.kendall_tau_a.argtypes=[ctypes.c_int,ctypes.c_int,dpointer,dpointer]
        _LIB.kendall_tau_a.restype=ctypes.c_int
    return _LIB


def fast_kendall_band(x,alpha=.05):
    """The same full-sample tau-a and Hoeffding baseline, computed faster."""
    x=np.ascontiguousarray(x,dtype=np.float64);n,p=x.shape;m=p*(p-1)//2
    if n<2 or not np.isfinite(x).all():raise ValueError('Complete sample required')
    tau=np.empty(m,np.float64)
    status=_minor_library().kendall_tau_a(n,p,x,tau)
    if status:raise ValueError('Kendall input failed validation')
    epsilon=float(np.sqrt(2*np.log(2*m/alpha)/(n//2)))
    return tau,np.maximum(-1,tau-epsilon),np.minimum(1,tau+epsilon),epsilon


def minor_envelopes(edges,p,allowed,forced):
    lo=np.zeros(p,np.int32);hi=lo.copy();difference=np.zeros((p,p),np.int32)
    args=[np.ascontiguousarray(v,dtype=np.int32) for v in [edges,forced,allowed]]
    status=_minor_library().envelopes(p,len(edges),*args,lo,hi,difference)
    if status:raise ValueError(f'Invalid certified tree family (status {status})')
    return lo,hi,difference


def conservation_ranks(lo,hi):
    """Sharp ranks over integer degree boxes intersected with sum d=2p-2.

    A threshold allocation gives the largest number above/equal a fixed degree;
    capacity allocation gives the smallest number strictly above. On a complete
    graph each positive integer vector with this sum is a tree degree vector.
    The box is still an outer relaxation of a restricted graphic minor.
    """
    lo=np.asarray(lo,int);hi=np.asarray(hi,int);p=len(lo);total=2*p-2
    if np.any(lo<1) or np.any(lo>hi) or lo.sum()>total or hi.sum()<total:
        raise ValueError('Infeasible tree-degree bounds')
    rlo=np.full(p,p,int);rhi=np.ones(p,int)
    for i in range(p):
        L=np.delete(lo,i);U=np.delete(hi,i)
        for s in range(lo[i],hi[i]+1):
            if s+L.sum()>total or s+U.sum()<total:continue
            # Worst rank: make as many other degrees as possible >= s.
            budget=total-s-int(L.sum())
            costs=np.sort(s-L[(L<s)&(U>=s)])
            extra=int(np.sum(np.cumsum(costs)<=budget))
            worst=1+int(np.sum(L>=s))+extra
            # Best rank: use non-outranking capacity before raising competitors.
            mandatory=L>s
            capacity=int(np.sum(np.where(mandatory,U,np.minimum(U,s))))
            remaining=max(0,total-s-capacity)
            increases=np.sort((U-s)[(~mandatory)&(U>s)])[::-1]
            add=0 if remaining==0 else int(np.searchsorted(np.cumsum(increases),remaining)+1)
            best=1+int(mandatory.sum())+add
            rlo[i]=min(rlo[i],best);rhi[i]=max(rhi[i],worst)
    return rlo,rhi


def project_order(ans,p):
    if not ans['relation'].any() and np.array_equal(ans['edges'],edge_index(p)):
        lo=np.ones(p,int);hi=np.full(p,p-1,int)
        difference=np.full((p,p),2-p,int);np.fill_diagonal(difference,0)
        rl=np.ones(p,int);rh=np.full(p,p,int)
        return {**ans,'allowed':np.ones(len(ans['edges']),bool),'forced':np.zeros(len(ans['edges']),bool),
            'degree_lower':lo,'degree_upper':hi,'degree_difference_lower':difference,
            'rank_lower':rl,'rank_upper':rh,'marginal_rank_lower':rl.copy(),'marginal_rank_upper':rh.copy(),
            'conservation_rank_lower':rl.copy(),'conservation_rank_upper':rh.copy(),
            'pair_rank_lower':rl.copy(),'pair_rank_upper':rh.copy()}
    allowed,forced=screen_edges(ans['relation'],ans['edges'],p)
    lo,hi,difference=minor_envelopes(ans['edges'],p,allowed,forced)
    marginal_lo=1+np.array([np.sum(np.delete(lo,i)>hi[i]) for i in range(p)])
    marginal_hi=p-np.array([np.sum(np.delete(hi,i)<lo[i]) for i in range(p)])
    pair_lo=1+np.sum(difference.T>0,axis=1)
    pair_hi=p-np.sum(difference>0,axis=1)
    conservation_lo,conservation_hi=conservation_ranks(lo,hi)
    rank_lo=np.maximum(pair_lo,conservation_lo)
    rank_hi=np.minimum(pair_hi,conservation_hi)
    assert np.all(rank_lo>=marginal_lo) and np.all(rank_hi<=marginal_hi)
    assert np.all(rank_lo<=rank_hi)
    return {**ans,'allowed':allowed,'forced':forced,'degree_lower':lo,'degree_upper':hi,
            'degree_difference_lower':difference,'rank_lower':rank_lo,'rank_upper':rank_hi,
            'marginal_rank_lower':marginal_lo,'marginal_rank_upper':marginal_hi,
            'conservation_rank_lower':conservation_lo,'conservation_rank_upper':conservation_hi,
            'pair_rank_lower':pair_lo,'pair_rank_upper':pair_hi}
