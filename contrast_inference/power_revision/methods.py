"""Finite-sample adaptive betting and independent-selection decision tests.

These are applications of established e-value, fixed-sequence and
intersection-union principles, not new generic multiple-testing theorems.
"""
from pathlib import Path
import sys,ctypes,subprocess
import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent/'statistical_papers'))
from order_inference import edge_index,fast_kendall_band,fixed_comparison_indices,_closure,project_order
from evalue_order import matching_loge,lambda_grid
from comparators import holm
from hub_inference import max_tree

_LIB=None
def adaptive_logs(h,a,b):
    global _LIB
    if _LIB is None:
        source=Path(__file__).with_name('adaptive_betting.cpp')
        target=source.with_suffix('.so')
        if not target.exists() or target.stat().st_mtime<source.stat().st_mtime:
            import os
            temp=target.with_name(f'adaptive_betting.{os.getpid()}.so')
            subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(temp)],check=True)
            temp.replace(target)
        _LIB=ctypes.CDLL(str(target));_LIB.adaptive_betting.restype=ctypes.c_int
        _LIB.adaptive_betting.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,
          np.ctypeslib.ndpointer(dtype=np.int8,flags='C_CONTIGUOUS'),
          np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS'),
          np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS'),
          np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')]
    h=np.ascontiguousarray(h,dtype=np.int8);a=np.ascontiguousarray(a,dtype=np.int32);b=np.ascontiguousarray(b,dtype=np.int32)
    result=np.empty(2*len(a));assert _LIB.adaptive_betting(len(h),h.shape[1],len(a),h,a,b,result)==0
    return result

def evidence(x,a,b,R=64,seed=61023001,adaptive=True,trained_stakes=None):
    """Both directed p-values. All permutations are independent of values.

    The first matching supplies P; zero signs use independent coordinate-wise
    fair coins, preserving E[H_e]=tau_a. E64/A64 use the original zero signs.
    """
    n,p=x.shape;N=n//2;edges=edge_index(p);rng=np.random.default_rng(seed)
    lambdas=lambda_grid(N);fixed=None;adapt=None;single=None;trained=None
    for r in range(R):
        y=x if r==0 else x[rng.permutation(n)]
        s=np.sign(y[:2*N:2]-y[1:2*N:2]).astype(np.int8)
        h=s[:,edges[:,0]]*s[:,edges[:,1]]
        if r==0:
            st=s.copy();tie=st==0
            # A separate RNG prevents tie frequency changing later matchings.
            st[tie]=np.random.default_rng(seed+10000000).choice([-1,1],size=tie.sum())
            hp=st[:,edges[:,0]]*st[:,edges[:,1]]
            d=hp[:,a]-hp[:,b];A=(d>0).sum(0);B=(d<0).sum(0)
            pair=np.r_[binom.sf(A-1,A+B,.5),binom.sf(B-1,A+B,.5)]
        # Blocked counts avoid an N x K temporary for large families.
        vals=np.empty(2*len(a));sv=np.empty(2*len(a));tv=np.empty(2*len(a))
        for start in range(0,len(a),2048):
            end=min(len(a),start+2048);d=h[:,a[start:end]]-h[:,b[start:end]]
            A=(d==2).sum(0);B=(d==-2).sum(0);C=(d==1).sum(0);D=(d==-1).sum(0)
            v=matching_loge(A,B,lambdas,C,D);L=end-start
            vals[start:end]=v[:L];vals[len(a)+start:len(a)+end]=v[L:]
            v=matching_loge(A,B,np.array([.25]),C,D)
            sv[start:end]=v[:L];sv[len(a)+start:len(a)+end]=v[L:]
            if trained_stakes is not None:
                lam=trained_stakes[start:end]
                tv[start:end]=A*np.log1p(lam)+B*np.log1p(-lam)+C*np.log1p(lam/2)+D*np.log1p(-lam/2)
                tv[len(a)+start:len(a)+end]=B*np.log1p(lam)+A*np.log1p(-lam)+D*np.log1p(lam/2)+C*np.log1p(-lam/2)
        fixed=vals if fixed is None else np.logaddexp(fixed,vals)
        single=sv if single is None else np.logaddexp(single,sv)
        if trained_stakes is not None:trained=tv if trained is None else np.logaddexp(trained,tv)
        if adaptive:
            av=adaptive_logs(h,a,b);adapt=av if adapt is None else np.logaddexp(adapt,av)
    out={'P':pair,'E64':np.exp(-np.maximum(0,fixed-np.log(R))),
         'S64':np.exp(-np.maximum(0,single-np.log(R)))}
    if adaptive:out['A64']=np.exp(-np.maximum(0,adapt-np.log(R)))
    if trained_stakes is not None:out['T64']=np.exp(-np.maximum(0,trained-np.log(R)))
    return out

def order_from_rejections(score,edges,a,b,reject,score_filter=True):
    relation=np.zeros((len(edges),len(edges)),bool)
    relation[np.asarray(a)[reject],np.asarray(b)[reject]]=True
    if score_filter:relation&=score[:,None]>score[None,:]
    raw=int(relation.sum());relation=_closure(relation,score)
    return dict(edges=edges,score=score,relation=relation,raw_order_count=raw,
                closed_order_count=int(relation.sum()))

def full_orders(x,R=64,seed=61023001,alpha=.05):
    p=x.shape[1];edges,a,b,_,_=fixed_comparison_indices(p,'local')
    tau=fast_kendall_band(x)[0];pv=evidence(x,a,b,R,seed)
    aa=np.r_[a,b];bb=np.r_[b,a]
    return {name:order_from_rejections(tau,edges,aa,bb,holm(values,alpha))
            for name,values in pv.items()}

def tree_constraints(tree,edges,p):
    """All fundamental-cycle comparisons f>e that certify this exact tree."""
    tree=set(map(int,tree));adj=[[] for _ in range(p)]
    for e in tree:
        u,v=map(int,edges[e]);adj[u].append((v,e));adj[v].append((u,e))
    pairs=[]
    for e,(u,v) in enumerate(edges):
        if e in tree:continue
        stack=[(int(u),-1,[])];path=None
        while stack:
            w,parent,route=stack.pop()
            if w==v:path=route;break
            for z,f in adj[w]:
                if z!=parent:stack.append((z,w,route+[f]))
        assert path is not None
        pairs.extend((f,e) for f in path)
    return np.array(pairs,dtype=int).T

def split_orders(x,R=64,seed=61023001,alpha=.05,fraction=.2):
    n,p=x.shape;ntrain=max(4,int(np.floor(n*fraction)))
    perm=np.random.default_rng(seed).permutation(n)
    train=x[perm[:ntrain]];test=x[perm[ntrain:]]
    edges=edge_index(p);tau=fast_kendall_band(train)[0]
    tree,degree=max_tree(tau,edges,p);a,b=tree_constraints(tree,edges,p)
    # A training-only sequence. Stop on the first failure, with no recycling.
    sequence=np.argsort(-(tau[a]-tau[b]),kind='stable');a=a[sequence];b=b[sequence]
    # Independent training estimates the null-centred second moment of Z.
    q=np.zeros(len(a));rr=np.random.default_rng(seed+2000000);N=ntrain//2
    for _ in range(8):
        t=train[rr.permutation(ntrain)];s=np.sign(t[:2*N:2]-t[1:2*N:2]);h=s[:,edges[:,0]]*s[:,edges[:,1]]
        for start in range(0,len(a),2048):
            end=min(start+2048,len(a));z=(h[:,a[start:end]]-h[:,b[start:end]])/2
            q[start:end]+=np.sum(z*z,axis=0)
    q=(1+q)/(1+8*N)
    stakes=np.clip((tau[a]-tau[b])/(2*q),.01,.5)
    pv=evidence(test,a,b,R,seed+1000000,trained_stakes=stakes);out={};diagnostics={}
    for name,values in pv.items():
        values=values[:len(a)]
        for mode in ['H','FS','IUT']:
            if mode=='H':reject=holm(values,alpha)
            elif mode=='FS':reject=np.logical_and.accumulate(values<=alpha)
            else:reject=np.full(len(a),bool(np.all(values<=alpha)))
            topological_score=np.isin(np.arange(len(edges)),tree).astype(float)
            ans=order_from_rejections(topological_score,edges,a,b,reject,score_filter=False)
            ans['selection_score']=tau
            out[mode+'-'+name]=ans
        diagnostics[name]=dict(global_tree_p=float(values.max()),sequence_passes=int(np.logical_and.accumulate(values<=alpha).sum()))
    return out,dict(ntrain=ntrain,ntest=n-ntrain,tree=tree.tolist(),tree_degree=degree.tolist(),
                   comparisons=len(a),stakes=stakes.tolist(),diagnostics=diagnostics)
