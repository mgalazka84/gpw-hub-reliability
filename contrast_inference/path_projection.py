"""Certified LP outer bounds incorporating cycle-path incompatibilities.

An order e>f excludes every path between the endpoints of e that contains f.
The path's edge incidence sum is therefore at most length(path)-1 in every
population-optimal tree. We insert all such length-two paths (triangles).
The LP also contains tree cardinality, degree and root-flow constraints.
Bounds are rounded only after validating a dual lower bound; no LP feasible
solution is used as an inferential certificate.
"""
import itertools
import numpy as np
from scipy.sparse import coo_matrix,vstack
from scipy.optimize import linprog


def path_constraints(relation,edges,p,allowed=None,forced=None,random_trees=8):
    index=np.full((p,p),-1,int)
    for e,(i,j) in enumerate(edges):index[i,j]=index[j,i]=e
    forbidden=set()
    for i,j,k in itertools.combinations(range(p),3):
        triangle=[index[i,j],index[i,k],index[j,k]]
        for z,closing in enumerate(triangle):
            path=[triangle[a] for a in range(3) if a!=z]
            if relation[closing,path].any():forbidden.add(tuple(sorted(path)))
    if allowed is not None and random_trees:
        # Deterministic, data-independent random priorities. Every selected
        # path is checked against the same simultaneous order certificates.
        # Path selection therefore requires no new multiple-testing budget.
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'statistical_papers'))
        from hub_inference import max_tree
        chosen_edges=np.flatnonzero(allowed);rng=np.random.default_rng(314159+p)
        for repetition in range(random_trees):
            weights=rng.random(len(chosen_edges))
            weights[forced[chosen_edges]]=p+1
            selected,_=max_tree(weights,edges[chosen_edges],p)
            selected=chosen_edges[selected];in_tree=np.zeros(len(edges),bool);in_tree[selected]=True
            adjacency=[[] for _ in range(p)]
            for e in selected:
                a,b=edges[e];adjacency[a].append((int(b),int(e)));adjacency[b].append((int(a),int(e)))
            parent=[];parent_edge=[]
            for root in range(p):
                par=np.full(p,-1,int);pe=np.full(p,-1,int);par[root]=root;queue=[root]
                for v in queue:
                    for u,e in adjacency[v]:
                        if par[u]<0:par[u]=v;pe[u]=e;queue.append(u)
                parent.append(par);parent_edge.append(pe)
            for closing,(a,b) in enumerate(edges):
                if in_tree[closing]:continue
                path=[];v=int(b)
                while v!=a:path.append(int(parent_edge[a][v]));v=int(parent[a][v])
                if relation[closing,path].any():forbidden.add(tuple(sorted(path)))
    return sorted(forbidden,key=lambda path:(len(path),path))


def _problem(edges,p,allowed,forced,paths):
    m=len(edges);variables=3*m;M=p-1
    rr=[];cc=[];vv=[];rhs=[]
    def add(indices,values,b):
        row=len(rhs);rr.extend([row]*len(indices));cc.extend(indices);vv.extend(values);rhs.append(b)
    for e in range(m):add([e,m+2*e,m+2*e+1],[-M,1,1],0)
    for i in range(p):
        incident=np.flatnonzero(np.any(edges==i,axis=1))
        add(incident.tolist(),[-1]*len(incident),-1)
    for path in paths:add(list(path),[1]*len(path),len(path)-1)
    A=coo_matrix((vv,(rr,cc)),shape=(len(rhs),variables),dtype=np.int64).tocsr();b=np.array(rhs,np.int64)
    rr=[];cc=[];vv=[]
    for e,(i,j) in enumerate(edges):
        rr.append(0);cc.append(e);vv.append(1)
        for node,sgn in [(i,1),(j,-1)]:
            rr.extend([1+node,1+node]);cc.extend([m+2*e,m+2*e+1]);vv.extend([sgn,-sgn])
    E=coo_matrix((vv,(rr,cc)),shape=(p+1,variables),dtype=np.int64).tocsr()
    eb=np.r_[p-1,p-1,-np.ones(p-1,dtype=np.int64)].astype(np.int64)
    lower=np.zeros(variables,np.int64);upper=np.r_[np.ones(m,dtype=np.int64),np.full(2*m,M,dtype=np.int64)]
    lower[:m]=forced.astype(np.int64);upper[:m]=allowed.astype(np.int64)
    return A,b,E,eb,lower,upper


def dual_bound(result,c,A,b,E,eb,lower,upper):
    """A rigorous rational certificate, independent of floating-point KKT.

    Quantize solver multipliers to integers over 2**20. The inequality
    multipliers remain nonpositive. All matrix coefficients, right hand sides
    and variable boxes are integer. Exact residual support therefore gives a
    valid rational lower bound, whether or not solver multipliers were exact.
    The generous magnitude limit prevents int64 overflow in sparse products.
    """
    scale=2**20
    yfloat=np.minimum(result.ineqlin.marginals,0);zfloat=result.eqlin.marginals
    if max(np.max(np.abs(yfloat),initial=0),np.max(np.abs(zfloat),initial=0))>1e6:
        raise ArithmeticError('Dual multiplier magnitude exceeds validated limit')
    y=np.floor(yfloat*scale).astype(np.int64);z=np.rint(zfloat*scale).astype(np.int64)
    maximum_box=int(np.max(upper,initial=0))
    conservative_magnitude=(scale*int(np.abs(np.rint(c)).sum())*maximum_box
        +int(np.max(np.abs(y),initial=0))*(int(np.abs(b).sum())+maximum_box*int(np.abs(A.data).sum()))
        +int(np.max(np.abs(z),initial=0))*(int(np.abs(eb).sum())+maximum_box*int(np.abs(E.data).sum())))
    if conservative_magnitude>=2**62:
        raise ArithmeticError('Exact sparse integer arithmetic could overflow')
    residual=np.rint(c).astype(np.int64)*scale-A.T@y-E.T@z
    box=np.where(residual>=0,residual*lower,residual*upper)
    numerator=int(b@y)+int(eb@z)+int(box.sum())
    iy=np.flatnonzero(y);iz=np.flatnonzero(z)
    certificate=dict(numerator=numerator,denominator=scale,
        y_indices=iy.tolist(),y_numerators=y[iy].tolist(),
        z_indices=iz.tolist(),z_numerators=z[iz].tolist())
    return certificate


def refine_degrees(ans,p,pairs=None):
    edges=ans['edges'];m=len(edges)
    paths=path_constraints(ans['relation'],edges,p,ans['allowed'],ans['forced'])
    A,b,E,eb,lower,upper=_problem(edges,p,ans['allowed'],ans['forced'],paths)
    lo=ans['degree_lower'].copy();hi=ans['degree_upper'].copy()
    diagnostics=[]
    for i in range(p):
        c=np.zeros(3*m);c[:m]=np.any(edges==i,axis=1)
        for direction in [1,-1]:
            objective=direction*c
            result=linprog(objective,A_ub=A,b_ub=b,A_eq=E,b_eq=eb,
                bounds=np.column_stack([lower,upper]),method='highs',options={'threads':1})
            if not result.success:raise RuntimeError(result.message)
            certificate=dual_bound(result,objective,A,b,E,eb,lower,upper)
            numerator=certificate['numerator'];scale=certificate['denominator']
            value=-((-numerator)//scale) if direction==1 else (-numerator)//scale
            improves=value>lo[i] if direction==1 else value<hi[i]
            if direction==1:lo[i]=max(lo[i],value)
            else:hi[i]=min(hi[i],value)
            diagnostics.append(dict(vertex=i,direction=direction,primal=float(result.fun),
                dual=numerator/scale,integer_bound=value,
                certificate=certificate if improves else None))
    differences=ans['degree_difference_lower'].copy()
    if pairs is not None:
        for i,j in pairs:
            c=np.zeros(3*m);c[:m]=np.any(edges==i,axis=1).astype(int)-np.any(edges==j,axis=1).astype(int)
            result=linprog(c,A_ub=A,b_ub=b,A_eq=E,b_eq=eb,
                bounds=np.column_stack([lower,upper]),method='highs',options={'threads':1})
            if not result.success:raise RuntimeError(result.message)
            certificate=dual_bound(result,c,A,b,E,eb,lower,upper)
            numerator=certificate['numerator'];scale=certificate['denominator']
            value=-((-numerator)//scale);improves=value>differences[i,j]
            differences[i,j]=max(differences[i,j],value)
            diagnostics.append(dict(vertex=i,other=j,direction=1,primal=float(result.fun),
                dual=numerator/scale,integer_bound=value,
                certificate=certificate if improves else None))
    from order_inference import conservation_ranks
    rl,rh=conservation_ranks(lo,hi)
    rl=np.maximum(rl,1+np.sum(differences.T>0,axis=1))
    rh=np.minimum(rh,p-np.sum(differences>0,axis=1))
    assert np.all(lo<=hi) and np.all(rl<=rh)
    return {**ans,'degree_lower':lo,'degree_upper':hi,'degree_difference_lower':differences,
            'rank_lower':rl,'rank_upper':rh,'path_constraints':len(paths),'lp_diagnostics':diagnostics}
