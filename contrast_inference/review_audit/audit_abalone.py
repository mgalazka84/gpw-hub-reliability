"""Reproduce the pooled-Abalone sharpness audit without changing calibration.

Run from any directory: python audit_abalone.py
Requires NumPy, pandas, SciPy, and a C++17 compiler. Uses the archived relations
from the accompanying repository, not a new statistical analysis.
"""
from pathlib import Path
import csv, hashlib, itertools, json, subprocess, sys, time
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from order_inference import edge_index, screen_edges, conservation_ranks
from path_projection import path_constraints, dual_bound
from run_comparative_simulations import verify_dual_python

NAMES=['Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']

def python_compatible(p,edges,selected,relation):
    """Independent edge-set enumeration + Boolean closure (not Pruefer/Kahn)."""
    adj=[[] for _ in range(p)]
    for e in selected:
        a,b=map(int,edges[e]);adj[a].append((b,int(e)));adj[b].append((a,int(e)))
    reached={0};todo=[0]
    for a in todo:
        for b,e in adj[a]:
            if b not in reached:reached.add(b);todo.append(b)
    if len(reached)!=p:return False
    reach=relation.copy()
    for e,(a,b) in enumerate(edges):
        if e in selected:continue
        parent={int(a):(None,None)};todo=[int(a)]
        for u in todo:
            for v,f in adj[u]:
                if v not in parent:parent[v]=(u,f);todo.append(v)
        v=int(b)
        while v!=a:
            v,f=parent[v];reach[f,e]=True
    for k in range(len(edges)):reach|=reach[:,k,None]&reach[None,k,:]
    return not np.trace(reach)

def exact_tree_polytope_lp(relation,edges,p):
    """Exact spanning-tree polytope + the manuscript's selected path cuts.

    The intersection remains an outer relaxation of compatible trees.
    Bounds use the existing integer/rational verifier, not float rounding.
    """
    m=len(edges);allowed,forced=screen_edges(relation,edges,p)
    paths=path_constraints(relation,edges,p,allowed,forced)
    mat=[];rhs=[]
    for bits in range(1,1<<p):
        n=bits.bit_count()
        if n<2 or n>=p:continue
        mat.append([int(bool(bits&(1<<int(a))) and bool(bits&(1<<int(b)))) for a,b in edges]);rhs.append(n-1)
    for path in paths:
        row=np.zeros(m,dtype=np.int64);row[list(path)]=1;mat.append(row);rhs.append(len(path)-1)
    A=csr_matrix(np.array(mat,dtype=np.int64));b=np.array(rhs,dtype=np.int64)
    E=csr_matrix(np.ones((1,m),dtype=np.int64));eb=np.array([p-1],dtype=np.int64)
    lower=forced.astype(np.int64);upper=allowed.astype(np.int64)
    incidence=np.array([np.any(edges==i,axis=1) for i in range(p)],dtype=np.int64)
    bounds=[];certs=[];primals=[]
    def minimum(c):
        result=linprog(c,A_ub=A,b_ub=b,A_eq=E,b_eq=eb,bounds=np.column_stack([lower,upper]),method='highs')
        if not result.success:raise RuntimeError(result.message)
        cert=dual_bound(result,c,A,b,E,eb,lower,upper)
        verify_dual_python(cert,c,A,b,E,eb,lower,upper)
        certs.append(cert);primals.append(float(result.fun))
        return -((-cert['numerator'])//cert['denominator'])
    for c in incidence:bounds.append([minimum(c),-minimum(-c)])
    differences=np.zeros((p,p),dtype=int)
    for i in range(p):
        for j in range(p):
            if i!=j:differences[i,j]=minimum(incidence[i]-incidence[j])
    lo,hi=np.array(bounds).T
    rl,rh=conservation_ranks(lo,hi)
    rl=np.maximum(rl,1+(differences.T>0).sum(axis=1))
    rh=np.minimum(rh,p-(differences>0).sum(axis=1))
    return dict(degree_lower=lo.tolist(),degree_upper=hi.tolist(),rank_lower=rl.tolist(),rank_upper=rh.tolist(),
                lp_objectives=len(certs),path_constraints=len(paths),rational_certificates=certs,
                unrounded_primal_objectives=primals)

def main():
    dest=HERE/'results';dest.mkdir(exist_ok=True)
    exe=HERE/'exhaustive_trees'
    subprocess.run(['g++','-O3','-std=c++17',str(HERE/'exhaustive_trees.cpp'),'-o',str(exe)],check=True)
    edges=edge_index(8);p=8;m=len(edges)
    source_e=ROOT/'results/evalue/cross_section/dataset00.json'
    source_p=ROOT/'results/cross_section/dataset00.json'
    spec=[('E64_local',source_e,'evalue_64','local'),('P_local',source_p,'paired_exact','local'),('E64_all',source_e,'evalue_64','all')]
    summaries=[];comparison=[]
    for label,source,method,family in spec:
        content=json.loads(source.read_text())
        record=next(c for c in content['certificates'] if c['method']==method and c['family']==family)
        rows=record['relation_rows'];rel=np.zeros((m,m),bool)
        for e,fs in enumerate(rows):rel[e,fs]=True
        pairs=np.argwhere(rel)
        manifest=dict(dataset='Abalone',stratum='pooled',p=p,n=4177,method=method,family=family,
            confidence=.95,edge_indexing='zero-based lexicographic unordered vertex pairs',
            relation_direction='row stronger than column; transitive closure',
            source_path=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            names=NAMES,edges=edges.tolist(),relation_rows=rows)
        (dest/f'{label}_relation.json').write_text(json.dumps(manifest,indent=2)+'\n')
        with (dest/f'{label}_relation.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['stronger_edge','stronger_u','stronger_v','weaker_edge','weaker_u','weaker_v'])
            for e,g in pairs:w.writerow([e,NAMES[edges[e,0]],NAMES[edges[e,1]],g,NAMES[edges[g,0]],NAMES[edges[g,1]]])
        inp=f'{p} {len(pairs)}\n'+''.join(f'{e} {f}\n' for e,f in pairs)
        begin=time.perf_counter()
        result=subprocess.run([str(exe)],input=inp,text=True,capture_output=True,check=True)
        elapsed=time.perf_counter()-begin
        header=['pruefer_code','edge_mask']+[f'd{i}' for i in range(p)]+[name for i in range(p) for name in [f'rank_lo{i}',f'rank_hi{i}']]
        (dest/f'{label}_compatible_trees.csv').write_text(','.join(header)+'\n'+result.stdout)
        df=pd.read_csv(dest/f'{label}_compatible_trees.csv')
        allowed,forced=screen_edges(rel,edges,p)
        mandatory=set(np.flatnonzero(forced));optional=np.flatnonzero(allowed&~forced)
        checked=set()
        for extra in itertools.combinations(optional,p-1-len(mandatory)):
            selected=mandatory|set(extra)
            if python_compatible(p,edges,selected,rel):checked.add(sum(1<<int(e) for e in selected))
        assert checked==set(df.edge_mask),'Independent enumerations disagree'
        assert len(df)==len(set(df.edge_mask)), 'Duplicate trees'
        bounds=dict(degree_lower=[int(df[f'd{i}'].min()) for i in range(p)],
                    degree_upper=[int(df[f'd{i}'].max()) for i in range(p)],
                    rank_lower=[int(df[f'rank_lo{i}'].min()) for i in range(p)],
                    rank_upper=[int(df[f'rank_hi{i}'].max()) for i in range(p)])
        strong=exact_tree_polytope_lp(rel,edges,p)
        (dest/f'{label}_strong_lp.json').write_text(json.dumps(strong,indent=2)+'\n')
        old=sorted([r for r in content['nodes'] if r['method']==method and r['family']==family],key=lambda r:r['vertex'])
        for i in range(p):
            for stage,answer in [('current_full_lp',{k:[r[k] for r in old] for k in bounds}),('exact_tree_polytope_lp',strong),('sharp_enumeration',bounds)]:
                comparison.append(dict(analysis=label,vertex=i,name=NAMES[i],stage=stage,**{k:answer[k][i] for k in bounds}))
        summaries.append(dict(analysis=label,enumerated=p**(p-2),compatible=len(df),closed_orders=len(pairs),
            independent_check=True,enumeration_seconds=elapsed,**bounds))
        print(label,json.dumps(summaries[-1]),flush=True)
    pd.DataFrame(comparison).to_csv(dest/'sharpness_comparison.csv',index=False)
    (dest/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')

if __name__=='__main__':main()
