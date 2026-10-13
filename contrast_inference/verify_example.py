from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'statistical_papers'))
from hub_inference import degree_envelopes,rank_envelopes,max_tree
from order_inference import edge_index,local_comparisons,project_order,_closure
p=8;edges=edge_index(p)
def corr(t):
 R=np.eye(p)
 for i,j in edges:
  if i==0 and j<6:r=.06+t
  elif j<6:r=.04+t
  elif (i,j)==(0,6):r=.10
  elif (i,j)==(6,7):r=.15
  else:r=0
  R[i,j]=R[j,i]=r
 return R
matrices=[corr(t) for t in [-.025,.025]];w=np.array([r[edges[:,0],edges[:,1]] for r in matrices]);z=np.arctanh(w);z-=z.mean(axis=1,keepdims=True)
# Centered derivatives: changing edge derivative exceeds 1-15/(28*(1-.085**2))>0;
# unchanged edge derivative is strictly negative. Thus extrema are endpoints.
assert 1-15/(28*(1-.085**2))>0
lo,hi=degree_envelopes(z.min(axis=0),z.max(axis=0),edges,p);rl,rh=rank_envelopes(lo,hi)
a,b=local_comparisons(edges,p);relation=np.zeros((len(edges),len(edges)),bool)
for e,f in zip(a,b):
 if np.all(w[:,e]>w[:,f]):relation[e,f]=True
 if np.all(w[:,f]>w[:,e]):relation[f,e]=True
score=w.mean(axis=0);relation=_closure(relation,score);ans=project_order(dict(edges=edges,relation=relation,score=score),p)
assert lo.tolist()==[2,1,1,1,1,1,2,1] and hi.tolist()==[6,5,5,5,5,5,2,1]
assert [rl[0],rh[0]]==[1,7] and ans['rank_upper'][0]==1 and ans['forced'].sum()==7
for t in np.linspace(-.025,.025,101):
 R=corr(t);assert np.linalg.eigvalsh(R).min()>0;_,d=max_tree(R[edges[:,0],edges[:,1]],edges,p);assert d.tolist()==[6,1,1,1,1,1,2,1]
result=dict(status='PASS',box_degree_lower=lo.tolist(),box_degree_upper=hi.tolist(),box_hub_rank=[int(rl[0]),int(rh[0])],order_hub_rank=[int(ans['rank_lower'][0]),int(ans['rank_upper'][0])],order_forced_edges=int(ans['forced'].sum()),minimum_endpoint_eigenvalue=min(float(np.linalg.eigvalsh(R).min()) for R in matrices))
(Path(__file__).parent/'results/separation_example.json').write_text(json.dumps(result,indent=2));print(result)
