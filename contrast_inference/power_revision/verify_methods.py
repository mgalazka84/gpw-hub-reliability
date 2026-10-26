from pathlib import Path
import itertools,json
import numpy as np
from methods import *
from evalue_order import evalue_order
def run():
 rng=np.random.default_rng(9184);checks=[]
 # Independent Python log-wealth recursion checks ties and both directions.
 for tied in [False,True]:
  x=rng.normal(size=(42,6));x=np.round(x) if tied else x
  edges,a,b,_,_=fixed_comparison_indices(6,'local');s=np.sign(x[::2]-x[1::2]);h=s[:,edges[:,0]]*s[:,edges[:,1]]
  got=adaptive_logs(h,a,b);expected=[]
  for direction in [1,-1]:
   for aa,bb in zip(a,b):
    S=0.;Q=4.;wealth=0.
    for z in direction*(h[:,aa]-h[:,bb])/2:
     lam=min(.5,max(0,S/Q));wealth+=np.log1p(lam*z);S+=z;Q+=z*z
    expected.append(wealth)
  assert np.allclose(got,expected,atol=1e-12);checks.append('adaptive recursion including ties '+str(tied))
 # Conditional fair-ternary null: sum probabilities over all 3^7 paths.
 for q in [.05,.3,.5]:
  mean=0
  for path in itertools.product([-1,0,1],repeat=7):
   prob=np.prod([q if z else 1-2*q for z in path]);S=0.;Q=4.;wealth=1.
   for z in path:
    lam=min(.5,max(0,S/Q));wealth*=1+lam*z;S+=z;Q+=z*z
   mean+=prob*wealth
  assert abs(mean-1)<1e-11;checks.append('exact null expectation q='+str(q))
 # On continuous data E64 must agree with the independent legacy code.
 x=rng.normal(size=(180,7));edges,a,b,_,_=fixed_comparison_indices(7,'local')
 actual=evidence(x,a,b,64,38112)['E64'];expected=evalue_order(x,seed=38112)['pvalues']
 assert np.allclose(actual,expected,atol=2e-12,rtol=2e-12);checks.append('E64 agrees with legacy implementation')
 # Every candidate-tree path comparison has a tree edge on the left only.
 for p in range(3,10):
  edges=edge_index(p);tau=rng.normal(size=len(edges));tree,degree=max_tree(tau,edges,p)
  a,b=tree_constraints(tree,edges,p)
  assert np.isin(a,tree).all() and not np.isin(b,tree).any() and (tau[a]>=tau[b]).all()
  ans=order_from_rejections(np.isin(np.arange(len(edges)),tree).astype(float),edges,a,b,np.ones(len(a),bool),False)
  out=project_order(ans,p)
  assert np.array_equal(out['degree_lower'],degree) and np.array_equal(out['degree_upper'],degree)
 checks.append('complete path certificate reproduces candidate degrees p=3..9')
 # Tied training scores must still admit a deterministic topological score.
 x=rng.integers(1,3,size=(80,6)).astype(float);out,meta=split_orders(x,R=4,seed=513)
 checks.append('split construction with tied training scores')
 result={'passed':True,'checks':checks}
 path=Path(__file__).parent/'results'/'method_verification.json';path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2))
if __name__=='__main__':run()
