from pathlib import Path
import sys,json,itertools
import numpy as np,pandas as pd
from methods import *
from application import load_data
sys.path.insert(0,str(ROOT/'review_audit'))
from audit_abalone import python_compatible,exact_tree_polytope_lp

def run():
 dest=Path(__file__).parent/'results'/'application';edges=edge_index(10);checks=[];summary=[]
 for dataset in ['adults','all_complete_ages']:
  for method in ['E64','FS-T64','A64','M-C']:
   path=dest/f'{dataset}_{method}.json';r=json.loads(path.read_text());a=r['lp'];rel=np.array(a['relation'],bool)
   forced=set(np.flatnonzero(a['forced']));optional=np.flatnonzero(np.array(a['allowed'])&~np.array(a['forced']))
   expected={tuple(sorted(t)) for t in r['sharp']['trees']};actual=set()
   for choice in itertools.combinations(optional,9-len(forced)):
    tree=forced|set(choice)
    if python_compatible(10,edges,tree,rel):actual.add(tuple(sorted(tree)))
   assert actual==expected
   ds=np.array([np.bincount(edges[list(t)].ravel(),minlength=10) for t in sorted(actual)])
   maximum=ds.max(axis=1);members=np.all(ds==maximum[:,None],axis=0)
   possible_unique=bool(np.any((ds==maximum[:,None]).sum(axis=1)==1))
   exact=dict(guaranteed_maximum_degree_items=['Q'+str(i+1) for i in np.flatnonzero(members)],
              unique_hub_possible=possible_unique,trees_checked=len(actual))
   # Independent strong ST-polytope comparison uses all subset inequalities.
   strong=exact_tree_polytope_lp(rel,edges,10)
   (dest/f'{dataset}_{method}_strong_lp.json').write_text(json.dumps(strong))
   exact['strong_polytope_matches_degree_bounds']=all(strong[k]==r['sharp'][k] for k in ['degree_lower','degree_upper'])
   exact['strong_polytope_matches_rank_bounds']=all(strong[k]==r['sharp'][k] for k in ['rank_lower','rank_upper'])
   for k in ['degree_lower','rank_lower']:assert np.all(np.array(strong[k])<=r['sharp'][k])
   for k in ['degree_upper','rank_upper']:assert np.all(np.array(strong[k])>=r['sharp'][k])
   summary.append(dict(dataset=dataset,method=method,**exact));checks.append(f'{dataset} {method}: independent tree sets and 110 rational LP bounds')
 sets,meta=load_data();ties={}
 for dataset,x in sets.items():
  tau=fast_kendall_band(x)[0];tie=np.array(meta[dataset]['pair_tie_probability'])
  taub=tau/np.sqrt((1-tie[edges[:,0]])*(1-tie[edges[:,1]]))
  ta,da=max_tree(tau,edges,10);tb,db=max_tree(taub,edges,10)
  ties[dataset]=dict(tau_a_tree=ta.tolist(),tau_b_tree=tb.tolist(),shared_edges=len(set(ta)&set(tb)),
                    tau_a_degrees=da.tolist(),tau_b_degrees=db.tolist(),
                    interpretation='Point-estimate sensitivity only. Tau-a confidence statements do not apply to tau-b.')
 out=dict(passed=True,checks=checks,shared_hub_conclusions=summary,tie_normalization_sensitivity=ties)
 (dest/'independent_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':run()
