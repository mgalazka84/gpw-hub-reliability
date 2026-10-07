"""Direct verification of tied-data Kendall tau-a and the hub/tree distinction."""
from pathlib import Path
import json,itertools
import numpy as np
from hub_inference import *

rng=np.random.default_rng(9281);maximum_error=0
for n in [5,11,30]:
 for _ in range(30):
    x=rng.integers(-2,3,size=(n,5));edges=edge_index(5)
    tau,_,_,_=kendall_finite_band(x)
    expected=np.mean([np.sign(x[a]-x[b])[edges[:,0]]*np.sign(x[a]-x[b])[edges[:,1]]
                      for a,b in itertools.combinations(range(n),2)],axis=0)
    maximum_error=max(maximum_error,float(np.max(np.abs(tau-expected))))
    assert np.allclose(tau,expected)
p=6;R=np.eye(p)
for j in [3,4,5]:R[0,j]=R[j,0]=.16
for i,j in [(0,1),(0,2),(1,2)]:R[i,j]=R[j,i]=.12
edges=edge_index(p);w=R[edges[:,0],edges[:,1]]
lo,hi=degree_envelopes(w-.005,w+.005,edges,p);rl,ru=rank_envelopes(lo,hi)
assert lo.tolist()==[4,1,1,1,1,1] and hi.tolist()==[5,2,2,1,1,1]
assert rl[0]==ru[0]==1 and np.linalg.eigvalsh(R)[0]>0
report=dict(status='passed',kendall_tied_max_error=maximum_error,
            example_min_eigenvalue=float(np.linalg.eigvalsh(R)[0]),
            example_degree_lower=lo.tolist(),example_degree_upper=hi.tolist(),
            example_rank_lower=rl.tolist(),example_rank_upper=ru.tolist())
(Path(__file__).parent/'results/finite_band_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
