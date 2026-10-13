"""Known nonelliptical population example: Pearson and Kendall target different trees."""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'statistical_papers'))
from hub_inference import max_tree
from order_inference import edge_index

def run():
    p=12;a=np.r_[.99,np.linspace(.45,.55,p-1)];scales=np.r_[2.5,np.linspace(.3,1.5,p-1)]
    gaussian=np.outer(a,a)+np.diag(1-a*a)
    pearson=np.expm1(scales[:,None]*scales[None,:]*gaussian)/np.sqrt(np.expm1(scales[:,None]**2)*np.expm1(scales[None,:]**2))
    kendall=2/np.pi*np.arcsin(gaussian);edges=edge_index(p)
    kt,kd=max_tree(kendall[edges[:,0],edges[:,1]],edges,p);pt,pd=max_tree(pearson[edges[:,0],edges[:,1]],edges,p)
    assert kd[0]==11 and pd[0]==1
    rng=np.random.default_rng(8831921);x=rng.normal(size=(30,p));y=np.exp(x*scales)
    np.testing.assert_array_equal(np.sign(x[:,None]-x[None,:]),np.sign(y[:,None]-y[None,:]))
    report=dict(status='PASS',p=p,loadings=a.tolist(),lognormal_scales=scales.tolist(),
        kendall_degree=kd.tolist(),pearson_degree=pd.tolist(),shared_tree_edges=len(set(kt)&set(pt)),
        interpretation='Positive monotone coordinate transformations preserve Kendall kernels but change Pearson tree ordering')
    (Path(__file__).parent/'results'/'estimand_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':run()
