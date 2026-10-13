"""Independent finite checks; never substitutes for the proofs."""
from pathlib import Path
import itertools,json,sys,warnings,gzip,time
import numpy as np
from scipy.stats import binom
from scipy.optimize import OptimizeWarning
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'statistical_papers'))
from hub_inference import kendall_finite_band,degree_envelopes,rank_envelopes
from order_inference import *
from path_projection import refine_degrees,path_constraints,_problem
warnings.filterwarnings('ignore',category=OptimizeWarning)
ROOT=Path(__file__).parent

def trees(p):
 edges=edge_index(p);index={tuple(e):i for i,e in enumerate(edges)};out=[]
 for code in itertools.product(range(p),repeat=p-2):
  d=np.ones(p,int)+np.bincount(code,minlength=p);selected=[]
  for j in code:
   i=int(np.flatnonzero(d==1)[0]);selected.append(index[tuple(sorted((i,j)))]);d[i]-=1;d[j]-=1
  selected.append(index[tuple(np.flatnonzero(d==1))]);row=np.zeros(len(edges),int);row[selected]=1;out.append(row)
 return np.array(out),edges

def verify_certificate(rec,edges,p,allowed,forced,paths,problem=None):
 A,b,E,h,l,u=_problem(edges,p,allowed,forced,paths) if problem is None else problem;m=len(edges)
 cert=rec['certificate'];scale=cert['denominator'];y=np.zeros(A.shape[0],np.int64);z=np.zeros(E.shape[0],np.int64)
 y[cert['y_indices']]=cert['y_numerators'];z[cert['z_indices']]=cert['z_numerators'];assert (y<=0).all()
 c=np.zeros(3*m,int);c[:m]=np.any(edges==rec['vertex'],axis=1)
 if 'other' in rec:c[:m]-=np.any(edges==rec['other'],axis=1).astype(int)
 c*=rec['direction']
 # Python integers independently avoid floating-point and fixed-width overflow.
 residual=[int(v)*scale for v in c]
 for mat,mult in [(A,y),(E,z)]:
  for i in np.flatnonzero(mult):
   for k in range(mat.indptr[i],mat.indptr[i+1]):residual[mat.indices[k]]-=int(mult[i])*int(mat.data[k])
 total=sum(int(v)*int(w) for v,w in zip(b,y))+sum(int(v)*int(w) for v,w in zip(h,z))
 total+=sum(min(v*int(a),v*int(b)) for v,a,b in zip(residual,l,u))
 assert total==cert['numerator']
 bound=-((-total)//scale) if rec['direction']==1 else (-total)//scale
 assert bound==rec['integer_bound']
 return True

def run():
 rng=np.random.default_rng(671390);counts=dict(minor_instances=0,conservation_instances=0,lp_instances=0,lp_certificates=0,full_kendall_comparisons=0,binomial_checks=0)
 for p in range(3,7):
  ts,edges=trees(p);inc=(edges[:,:,None]==np.arange(p)).any(axis=1).astype(int);degrees=ts@inc
  for rep in range(75):
   witness=ts[rng.integers(len(ts))];forced=(witness>0)&(rng.random(len(edges))<.3);allowed=(witness>0)|(rng.random(len(edges))<.6)
   valid=(ts[:,forced].sum(axis=1)==forced.sum())&(ts[:,~allowed].sum(axis=1)==0);d=degrees[valid]
   lo,hi,diff=minor_envelopes(edges,p,allowed,forced)
   assert np.array_equal(lo,d.min(axis=0)) and np.array_equal(hi,d.max(axis=0))
   assert np.array_equal(diff,(d[:,:,None]-d[:,None,:]).min(axis=0));counts['minor_instances']+=1
   rel=(degrees>=lo).all(axis=1)&(degrees<=hi).all(axis=1);dd=degrees[rel]
   rl,rh=conservation_ranks(lo,hi)
   true_lo=(1+(dd[:,None,:]>dd[:,:,None]).sum(axis=2)).min(axis=0)
   true_hi=(dd[:,None,:]>=dd[:,:,None]).sum(axis=2).max(axis=0)
   assert np.array_equal(rl,true_lo) and np.array_equal(rh,true_hi);counts['conservation_instances']+=1
  for rep in range(8):
   w=rng.integers(0,5,len(edges));relation=(w[:,None]>w[None,:])&(rng.random((len(edges),len(edges)))<.25)
   relation=_closure_public(relation,w)
   ans=project_order(dict(relation=relation,score=w,edges=edges),p)
   lp=refine_degrees(ans,p,[(i,j) for i in range(p) for j in range(p) if i!=j]);weights=ts@w;opt=ts[weights==weights.max()];od=degrees[weights==weights.max()]
   paths=path_constraints(relation,edges,p,ans['allowed'],ans['forced'])
   for path in paths:assert np.all(opt[:,path].sum(axis=1)<=len(path)-1)
   assert np.all(lp['degree_lower']<=od.min(axis=0)) and np.all(lp['degree_upper']>=od.max(axis=0))
   assert np.all(lp['degree_difference_lower']<=(od[:,:,None]-od[:,None,:]).min(axis=0))
   for rec in lp['lp_diagnostics']:
    if rec['certificate'] is not None:verify_certificate(rec,edges,p,ans['allowed'],ans['forced'],paths);counts['lp_certificates']+=1
   counts['lp_instances']+=1
 for rep in range(30):
  x=rng.normal(size=(10+rep,5)) if rep%2 else rng.integers(-2,3,size=(10+rep,5)).astype(float)
  if rep%3==0:x[:,0]=1
  assert np.allclose(fast_kendall_band(x)[0],kendall_finite_band(x)[0],atol=2e-15);counts['full_kendall_comparisons']+=1
 for N in [0,1,2,5,20,100,1000]:
  k=np.arange(N+1);pv=binom.sf(k-1,N,.5)
  for q in [0,.1,.49,.5]:
   pm=binom.pmf(k,N,q)
   for level in [.0001,.001,.01,.05,.1,.5,1]:assert pm[pv<=level].sum()<=level+1e-12;counts['binomial_checks']+=1
 # Direct discordance counts independently reproduce optimized Gram computation.
 for rep in range(30):
  x=rng.normal(size=(40,7));ans=kendall_order(x,close=False);edges=ans['edges'];s=np.sign(x[::2]-x[1::2]);h=s[:,edges[:,0]]*s[:,edges[:,1]];a,b=ans['comparison_a'],ans['comparison_b']
  A=((h[:,a]==1)&(h[:,b]==-1)).sum(axis=0);B=((h[:,a]==-1)&(h[:,b]==1)).sum(axis=0)
  assert np.array_equal(ans['pvalues'],np.r_[binom.sf(A-1,A+B,.5),binom.sf(B-1,A+B,.5)])
 counts['direct_discordance_comparisons']=30
 counts['status']='PASS';(ROOT/'results/verification.json').write_text(json.dumps(counts,indent=2));print(counts,flush=True)

def _closure_public(r,w):
 from order_inference import _closure
 return _closure(r,w)

def gpw():
 start=time.time();windows=0;certificates=0
 with gzip.open(ROOT/'results/gpw_refined_certificates.jsonl.gz','rt') as f:
  for line in f:
   obj=json.loads(line);p=obj['p'];edges=edge_index(p);m=len(edges);r=np.zeros((m,m),bool)
   if obj['order_pairs']:
    a,b=np.array(obj['order_pairs']).T;r[a,b]=True
   allowed=np.zeros(m,bool);allowed[obj['allowed']]=True;forced=np.zeros(m,bool);forced[obj['forced']]=True
   aa,ff=screen_edges(r,edges,p);assert np.array_equal(aa,allowed) and np.array_equal(ff,forced)
   paths=path_constraints(r,edges,p,allowed,forced)
   problem=_problem(edges,p,allowed,forced,paths)
   for rec in obj['certificates']:verify_certificate(rec,edges,p,allowed,forced,paths,problem);certificates+=1
   windows+=1
   if windows%60==0:print('GPW certificates',windows,certificates,round(time.time()-start),flush=True)
 result=dict(status='PASS',windows=windows,certificates=certificates,arithmetic='independent Python integer dual support verification')
 (ROOT/'results/gpw_certificate_verification.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':
 if '--gpw' in sys.argv:gpw()
 else:run()
