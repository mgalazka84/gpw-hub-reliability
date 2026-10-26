"""Specified RSE application. Download raw data separately; do not redistribute."""
from pathlib import Path
import json,time,zipfile,io,hashlib,math,itertools
import numpy as np,pandas as pd
from methods import *
from comparators import kendall_projection,multiplier_orders
from run_comparative_simulations import certify_lp
BASE=Path(__file__).parent

def sharp_if_small(ans,p,limit=500000):
 """Exact compatible-tree bounds by restricted edge-set enumeration.

  The restriction to allowed edges and forced edges is lossless relative to
  the input relation. Acyclicity of strict orders plus weak path-optimality
  inequalities is necessary and sufficient; see review_audit/README.txt.
 """
 edges=ans['edges'];rel=ans['relation'];m=len(edges)
 forced=np.flatnonzero(ans['forced']);optional=np.flatnonzero(ans['allowed']&~ans['forced'])
 need=p-1-len(forced);count=math.comb(len(optional),need)
 if count>limit:return dict(status='not_enumerated',candidate_subsets=count)
 dlo=np.full(p,p-1);dhi=np.ones(p,int);rlo=np.full(p,p);rhi=np.ones(p,int);accepted=[]
 relation_arcs=[list(map(int,np.flatnonzero(row))) for row in rel]
 for choice in itertools.combinations(optional,need):
  tree=np.r_[forced,choice].astype(int);parent=list(range(p));adj=[[] for _ in range(p)];ok=True
  def root(i):
   while parent[i]!=i:i=parent[i]
   return i
  for e in tree:
   u,v=map(int,edges[e]);a,b=root(u),root(v)
   if a==b:ok=False;break
   parent[a]=b;adj[u].append((v,int(e)));adj[v].append((u,int(e)))
  if not ok:continue
  arcs=[v.copy() for v in relation_arcs];treeset=set(map(int,tree))
  for e,(u,v) in enumerate(edges):
   if e in treeset:continue
   stack=[(int(u),-1,[])];route=None
   while stack:
    w,last,path=stack.pop()
    if w==v:route=path;break
    for z,f in adj[w]:
     if z!=last:stack.append((z,w,path+[f]))
   assert route is not None
   for f in route:arcs[f].append(e)
  indegree=np.zeros(m,int)
  for aa in arcs:
   for b in aa:indegree[b]+=1
  queue=list(np.flatnonzero(indegree==0));visited=0
  while queue:
   a=queue.pop();visited+=1
   for b in arcs[a]:
    indegree[b]-=1
    if indegree[b]==0:queue.append(b)
  if visited!=m:continue
  degree=np.bincount(edges[tree].ravel(),minlength=p)
  lo=1+(degree[None,:]>degree[:,None]).sum(1);hi=(degree[None,:]>=degree[:,None]).sum(1)
  dlo=np.minimum(dlo,degree);dhi=np.maximum(dhi,degree);rlo=np.minimum(rlo,lo);rhi=np.maximum(rhi,hi)
  accepted.append(tree.tolist())
 assert accepted,'No compatible tree; cannot report a certificate'
 return dict(status='exact',candidate_subsets=count,compatible_trees=len(accepted),
  degree_lower=dlo.tolist(),degree_upper=dhi.tolist(),rank_lower=rlo.tolist(),rank_upper=rhi.tolist(),trees=accepted)

def load_data():
 raw=BASE/'data'/'RSE.zip'
 with zipfile.ZipFile(raw) as z:
  data=z.read('RSE/data.csv');book=z.read('RSE/codebook.txt').decode()
 df=pd.read_csv(io.BytesIO(data),sep='\t');items=[f'Q{i}' for i in range(1,11)]
 complete=df[items].isin([1,2,3,4]).all(axis=1)
 adult=df.age.between(18,100)
 meta=dict(source='https://openpsychometrics.org/_rawdata/RSE.zip',raw_zip_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
  source_rows=len(df),complete_rows=int(complete.sum()),complete_adults=int((complete&adult).sum()),
  incomplete_rows=int((~complete).sum()),age_exclusions_among_complete=int((complete&~adult).sum()),
  reverse_coded=['Q3','Q5','Q8','Q9','Q10'],criterion='items 1..4 and age 18..100; preserve duplicate answer patterns',
  license_provenance='Original source explicitly releases anonymous records for research; IRW lists its RSE release CC BY 4.0. Raw records omitted from redistribution.',
  primary_question='Exclude items from highest degree in the signed tau-a maximum spanning tree after direction coding.',
  iid_limitation='Anonymous convenience responses. Independent identical respondents are a working model; repeat participation cannot be checked.')
 sets={}
 for name,mask in [('adults',complete&adult),('all_complete_ages',complete)]:
  x=df.loc[mask,items].to_numpy(float)
  x[:,[2,4,7,8,9]]=5-x[:,[2,4,7,8,9]]
  sets[name]=x
  _,cnt=np.unique(x,axis=0,return_counts=True)
  tie=[float(sum(c*(c-1) for c in np.unique(x[:,i],return_counts=True)[1])/(len(x)*(len(x)-1))) for i in range(10)]
  meta[name]=dict(n=len(x),unique_answer_profiles=len(cnt),rows_beyond_first_profile=int((cnt-1).sum()),
                 pair_tie_probability=tie)
 return sets,meta

def serialize(ans):
 keys=['relation','degree_lower','degree_upper','rank_lower','rank_upper','allowed','forced','raw_order_count','closed_order_count']
 return {k:ans[k].tolist() if hasattr(ans[k],'tolist') else ans[k] for k in keys}

def run():
 dest=BASE/'results'/'application';dest.mkdir(parents=True,exist_ok=True)
 sets,meta=load_data();(dest/'provenance.json').write_text(json.dumps(meta,indent=2));rows=[];records=[]
 for name,x in sets.items():
  seed=61023117;p=x.shape[1];edges=edge_index(p);tau=fast_kendall_band(x)[0];tree,degree=max_tree(tau,edges,p)
  out={};t=time.perf_counter();full=full_orders(x,seed=seed+100000);fulltime=time.perf_counter()-t
  t=time.perf_counter();split,diag=split_orders(x,seed=seed);splittime=time.perf_counter()-t
  out.update(full);out.update(split)
  t=time.perf_counter();ts,phi=kendall_projection(x);mc,_=multiplier_orders(ts,phi,p,B=1999,seed=seed+300000)
  out['M-C']=mc;mctime=time.perf_counter()-t
  metadata=dict(dataset=name,n=len(x),point_tree=tree.tolist(),point_degrees=degree.tolist(),tau=tau.tolist(),split=diag,
                combined_full_seconds=fulltime,combined_split_seconds=splittime,multiplier_seconds=mctime)
  (dest/f'{name}_design.json').write_text(json.dumps(metadata,indent=2))
  for method,raw in out.items():
   t=time.perf_counter();minor=project_order(raw,p);lp=certify_lp(minor,p,degree_pairs=False)
   projectiontime=time.perf_counter()-t
   # Exact audit for predeclared primary and main general-purpose comparator.
   sharp=sharp_if_small(lp,p) if method in ['FS-T64','E64','A64','M-C'] else dict(status='not_requested')
   final=sharp if sharp['status']=='exact' else lp
   result=dict(dataset=name,method=method,n=len(x),minor=serialize(minor),lp=serialize(lp),sharp=sharp,
               projection_seconds=projectiontime,lp_solves=lp.get('lp_solves',0),
               verified_certificates=lp.get('python_verified_certificates',0))
   (dest/f'{name}_{method}.json').write_text(json.dumps(result))
   for i in range(p):
    rows.append(dict(dataset=name,method=method,item='Q'+str(i+1),point_degree=int(degree[i]),
       degree_lower=int(final['degree_lower'][i]),degree_upper=int(final['degree_upper'][i]),
       rank_lower=int(final['rank_lower'][i]),rank_upper=int(final['rank_upper'][i]),
       sharp=sharp['status']=='exact',minor_rank_lower=int(minor['rank_lower'][i]),
       minor_rank_upper=int(minor['rank_upper'][i]),lp_rank_lower=int(lp['rank_lower'][i]),lp_rank_upper=int(lp['rank_upper'][i])))
   records.append(dict(dataset=name,method=method,n=len(x),orders=raw['raw_order_count'],
     exact_degrees=int(sum(a==b for a,b in zip(final['degree_lower'],final['degree_upper']))),
     excluded=int(sum(a>1 for a in final['rank_lower'])),
     certified_hubs=int(sum(a==1 for a in final['rank_upper'])),
     mean_degree_width=float(np.mean(np.array(final['degree_upper'])-final['degree_lower'])),
     mean_rank_width=float(np.mean(np.array(final['rank_upper'])-final['rank_lower'])),
     forced_edges=int(minor['forced'].sum()),allowed_edges=int(minor['allowed'].sum()),
     projection_seconds=projectiontime,sharp_status=sharp['status'],
     compatible_trees=sharp.get('compatible_trees',None)))
   print(name,method,'excluded',records[-1]['excluded'],'exact degrees',records[-1]['exact_degrees'],'sharp',sharp['status'],flush=True)
 pd.DataFrame(rows).to_csv(dest/'nodes.csv',index=False);pd.DataFrame(records).to_csv(dest/'summary.csv',index=False)
 print('DONE',json.dumps(meta,indent=2),flush=True)
if __name__=='__main__':run()
