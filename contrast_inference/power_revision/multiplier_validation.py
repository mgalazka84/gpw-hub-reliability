"""Strong asymptotic comparator on exactly the frozen validation datasets."""
from validate import *
from comparators import kendall_projection,multiplier_orders
def comparator_job(job):
 ci,cell,rep,config=job;seed=config['seed']+ci*100000+rep;p=cell['p'];x,corr=make_sample(cell,seed)
 edges=edge_index(p);truth=2/np.pi*np.arcsin(corr[edges[:,0],edges[:,1]])
 _,d=max_tree(truth,edges,p);unique=cell['model'] not in ['independent','discrete_independent','equicorrelated','tied_hubs'] and (d==d.max()).sum()==1
 t=time.perf_counter();tau,phi=kendall_projection(x);raw,_=multiplier_orders(tau,phi,p,B=999,seed=seed+40000000);cal=time.perf_counter()-t
 t=time.perf_counter();ans=project_order(raw,p);proj=time.perf_counter()-t
 row=dict(cell=ci,rep=rep,seed=seed,p=p,n=cell['n'],model=cell['model'],method='M-C',
  false_order=int(np.any(raw['relation']&(truth[:,None]<=truth[None,:]+1e-12))),
  hub=int(ans['rank_upper'][np.argmax(d)]==1) if unique else np.nan,
  certified_hubs=int((ans['rank_upper']==1).sum()),excluded=int((ans['rank_lower']>1).sum()),
  exact_degrees=int((ans['degree_lower']==ans['degree_upper']).sum()),
  degree_width=float(np.mean(ans['degree_upper']-ans['degree_lower'])),rank_width=float(np.mean(ans['rank_upper']-ans['rank_lower'])),
  orders=raw['raw_order_count'],combined_calibration_seconds=cal,projection_seconds=proj)
 return ci,rep,row
def run():
 dest=DEST.parent/'multiplier_validation';dest.mkdir(exist_ok=True);config=design()
 (dest/'design.json').write_text(json.dumps(dict(base_design=config,B=999,seed_offset=40000000,
   status='All frozen validation cells, matched samples. No changes to selected exact methods.'),indent=2))
 jobs=[(ci,c,r,config) for ci,c in enumerate(config['cells']) for r in range(c['repetitions']) if not (dest/f'cell{ci:02d}_{r:04d}.json').exists()]
 start=time.time()
 with ProcessPoolExecutor(max_workers=6) as ex:
  for k,f in enumerate(as_completed([ex.submit(comparator_job,j) for j in jobs]),1):
   ci,r,row=f.result();(dest/f'cell{ci:02d}_{r:04d}.json').write_text(json.dumps(row))
   if k%500==0:print(k,len(jobs),round(time.time()-start,1),flush=True)
 df=pd.DataFrame([json.loads(f.read_text()) for f in sorted(dest.glob('cell*.json'))]);df.to_csv(dest/'replications.csv.gz',index=False)
 df.groupby(['model','n','method']).mean(numeric_only=True).reset_index().to_csv(dest/'summary.csv',index=False)
 print('DONE multiplier',len(df),flush=True)
if __name__=='__main__':run()
