"""Fixed annual sensitivity grid, liquidity selection and forward diagnostics.

All bootstrap endpoints are nominal. Changing the universe changes the tree
estimand. The future sample is never used for eligibility or stock selection.
"""
from pathlib import Path
import sys,argparse,json,hashlib,time,gzip,warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'statistical_papers'))
from hub_inference import bootstrap_covariances,correlation_edges,transform_cov,max_tree
from order_inference import edge_index,bootstrap_order,project_order
from path_projection import refine_degrees
ROOT=Path(__file__).parent;DATA={};CAPS=[10,20,40];SIZES=[252,504,1008]

def load(root):
 root=Path(root)
 panel=pd.read_csv(root/'data/close_simple_returns.csv.gz',index_col=0,parse_dates=True)
 vol=pd.read_csv(root/'data/volume_panel.csv.gz',index_col=0,parse_dates=True)
 close=pd.read_csv(root/'data/close_panel.csv.gz',index_col=0,parse_dates=True)
 wig=pd.read_csv(root/'data/wig_ohlc.csv',index_col=0,parse_dates=True)
 sel=pd.read_csv(root/'results/hub_selections.csv.gz',parse_dates=['origin_date'])
 sel=sel[(sel.convention=='close')&(sel.network=='raw')]
 factor=np.log(wig.close/wig.close.shift()).reindex(panel.index)
 return dict(panel=panel,vol=vol,close=close,selection=sel,factor=factor)

def design(root,B=999):
 d=load(root);sel=d['selection'];panel=d['panel'];orig=pd.Series(sorted(sel.origin_date.unique()));annual=orig.groupby(orig.dt.year).max()
 sets=[];jobs=[];exclusions=[]
 for oi,origin in enumerate(annual):
  loc=panel.index.get_loc(origin);names=sel[sel.origin_date==origin];base=names['isin'].tolist();available=[n for n in SIZES if loc+1>=n]
  longest=max(available);ix=panel.index[loc-longest+1:loc+1]
  ret=panel[base].loc[ix];vol=d['vol'][base].reindex(ix);close=d['close'][base].reindex(ix)
  good=ret.notna().all()&vol.notna().all()&close.notna().all()&(ret>-1).all()&((vol>0).mean()>=.9)&((ret==0).mean()<=.15)&(ret.var()>0)
  eligible=good[good].index.tolist();liquidity=(vol[eligible]*close[eligible]).median()
  ordered=sorted(eligible,key=lambda isin:(-float(liquidity[isin]),isin));origin_s=str(origin.date())
  eligible_caps=[cap for cap in CAPS if len(ordered)>=cap]
  for cap in CAPS:
   if len(ordered)<cap:
    for n in available:exclusions.append(dict(origin=origin_s,lookback=n,cap=cap,reason='fewer than cap eligible stocks on common longest lookback',eligible=len(ordered)))
   else:
    for rank,isin in enumerate(ordered[:cap],1):sets.append(dict(origin=origin_s,cap=cap,isin=isin,name=names.set_index('isin').loc[isin,'name_at_snapshot'],liquidity_rank=rank,selection_lookback=longest,eligible_count=len(ordered)))
  for n in SIZES:
   if n not in available:
    for cap in CAPS:exclusions.append(dict(origin=origin_s,lookback=n,cap=cap,reason='insufficient calendar history',eligible=0))
   else:
    jobs.append(dict(origin=origin_s,lookback=n,block=10,caps=eligible_caps,stocks=ordered[:max(eligible_caps)],name_map=names.set_index('isin').name_at_snapshot.to_dict(),seed=2306171+oi*100000+n*10+10,analysis='primary',selection_lookback=longest))
  if 1008 in available and 20 in eligible_caps:
   for block in [5,20]:jobs.append(dict(origin=origin_s,lookback=1008,block=block,caps=[20],stocks=ordered[:20],name_map=names.set_index('isin').name_at_snapshot.to_dict(),seed=2306171+oi*100000+1008*10+block,analysis='block_sensitivity',selection_lookback=longest))
 source_names=['data/close_simple_returns.csv.gz','data/volume_panel.csv.gz','data/close_panel.csv.gz','data/wig_ohlc.csv','results/hub_selections.csv.gz']
 config=dict(version='0.4',status='retrospective sensitivity design fixed before extended outcomes',origin_rule='last frozen origin in every represented calendar year',origins=[str(x.date()) for x in annual],lookbacks=SIZES,caps=CAPS,bootstrap_repetitions=B,blocks=dict(primary=10,sensitivity=[5,20],sensitivity_scope='raw, cap20, lookback1008'),alpha=.05,forward_sessions=252,seed=2306171,stock_rule='nested top median supplied-price times volume on longest calendar-available lookback; same stock set across shorter lookbacks',eligibility='complete returns, prices and volumes over longest window; positive volume on >=90% of sessions; <=15% zero returns; positive variance; origin membership from frozen eligible list',selection_lookback='1008 if calendar permits; otherwise 504',random_path_trees=8,exclusions=exclusions,jobs=jobs,source_sha256={name:hashlib.sha256((Path(root)/name).read_bytes()).hexdigest() for name in source_names},interpretation='nominal per-window Pearson bootstrap; no finite-sample guarantee or correction for the sensitivity grid; forward agreement is stability, not population coverage')
 (ROOT/'results/gpw_extended_design.json').write_text(json.dumps(config,indent=2))
 pd.DataFrame(sets).to_csv(ROOT/'results/gpw_extended_universes.csv',index=False)
 pd.DataFrame(exclusions).to_csv(ROOT/'results/gpw_extended_exclusions.csv',index=False)
 return config

def initialize(root,B):
 warnings.filterwarnings('ignore',category=OptimizeWarning);DATA.update(load(root));DATA['B']=B

def one(job):
 origin=pd.Timestamp(job['origin']);panel=DATA['panel'];loc=panel.index.get_loc(origin);n=job['lookback'];stocks=job['stocks'];q=len(stocks)
 x=np.log1p(panel[stocks].iloc[loc-n+1:loc+1].to_numpy());f=DATA['factor'].iloc[loc-n+1:loc+1].to_numpy()
 assert len(x)==n and np.isfinite(x).all() and np.isfinite(f).all()
 obs,boot=bootstrap_covariances(np.column_stack([x,f]),DATA['B'],job['block'],np.random.default_rng(job['seed']))
 records=[];nodes=[];certs=[]
 for p in job['caps']:
  positions=np.r_[np.arange(p),q];subobs=obs[np.ix_(positions,positions)];subboot=boot[:,positions,:][:,:,positions];edges=edge_index(p)
  kinds=['raw','demeaned','factor_residual'] if job['analysis']=='primary' else ['raw']
  future_x=panel[stocks[:p]].iloc[loc+1:loc+253].to_numpy();future_f=DATA['factor'].iloc[loc+1:loc+253].to_numpy()
  future_available=len(future_x)==252 and np.isfinite(future_x).all() and np.isfinite(future_f).all() and (future_x>-1).all()
  if future_available:
   future_data=np.column_stack([np.log1p(future_x),future_f]);future_data-=future_data.mean(axis=0);fcov=future_data.T@future_data/252
  for kind in kinds:
   r=correlation_edges(transform_cov(subobs,p,kind),edges);br=correlation_edges(transform_cov(subboot,p,kind),edges)
   tree,degree=max_tree(r,edges,p);anchor=int(np.argmax(degree));base=project_order(bootstrap_order(r,br,p),p)
   pairs=[(anchor,j) for j in range(p) if j!=anchor]+[(j,anchor) for j in range(p) if j!=anchor]
   ans=refine_degrees(base,p,pairs);counts=np.zeros(p,int)
   for weights in br:
    _,bd=max_tree(weights,edges,p);counts[bd==bd.max()]+=1
   future_degree=None;future_agreement=np.nan;future_rank_low=np.nan;future_rank_hi=np.nan
   if future_available:
    fr=correlation_edges(transform_cov(fcov,p,kind),edges);_,future_degree=max_tree(fr,edges,p)
    a,b=np.where(ans['relation']);future_agreement=float(np.mean(fr[a]>fr[b])) if len(a) else np.nan
    future_rank_low=int(1+(future_degree>future_degree[anchor]).sum());future_rank_hi=int((future_degree>=future_degree[anchor]).sum())
   pair=ans['degree_difference_lower'];strict=pair>0;np.fill_diagonal(strict,False)
   observed_rl=1+(degree[None,:]>degree[:,None]).sum(axis=1);observed_rh=(degree[None,:]>=degree[:,None]).sum(axis=1)
   certified_rank_fraction=float(np.mean((ans['rank_lower']>1)|(ans['rank_upper']<p)))
   record=dict(origin=job['origin'],lookback=n,cap=p,block=job['block'],network=kind,analysis=job['analysis'],seed=job['seed'],selection_lookback=job['selection_lookback'],anchor_isin=stocks[anchor],anchor_name=job['name_map'][stocks[anchor]],anchor_observed_degree=int(degree[anchor]),anchor_rank_lower=int(ans['rank_lower'][anchor]),anchor_rank_upper=int(ans['rank_upper'][anchor]),anchor_bootstrap_top_tied_frequency=float(counts[anchor]/DATA['B']),order_count=ans['raw_order_count'],closed_order_count=ans['closed_order_count'],forced_edges=int(ans['forced'].sum()),allowed_edges=int(ans['allowed'].sum()),path_constraints=ans['path_constraints'],critical=ans['critical'],minor_degree_width=float(np.mean(base['degree_upper']-base['degree_lower'])),mean_degree_width=float(np.mean(ans['degree_upper']-ans['degree_lower'])),mean_rank_width=float(np.mean(ans['rank_upper']-ans['rank_lower'])),rank_width_fraction=float(np.mean(ans['rank_upper']-ans['rank_lower'])/(p-1)),nonfull_rank_fraction=certified_rank_fraction,top1_candidates=int((ans['rank_lower']==1).sum()),top1_certified=int((ans['rank_upper']==1).sum()),topdecile_certified=int((ans['rank_upper']<=int(np.ceil(.1*p))).sum()),strict_degree_pairs=int(strict.sum()),future_available=bool(future_available),future_anchor_best_rank=future_rank_low,future_anchor_worst_rank=future_rank_hi,future_order_agreement=future_agreement,nontrivial_dual_certificates=sum(x['certificate'] is not None for x in ans['lp_diagnostics']))
   records.append(record)
   for i,isin in enumerate(stocks[:p]):nodes.append(dict(origin=job['origin'],lookback=n,cap=p,block=job['block'],network=kind,analysis=job['analysis'],isin=isin,name=job['name_map'][isin],degree=int(degree[i]),degree_lower=int(ans['degree_lower'][i]),degree_upper=int(ans['degree_upper'][i]),rank_lower=int(ans['rank_lower'][i]),rank_upper=int(ans['rank_upper'][i]),bootstrap_top_tied_frequency=float(counts[i]/DATA['B']),is_anchor=i==anchor,future_degree=int(future_degree[i]) if future_degree is not None else np.nan))
   certs.append(dict(origin=job['origin'],lookback=n,cap=p,block=job['block'],network=kind,analysis=job['analysis'],p=p,stocks=stocks[:p],order_pairs=np.column_stack(np.where(ans['relation'])).tolist(),allowed=np.flatnonzero(ans['allowed']).tolist(),forced=np.flatnonzero(ans['forced']).tolist(),certificates=[x for x in ans['lp_diagnostics'] if x['certificate'] is not None]))
 return records,nodes,certs

def run(root,B=999,workers=6):
 config=design(root,B);jobs=config['jobs'];dest=ROOT/'results';rows=[];nodes=[];start=time.time()
 with gzip.open(dest/'gpw_extended_certificates.jsonl.gz','wt') as stream:
  with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(str(root),B)) as pool:
   for i,(rr,nn,cc) in enumerate(pool.map(one,jobs,chunksize=1)):
    rows.extend(rr);nodes.extend(nn)
    for c in cc:stream.write(json.dumps(c,separators=(',',':'))+'\n')
    if (i+1)%3==0 or i+1==len(jobs):
     pd.DataFrame(rows).to_csv(dest/'gpw_extended_windows.csv',index=False);pd.DataFrame(nodes).to_csv(dest/'gpw_extended_nodes.csv.gz',index=False,compression='gzip');stream.flush()
     df=pd.DataFrame(rows);print('extended',i+1,'/',len(jobs),'analyses',len(df),'nonfull',int((df.nonfull_rank_fraction>0).sum()),'hubs',int(df.top1_certified.sum()),'seconds',round(time.time()-start),flush=True)
 df=pd.DataFrame(rows);group=['analysis','lookback','cap','block','network'];metrics=[c for c in df.select_dtypes(include=['number','bool']) if c not in group]
 summary=df.groupby(group)[metrics].mean().reset_index();counts=df.groupby(group).size().rename('windows').reset_index();summary=summary.merge(counts,on=group);summary.to_csv(dest/'gpw_extended_summary.csv',index=False)
 print(summary[['analysis','lookback','cap','block','network','windows','nonfull_rank_fraction','top1_certified','anchor_rank_upper']].to_string(index=False),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--data-root',required=True);a.add_argument('--bootstrap',type=int,default=999);a.add_argument('--workers',type=int,default=6);x=a.parse_args();run(x.data_root,x.bootstrap,x.workers)
