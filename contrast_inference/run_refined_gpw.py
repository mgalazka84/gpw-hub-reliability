"""All frozen GPW windows, all representations, with rational LP witnesses."""
from pathlib import Path
import argparse,hashlib,json,time,sys,gzip,warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'statistical_papers'))
from hub_inference import bootstrap_covariances,correlation_edges,transform_cov,max_tree
from order_inference import edge_index,bootstrap_order,project_order
from path_projection import refine_degrees

ROOT=Path(__file__).parent;DATA={}
def initialize(data_root,B,block):
 warnings.filterwarnings('ignore',category=OptimizeWarning)
 root=Path(data_root)
 panel=pd.read_csv(root/'data/close_simple_returns.csv.gz',index_col=0,parse_dates=True)
 wig=pd.read_csv(root/'data/wig_ohlc.csv',index_col=0,parse_dates=True)
 selection=pd.read_csv(root/'results/hub_selections.csv.gz',parse_dates=['origin_date'])
 DATA.update(panel=panel,factor=np.log(wig.close/wig.close.shift()).reindex(panel.index),selection=selection,
  selected=selection[(selection.convention=='close')&(selection.network=='raw')],B=B,block=block)

def one_origin(origin):
 panel=DATA['panel'];names=DATA['selected'][DATA['selected'].origin_date==origin]
 stocks=names['isin'].tolist();p=len(stocks);edges=edge_index(p);loc=panel.index.get_loc(origin)
 x=np.log1p(panel[stocks].iloc[loc-251:loc+1].to_numpy());f=DATA['factor'].iloc[loc-251:loc+1].to_numpy()
 if len(x)!=252 or not np.isfinite(x).all() or not np.isfinite(f).all():raise ValueError('Frozen window failed complete-data check')
 obs,boots=bootstrap_covariances(np.column_stack([x,f]),DATA['B'],DATA['block'],np.random.default_rng(803177+loc))
 records=[];nodes=[];certificates=[]
 for kind in ['raw','demeaned','factor_residual']:
  r=correlation_edges(transform_cov(obs,p,kind),edges);br=correlation_edges(transform_cov(boots,p,kind),edges)
  tree,degree=max_tree(r,edges,p);anchor=int(np.argmax(degree))
  base=project_order(bootstrap_order(r,br,p),p)
  pairs=[(anchor,j) for j in range(p) if j!=anchor]+[(j,anchor) for j in range(p) if j!=anchor]
  ans=refine_degrees(base,p,pairs=pairs)
  k=int(np.ceil(.1*p));origin_s=str(pd.Timestamp(origin).date())
  record=dict(origin=origin_s,network=kind,p=p,k=k,anchor_isin=stocks[anchor],anchor_observed_degree=int(degree[anchor]),
   order_count=ans['raw_order_count'],closed_order_count=ans['closed_order_count'],forced_edges=int(ans['forced'].sum()),
   allowed_edges=int(ans['allowed'].sum()),path_constraints=ans['path_constraints'],critical=ans['critical'],
   minor_mean_degree_width=float(np.mean(base['degree_upper']-base['degree_lower'])),
   mean_degree_width=float(np.mean(ans['degree_upper']-ans['degree_lower'])),
   mean_rank_width=float(np.mean(ans['rank_upper']-ans['rank_lower'])),
   full_rank_fraction=float(np.mean((ans['rank_lower']==1)&(ans['rank_upper']==p))),
   top1_candidates=int((ans['rank_lower']==1).sum()),top1_certified=int((ans['rank_upper']==1).sum()),
   topk_certified=int((ans['rank_upper']<=k).sum()),anchor_rank_lower=int(ans['rank_lower'][anchor]),anchor_rank_upper=int(ans['rank_upper'][anchor]),
   nontrivial_dual_certificates=sum(d['certificate'] is not None for d in ans['lp_diagnostics']))
  records.append(record)
  for i,isin in enumerate(stocks):
   nodes.append(dict(origin=origin_s,network=kind,isin=isin,name=names.name_at_snapshot.iloc[i],
    observed_degree=int(degree[i]),degree_lower=int(ans['degree_lower'][i]),degree_upper=int(ans['degree_upper'][i]),
    rank_lower=int(ans['rank_lower'][i]),rank_upper=int(ans['rank_upper'][i]),is_anchor=i==anchor))
  # Derived orders and integer witnesses, without prices or return panels.
  certificate_record=dict(origin=origin_s,network=kind,p=p,stocks=stocks,
   order_pairs=np.column_stack(np.where(ans['relation'])).tolist(),
   allowed=np.flatnonzero(ans['allowed']).tolist(),forced=np.flatnonzero(ans['forced']).tolist(),
   certificates=[d for d in ans['lp_diagnostics'] if d['certificate'] is not None])
  certificates.append(certificate_record)
 return records,nodes,certificates

def run(root,B=999,block=10,workers=4,stride=1,output='gpw_refined'):
 root=Path(root);dest=ROOT/'results';dest.mkdir(exist_ok=True)
 selection=pd.read_csv(root/'results/hub_selections.csv.gz',parse_dates=['origin_date'])
 origins=sorted(selection[(selection.convention=='close')&(selection.network=='raw')].origin_date.unique())[::stride]
 sources=['data/close_simple_returns.csv.gz','data/wig_ohlc.csv','results/hub_selections.csv.gz']
 config=dict(origins=len(origins),networks=['raw','demeaned','factor_residual'],bootstrap_repetitions=B,block=block,
  alpha=.05,lookback=252,seed=803177,random_path_trees=8,
  projection='minor pair extrema, cycle-path LP, rational dual bounds, integer conservation ranks',
  anchor='largest estimated tree degree, first frozen-universe index for a tie',
  source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources},
  guarantee='nominal, fixed-window bootstrap; no IID finite-sample guarantee for GPW returns')
 (dest/(output+'_design.json')).write_text(json.dumps(config,indent=2))
 rows=[];nodes=[];start=time.time()
 with gzip.open(dest/(output+'_certificates.jsonl.gz'),'wt') as stream:
  with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(str(root),B,block)) as executor:
   for i,(records,newnodes,certs) in enumerate(executor.map(one_origin,origins,chunksize=1)):
    rows.extend(records);nodes.extend(newnodes)
    for certificate in certs:stream.write(json.dumps(certificate,separators=(',',':'))+'\n')
    if (i+1)%5==0 or i+1==len(origins):
     pd.DataFrame(rows).to_csv(dest/(output+'_windows.csv'),index=False)
     pd.DataFrame(nodes).to_csv(dest/(output+'_nodes.csv.gz'),index=False,compression='gzip')
     stream.flush();print(f'{output}: {i+1}/{len(origins)}, {time.time()-start:.1f}s',flush=True)
 df=pd.DataFrame(rows)
 metrics=[c for c in df if c not in ['origin','network','anchor_isin']]
 summary=df.groupby('network')[metrics].mean().reset_index()
 summary['origins']=len(origins)
 summary['windows_with_nonfull_ranks']=df.groupby('network').full_rank_fraction.apply(lambda s:int((s<1).sum())).to_numpy()
 summary['windows_certified_top1']=df.groupby('network').top1_certified.apply(lambda s:int((s>0).sum())).to_numpy()
 summary['windows_certified_topk']=df.groupby('network').topk_certified.apply(lambda s:int((s>0).sum())).to_numpy()
 summary.to_csv(dest/(output+'_summary.csv'),index=False)
 print(summary.to_string(index=False),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--data-root',required=True);parser.add_argument('--bootstrap',type=int,default=999)
 parser.add_argument('--block',type=int,default=10);parser.add_argument('--workers',type=int,default=4);parser.add_argument('--stride',type=int,default=1)
 parser.add_argument('--output',default='gpw_refined');a=parser.parse_args();run(a.data_root,a.bootstrap,a.block,a.workers,a.stride,a.output)
