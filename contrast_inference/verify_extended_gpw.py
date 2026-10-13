"""Independent audit of universes, forward diagnostics and rational witnesses."""
from pathlib import Path
import json,gzip,hashlib,sys,time
import numpy as np
import pandas as pd
from order_inference import edge_index,screen_edges
from path_projection import path_constraints,_problem
from verify_revision import verify_certificate
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'statistical_papers'))
from hub_inference import max_tree,correlation_edges,transform_cov
ROOT=Path(__file__).parent;R=ROOT/'results'

def run(data_root):
 root=Path(data_root);config=json.loads((R/'gpw_extended_design.json').read_text())
 for name,digest in config['source_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
 windows=pd.read_csv(R/'gpw_extended_windows.csv');nodes=pd.read_csv(R/'gpw_extended_nodes.csv.gz');uni=pd.read_csv(R/'gpw_extended_universes.csv');ex=pd.read_csv(R/'gpw_extended_exclusions.csv')
 key=['origin','lookback','cap','block','network','analysis'];assert not windows.duplicated(key).any();assert not nodes.duplicated(key+['isin']).any()
 expected=sum(len(j['caps'])*(3 if j['analysis']=='primary' else 1) for j in config['jobs']);assert len(windows)==expected==466;assert (windows.analysis=='primary').sum()==432
 panel=pd.read_csv(root/'data/close_simple_returns.csv.gz',index_col=0,parse_dates=True);vol=pd.read_csv(root/'data/volume_panel.csv.gz',index_col=0,parse_dates=True);close=pd.read_csv(root/'data/close_panel.csv.gz',index_col=0,parse_dates=True);wig=pd.read_csv(root/'data/wig_ohlc.csv',index_col=0,parse_dates=True);factor=np.log(wig.close/wig.close.shift()).reindex(panel.index)
 sel=pd.read_csv(root/'results/hub_selections.csv.gz',parse_dates=['origin_date']);sel=sel[(sel.convention=='close')&(sel.network=='raw')]
 origins=pd.Series(sorted(sel.origin_date.unique()));annual=origins.groupby(origins.dt.year).max();assert annual.dt.date.astype(str).tolist()==config['origins']
 for origin in config['origins']:
  date=pd.Timestamp(origin);loc=panel.index.get_loc(date);lookback=1008 if loc+1>=1008 else 504;ix=panel.index[loc-lookback+1:loc+1];base=sel[sel.origin_date==date]['isin'].tolist();eligible=[];liq={}
  for isin in base:
   rr=panel[isin].loc[ix];vv=vol[isin].reindex(ix);cc=close[isin].reindex(ix)
   if rr.isna().any() or vv.isna().any() or cc.isna().any() or (rr<=-1).any():continue
   if (vv>0).mean()<.9 or (rr==0).mean()>.15 or rr.var()<=0:continue
   eligible.append(isin);liq[isin]=float(np.median(vv.to_numpy()*cc.to_numpy()))
  ordering=sorted(eligible,key=lambda i:(-liq[i],i))
  for cap in [10,20,40]:
   subset=uni[(uni.origin==origin)&(uni.cap==cap)].sort_values('liquidity_rank')
   if len(ordering)>=cap:assert subset['isin'].tolist()==ordering[:cap]
   else:assert len(subset)==0
 for values,d in nodes.groupby(key):
  mask=np.ones(len(windows),bool)
  for k,v in zip(key,values):mask&=windows[k].to_numpy()==v
  row=windows[mask].iloc[0];assert len(d)==row.cap
  assert ((d.degree_lower<=d.degree)&(d.degree_upper>=d.degree)).all();assert ((d.degree_lower>=1)&(d.degree_upper<=row.cap-1)).all()
  assert ((d.rank_lower>=1)&(d.rank_lower<=d.rank_upper)&(d.rank_upper<=row.cap)).all()
  assert np.isclose((d.degree_upper-d.degree_lower).mean(),row.mean_degree_width)
  assert np.isclose((d.rank_upper-d.rank_lower).mean(),row.mean_rank_width)
  assert np.isclose(((d.rank_lower>1)|(d.rank_upper<row.cap)).mean(),row.nonfull_rank_fraction)
 start=time.time();counter=0;certificates=0;forward=0
 with gzip.open(R/'gpw_extended_certificates.jsonl.gz','rt') as stream:
  for line in stream:
   obj=json.loads(line);p=obj['p'];edges=edge_index(p);m=len(edges);relation=np.zeros((m,m),bool)
   if obj['order_pairs']:a,b=np.array(obj['order_pairs']).T;relation[a,b]=True
   allowed=np.zeros(m,bool);allowed[obj['allowed']]=True;forced=np.zeros(m,bool);forced[obj['forced']]=True
   aa,ff=screen_edges(relation,edges,p);assert np.array_equal(aa,allowed) and np.array_equal(ff,forced)
   paths=path_constraints(relation,edges,p,allowed,forced);problem=_problem(edges,p,allowed,forced,paths)
   for rec in obj['certificates']:verify_certificate(rec,edges,p,allowed,forced,paths,problem);certificates+=1
   mask=np.ones(len(windows),bool)
   for k in key:mask&=windows[k].to_numpy()==obj[k]
   row=windows[mask].iloc[0];assert certificates>=len(obj['certificates']);assert len(paths)==row.path_constraints
   if row.future_available:
    loc=panel.index.get_loc(pd.Timestamp(obj['origin']));x=np.column_stack([np.log1p(panel[obj['stocks']].iloc[loc+1:loc+253].to_numpy()),factor.iloc[loc+1:loc+253].to_numpy()]);assert len(x)==252 and np.isfinite(x).all();x-=x.mean(axis=0);cov=x.T@x/252;r=correlation_edges(transform_cov(cov,p,obj['network']),edges);_,degree=max_tree(r,edges,p);a,b=np.where(relation)
    rate=float(np.mean(r[a]>r[b])) if len(a) else np.nan
    assert (np.isnan(rate) and np.isnan(row.future_order_agreement)) or np.isclose(rate,row.future_order_agreement)
    anchor=obj['stocks'].index(row.anchor_isin);assert row.future_anchor_best_rank==1+(degree>degree[anchor]).sum();assert row.future_anchor_worst_rank==(degree>=degree[anchor]).sum();forward+=1
   counter+=1
   if counter%60==0:print('verify extended',counter,certificates,'seconds',round(time.time()-start),flush=True)
 assert counter==len(windows) and certificates==int(windows.nontrivial_dual_certificates.sum())
 result=dict(status='PASS',analyses=counter,primary_analyses=432,block_sensitivity_analyses=34,vertex_records=len(nodes),planned_primary_analyses=513,excluded_primary_analyses=len(ex)*3,source_hashes=5,stock_universes_independently_checked=int(uni.groupby(['origin','cap']).ngroups),forward_analyses_recomputed=forward,rational_certificates=certificates)
 (R/'gpw_extended_verification.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':run(sys.argv[1])
