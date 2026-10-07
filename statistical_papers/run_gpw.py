"""Reanalyse the frozen complete GPW windows; no new market data are invented."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import pandas as pd
from hub_inference import *

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(data_root,B=999,block=10,stride=1,output='gpw'):
    data_root=Path(data_root)
    panel=pd.read_csv(data_root/'data/close_simple_returns.csv.gz',index_col=0,parse_dates=True)
    wig=pd.read_csv(data_root/'data/wig_ohlc.csv',index_col=0,parse_dates=True)
    factor=np.log(wig.close/wig.close.shift()).reindex(panel.index)
    selection=pd.read_csv(data_root/'results/hub_selections.csv.gz',parse_dates=['origin_date'])
    selected=selection[(selection.convention=='close')&(selection.network=='raw')]
    origins=sorted(selected.origin_date.unique())[::stride]
    dest=Path(__file__).parent/'results';dest.mkdir(exist_ok=True)
    source_paths=['data/close_simple_returns.csv.gz','data/wig_ohlc.csv','results/hub_selections.csv.gz']
    config=dict(bootstrap_repetitions=B,block=block,alpha=.05,origins=len(origins),
                seed=803177,return_type='close-to-close log price returns',lookback=252,
                universe='Frozen previously audited past-only eligible stocks at each origin',
                band='Centered Fisher-z bootstrap max standardized deviation',
                coverage_scope='Per window and representation, simultaneous across all nodes; not across dates or representations',
                source_sha256={f:digest(data_root/f) for f in source_paths})
    (dest/(output+'_design.json')).write_text(json.dumps(config,indent=2))
    rows=[];nodes=[];start=time.time()
    for oi,origin in enumerate(origins):
        names=selected[selected.origin_date==origin]
        stocks=names['isin'].tolist();p=len(stocks)
        loc=panel.index.get_loc(origin)
        x=np.log1p(panel[stocks].iloc[loc-251:loc+1].to_numpy())
        f=factor.iloc[loc-251:loc+1].to_numpy()
        assert len(x)==252 and np.isfinite(x).all() and np.isfinite(f).all()
        obs,boots=bootstrap_covariances(np.column_stack([x,f]),B,block,np.random.default_rng(803177+loc))
        for kind in ['raw','demeaned','factor_residual']:
            ans=analyse_covariances(obs,boots,p,kind,selection=False)
            old=selection[(selection.origin_date==origin)&(selection.convention=='close')&(selection.network==kind)].set_index('isin').loc[stocks]
            assert np.array_equal(old.degree.to_numpy(),ans['degree'])
            stable=old.bootstrap_selection.to_numpy()>=.8
            row=dict(origin=str(pd.Timestamp(origin).date()),network=kind,p=p,k=ans['k'],
                     mean_degree_width=float(np.mean(ans['degree_upper']-ans['degree_lower'])),
                     mean_rank_width=float(np.mean(ans['rank_upper']-ans['rank_lower'])),
                     top1_candidate_size=int((ans['rank_lower']<=1).sum()),
                     top1_certified=int((ans['rank_upper']<=1).sum()),
                     topk_candidate_size=int(ans['candidate'].sum()),
                     topk_certified=int(ans['certified'].sum()),
                     exact_degree_count=int((ans['degree_lower']==ans['degree_upper']).sum()),
                     stable_bootstrap_count=int(stable.sum()),
                     stable_and_certified=int((stable&ans['certified']).sum()),critical=ans['critical'])
            rows.append(row)
            for i,isin in enumerate(stocks):
                nodes.append(dict(origin=row['origin'],network=kind,isin=isin,name=names.name_at_snapshot.iloc[i],
                    observed_degree=int(ans['degree'][i]),degree_lower=int(ans['degree_lower'][i]),degree_upper=int(ans['degree_upper'][i]),
                    rank_lower=int(ans['rank_lower'][i]),rank_upper=int(ans['rank_upper'][i]),
                    bootstrap_selection_original=float(old.bootstrap_selection.iloc[i]),
                    candidate_topk=bool(ans['candidate'][i]),certified_topk=bool(ans['certified'][i])))
        if (oi+1)%10==0 or oi+1==len(origins):
            pd.DataFrame(rows).to_csv(dest/(output+'_windows.csv'),index=False)
            pd.DataFrame(nodes).to_csv(dest/(output+'_nodes.csv.gz'),index=False,compression='gzip')
            print(f'{output}: {oi+1}/{len(origins)} windows, {time.time()-start:.1f}s',flush=True)
    df=pd.DataFrame(rows)
    summary=df.groupby('network').agg(origins=('origin','size'),mean_p=('p','mean'),
       mean_degree_width=('mean_degree_width','mean'),mean_rank_width=('mean_rank_width','mean'),
       mean_top1_candidates=('top1_candidate_size','mean'),mean_topk_candidates=('topk_candidate_size','mean'),
       origins_certified_top1=('top1_certified',lambda x:int((x>0).sum())),
       origins_certified_topk=('topk_certified',lambda x:int((x>0).sum())),
       mean_certified_topk=('topk_certified','mean'),
       origins_stable_bootstrap=('stable_bootstrap_count',lambda x:int((x>0).sum())),
       stable_nodes=('stable_bootstrap_count','sum'),stable_certified_nodes=('stable_and_certified','sum')).reset_index()
    summary.to_csv(dest/(output+'_summary.csv'),index=False)
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--bootstrap',type=int,default=999)
    p.add_argument('--block',type=int,default=10);p.add_argument('--stride',type=int,default=1);p.add_argument('--output',default='gpw')
    a=p.parse_args();run(a.data_root,a.bootstrap,a.block,a.stride,a.output)
