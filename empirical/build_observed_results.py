"""Retrospective GPW analysis on observed prices: explicitly exploratory."""
from pathlib import Path
from datetime import datetime, timezone
import json
import sys
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'submission'))
from network_methods import correlation, hub_stability, diversification_loss
from forecast_evaluation import evaluate, paired_loss_interval

RESULTS=BASE/'results'
RESULTS.mkdir(exist_ok=True)
KINDS=['raw','demeaned','factor_residual']
CONVENTIONS=['close','intraday']
CONFIG=dict(design_recorded_at=datetime.now(timezone.utc).isoformat(),
    status='Retrospective exploratory observed-price analysis; not a prospective preregistration',
    primary_return_convention='close-to-close price returns as supplied, not certified total returns',
    sensitivity_return_convention='open-to-close price returns, excluding overnight risk and cash-dividend income',
    universe='Latest archived WIG20 and mWIG40 quarterly portfolio strictly before the origin; extraordinary changes not reconciled',
    first_origin_not_before='2007-03-19',data_cutoff='2025-09-19',lookback_sessions=252,
    stride_sessions=20,horizon_sessions=20,minimum_assets=20,
    eligibility='Same stocks in both conventions: complete past 252 returns and volumes, positive volume >=90%, zero close returns <=15%, both log-return variances positive',
    missing_policy='No filling. Select assets using past observations only. Any incomplete future asset return makes the entire origin target missing in both conventions.',
    factor='Single contemporaneous WIG log return of the same return convention, with intercept; refitted in every bootstrap sample',
    bootstrap_repetitions=500,bootstrap_block_sessions=10,hub_fraction=.10,
    seed=8317,validation_years=[2015,2018],test_years=[2019,2025],
    minimum_mature_training_origins=60,ridge_alphas=[.01,.1,1.,10.,100.],
    primary_network_comparison='Market-residual degree versus market-residual degree plus reliability',
    secondary_networks=['raw','cross-sectionally demeaned'],
    extra_benchmark='EWMA covariance with fixed lambda 0.94 over the past 252 simple returns',
    loss_interval='Paired origin-level loss difference; 5000 circular block replicates, primary block 6, sensitivity 3 and 12',
    interpretation='Conditional uncertainty in the downloaded quarterly-snapshot universe, not proof of causal or structural hubs; evaluation uses historical data vintage downloaded in 2026')

def ewma_y(x,decay=.94):
    weights=decay**np.arange(len(x)-1,-1,-1,dtype=float)
    weights/=weights.sum()
    z=x-(weights[:,None]*x).sum(axis=0)
    cov=(z*weights[:,None]).T@z
    w=np.full(x.shape[1],1/x.shape[1])
    return float(np.clip((w@cov@w)/(w@np.sqrt(np.diag(cov)))**2,0,1))

def build():
    config_path=RESULTS/'observed_design.json'
    if not config_path.exists():
        config_path.write_text(json.dumps(CONFIG,indent=2))
    config=json.loads(config_path.read_text())
    if config['lookback_sessions']!=252 or config['bootstrap_repetitions']!=500:
        raise ValueError('Code/design disagreement')
    data=BASE/'data'
    panels={c:pd.read_csv(data/f'{c}_simple_returns.csv.gz',index_col=0,parse_dates=True) for c in CONVENTIONS}
    volume=pd.read_csv(data/'volume_panel.csv.gz',index_col=0,parse_dates=True)
    wig=pd.read_csv(data/'wig_ohlc.csv',index_col=0,parse_dates=True)
    market={'close':np.log(wig.close/wig.close.shift()),'intraday':np.log(wig.close/wig.open)}
    calendar=panels['close'].index
    snapshots=pd.read_csv(BASE.parent/'submission/data/constituents_snapshots.csv',parse_dates=['snapshot_revision_date'])
    revisions=sorted(snapshots.snapshot_revision_date.unique())
    start=int(calendar.searchsorted(pd.Timestamp('2007-03-19')))
    positions=list(range(start,len(calendar)-20,20))
    rows={c:[] for c in CONVENTIONS}; coverage=[]; nodes=[]; edges=[]; exclusions=[]
    for number,pos in enumerate(positions):
        origin=calendar[pos];end=calendar[pos+20]
        revision=max(d for d in revisions if pd.Timestamp(d)<origin)
        members=snapshots.loc[snapshots.snapshot_revision_date==revision].drop_duplicates('isin')
        candidates=members['isin'].tolist()
        eligible=[]
        for isin in candidates:
            reason=[]
            if isin not in volume:
                reason.append('unavailable_source_series')
            else:
                pasts={c:panels[c][isin].iloc[pos-251:pos+1].to_numpy(float) for c in CONVENTIONS}
                v=volume[isin].iloc[pos-251:pos+1].to_numpy(float)
                if any(len(x)!=252 or not np.isfinite(x).all() or np.any(x<=-1) for x in pasts.values()):reason.append('incomplete_past_return_window')
                if len(v)!=252 or not np.isfinite(v).all():reason.append('incomplete_past_volume')
                elif np.mean(v>0)<.9:reason.append('insufficient_positive_volume')
                if np.mean(pasts['close']==0)>.15:reason.append('too_many_zero_close_returns')
                if any(np.nanstd(np.log1p(x))<=1e-12 for x in pasts.values()):reason.append('constant_returns')
            if reason:
                exclusions.append(dict(origin_date=str(origin.date()),isin=isin,reasons=';'.join(reason)))
            else:eligible.append(isin)
        coverage.append(dict(origin_date=str(origin.date()),label_end_date=str(end.date()),
            snapshot_revision_date=str(pd.Timestamp(revision).date()),snapshot_members=len(candidates),
            downloaded_members=sum(i in volume for i in candidates),eligible_assets=len(eligible),
            future_complete=False,origin_retained=len(eligible)>=20))
        if len(eligible)<20:
            print(f'Origin {number+1}/{len(positions)} {origin.date()}: only {len(eligible)} assets; excluded',flush=True)
            continue
        future={c:panels[c][eligible].iloc[pos+1:pos+21].to_numpy(float) for c in CONVENTIONS}
        complete=all(np.isfinite(f).all() for f in future.values())
        coverage[-1]['future_complete']=bool(complete)
        zero_volume=float(np.mean(volume[eligible].iloc[pos-251:pos+1].to_numpy()==0))
        n=len(eligible);w=np.full(n,1/n)
        for ci,convention in enumerate(CONVENTIONS):
            simple=panels[convention][eligible].iloc[pos-251:pos+1].to_numpy(float)
            log=np.log1p(simple)
            f=market[convention].iloc[pos-251:pos+1].to_numpy(float)
            if not np.isfinite(f).all():raise ValueError('Incomplete market factor')
            corr=correlation(log)
            row=dict(origin_date=str(origin.date()),label_end_date=str(end.date()),
                snapshot_revision_date=str(pd.Timestamp(revision).date()),n_assets=n,
                future_y=diversification_loss(future[convention],w) if complete else np.nan,
                current_y20=diversification_loss(simple[-20:],w),
                current_y252=diversification_loss(simple,w),
                mean_correlation=float((corr.sum()-n)/(n*(n-1))),
                pc1_share=float(np.linalg.eigvalsh(corr)[-1]/n),
                market_vol20=float(np.std(f[-20:],ddof=1)),market_vol252=float(np.std(f,ddof=1)),
                zero_volume_share=zero_volume,ewma_y=ewma_y(simple))
            for ki,kind in enumerate(KINDS):
                # The identical seed gives common resampled days across representations.
                result=hub_stability(log,f,kind,500,10,8317+number*17+ci*100003)
                row['degree_'+kind]=result.degree_concentration
                row['reliability_'+kind]=result.reliability_share
                row['discount_'+kind]=result.degree_concentration-result.reliability_share
                row['max_selection_'+kind]=float(result.selection.max())
                row['stable_count_'+kind]=int((result.selection>=.8).sum())
                for j,isin in enumerate(eligible):
                    source_name=members.loc[members['isin']==isin,'source_name'].iloc[0]
                    nodes.append(dict(origin_date=str(origin.date()),convention=convention,network=kind,
                        isin=isin,name_at_snapshot=source_name,degree=int(result.degree[j]),
                        bootstrap_selection=float(result.selection[j]),log_return_sd=float(np.std(log[:,j],ddof=1))))
                for a,b in result.edges:
                    edges.append(dict(origin_date=str(origin.date()),convention=convention,network=kind,
                        isin_a=eligible[a],isin_b=eligible[b],bootstrap_frequency=float(result.edge_frequency[a,b])))
            rows[convention].append(row)
        if (number+1)%10==0 or number+1==len(positions):
            print(f'Networks {number+1}/{len(positions)}; {origin.date()}; N={n}; future complete={complete}',flush=True)
            for c in CONVENTIONS:pd.DataFrame(rows[c]).to_csv(RESULTS/f'features_{c}.csv',index=False)
            pd.DataFrame(coverage).to_csv(RESULTS/'origin_coverage.csv',index=False)
            pd.DataFrame(exclusions).to_csv(RESULTS/'eligibility_exclusions.csv',index=False)
            pd.DataFrame(nodes).to_csv(RESULTS/'hub_selections.csv.gz',index=False,compression='gzip')
            pd.DataFrame(edges).to_csv(RESULTS/'original_tree_edges.csv.gz',index=False,compression='gzip')
    for c in CONVENTIONS:pd.DataFrame(rows[c]).to_csv(RESULTS/f'features_{c}.csv',index=False)
    pd.DataFrame(coverage).to_csv(RESULTS/'origin_coverage.csv',index=False)
    pd.DataFrame(exclusions).to_csv(RESULTS/'eligibility_exclusions.csv',index=False)
    pd.DataFrame(nodes).to_csv(RESULTS/'hub_selections.csv.gz',index=False,compression='gzip')
    pd.DataFrame(edges).to_csv(RESULTS/'original_tree_edges.csv.gz',index=False,compression='gzip')

def forecast():
    all_results={};scores=[];descriptive=[]
    for convention in CONVENTIONS:
        frame=pd.read_csv(RESULTS/f'features_{convention}.csv',parse_dates=['origin_date','label_end_date'])
        for kind in KINDS:
            df=frame.copy()
            df['degree_residual']=df['degree_'+kind]
            df['reliability_residual']=df['reliability_'+kind]
            predictions,config,result=evaluate(df)
            predictions['ewma']=frame.ewma_y
            test=predictions.origin_date.dt.year.between(2019,2025)
            final=predictions.loc[test].dropna(subset=['future_y','base','degree','reliability','persistence','ewma'])
            if len(final)!=result['test_origins']:raise AssertionError('Benchmark sample mismatch')
            result['mse']['ewma']=float(((final.ewma-final.future_y)**2).mean())
            result['reliability_vs_base_interval']=paired_loss_interval(((final.base-final.future_y)**2-(final.reliability-final.future_y)**2).to_numpy(),6)
            result['relative_mse_reduction_vs_degree']=1-result['mse']['reliability']/result['mse']['degree']
            predictions.to_csv(RESULTS/f'predictions_{convention}_{kind}.csv',index=False)
            (RESULTS/f'forecast_config_{convention}_{kind}.json').write_text(json.dumps(config,indent=2))
            all_results[convention+'_'+kind]=result
            for model,mse in result['mse'].items():scores.append(dict(convention=convention,network=kind,model=model,mse=mse,test_origins=len(final)))
            descriptive.append(dict(convention=convention,network=kind,origins=len(frame),
                first_origin=str(frame.origin_date.min().date()),last_origin=str(frame.origin_date.max().date()),
                mean_N=float(frame.n_assets.mean()),min_N=int(frame.n_assets.min()),max_N=int(frame.n_assets.max()),
                mean_C=float(frame['degree_'+kind].mean()),mean_H=float(frame['reliability_'+kind].mean()),
                mean_uncertainty_discount=float(frame['discount_'+kind].mean()),
                mean_max_selection=float(frame['max_selection_'+kind].mean()),
                origins_with_selection_at_least_80=int((frame['stable_count_'+kind]>0).sum()),
                mean_stable_hub_count=float(frame['stable_count_'+kind].mean())))
    (RESULTS/'forecast_results_all.json').write_text(json.dumps(all_results,indent=2))
    pd.DataFrame(scores).to_csv(RESULTS/'forecast_scores.csv',index=False)
    pd.DataFrame(descriptive).to_csv(RESULTS/'network_summary.csv',index=False)
    print(json.dumps(all_results,indent=2),flush=True)

if __name__=='__main__':
    if '--forecast-only' not in sys.argv:build()
    forecast()
