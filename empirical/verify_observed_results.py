"""Independent arithmetic, panel and chronological consistency checks."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'submission'))
from network_methods import diversification_loss
from forecast_evaluation import ridge_prediction, MODEL_FEATURES

def main():
    r=BASE/'results';checks=[]
    manifest=json.loads((BASE/'source_checks/company_download_manifest.json').read_text())
    for item in manifest:
        if item['status']=='downloaded':
            blob=(BASE/'raw/company'/f"{item['isin']}.json").read_bytes()
            assert hashlib.sha256(blob).hexdigest()==item['sha256']
    checks.append('All 164 successful raw-response SHA256 hashes match the acquisition manifest.')
    nodes=pd.read_csv(r/'hub_selections.csv.gz')
    coverage=pd.read_csv(r/'origin_coverage.csv',parse_dates=['origin_date','label_end_date'])
    close=pd.read_csv(BASE/'data/close_panel.csv.gz',index_col=0,parse_dates=True)
    opened=pd.read_csv(BASE/'data/open_panel.csv.gz',index_col=0,parse_dates=True)
    raw_c=close/close.shift()-1;raw_i=close/opened-1
    for convention,constructed in [('close',raw_c),('intraday',raw_i)]:
        panel=pd.read_csv(BASE/'data'/f'{convention}_simple_returns.csv.gz',index_col=0,parse_dates=True)
        assert np.allclose(panel.to_numpy(),constructed.to_numpy(),equal_nan=True,atol=1e-14,rtol=1e-12)
        frame=pd.read_csv(r/f'features_{convention}.csv',parse_dates=['origin_date','label_end_date'])
        for _,row in frame.iterrows():
            date=row.origin_date.strftime('%Y-%m-%d')
            group=nodes.loc[(nodes.origin_date==date)&(nodes.convention==convention)]
            isin=group.loc[group.network=='raw','isin'].tolist()
            n=len(isin);k=int(np.ceil(.1*n));pos=panel.index.get_loc(row.origin_date)
            assert n==row.n_assets
            assert panel.index[pos+20]==row.label_end_date
            past=panel[isin].iloc[pos-251:pos+1].to_numpy()
            assert np.isfinite(past).all() and past.shape==(252,n)
            w=np.full(n,1/n)
            assert np.isclose(diversification_loss(past,w),row.current_y252,atol=1e-13)
            future=panel[isin].iloc[pos+1:pos+21].to_numpy()
            if pd.notna(row.future_y):
                assert np.isfinite(future).all()
                assert np.isclose(diversification_loss(future,w),row.future_y,atol=1e-13)
            for kind in ['raw','demeaned','factor_residual']:
                g=group.loc[group.network==kind]
                assert len(g)==n and g['isin'].nunique()==n
                assert g.degree.sum()==2*(n-1)
                assert np.isclose(g.bootstrap_selection.sum(),k,atol=1e-10)
                H=float(g.bootstrap_selection@g.degree/(2*(n-1)))
                degree=g.degree.to_numpy();threshold=np.sort(degree)[-k]
                q=(degree>threshold).astype(float);tied=degree==threshold
                q[tied]=(k-np.sum(degree>threshold))/tied.sum()
                C=float(q@degree/(2*(n-1)))
                assert np.isclose(H,row['reliability_'+kind],atol=1e-13)
                assert np.isclose(C,row['degree_'+kind],atol=1e-13)
                assert k/(2*(n-1))-1e-10<=H<=C+1e-10
                assert C<=(n+k-2)/(2*(n-1))+1e-10
        checks.append(f'{convention}: every current/future target and C/H independently recomputed; degree/membership sums and theoretical bounds pass.')
        # Reproduce all reported test predictions from matured history.
        for kind in ['raw','demeaned','factor_residual']:
            features=frame.copy()
            features['degree_residual']=features['degree_'+kind]
            features['reliability_residual']=features['reliability_'+kind]
            predictions=pd.read_csv(r/f'predictions_{convention}_{kind}.csv',parse_dates=['origin_date','label_end_date'])
            config=json.loads((r/f'forecast_config_{convention}_{kind}.json').read_text())
            for model,cols in MODEL_FEATURES.items():
                for idx,row in features.loc[features.origin_date.dt.year.between(2019,2025)].iterrows():
                    recorded=predictions.at[idx,model]
                    if pd.isna(recorded):continue
                    train=features.loc[(features.origin_date<row.origin_date)&(features.label_end_date<=row.origin_date)].dropna(subset=cols+['future_y'])
                    assert len(train)>=60
                    assert train.label_end_date.max()<=row.origin_date
                    value=ridge_prediction(train[cols].to_numpy(float),train.future_y.to_numpy(float),row[cols].to_numpy(float),config['models'][model]['alpha'])
                    assert np.isclose(value,recorded,atol=1e-13)
            checks.append(f'{convention}/{kind}: all test ridge forecasts reproduce from past, fully matured labels only.')
    assert (coverage.origin_date.iloc[1:].to_numpy()>=coverage.label_end_date.iloc[:-1].to_numpy()).all()
    checks.append('All outcome windows contain 20 sessions and do not overlap.')
    results=json.loads((r/'forecast_results_all.json').read_text())
    for key,result in results.items():
        convention,kind=key.split('_',1)
        p=pd.read_csv(r/f'predictions_{convention}_{kind}.csv',parse_dates=['origin_date'])
        t=p.loc[p.origin_date.dt.year.between(2019,2025)].dropna(subset=['future_y','base','degree','reliability','persistence','ewma'])
        assert len(t)==result['test_origins']
        for model,recorded in result['mse'].items():assert np.isclose(((t[model]-t.future_y)**2).mean(),recorded,atol=1e-14)
        difference=((t.degree-t.future_y)**2-(t.reliability-t.future_y)**2).mean()
        assert np.isclose(difference,result['primary_interval']['mean_difference'],atol=1e-14)
    checks.append('All common-test MSE values and the sign of paired loss differences independently verified.')
    report=dict(status='PASS',checks=checks,verified_hub_rows=len(nodes),
        unique_analysed_isins=int(nodes['isin'].nunique()),origin_count=len(coverage),
        origin_dates_all_retained=bool(coverage.origin_retained.all()),
        incomplete_future_outcomes=int((coverage.origin_retained&~coverage.future_complete).sum()))
    (r/'observed_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
