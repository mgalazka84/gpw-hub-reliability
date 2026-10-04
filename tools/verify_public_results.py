"""Verify derived results without claiming to re-audit excluded raw prices."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'submission'))
from forecast_evaluation import MODEL_FEATURES, ridge_prediction, paired_loss_interval

def main():
    results=ROOT/'empirical/results'
    nodes=pd.read_csv(results/'hub_selections.csv.gz')
    coverage=pd.read_csv(results/'origin_coverage.csv',parse_dates=['origin_date','label_end_date'])
    recorded=json.loads((results/'forecast_results_all.json').read_text())
    assert len(coverage)==231 and nodes['isin'].nunique()==155
    assert int((~coverage.future_complete).sum())==42
    assert (coverage.origin_date.iloc[1:].to_numpy()>=coverage.label_end_date.iloc[:-1].to_numpy()).all()
    checks=[]
    for convention in ['close','intraday']:
        frame=pd.read_csv(results/f'features_{convention}.csv',parse_dates=['origin_date','label_end_date'])
        assert len(frame)==231
        for kind in ['raw','demeaned','factor_residual']:
            groups=nodes.loc[(nodes.convention==convention)&(nodes.network==kind)].groupby('origin_date')
            for _,row in frame.iterrows():
                g=groups.get_group(row.origin_date.strftime('%Y-%m-%d'))
                n=len(g);k=int(np.ceil(.1*n));degree=g.degree.to_numpy()
                assert n==row.n_assets and g['isin'].nunique()==n
                assert degree.sum()==2*(n-1)
                assert np.isclose(g.bootstrap_selection.sum(),k,atol=1e-10)
                threshold=np.sort(degree)[-k]
                membership=(degree>threshold).astype(float)
                tied=degree==threshold
                membership[tied]=(k-(degree>threshold).sum())/tied.sum()
                C=membership@degree/(2*(n-1))
                H=g.bootstrap_selection.to_numpy()@degree/(2*(n-1))
                assert np.isclose(C,row['degree_'+kind],atol=1e-13)
                assert np.isclose(H,row['reliability_'+kind],atol=1e-13)
                assert k/(2*(n-1))-1e-10<=H<=C+1e-10
                assert C<=(n+k-2)/(2*(n-1))+1e-10
            features=frame.copy()
            features['degree_residual']=features['degree_'+kind]
            features['reliability_residual']=features['reliability_'+kind]
            predictions=pd.read_csv(results/f'predictions_{convention}_{kind}.csv',parse_dates=['origin_date','label_end_date'])
            config=json.loads((results/f'forecast_config_{convention}_{kind}.json').read_text())
            for model,cols in MODEL_FEATURES.items():
                for idx,row in features.loc[features.origin_date.dt.year.between(2019,2025)].iterrows():
                    value=predictions.at[idx,model]
                    if pd.isna(value):continue
                    train=features.loc[(features.origin_date<row.origin_date)&(features.label_end_date<=row.origin_date)].dropna(subset=cols+['future_y'])
                    assert len(train)>=60 and train.label_end_date.max()<=row.origin_date
                    reproduced=ridge_prediction(train[cols].to_numpy(float),train.future_y.to_numpy(float),row[cols].to_numpy(float),config['models'][model]['alpha'])
                    assert np.isclose(reproduced,value,atol=1e-13)
            test=predictions.loc[predictions.origin_date.dt.year.between(2019,2025)].dropna(subset=['future_y','base','degree','reliability','persistence','ewma'])
            result=recorded[convention+'_'+kind]
            assert len(test)==result['test_origins']==73
            for model,mse in result['mse'].items():
                assert np.isclose(((test[model]-test.future_y)**2).mean(),mse,atol=1e-14)
            differences=((test.degree-test.future_y)**2-(test.reliability-test.future_y)**2).to_numpy()
            for expected in [result['primary_interval']]+result['sensitivity_intervals']:
                interval=paired_loss_interval(differences,expected['block_origins'])
                for key in ['mean_difference','ci95_low','ci95_high']:
                    assert np.isclose(interval[key],expected[key],atol=1e-13)
            checks.append(convention+'/'+kind+': concentration, reliability, chronological forecasts, MSE and paired intervals verified')
    output=dict(status='PASS',scope='Verification from public derived inputs; raw quotation hashes and price-to-return calculations are not re-audited by this offline check.',origins=len(coverage),stocks=int(nodes['isin'].nunique()),complete_test_origins=73,checks=checks)
    (ROOT/'public_verification.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()

