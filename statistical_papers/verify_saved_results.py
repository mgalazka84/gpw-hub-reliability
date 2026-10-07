"""Audit saved outputs and reproduce representative Monte Carlo draws."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from hub_inference import *
from run_simulations import sample

def main():
    root=Path(__file__).parent/'results';verified=0
    for prefix,cells in [('simulation',15),('simulation_shapes',6)]:
        d=pd.read_csv(root/(prefix+'_replications.csv'));s=pd.read_csv(root/(prefix+'_summary.csv'))
        assert len(d)==500*cells and len(s)==cells
        assert not d.duplicated(['model','n','rep']).any()
        assert (d.groupby(['model','n']).size()==500).all()
        grouped=d.groupby(['model','n'])
        for col in ['band_coverage','degree_coverage','rank_coverage','mean_degree_width','top1_candidate_size','top1_certified']:
            expect=grouped[col].mean();actual=s.set_index(['model','n'])[col]
            assert np.allclose(expect,actual.reindex(expect.index)),(prefix,col)
        assert (d.degree_coverage>=d.band_coverage).all()
        assert (d.rank_coverage>=d.degree_coverage).all()
        # Independently regenerate one frozen full-bootstrap draw in every cell.
        for (_,n),part in grouped:
            row=part.iloc[137];rng=np.random.default_rng(int(row.seed))
            x,corr=sample(row.model,int(n),12,rng)
            obs,boots=bootstrap_covariances(x,999,int(row.block),rng)
            ans=analyse_covariances(obs,boots,12,selection=False)
            assert np.isclose(np.mean(ans['degree_upper']-ans['degree_lower']),row.mean_degree_width)
            assert int(np.sum(ans['rank_lower']<=1))==row.top1_candidate_size
            verified+=1
    k=pd.read_csv(root/'kendall_replications.csv');assert len(k)==3000
    g=pd.read_csv(root/'gpw_windows.csv');nodes=pd.read_csv(root/'gpw_nodes.csv.gz')
    assert len(g)==693 and g.origin.nunique()==231 and nodes['isin'].nunique()==155
    groups=nodes.groupby(['origin','network'])
    for r in g.itertuples():
        a=groups.get_group((r.origin,r.network));assert len(a)==r.p
        assert np.isclose((a.degree_upper-a.degree_lower).mean(),r.mean_degree_width)
        assert np.isclose((a.rank_upper-a.rank_lower).mean(),r.mean_rank_width)
        assert (a.degree_lower<=a.observed_degree).all() and (a.observed_degree<=a.degree_upper).all()
        assert (a.rank_lower==1).all() and (a.rank_upper==r.p).all()
        assert (a.observed_degree.sum()==2*(r.p-1))
    report=dict(status='passed',primary_monte_carlo_draws=7500,shape_draws=3000,kendall_draws=3000,
                representative_draws_regenerated=verified,gpw_windows=231,representations=3,
                unique_gpw_securities=155,gpw_node_records=len(nodes),
                scope='Independent aggregate checks and representative seed reruns; not a new raw-data acquisition')
    (root/'final_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
