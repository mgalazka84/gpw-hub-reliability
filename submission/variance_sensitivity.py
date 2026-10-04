"""Exploratory sensitivity to variance contrast, added after the initial pilot.

Common innovations and bootstrap seeds pair the contrast levels within a Monte
Carlo replication. Each contrast still has independent Monte Carlo samples.
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from network_methods import hub_stability

BASE = Path(__file__).parent


def main():
    seed, mc, bootstrap, n, t, block = 84261003, 100, 500, 60, 252, 10
    ratios = [2., 4., 8., 16.]
    records = []
    start = time.monotonic()
    sequences = np.random.SeedSequence(seed).spawn(mc)
    for rep in range(mc):
        rng = np.random.default_rng(sequences[rep])
        eps = rng.normal(size=(t,n))
        bseed = int(rng.integers(0,2**32-1))
        for ratio in ratios:
            sigma = np.geomspace(1,ratio,n)
            result = hub_stability(eps*sigma,None,"demeaned",bootstrap,block,bseed)
            records.append(dict(replication=rep,standard_deviation_ratio=ratio,
                max_selection=float(result.selection.max()),max_degree=int(result.degree.max()),
                reliability_share=result.reliability_share,degree_concentration=result.degree_concentration))
        if (rep+1)%20 == 0:
            print(f"Variance sensitivity {rep+1}/{mc}; {time.monotonic()-start:.1f}s",flush=True)
    data = pd.DataFrame(records)
    output = BASE/"results"
    data.to_csv(output/"variance_sensitivity_replications.csv",index=False)
    summaries = []
    for ratio,group in data.groupby("standard_deviation_ratio"):
        prop = float((group.max_selection>=.8).mean())
        z = 1.95996398454
        center = (prop+z*z/(2*mc))/(1+z*z/mc)
        half = z*np.sqrt(prop*(1-prop)/mc+z*z/(4*mc*mc))/(1+z*z/mc)
        summaries.append(dict(standard_deviation_ratio=ratio,replications=mc,
            mean_max_selection=float(group.max_selection.mean()),
            mc_se_max_selection=float(group.max_selection.std(ddof=1)/np.sqrt(mc)),
            frequency_ge_080=prop,wilson_low=max(0,float(center-half)),
            wilson_high=min(1,float(center+half))))
    summary = pd.DataFrame(summaries)
    summary.to_csv(output/"variance_sensitivity_summary.csv",index=False)
    (output/"variance_sensitivity_config.json").write_text(json.dumps(dict(seed=seed,mc=mc,
        bootstrap=bootstrap,n=n,t=t,block=block,ratios=ratios,
        status="exploratory; added after the initial diagnostic pilot",
        common_random_numbers=True,elapsed_seconds=time.monotonic()-start),indent=2))
    print(summary.to_string(index=False),flush=True)


if __name__=="__main__":
    main()
