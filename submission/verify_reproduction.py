"""Reproduce saved first-sample asset results and check frozen source hashes."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from network_methods import hub_stability
from run_simulations import CASES,KINDS,dgp

def main():
    base=Path(__file__).parent
    conf=json.loads((base/"results/simulation_config.json").read_text())
    rows=pd.read_csv(base/"results/simulation_replications.csv")
    sequence=np.random.SeedSequence(conf["seed"]).spawn(len(CASES)*conf["mc"])
    lines=[]
    for ci,case in enumerate(CASES):
        rng=np.random.default_rng(sequence[ci*conf["mc"]])
        x,f,_,_=dgp(case,conf["n"],conf["t"],rng)
        for kind in KINDS:
            seed=int(rng.integers(0,2**32-1))
            observed=rows.loc[(rows.case==case)&(rows.replication==0)&(rows["transform"]==kind)].iloc[0]
            assert seed==observed.seed
            result=hub_stability(x,f,kind,conf["bootstrap"],conf["block"],seed)
            example=pd.read_csv(base/f"results/example_{case}_{kind}.csv")
            np.testing.assert_array_equal(result.degree,example.degree)
            np.testing.assert_allclose(result.selection,example.selection,rtol=0,atol=1e-15)
            np.testing.assert_allclose([result.selection.max(),result.degree_concentration,result.reliability_share],
                [observed.max_selection,observed.degree_concentration,observed.reliability_share],rtol=0,atol=1e-15)
            lines.append(f"PASS {case}/{kind}: all 60 degrees and selection frequencies reproduced; seed {seed}")
    for name,expected in conf["source_hashes"].items():
        actual=hashlib.sha256((base/"frozen"/name).read_bytes()).hexdigest()
        assert actual==expected
        lines.append(f"PASS frozen/{name}: SHA-256 equals production-run record")
    (base/"results/reproduction_checks.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":main()
