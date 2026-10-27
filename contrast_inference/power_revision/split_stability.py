"""Descriptive split sensitivity; no pooling or selection of a favourable split.

Run after the application, with design.json written before any new outcome.
All original method implementations and the original adult dataset are reused.
"""
from pathlib import Path
import argparse, hashlib, json, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from methods import split_orders, project_order, edge_index
from application import load_data, sharp_if_small

BASE = Path(__file__).parent
DEST = BASE / 'results' / 'split_stability'

def describe(ans, p=10):
    """Use the same minor projection for every method and split.

    Membership is a sufficient certificate from simultaneous degree bounds;
    exact enumeration is only an audit and never supplies selected endpoints.
    """
    lo, hi = np.asarray(ans['degree_lower']), np.asarray(ans['degree_upper'])
    rl, rh = np.asarray(ans['rank_lower']), np.asarray(ans['rank_upper'])
    members = [i for i in range(p) if lo[i] >= np.max(np.delete(hi, i))]
    excluded = list(map(int, np.flatnonzero(rl > 1)))
    return dict(degree_lower=lo.tolist(), degree_upper=hi.tolist(),
                rank_lower=rl.tolist(), rank_upper=rh.tolist(),
                fixed_degrees=int(np.sum(lo == hi)), guaranteed_members=members,
                excluded_items=excluded, unique_hub_excluded=len(members) >= 2)

def job(pair):
    index, seed = pair
    dest = DEST / 'splits' / f'{index:03d}.json'
    if dest.exists():
        old = json.loads(dest.read_text())
        assert old['seed'] == seed
        return old
    x = load_data()[0]['adults']
    start = time.perf_counter()
    variants, diag = split_orders(x, seed=seed)
    results = {}
    for method, raw in variants.items():
        ans = project_order(raw, 10)
        stats = describe(ans)
        stats['raw_orders'] = raw['raw_order_count']
        # Audit the featured method whenever screening makes enumeration cheap.
        if method == 'FS-T64':
            sharp = sharp_if_small(ans, 10, limit=5000)
            stats['sharp_audit'] = sharp
            if sharp['status'] == 'exact':
                for k in ['degree_lower', 'rank_lower']:
                    assert np.all(np.asarray(ans[k]) <= sharp[k])
                for k in ['degree_upper', 'rank_upper']:
                    assert np.all(np.asarray(ans[k]) >= sharp[k])
                ds = np.array([np.bincount(ans['edges'][t].ravel(), minlength=10)
                               for t in sharp['trees']])
                exact_members = set(np.flatnonzero(np.all(ds == ds.max(1)[:, None], axis=0)))
                assert set(stats['guaranteed_members']) <= exact_members
        results[method] = stats
    out = dict(split=index, seed=seed, ntrain=diag['ntrain'], ntest=diag['ntest'],
               methods=results, elapsed_seconds=time.perf_counter()-start)
    dest.write_text(json.dumps(out))
    return out

def aggregate(records, design):
    original = json.loads((BASE/'results/application/adults_E64.json').read_text())
    reference = describe(original['minor'])
    # Main-application minor endpoints are already exact for E64.
    assert all(original['minor'][k] == original['sharp'][k]
               for k in ['degree_lower','degree_upper','rank_lower','rank_upper'])
    rows=[]
    for record in records:
        for method,s in record['methods'].items():
            dominates = all(np.all(np.asarray(s[k]) >= reference[k])
                            for k in ['degree_lower','rank_lower'])
            dominates &= all(np.all(np.asarray(s[k]) <= reference[k])
                             for k in ['degree_upper','rank_upper'])
            dominates &= set(reference['guaranteed_members']) <= set(s['guaranteed_members'])
            row = dict(split=record['split'],seed=record['seed'],method=method,
                       fixed_degrees=s['fixed_degrees'],guaranteed_hubs=len(s['guaranteed_members']),
                       all_three_shared=set([0,2,9]) <= set(s['guaranteed_members']),
                       unique_hub_excluded=s['unique_hub_excluded'],
                       at_least_six_fixed=s['fixed_degrees']>=6,
                       weakly_dominates_E64=bool(dominates),
                       excluded=len(s['excluded_items']),raw_orders=s['raw_orders'])
            for i in range(10): row[f'Q{i+1}_hub']=i in s['guaranteed_members']
            rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(DEST/'per_split.csv',index=False)
    summary=[]
    for method,g in frame.groupby('method',sort=True):
        r=dict(method=method,splits=len(g),fixed_degrees_min=int(g.fixed_degrees.min()),
               fixed_degrees_q1=float(g.fixed_degrees.quantile(.25)),
               fixed_degrees_median=float(g.fixed_degrees.median()),
               fixed_degrees_q3=float(g.fixed_degrees.quantile(.75)),
               fixed_degrees_max=int(g.fixed_degrees.max()))
        for c in ['all_three_shared','unique_hub_excluded','at_least_six_fixed','weakly_dominates_E64']:
            r[c+'_count']=int(g[c].sum());r[c+'_percent']=100*float(g[c].mean())
        for i in range(10):r[f'Q{i+1}_hub_count']=int(g[f'Q{i+1}_hub'].sum())
        summary.append(r)
    pd.DataFrame(summary).to_csv(DEST/'summary.csv',index=False)
    (DEST/'summary.json').write_text(json.dumps(dict(design=design,reference_E64=reference,methods=summary),indent=2))
    print(pd.DataFrame(summary)[['method','fixed_degrees_median','all_three_shared_count',
          'unique_hub_excluded_count','weakly_dominates_E64_count']].to_string(index=False),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();DEST.mkdir(exist_ok=True);(DEST/'splits').mkdir(exist_ok=True)
    seeds=[int(x) for x in np.random.SeedSequence(2026102701).generate_state(100,dtype=np.uint32)]
    design=dict(purpose='Post-application descriptive split sensitivity; not a new confirmatory analysis',
        B=100,seed_generator='numpy SeedSequence(2026102701).generate_state(100,uint32)',seeds=seeds,
        fraction=.2,alpha=.05,matchings=64,dataset='complete adult Rosenberg, n=35079',
        original_split_seed=61023117,raw_sha256=hashlib.sha256((BASE/'data/RSE.zip').read_bytes()).hexdigest(),
        projection='same minor and conservation projection for all methods; no selection by exact-audit success',
        reference='original full-sample E64, seed 61123117; fixed once, not rerun on verification subsets',
        dominance='all degree and rank intervals are subsets of original E64 intervals and all E64 guaranteed hub members are retained',
        guarantee='individual fixed procedures only; no simultaneous guarantee over methods/splits, no pooled conclusions',
        scope='all 15 split variants reported; all 100 draws retained regardless of result')
    path=DEST/'design.json'
    if path.exists(): assert json.loads(path.read_text())==design
    else: path.write_text(json.dumps(design,indent=2))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records=[]
        for r in pool.map(job,enumerate(seeds)):
            records.append(r)
            if len(records)%10==0:print(f'Completed {len(records)}/100 splits',flush=True)
    aggregate(records,design)

if __name__=='__main__':main()
