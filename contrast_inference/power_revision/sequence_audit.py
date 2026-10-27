"""Diagnose the archived 100 splits and time the unchanged questionnaire pipeline.

No selection, pooling or retesting of conclusions. Every rerun must reproduce
the archived FS-T64 relation size and minor endpoints. Timings cover all 15
split variants' shared calibration, then FS-T64 projection/audit stages.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import argparse, gzip, hashlib, json, os, platform, sys, time
import numpy as np
import scipy
import methods
from application import load_data, sharp_if_small
from split_stability import describe

BASE = Path(__file__).resolve().parent
DEST = BASE / 'results' / 'sequence_audit'

def capture(x, seed):
    original = methods.evidence
    captured = {}
    def wrapper(data, a, b, *args, **kwargs):
        captured['selection_seconds'] = time.perf_counter() - begin
        t = time.perf_counter()
        values = original(data, a, b, *args, **kwargs)
        captured['evidence_seconds'] = time.perf_counter() - t
        captured.update(a=a.copy(), b=b.copy(), pvalues=values['T64'][:len(a)].copy())
        return values
    methods.evidence = wrapper
    begin = time.perf_counter()
    try:
        variants, diag = methods.split_orders(x, seed=seed)
    finally:
        methods.evidence = original
    captured['order_construction_seconds'] = time.perf_counter() - begin - captured['selection_seconds'] - captured['evidence_seconds']
    return variants, diag, captured

def job(index):
    old = json.loads((BASE/'results/split_stability/splits'/f'{index:03d}.json').read_text())
    x = load_data()[0]['adults']
    variants, diag, c = capture(x, old['seed'])
    ans = methods.project_order(variants['FS-T64'], 10)
    stats = describe(ans)
    for key, value in stats.items():
        assert value == old['methods']['FS-T64'][key], (index,key)
    assert variants['FS-T64']['raw_order_count'] == old['methods']['FS-T64']['raw_orders']
    pv = c['pvalues']; failed = np.flatnonzero(pv > .05)
    stop = int(failed[0]) if len(failed) else len(pv)
    assert stop == diag['diagnostics']['T64']['sequence_passes']
    edges = methods.edge_index(10)
    a, b = c['a'], c['b']
    tau = variants['FS-T64']['selection_score']
    pair = None if stop == len(pv) else [int(a[stop]),int(b[stop])]
    return dict(split=index, seed=old['seed'], comparisons=len(pv), passed=stop,
        stop_position=None if pair is None else stop+1,
        stronger_edge=None if pair is None else (edges[pair[0]]+1).tolist(),
        weaker_edge=None if pair is None else (edges[pair[1]]+1).tolist(),
        stop_pvalue=None if pair is None else float(pv[stop]),
        stop_training_gap=None if pair is None else float(tau[pair[0]]-tau[pair[1]]),
        later_nominal_passes=int(np.sum(pv[stop+1:] <= .05)),
        degree_lower=stats['degree_lower'], degree_upper=stats['degree_upper'],
        rank_lower=stats['rank_lower'], rank_upper=stats['rank_upper'],
        guaranteed_members=stats['guaranteed_members'],
        sequence_pairs=np.column_stack([a,b]).tolist(),pvalues=pv.tolist())

def aggregate(records):
    reference = json.loads((BASE/'results/split_stability/summary.json').read_text())['reference_E64']
    paircounts = Counter((tuple(r['stronger_edge']),tuple(r['weaker_edge'])) for r in records if r['stop_position'] is not None)
    positioncounts = Counter(r['stop_position'] for r in records)
    worse = dict(degree_intervals=0,rank_intervals=0,lost_guaranteed_member=0,any=0)
    for r in records:
        d=any(a<b for a,b in zip(r['degree_lower'],reference['degree_lower'])) or any(a>b for a,b in zip(r['degree_upper'],reference['degree_upper']))
        q=any(a<b for a,b in zip(r['rank_lower'],reference['rank_lower'])) or any(a>b for a,b in zip(r['rank_upper'],reference['rank_upper']))
        h=not set(reference['guaranteed_members']) <= set(r['guaranteed_members'])
        worse['degree_intervals']+=d;worse['rank_intervals']+=q
        worse['lost_guaranteed_member']+=h;worse['any']+=d or q or h
    assert worse['any']==39
    z=1.959963984540054;n=100;ph=.16
    centre=(ph+z*z/(2*n))/(1+z*z/n)
    radius=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/(1+z*z/n)
    summary=dict(all_100_archived_FS_T64_results_reproduced=True,
        split_sensitivity_Wilson_95=[float(centre-radius),float(centre+radius)],
        scope='Conditional-on-observed-data Monte Carlo uncertainty over independent split and matching seeds; not population coverage',
        E64_losses=worse,first_failure_pairs=[dict(stronger=list(a),weaker=list(b),count=v) for (a,b),v in paircounts.most_common()],
        stop_positions=[dict(position=k,count=v) for k,v in sorted(positioncounts.items(),key=lambda x:-1 if x[0] is None else x[0])],
        later_nominal_passes=dict(runs=int(sum(r['later_nominal_passes']>0 for r in records)),
          median=float(np.median([r['later_nominal_passes'] for r in records])),
          max=max(r['later_nominal_passes'] for r in records)),
        warning='Later p-values are diagnostic only and cannot be appended to the fixed-sequence rejection set.')
    (DEST/'sequence_records.json').write_text(json.dumps(records,indent=2)+'\n')
    (DEST/'sequence_records.json.gz').write_bytes(gzip.compress(json.dumps(records,separators=(',',':')).encode(),mtime=0))
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

def benchmark(repeats):
    sys.path.insert(0,str(BASE.parent/'review_audit'))
    from audit_abalone import exact_tree_polytope_lp
    from threadpoolctl import threadpool_limits, threadpool_info
    sets,meta=load_data();x=sets['adults'];rows=[]
    with threadpool_limits(limits=1):
        # One untimed full warm-up avoids charging lazy shared-library setup.
        vs,_,_=capture(x,61023117);a=methods.project_order(vs['FS-T64'],10)
        exact_tree_polytope_lp(a['relation'],a['edges'],10);sharp_if_small(a,10)
        for i in range(repeats):
            vs,diag,c=capture(x,61023117)
            t=time.perf_counter();a=methods.project_order(vs['FS-T64'],10);minor=time.perf_counter()-t
            t=time.perf_counter();strong=exact_tree_polytope_lp(a['relation'],a['edges'],10);lp=time.perf_counter()-t
            t=time.perf_counter();sharp=sharp_if_small(a,10);enumeration=time.perf_counter()-t
            assert sharp['compatible_trees']==2 and strong['lp_objectives']==110
            for k in ['degree_lower','degree_upper','rank_lower','rank_upper']:
                assert a[k].tolist()==strong[k]==sharp[k]
            row={k:float(c[k]) for k in ['selection_seconds','evidence_seconds','order_construction_seconds']}
            row.update(minor_seconds=minor,strong_lp_with_rational_checks_seconds=lp,enumeration_seconds=enumeration)
            rows.append(row)
        pools=threadpool_info()
    cpu=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),'unknown')
    result=dict(repeats=repeats,seed=61023117,n=35079,p=10,comparisons=112,matchings=64,
       warmup_runs=1,wall_clock='time.perf_counter; sequential, no concurrent benchmark workers',
       excludes='data retrieval/loading, imports and C++ compilation; all raw records already in memory',
       calibration_scope='Shared training and evidence for all five evidence types and all 15 split variants; not an optimized FS-T64-only implementation',
       projection_scope='FS-T64 only: minor; exact spanning-tree-polytope/path LP with 110 objectives and rational checks; exact enumeration',
       cpu=cpu,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
       thread_pools=pools,environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
       raw_sha256=meta['raw_zip_sha256'],runs=rows,
       median_seconds={k:float(np.median([r[k] for r in rows])) for k in rows[0]},
       range_seconds={k:[min(r[k] for r in rows),max(r[k] for r in rows)] for k in rows[0]})
    (DEST/'timings.json').write_text(json.dumps(result,indent=2)+'\n')
    print('TIMINGS',json.dumps(result['median_seconds']),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--benchmark-repeats',type=int,default=5);args=parser.parse_args()
    DEST.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records=[]
        for r in pool.map(job,range(100)):
            records.append(r)
            if len(records)%20==0:print(f'Checked {len(records)}/100 sequences',flush=True)
    aggregate(records)
    benchmark(args.benchmark_repeats)

if __name__=='__main__':main()
