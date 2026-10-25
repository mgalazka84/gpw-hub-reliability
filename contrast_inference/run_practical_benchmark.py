"""Serial single-core timing audit of cheap and full certified projections.

All 12 measurement populations and the first replication of every p=12,
n>=1008 simulation cell are included, for P and E64, local comparisons.
Timing repetitions are repeated computations, NOT independent datasets.
Each case/stage runs in its own process, with one untimed warm-up and three
measured runs. Calibration and projection are measured separately. Source
data loading and interpreter startup are excluded; certificate checking is in.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import sys,json,time,subprocess,platform,resource,argparse,hashlib,gzip
import numpy as np
import pandas as pd
import scipy
from order_inference import edge_index,project_order,kendall_order
from run_comparative_simulations import certify_lp,design,sample
from evalue_order import evalue_order
from audit_practical_value import metrics

ROOT=Path(__file__).resolve().parent;DEST=ROOT/'results'/'practical_value'/'benchmark'
MODES=['calibration','minor','degree_lp','full_lp']

def array_result(ans):
    return {k:ans[k].astype(int).tolist() for k in ['degree_lower','degree_upper','rank_lower','rank_upper']}

def relation(ans):return [np.flatnonzero(r).tolist() for r in ans['relation']]

def load_x(case):
    if case['source']=='measurement':
        from run_cross_section import datasets
        _,_,x,_=datasets()[case['index']]
        return x[np.random.default_rng(case['seed']).permutation(len(x))]
    rng=np.random.default_rng(case['seed'])
    if case['model']=='discrete_independent':return rng.integers(-1,2,size=(case['n'],case['p'])).astype(float)
    return sample(case['model'],case['n'],case['p'],rng)[0]

def calibrate(x,case):
    if case['method']=='paired_exact':return kendall_order(x,rng=np.random.default_rng(case['test_seed']),family='local')
    return evalue_order(x,seed=case['test_seed'],matchings=64,family='local')

def prepare():
    DEST.mkdir(parents=True,exist_ok=True);(DEST/'cases').mkdir(exist_ok=True);(DEST/'runs').mkdir(exist_ok=True)
    cases=[]
    for folder in [ROOT/'results'/'cross_section',ROOT/'results'/'evalue'/'cross_section']:
        for fn in sorted(folder.glob('dataset*.json')):
            j=json.loads(fn.read_text());index=int(fn.stem[-2:])
            for row in j['rows']:
                if row['family']!='local' or row['method'] not in ['paired_exact','evalue_64']:continue
                cert=next(c for c in j['certificates'] if c['method']==row['method'] and c['family']=='local')
                nd=sorted([v for v in j['nodes'] if v['method']==row['method'] and v['family']=='local'],key=lambda v:v['vertex'])
                cases.append(dict(source='measurement',index=index,dataset=row['dataset'],stratum=row['stratum'],n=row['n'],p=row['p'],
                    method=row['method'],seed=row['seed'],test_seed=row['seed']+(10000000 if row['method']=='paired_exact' else 20000000),
                    relation_rows=cert['relation_rows'],expected_full={k:[v[k] for v in nd] for k in ['degree_lower','degree_upper','rank_lower','rank_upper']}))
    config=design()
    for index,cell in enumerate(config['cells']):
        if cell['p']!=12 or cell['n']<1008:continue
        for method in ['paired_exact','evalue_64']:
            seed=config['seed']+index*100000
            case=dict(source='simulation',index=index,model=cell['model'],n=cell['n'],p=cell['p'],method=method,
                seed=seed,test_seed=seed+(40000000 if method=='paired_exact' else 41000000),rep=0)
            raw=calibrate(load_x(case),case);case['relation_rows']=relation(raw);cases.append(case)
    assert len(cases)==50
    for i,case in enumerate(cases):
        case['case_id']=i;(DEST/'cases'/f'case{i:02d}.json').write_text(json.dumps(case))
    cpu=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),'unknown')
    plan=dict(version='0.6',scope='All 12 measurement populations and first replication of all 13 p=12 n>=1008 simulation cells; P/E64, local family',
        cases=len(cases),measurement_cases=24,simulation_cases=26,modes=MODES,warmups=1,timing_repetitions=3,
        scheduling='Sequential fresh process per case/stage, pinned to the first available CPU; one BLAS/OpenMP/HiGHS thread',
        included='Calibration or minor plus indicated LP, including rational-certificate validation',
        excluded='Input loading, interpreter startup and one warm-up',
        memory='Linux ru_maxrss per fresh worker in KiB, including Python imports, input and warm-up; not incremental solver memory',
        hardware=dict(cpu=cpu,platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        reporting='Timing repetitions are not Monte Carlo replicates; these timings are hardware- and instance-specific')
    (DEST/'design.json').write_text(json.dumps(plan,indent=2));print('Prepared',len(cases),'cases',flush=True)

def worker(case_file,mode):
    case=json.loads(Path(case_file).read_text());p=case['p'];edges=edge_index(p)
    if hasattr(os,'sched_getaffinity'):
        cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    else:cpu=None
    rel=np.zeros((len(edges),len(edges)),bool)
    for i,r in enumerate(case['relation_rows']):rel[i,r]=True
    raw=dict(edges=edges,relation=rel)
    x=load_x(case) if mode=='calibration' else None
    times=[];answer=None
    for rep in range(4):
        t=time.perf_counter();minor_s=lp_s=0
        if mode=='calibration':
            ans=calibrate(x,case);total=time.perf_counter()-t
            assert np.array_equal(ans['relation'],rel),'Calibration differs from archived relation'
            answer=dict(raw_orders=ans['raw_order_count'],closed_orders=int(ans['relation'].sum()))
        else:
            ans=project_order(raw,p);minor_s=time.perf_counter()-t
            if mode!='minor':
                t_lp=time.perf_counter();ans=certify_lp(ans,p,degree_pairs=(mode=='full_lp'));lp_s=time.perf_counter()-t_lp
            total=time.perf_counter()-t
            current={**array_result(ans),**metrics(ans),'lp_solves':ans.get('lp_solves',0),
                'verified_certificates':ans.get('python_verified_certificates',0)}
            if answer is not None:assert current==answer,'Nondeterministic endpoints'
            answer=current
            if mode=='full_lp' and 'expected_full' in case:
                assert array_result(ans)==case['expected_full'],'Full projection differs from archived endpoints'
        if rep:times.append(dict(repetition=rep,total_seconds=total,minor_seconds=minor_s,lp_seconds=lp_s))
    out=dict(case_id=case['case_id'],mode=mode,cpu_affinity=cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        timings=times,answer=answer,case_sha256=hashlib.sha256(Path(case_file).read_bytes()).hexdigest())
    target=DEST/'runs'/f"case{case['case_id']:02d}_{mode}.json";target.write_text(json.dumps(out));print(target.name,flush=True)

def run():
    cases=sorted((DEST/'cases').glob('case*.json'));assert len(cases)==50
    start=time.time()
    for i,case in enumerate(cases):
        for mode in MODES:
            target=DEST/'runs'/f'{case.stem}_{mode}.json'
            if target.exists():continue
            subprocess.run([sys.executable,__file__,'--worker',str(case),'--mode',mode],check=True,stdout=subprocess.DEVNULL)
        print(f'Benchmark {i+1}/{len(cases)}, elapsed {time.time()-start:.1f}s',flush=True)
    report()

def report(from_records=False):
    rows=[];checks=[]
    case_files=[] if from_records else sorted((DEST/'cases').glob('case*.json'))
    if case_files:
        cases=[json.loads(cf.read_text()) for cf in case_files]
        runs=[json.loads(f.read_text()) for f in sorted((DEST/'runs').glob('case*.json'))]
        for name,records in [('cases',cases),('runs',runs)]:
            with gzip.open(DEST/(name+'.jsonl.gz'),'wt') as f:
                for record in records:f.write(json.dumps(record)+'\n')
    else:
        with gzip.open(DEST/'cases.jsonl.gz','rt') as f:cases=[json.loads(s) for s in f]
        with gzip.open(DEST/'runs.jsonl.gz','rt') as f:runs=[json.loads(s) for s in f]
    run_map={(r['case_id'],r['mode']):r for r in runs}
    assert len(cases)==50 and len(run_map)==200
    for case in cases:
        answers={}
        for mode in MODES:
            r=run_map[case['case_id'],mode];assert len(r['timings'])==3
            assert r['case_sha256']==hashlib.sha256(json.dumps(case).encode()).hexdigest()
            answers[mode]=r['answer']
            meta={k:case[k] for k in ['case_id','source','n','p','method']}
            meta.update(dataset=case.get('dataset',''),stratum=case.get('stratum',''),model=case.get('model',''))
            for t in r['timings']:
                rows.append({**meta,'mode':mode,**t,'peak_rss_kib':r['peak_rss_kib'],
                    **{k:v for k,v in r['answer'].items() if not isinstance(v,list)}})
        d,f=answers['degree_lp'],answers['full_lp']
        assert d['degree_lower']==f['degree_lower'] and d['degree_upper']==f['degree_upper']
        assert np.all(np.array(d['rank_lower'])<=f['rank_lower']) and np.all(np.array(d['rank_upper'])>=f['rank_upper'])
        checks.append(dict(case_id=case['case_id'],equal_degrees=True,equal_hub_exclusions=bool(np.array_equal(np.array(d['rank_lower'])>1,np.array(f['rank_lower'])>1)),
            equal_rank_endpoints=d['rank_lower']==f['rank_lower'] and d['rank_upper']==f['rank_upper']))
    pd.DataFrame(rows).to_csv(DEST/'timings.csv',index=False)
    (DEST/'verification.json').write_text(json.dumps(dict(status='PASS',cases=len(checks),measured_runs=len(rows),
        calibration_matches_archived_relations=True,degree_bounds_identical=True,checks=checks),indent=2))
    print('Benchmark PASS',len(checks),'cases',len(rows),'measured runs',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--run',action='store_true');a.add_argument('--report',action='store_true')
    a.add_argument('--worker');a.add_argument('--mode',choices=MODES);a.add_argument('--from-records',action='store_true');args=a.parse_args()
    if args.prepare:prepare()
    if args.worker:worker(args.worker,args.mode)
    if args.run:run()
    if args.report:report(args.from_records)
