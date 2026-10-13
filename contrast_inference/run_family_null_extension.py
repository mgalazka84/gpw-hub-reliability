"""Extend complete-edge-family null calibration to 1000 matched datasets."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,time
from run_comparative_simulations import design,worker,DEST

def run(workers=2):
    config=design();config['family_override']=['all'];config['lp_repetitions']=0
    config['extension']='Complete family, same null datasets 100..999; base run supplies 0..99'
    dest=DEST/'family_null_extension';dest.mkdir(parents=True,exist_ok=True)
    (dest/'design.json').write_text(json.dumps(config,indent=2));jobs=[]
    for ci,cell in enumerate(config['cells']):
        if cell['p']==12 and cell['model'] in ['independent','discrete_independent']:
            jobs.extend((ci,cell,rep,config) for rep in range(100,1000) if not (dest/f'cell{ci:02d}_{rep:04d}.json').exists())
    start=time.time();done=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(worker,j) for j in jobs]):
            ci,rep,rows=future.result();(dest/f'cell{ci:02d}_{rep:04d}.json').write_text(json.dumps(rows));done+=1
            if done%100==0:print('family null',done,len(jobs),time.time()-start,flush=True)
    print('COMPLETED family null',done,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=2);args=a.parse_args();run(args.workers)
