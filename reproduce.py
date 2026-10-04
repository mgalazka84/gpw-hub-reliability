"""Reproduce public derived results, or the separate price-level workflow."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent

def run(*arguments):
    environment=os.environ.copy()
    environment['OPENBLAS_NUM_THREADS']='1'
    environment['OMP_NUM_THREADS']='1'
    subprocess.run([sys.executable,*arguments],cwd=ROOT,env=environment,check=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-prices',action='store_true',help='Requires locally acquired inputs; rebuild all empirical networks and forecasts')
    args=parser.parse_args()
    run('-m','unittest','discover','-s','submission','-p','test_methods.py','-v')
    if args.from_prices:
        required=[ROOT/'empirical/raw/company',ROOT/'empirical/source_checks/wig_bankier_chart.json',ROOT/'submission/data/constituents_snapshots.csv']
        missing=[str(p.relative_to(ROOT)) for p in required if not p.exists()]
        if missing:
            parser.error('Acquire inputs first with python tools/acquire_data.py. Missing: '+', '.join(missing))
        run('empirical/prepare_observed.py')
        run('empirical/build_observed_results.py')
        run('empirical/verify_observed_results.py')
    run('tools/verify_public_results.py')
    run('empirical/plot_observed_results.py')
    run('final/build_integrated_manuscript.py')
    print('Completed. Figures: empirical/figures; manuscript: final/deliverables; check: public_verification.json')

if __name__=='__main__':main()

