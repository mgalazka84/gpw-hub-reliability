"""Acquire source inputs separately; no third-party prices are bundled in Git."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    for executable in ['curl','pdftotext']:
        if shutil.which(executable) is None:
            raise SystemExit(f'Required external command not found: {executable}')
    environment=os.environ.copy();environment['OPENBLAS_NUM_THREADS']='1'
    def run(*args):subprocess.run([sys.executable,*args],cwd=ROOT,env=environment,check=True)
    run('submission/download_constituents.py','--manifest','submission/data/constituents_manifest.json')
    run('empirical/download_bankier.py')
    source='https://api.bankier.pl/quotes/public/gpw-indices-section-chart/?symbols=WIG&intraday=false&max_period=true'
    destination=ROOT/'empirical/source_checks/wig_bankier_chart.json'
    if not destination.exists():
        response=subprocess.run(['curl','--silent','--show-error','--fail','--location','--max-time','40',source],check=True,capture_output=True)
        payload=json.loads(response.stdout)
        if not payload.get('data') or not payload['data'][0].get('data'):
            raise ValueError('No WIG history in source response')
        destination.write_bytes(response.stdout)
    record=dict(url=source,retrieved_at=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    (destination.parent/'wig_acquisition.json').write_text(json.dumps(record,indent=2))
    run('empirical/prepare_observed.py')
    print('Inputs acquired locally. Run: python reproduce.py --from-prices')
    print('Later data vintages can differ from the manuscript vintage of 3 October 2026.')

if __name__=='__main__':main()
