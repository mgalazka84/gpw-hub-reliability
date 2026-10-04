"""Ordinary requests to the public chart endpoint used by Bankier's own page."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import threading
import time
import pandas as pd

BASE=Path(__file__).parent
RAW=BASE/"raw/company"
RAW.mkdir(parents=True,exist_ok=True)
(BASE/"source_checks").mkdir(parents=True,exist_ok=True)
STOP=threading.Event()

def fetch(isin):
    url=f"https://api.bankier.pl/quotes/public/company-profile-chart/{isin}/?intraday=false&max_period=true"
    path=RAW/f"{isin}.json"
    stamp=datetime.now(timezone.utc).isoformat()
    if STOP.is_set():return dict(isin=isin,url=url,status="rate_limit_stop")
    try:
        if not path.exists():
            call=subprocess.run(["curl","-sS","-L","--max-time","35","-w","\n%{http_code}",url],capture_output=True)
            if call.returncode:raise ValueError(f"curl exit {call.returncode}: {call.stderr.decode()[:180]}")
            body,code=call.stdout.rsplit(b"\n",1)
            if code==b"429":STOP.set();raise ValueError("HTTP 429; further requests stopped")
            if code!=b"200":raise ValueError("HTTP "+code.decode())
            payload=json.loads(body)
            if payload.get("isin")!=isin:raise ValueError("Returned ISIN differs from requested security")
            path.write_bytes(body)
        body=path.read_bytes();payload=json.loads(body)
        if payload.get("isin")!=isin:raise ValueError("Cached ISIN mismatch")
        main=payload.get("main",[])
        if not main:raise ValueError("No historical OHLC observations returned")
        dates=pd.to_datetime([row[0] for row in main],unit="ms",utc=True)
        return dict(isin=isin,url=url,status="downloaded",rows=len(main),
            first_date=str(dates.min().date()),last_date=str(dates.max().date()),
            symbol=payload.get("symbol"),name=payload.get("nazwa_spolki"),
            provider_source_field=payload.get("source"),sha256=hashlib.sha256(body).hexdigest(),retrieved_at=stamp)
    except Exception as error:
        return dict(isin=isin,url=url,status="unavailable",error=str(error),retrieved_at=stamp)
    finally:time.sleep(.25)

def main():
    snapshots=pd.read_csv(BASE.parent/"submission/data/constituents_snapshots.csv",dtype={"isin":str})
    isins=sorted(snapshots["isin"].unique())
    for short,isin in [("pko","PLPKO0000016"),("tvn","PLTVN0000017")]:
        source=BASE/f"source_checks/{short}_bankier_chart.json"
        if source.exists() and not (RAW/f"{isin}.json").exists():shutil.copy2(source,RAW/f"{isin}.json")
    records=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs={pool.submit(fetch,isin):isin for isin in isins}
        for job in as_completed(jobs):
            record=job.result();records.append(record)
            if len(records)%15==0 or len(records)==len(isins):
                print(f"GPW OHLC: {len(records)}/{len(isins)}; downloaded {sum(r['status']=='downloaded' for r in records)}",flush=True)
                (BASE/"source_checks/company_download_manifest.json").write_text(json.dumps(sorted(records,key=lambda r:r["isin"]),indent=2))
    (BASE/"source_checks/company_download_manifest.json").write_text(json.dumps(sorted(records,key=lambda r:r["isin"]),indent=2))
    summary=dict(requested=len(isins),downloaded=sum(r["status"]=="downloaded" for r in records),
        unavailable=sum(r["status"]!="downloaded" for r in records),
        downloaded_rows=sum(r.get("rows",0) for r in records),source="Bankier public company-profile-chart endpoint",
        adjustment_policy="Not certified; OHLC and volume as supplied",raw_data_redistribution_license_not_established=True)
    (BASE/"source_checks/download_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)

if __name__=="__main__":main()
