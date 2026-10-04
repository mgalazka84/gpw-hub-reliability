"""Audit observed public OHLC, without certifying cash-dividend adjustments."""
from pathlib import Path
import hashlib
import gzip
import json
import re
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
CHECKS = BASE / 'source_checks'
DATA.mkdir(exist_ok=True)
MONTHS = dict(stycznia=1,lutego=2,marca=3,kwietnia=4,maja=5,czerwca=6,
              lipca=7,sierpnia=8,wrzesnia=9,pazdziernika=10,listopada=11,grudnia=12)

def write_gzip(frame,path):
    content=frame.to_csv().encode('utf-8')
    blob=gzip.compress(content,compresslevel=6,mtime=0)
    path.write_bytes(blob)
    if gzip.decompress(path.read_bytes())!=content:
        raise IOError('Compressed panel round-trip failed: '+str(path))

def bars(rows):
    if any(len(row) != 5 for row in rows):
        raise ValueError('Unexpected OHLC row width')
    frame = pd.DataFrame(rows, columns=['timestamp','open','high','low','close'])
    dates = pd.to_datetime(frame.pop('timestamp'),unit='ms',utc=True)
    if not (dates == dates.dt.normalize()).all():
        raise ValueError('Daily chart timestamp is not midnight UTC')
    frame.index = dates.dt.tz_localize(None)
    frame.index.name = 'date'
    if frame.index.duplicated().any():
        raise ValueError('Duplicate session dates')
    if not frame.index.is_monotonic_increasing:
        raise ValueError('Unordered session dates')
    finite = np.isfinite(frame).all(axis=1)
    positive = frame.gt(0).all(axis=1)
    within = ((frame.low <= frame[['open','close']].min(axis=1)+1e-8)
              & (frame.high+1e-8 >= frame[['open','close']].max(axis=1))
              & (frame.high+1e-8 >= frame.low))
    valid = finite & positive & within
    return frame, valid

def reference_date(name):
    text = (BASE.parent/'submission/data/constituents'/name.replace('.pdf','.txt')).read_text()
    prefix = text[:1000].lower().replace('ś','s').replace('ź','z').replace('ń','n')
    match = re.search(r'wg\s+stanu\s+na\s+(\d{1,2})\s+([a-z]+)\s+(\d{4})',prefix)
    if not match:
        return None
    day, month, year = match.groups()
    return pd.Timestamp(int(year),MONTHS[month],int(day))

def main():
    wig_path = CHECKS/'wig_bankier_chart.json'
    wig_payload = json.loads(wig_path.read_text())
    wig, valid = bars(wig_payload['data'][0]['data'])
    if not valid.loc['2005-01-01':'2025-09-19'].all():
        raise ValueError('Invalid benchmark OHLC in analysis calendar')
    calendar = wig.loc['2005-01-01':'2025-09-19'].index
    wig.loc[calendar].to_csv(DATA/'wig_ohlc.csv')
    raw_frames, checks, rejected = {}, [], []
    for path in sorted((BASE/'raw/company').glob('*.json')):
        payload = json.loads(path.read_text())
        if not payload.get('main'):
            continue  # Logged as unavailable in company_download_manifest.json.
        frame, valid = bars(payload['main'])
        volumes = pd.DataFrame(payload['volume'],columns=['timestamp','volume'])
        volumes.index = pd.to_datetime(volumes.pop('timestamp'),unit='ms',utc=True).dt.tz_localize(None)
        if volumes.index.duplicated().any():
            raise ValueError(f'Duplicate volume dates: {path.stem}')
        frame['volume'] = volumes.volume.reindex(frame.index)
        valid &= frame.volume.notna() & np.isfinite(frame.volume) & frame.volume.ge(0)
        for date, row in frame.loc[~valid].iterrows():
            rejected.append(dict(isin=path.stem,date=str(date.date()),**row.to_dict()))
        # Reject malformed bars explicitly; no imputation or forward filling.
        clean = frame.where(valid, np.nan)
        raw_frames[path.stem] = clean
        checks.append(dict(isin=path.stem,symbol=payload.get('symbol'),rows=len(frame),
            valid_rows=int(valid.sum()),rejected_rows=int((~valid).sum()),
            analysis_rows=int(clean.reindex(calendar).close.notna().sum()),
            first_date=str(frame.index.min().date()),last_date=str(frame.index.max().date()),
            ended_before_2025_cutoff=bool(frame.index.max()<calendar.max()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    pd.DataFrame(checks).to_csv(CHECKS/'ohlc_audit.csv',index=False)
    pd.DataFrame(rejected,columns=['isin','date','open','high','low','close','volume']).to_csv(CHECKS/'rejected_bars.csv',index=False)
    panels = {}
    for field in ['open','high','low','close','volume']:
        panels[field] = pd.DataFrame({isin:f[field].reindex(calendar) for isin,f in raw_frames.items()},index=calendar)
        write_gzip(panels[field],DATA/f'{field}_panel.csv.gz')
    close_simple = panels['close'].div(panels['close'].shift())-1
    intraday_simple = panels['close'].div(panels['open'])-1
    write_gzip(close_simple,DATA/'close_simple_returns.csv.gz')
    write_gzip(intraday_simple,DATA/'intraday_simple_returns.csv.gz')
    snapshots = pd.read_csv(BASE.parent/'submission/data/constituents_snapshots.csv')
    comparisons=[]
    dates={source:reference_date(source) for source in snapshots.source_pdf.unique()}
    for _,row in snapshots.iterrows():
        date=dates[row.source_pdf]
        reference=row.price_reference
        if date is None or pd.isna(reference):
            continue
        series=raw_frames.get(row['isin'])
        quote=np.nan if series is None or date not in series.index else series.at[date,'close']
        ratio=quote/reference if pd.notna(quote) else np.nan
        comparisons.append(dict(source_pdf=row.source_pdf,reference_date=str(date.date()),
            isin=row['isin'],gpw_reference=reference,bankier_close=quote,ratio=ratio,
            exact_to_rounding=bool(pd.notna(quote) and abs(quote-reference)<=max(.011,reference*.00011))))
    comp=pd.DataFrame(comparisons)
    comp['integer_price_rounding_match']=False
    mask=comp.source_pdf.isin(['2025_06_20_WIG20.pdf','2025_09_19_WIG20.pdf']) & comp.bankier_close.notna()
    comp.loc[mask,'integer_price_rounding_match']=(comp.loc[mask,'gpw_reference']-comp.loc[mask,'bankier_close']).abs()<=.500001
    factors=np.array([.04,.1,.2,.25,.5,1.6,2,3,4,5,8,10,16,20,25,50,100,120])
    comp['ratio_near_common_denomination_factor']=np.min(abs(comp.ratio.to_numpy()[:,None]/factors-1),axis=1)<.001
    comp['ratio_near_compounded_denomination_factor']=(comp['isin']=='PLDWORY00019') & ((comp.ratio/(1/67.2)-1).abs()<.003)
    comp.to_csv(CHECKS/'gpw_reference_comparisons.csv',index=False)
    # Large overnight movements are flags, not automatically deleted as errors.
    jumps=close_simple.stack().loc[lambda x: x.abs()>.35].reset_index()
    jumps.columns=['date','isin','close_to_close_return']
    jumps.to_csv(CHECKS/'large_price_movements.csv',index=False)
    valid_comp=comp.dropna(subset=['bankier_close'])
    summary=dict(downloaded_series=len(raw_frames),calendar_sessions=len(calendar),
        first_calendar_date=str(calendar.min().date()),last_calendar_date=str(calendar.max().date()),
        observed_company_bars=sum(c['rows'] for c in checks),rejected_bars=len(rejected),
        series_ended_before_cutoff=sum(c['ended_before_2025_cutoff'] for c in checks),
        gpw_reference_rows=len(comp),gpw_reference_comparable=len(valid_comp),
        gpw_reference_exact=int(valid_comp.exact_to_rounding.sum()),
        large_close_to_close_movements=len(jumps),
        nonmatching_reference_prices_explanation='Historical denomination adjustments possible; cash-dividend adjustment policy not certified',
        no_imputation=True,raw_price_source='Observed Bankier public chart responses',
        wig_raw_sha256=hashlib.sha256(wig_path.read_bytes()).hexdigest())
    summary['additional_matches_at_integer_price_precision']=int((comp.integer_price_rounding_match&~comp.exact_to_rounding).sum())
    summary['nonmatching_ratios_near_denomination_factors']=int((comp.ratio_near_common_denomination_factor&~comp.exact_to_rounding&~comp.integer_price_rounding_match).sum())
    summary['nonmatching_compounded_denomination_ratios']=int((comp.ratio_near_compounded_denomination_factor&~comp.exact_to_rounding).sum())
    explained=comp.exact_to_rounding|comp.integer_price_rounding_match|comp.ratio_near_common_denomination_factor|comp.ratio_near_compounded_denomination_factor
    summary['unresolved_reference_mismatches']=int((comp.bankier_close.notna()&~explained).sum())
    summary['denomination_ratios_are_diagnostic_not_certified_corporate_action_audit']=True
    (CHECKS/'observed_data_audit.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
