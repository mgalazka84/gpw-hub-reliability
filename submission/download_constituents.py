"""Retrieve public, official quarterly WIG20/mWIG40 portfolio PDFs.

This is a snapshot catalogue, not a complete point-in-time membership history:
extraordinary replacements and actual effective timing remain to be reconciled.
PDFs stay local; share the manifest and extraction, not third-party source PDFs.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urljoin
from urllib.request import urlopen

BASE = Path(__file__).parent
PAGE = "https://gpwbenchmark.pl/historyczne-portfele-indeksow"


def parse(path: Path, index: str, effective: str):
    text_path = path.with_suffix(".txt")
    subprocess.run(["pdftotext", "-layout", str(path), str(text_path)], check=True)
    text = text_path.read_text()
    # Exclude every reserve table; multi-page repeated main headings are harmless.
    text = re.split(r"Lista rezerwowa|Reserve list", text)[0]
    has_price = bool(re.search(r"Kurs|Price", text))
    has_weight = bool(re.search(r"Udział|Share", text))
    pattern = re.compile(r"^\s*(\d+)\s+([A-Z]{2}[A-Z0-9]{9}\d)\s+(.+?)\s*$")
    rows = []
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        rank, isin, remainder = match.groups()
        columns = re.split(r"\s{2,}", remainder.strip())
        if len(columns) == 2 and not has_price:
            name, shares = columns
            price, weight = "", ""
        elif len(columns) == 3 and has_price and has_weight and "," in columns[2]:
            # Old fixed-width tables sometimes leave one space between price and shares.
            merged = re.fullmatch(r"([\d ]+,\d+)\s+([\d ]+)", columns[1])
            if not merged:
                continue
            name, weight = columns[0], columns[2]
            price, shares = merged.groups()
        elif len(columns) in [3, 4]:
            name, price, shares = columns[:3]
            weight = columns[3] if len(columns) == 4 else ""
        else:
            continue
        rows.append(dict(snapshot_revision_date=effective, index=index, rank=int(rank),
                         isin=isin, source_name=name.strip(),
                         price_reference=float(price.replace(" ", "").replace(",", ".")) if price else None,
                         shares=int(shares.replace(" ", "")),
                         weight_percent=float(weight.replace(" ", "").replace(",", ".")) if weight else None,
                         source_pdf=path.name))
    return rows


def retrieve(item):
    url, index, effective = item
    destination = BASE / "data" / "constituents" / url.rsplit("/", 1)[-1]
    try:
        if not destination.exists():
            # Bounded ordinary retries for interrupted downloads; no access-wall bypass.
            for attempt in range(3):
                try:
                    fetched = subprocess.run(["curl", "--silent", "--show-error", "--fail",
                        "--location", "--max-time", "35", url], capture_output=True, check=True)
                    payload = fetched.stdout
                    break
                except subprocess.CalledProcessError:
                    if attempt == 2:
                        raise
            if not payload.startswith(b"%PDF"):
                raise ValueError("Source returned non-PDF content")
            destination.write_bytes(payload)
        payload = destination.read_bytes()
        rows = parse(destination, index, effective)
        expected = 20 if index == "WIG20" else 40
        ranks = sorted(r["rank"] for r in rows)
        all_weights = len(rows) > 0 and all(r["weight_percent"] is not None for r in rows)
        weight = sum(r["weight_percent"] for r in rows) if all_weights else None
        # Early sources publish weights rounded to two decimals.
        tolerance = .005 * expected + 1e-8
        identities_valid = ranks == list(range(1, expected+1)) and len({r["isin"] for r in rows}) == expected
        status = "validated" if identities_valid and (weight is None or abs(weight-100) <= tolerance) else "review_required"
        for row in rows:
            row["portfolio_status"] = status
        return dict(url=url, snapshot_revision_date=effective, index=index, status=status,
                    records=len(rows), weight_sum=round(weight, 6) if weight is not None else None,
                    weights_present=all_weights,
                    sha256=hashlib.sha256(payload).hexdigest(), file=destination.name), rows
    except Exception as error:
        return dict(url=url, snapshot_revision_date=effective, index=index, status="failed",
                    error=str(error)), []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--manifest", type=Path, help="Use the recorded source URL list")
    args = parser.parse_args()
    output = BASE / "data" / "constituents"
    output.mkdir(parents=True, exist_ok=True)
    (BASE / "source_checks").mkdir(parents=True, exist_ok=True)
    page_path = BASE / "source_checks" / "constituents.html"
    if not args.manifest and not page_path.exists():
        with urlopen(PAGE, timeout=30) as response:
            page_path.write_bytes(response.read())
    html = "" if args.manifest else page_path.read_text()
    urls = ([item["url"] for item in json.loads(args.manifest.read_text())] if args.manifest else
            sorted(set(urljoin(PAGE, href) for href in re.findall(r'href="([^"]+\.pdf)"', html))))
    candidates = []
    for url in urls:
        match = re.search(r"(\d{4})_(\d{2})_(\d{2})_(WIG20|mWIG40)\.pdf$", url)
        if match and 2006 <= int(match[1]) <= 2025:
            year, month, day, index = match.groups()
            candidates.append((url, index, f"{year}-{month}-{day}"))
    if args.limit:
        candidates = candidates[:args.limit]
    records, manifests = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        for position, (manifest, rows) in enumerate(executor.map(retrieve, candidates), 1):
            manifests.append(manifest); records.extend(rows)
            if position % 20 == 0:
                print(f"Official portfolios processed: {position}/{len(candidates)}", flush=True)
    records.sort(key=lambda r: (r["snapshot_revision_date"], r["index"], r["rank"]))
    if records:
        with (BASE / "data" / "constituents_snapshots.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    (BASE / "data" / "constituents_manifest.json").write_text(json.dumps(manifests, indent=2))
    summary = dict(portfolios=len(manifests), records=len(records),
                   unique_isins=len(set(r["isin"] for r in records)),
                   validated=sum(m["status"] == "validated" for m in manifests),
                   review_required=sum(m["status"] == "review_required" for m in manifests),
                   failed=sum(m["status"] == "failed" for m in manifests))
    (BASE / "data" / "constituents_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
