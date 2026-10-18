#!/usr/bin/env python3
"""Turn the private source files into the anonymized files the notebooks and scripts read.

Runs only on my machine. It needs a name map (data/private/clients.json, never committed)
that turns each real business name into the sector label used everywhere in this repo.
Nothing in this script names a business.

    python src/prepare_data.py --source /path/to/practicum

Reads from the practicum tree:
  week-6/analysis/data/gsc_history_device_query.parquet   Search Console, 16 months, date x device x query
  week-6/analysis/data/ga4_history_channel.parquet        GA4 sessions, date x channel
  data/processed/leadsnap_geogrids.parquet                Maps scans, one row per grid point
  data/processed/leadsnap_geogrids_full.parquet           same scans with the API status fields (older pull)
  data/processed/historical_geogrids.parquet              older scans from a different tool, June to August
  week-4/analysis/scan_triage.parquet                     the Week 4 usable / not-usable call per scan
Writes to data/private/ (git-ignored):
  gsc_daily.parquet, ga4.parquet, geogrids.parquet, geogrids_status.parquet, historical_geogrids.parquet, triage.parquet
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "private"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="path to the practicum tree")
    a = ap.parse_args()
    src = Path(a.source)
    cfg = json.loads((OUT / "clients.json").read_text())
    labels, default = cfg["labels"], cfg["default_label"]
    relabel = lambda s: s.map(lambda v: labels.get(v, default))

    g = pd.read_parquet(src / "week-6/analysis/data/gsc_history_device_query.parquet")
    g["client"] = relabel(g["client"])
    g.to_parquet(OUT / "gsc_daily.parquet", index=False)
    print(f"gsc_daily: {len(g):,} rows, {g.date.min().date()} to {g.date.max().date()}, {g.client.nunique()} clients")

    a4 = pd.read_parquet(src / "week-6/analysis/data/ga4_history_channel.parquet")
    a4["client"] = relabel(a4["client"])
    a4.to_parquet(OUT / "ga4.parquet", index=False)
    print(f"ga4: {len(a4):,} rows, {a4.date.min().date()} to {a4.date.max().date()}")

    geo = pd.read_parquet(src / "data/processed/leadsnap_geogrids.parquet")
    geo["business_name"] = relabel(geo["business_name"])
    ids = {v: f"place_anon_{i:03d}" for i, v in enumerate(sorted(geo["place_id"].dropna().unique()))}
    geo["place_id"] = geo["place_id"].map(ids)
    geo.to_parquet(OUT / "geogrids.parquet", index=False)
    print(f"geogrids: {len(geo):,} rows, {geo.heatmap_id.nunique()} scans, {geo.business_name.nunique()} businesses")

    full = pd.read_parquet(src / "data/processed/leadsnap_geogrids_full.parquet")
    full["business_name"] = relabel(full["business_name"])
    st = (full.groupby("heatmap_id")
              .agg(business_name=("business_name", "first"), status=("status", "first"),
                   points=("point_index", "size"), processed=("is_processed", "sum"),
                   ranked=("rank", lambda s: int(s.notna().sum())))
              .reset_index())
    st.to_parquet(OUT / "geogrids_status.parquet", index=False)
    print(f"geogrids_status: {len(st)} scans with API status fields")

    hist = pd.read_parquet(src / "data/processed/historical_geogrids.parquet")
    hist["business_name"] = relabel(hist["business_name"])
    hist = hist.drop(columns=[c for c in ("competitor_name", "client_id", "scan_file") if c in hist.columns])
    hist.to_parquet(OUT / "historical_geogrids.parquet", index=False)
    print(f"historical_geogrids: {len(hist):,} rows")

    tri = pd.read_parquet(src / "week-4/analysis/scan_triage.parquet")
    tri["sector"] = relabel(tri["sector"])
    tri.to_parquet(OUT / "triage.parquet", index=False)
    print(f"triage: {len(tri)} scans, usable {int(tri.usable.sum())}")


if __name__ == "__main__":
    main()
