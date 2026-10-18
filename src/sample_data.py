#!/usr/bin/env python3
"""Make small fake datasets with the same columns as the real ones.

The real data belongs to the businesses and is not in this repo. These files exist
so the two model scripts run end to end. The numbers they produce mean nothing.

    python src/sample_data.py            # writes data/sample/geogrids.parquet and gsc_daily.parquet
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample"
DEG_LAT_MI = 69.055


def geogrids(rng):
    """One row per grid point. A scan is one business, one keyword, one date, a 13x13 grid."""
    rows, hid = [], 1000
    plan = [
        # business, center, (keyword, date, spacing_mi, r50_mi or None for 'top 3 everywhere')
        ("Sample barbershop", (39.74, -105.00), [("barber near me", "2026-07-12", 0.5, 1.2),
                                                 ("barber city", "2026-09-07", 0.3, 2.0),
                                                 ("barber city", "2026-10-11", 0.5, None)]),
        ("Sample dance studio", (39.72, -104.95), [("dance classes city", "2026-08-12", 0.9, 4.0),
                                                   ("dance classes city", "2026-09-07", 0.3, 3.8)]),
        ("Sample clinic", (39.66, -104.95), [("clinic specialty city", "2026-09-08", 0.3, 2.9),
                                             ("clinic specialty city", "2026-09-29", 0.3, 2.7),
                                             ("clinic generic", "2026-09-29", 0.3, 0.0)]),
    ]
    for business, (clat, clng), scans in plan:
        for keyword, date, spacing, r50 in scans:
            hid += 1
            n = 13
            for i in range(n):
                for j in range(n):
                    dy = (i - n // 2) * spacing
                    dx = (j - n // 2) * spacing
                    dist = float(np.hypot(dx, dy))
                    lat = clat + dy / DEG_LAT_MI
                    lng = clng + dx / (DEG_LAT_MI * np.cos(np.radians(clat)))
                    if r50 is None:
                        p = 0.98
                    elif r50 == 0.0:
                        p = 0.0
                    else:
                        p = 0.95 / (1.0 + np.exp(1.8 * (dist - r50)))
                    top3 = rng.random() < p
                    rank = int(rng.integers(1, 4)) if top3 else int(rng.integers(4, 21))
                    rows.append(dict(heatmap_id=hid, business_name=business, keyword=keyword,
                                     scan_date=date, grid_size=n, center_lat=clat, center_lng=clng,
                                     lat=lat, lng=lng, distance_miles=dist, rank=rank, is_top3=int(top3)))
    return pd.DataFrame(rows)


def gsc_daily(rng):
    """One row per client, day, device and search query, like a Search Console export."""
    clients = {
        "Sample barbershop": ("treated", 9.0),
        "Sample insulation contractor": ("control", 6.0),
        "Sample dental practice": ("control", 14.0),
        "Sample consultancy": ("control", 4.0),
        "Sample dance studio": ("control", 20.0),
        "Sample clinic": ("control", 12.0),
    }
    days = pd.date_range("2026-07-01", "2026-10-15", freq="D")
    rows = []
    for client, (role, level) in clients.items():
        for d in days:
            weekday = 1.0 if d.dayofweek < 5 else 0.75
            season = 1.0 + 0.15 * np.sin(2 * np.pi * d.dayofyear / 60.0)
            for q in range(1, 7):
                for device in ("MOBILE", "DESKTOP"):
                    lam = level * weekday * season / (q * (1.0 if device == "MOBILE" else 1.6))
                    imp = int(rng.poisson(lam))
                    if imp == 0:
                        continue
                    clicks = int(rng.binomial(imp, 0.04))
                    pos = float(np.clip(rng.normal(8 + q, 3), 1, 60))
                    rows.append(dict(client=client, role=role, date=d, device=device,
                                     query=f"query {q}", clicks=clicks, impressions=imp, position=round(pos, 1)))
            # a rank tracker hitting the clinic from a desktop every day: zero clicks, position ~1
            if client == "Sample clinic":
                rows.append(dict(client=client, role=role, date=d, device="DESKTOP", query="query 1",
                                 clicks=0, impressions=int(rng.integers(25, 45)), position=1.1))
    return pd.DataFrame(rows)


def make():
    rng = np.random.default_rng(692)
    OUT.mkdir(parents=True, exist_ok=True)
    g = geogrids(rng)
    s = gsc_daily(rng)
    g.to_parquet(OUT / "geogrids.parquet", index=False)
    s.to_parquet(OUT / "gsc_daily.parquet", index=False)
    print(f"wrote {OUT / 'geogrids.parquet'}  ({len(g):,} grid points, {g.heatmap_id.nunique()} scans)")
    print(f"wrote {OUT / 'gsc_daily.parquet'}  ({len(s):,} rows, {s.client.nunique()} clients)")


if __name__ == "__main__":
    make()
