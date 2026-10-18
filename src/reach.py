#!/usr/bin/env python3
"""How far from the storefront does Google Maps still show the business in the top 3?

A scan is one business, one keyword, one date, and a grid of search points around the
address (usually 13 x 13). For each scan we fit the chance of being in the top 3 against
distance from the center and find r50: the distance where that chance falls to half of
what it is at the center.

Not every scan can be fitted, and we say which:
  ceiling  top 3 at every point. r50 is somewhere past the grid, so we report
           "at least <farthest point>" instead of a number.
  floor    top 3 nowhere. Reach inside this grid is zero.
  sparse   fewer than 3 top-3 points. Nothing to fit.
  fitted   a radius was estimated.
  nofit    enough points but neither estimator gave anything usable.

Two estimators run on every fittable scan and both are printed. Kriging walks out from
the center in rings over a kriged surface until the top-3 probability halves; the radii
on the slides are the medians of this one. Logistic is a plain fit of top 3 on distance,
the simpler check. Where both fit they usually land within about half a mile of each other.

A chart of the per-business summary is written to results/run/reach.png.

    python src/reach.py                              # runs on data/sample/geogrids.parquet
    python src/reach.py path/to/geogrids.parquet     # same columns, your own file
    python src/reach.py file.parquet --until 2026-09-20
    python src/reach.py file.parquet --businesses "Specialty Barbershop" "Latin Dance Studio"
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample" / "geogrids.parquet"

MIN_TOP3 = 3        # fewer top-3 points than this and there is nothing to fit
MIN_GRID_MI = 0.5   # a grid smaller than this cannot show decay
MAX_GRID_MI = 60.0  # a grid bigger than this spans other metros
DEG_LAT_MI = 69.055

try:
    from pykrige.ok import OrdinaryKriging
except ImportError:  # the logistic fit still works
    OrdinaryKriging = None


def r50_logistic(scan):
    """Plain logistic regression of top-3 on distance."""
    y = scan["is_top3"].to_numpy().astype(int)
    if y.sum() < MIN_TOP3 or y.sum() == len(y):
        return None
    x = scan["distance_miles"].to_numpy().reshape(-1, 1)
    m = LogisticRegression().fit(x, y)
    b0, b1 = float(m.intercept_[0]), float(m.coef_[0][0])
    if b1 >= 0:
        return None
    p0 = 1.0 / (1.0 + np.exp(-b0))
    target = p0 / 2.0
    r = (np.log(target / (1.0 - target)) - b0) / b1
    return float(r) if 0 < r <= MAX_GRID_MI else None


def r50_kriging(scan):
    """Ordinary kriging over the grid, then walk out from the center in rings until the
    kriged top-3 probability halves. This is the estimator the talk's numbers came from."""
    if OrdinaryKriging is None:
        return None
    y = scan["is_top3"].to_numpy().astype(float)
    lon, lat = scan["lng"].to_numpy(), scan["lat"].to_numpy()
    if len(np.unique(lon)) < 4 or len(np.unique(lat)) < 4:
        return None
    try:
        ok = OrdinaryKriging(lon, lat, y, variogram_model="spherical",
                             enable_plotting=False, coordinates_type="geographic")
    except Exception:
        return None
    clat, clon = float(scan["center_lat"].iloc[0]), float(scan["center_lng"].iloc[0])
    grid_mi = float(scan["distance_miles"].max())
    rings = np.arange(0.0, grid_mi + 0.001, max(grid_mi / 100.0, 0.05))
    bearings = np.deg2rad(np.arange(0, 360, 45))
    prof = []
    for r_mi in rings:
        dlat = (r_mi / DEG_LAT_MI) * np.cos(bearings)
        dlon = (r_mi / (DEG_LAT_MI * np.cos(np.deg2rad(clat)))) * np.sin(bearings)
        try:
            z, _ = ok.execute("points", clon + dlon, clat + dlat)
        except Exception:
            return None
        prof.append(float(np.mean(z)))
    prof = np.clip(np.array(prof), 0.0, 1.0)
    p0 = prof[0]
    if p0 <= 0.02:
        return None
    below = np.where(prof <= p0 / 2.0)[0]
    if len(below) == 0 or below[0] == 0:
        return None
    i = below[0]
    target = p0 / 2.0
    r_lo, r_hi, p_lo, p_hi = rings[i - 1], rings[i], prof[i - 1], prof[i]
    if p_lo == p_hi:
        return float(r_hi)
    return float(r_lo + (p_lo - target) / (p_lo - p_hi) * (r_hi - r_lo))


def classify(scan):
    n = len(scan)
    t3 = int(scan["is_top3"].sum())
    grid_mi = round(float(scan["distance_miles"].max()), 2)
    base = dict(points=n, top3=t3, grid_mi=grid_mi, r50_mi=None, r50_logistic_mi=None)
    if t3 == n:
        return dict(base, state="ceiling", at_least_mi=grid_mi)
    if t3 == 0:
        return dict(base, state="floor")
    if t3 < MIN_TOP3:
        return dict(base, state="sparse")
    rk = r50_kriging(scan)
    rl = r50_logistic(scan)
    if rk is not None and not (0 < rk <= MAX_GRID_MI):
        rk = None
    if rk is None and rl is None:
        return dict(base, state="nofit")
    return dict(base, state="fitted",
                r50_mi=round(rk, 2) if rk is not None else None,
                r50_logistic_mi=round(rl, 2) if rl is not None else None)


def load(path, until=None):
    df = pd.read_parquet(path)
    df["scan_date"] = pd.to_datetime(df["scan_date"], format="ISO8601", utc=True).dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["distance_miles", "is_top3", "lat", "lng"])
    if until:
        df = df[df["scan_date"] <= pd.Timestamp(until)]
    keep = []
    for hid, s in df.groupby("heatmap_id"):
        g = float(s["distance_miles"].max())
        if MIN_GRID_MI <= g <= MAX_GRID_MI:
            keep.append(hid)
    return df[df["heatmap_id"].isin(keep)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=str(SAMPLE))
    ap.add_argument("--until", help="only scans on or before this date, e.g. 2026-09-20")
    ap.add_argument("--businesses", nargs="+", help="only these business labels (default: all in the file)")
    a = ap.parse_args()

    if a.path == str(SAMPLE) and not SAMPLE.exists():
        sys.path.insert(0, str(ROOT / "src"))
        import sample_data
        sample_data.make()
    if OrdinaryKriging is None:
        print("note: pykrige not installed, using the logistic fit only\n")

    df = load(a.path, a.until)
    if a.businesses:
        df = df[df["business_name"].isin(a.businesses)]
    rows = []
    for hid, s in df.groupby("heatmap_id"):
        c = classify(s)
        rows.append(dict(business=s["business_name"].iloc[0], date=s["scan_date"].iloc[0].date(),
                         keyword=s["keyword"].iloc[0][:32], **c))
    t = pd.DataFrame(rows).sort_values(["business", "date", "keyword"])

    print("Per scan")
    with pd.option_context("display.width", 160, "display.max_rows", 500):
        print(t.to_string(index=False))

    print("\nPer business")
    summary = []
    for b, g in t.groupby("business"):
        krig = g["r50_mi"].dropna()
        logi = g["r50_logistic_mi"].dropna()
        ceil = g[g.state == "ceiling"]
        line = f"  {b}: {len(g)} scans"
        if len(krig):
            line += f", kriging median {krig.median():.1f} mi ({len(krig)} scans)"
        if len(logi):
            line += f", logistic median {logi.median():.1f} mi ({len(logi)} scans)"
        if len(ceil):
            line += f", top 3 everywhere on {len(ceil)} scan(s), farthest point {ceil['at_least_mi'].max():.1f} mi"
        counts = g.state.value_counts().to_dict()
        line += f"  ({', '.join(f'{k} {v}' for k, v in counts.items())})"
        print(line)
        summary.append(dict(business=b, kriging=krig.median() if len(krig) else None,
                            logistic=logi.median() if len(logi) else None,
                            ceiling=ceil["at_least_mi"].max() if len(ceil) else None))
    draw(summary)


def draw(summary):
    """Per-business reach as horizontal bars: kriging median, logistic median, and a marker
    where a scan was top 3 across the whole grid (reach at least that far)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = ROOT / "results" / "run"
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in summary if r["kriging"] or r["logistic"] or r["ceiling"]]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9, 0.9 * len(rows) + 1.5))
    ys = np.arange(len(rows))
    ax.barh(ys + 0.18, [r["kriging"] or 0 for r in rows], height=0.34, color="#2563eb", label="kriging median r50")
    ax.barh(ys - 0.18, [r["logistic"] or 0 for r in rows], height=0.34, color="#93c5fd", label="logistic median r50")
    for y, r in zip(ys, rows):
        if r["ceiling"]:
            ax.plot([r["ceiling"]], [y], marker="|", markersize=22, color="#ea580c", mew=3)
            ax.text(r["ceiling"] + 0.08, y, f"top 3 across the whole grid, {r['ceiling']:.1f} mi", va="center", fontsize=9, color="#ea580c")
    ax.set_yticks(ys)
    ax.set_yticklabels([r["business"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("miles from the address")
    ax.set_title("How far the Maps top-3 reach goes, per business")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out / "reach.png"
    fig.savefig(path, dpi=150)
    print(f"\nchart: {path}")


if __name__ == "__main__":
    main()
