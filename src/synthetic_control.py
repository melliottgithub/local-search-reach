#!/usr/bin/env python3
"""Did the website changes move search impressions? A synthetic control check.

Build a "what would have happened anyway" line for the business that got the change,
from businesses that did not get one (the donors). Ridge regression of the treated
business's daily log impressions on the donors' plus day of week, fitted on the days
before the change only. Compare the days after the change with that line.

Then the placebos. The same model is run on fake change dates (2 to 6 weeks earlier)
and on each donor as if it had been treated. If those "effects" are as big as the real
one, the real one is noise.

Rank-tracker hits are removed first: desktop rows with zero clicks, 20 or more
impressions and average position 2.5 or better. That is the rank tracker used for the Maps scans, not people.

A chart of actual vs expected (7-day averages) is written to results/run/synthetic_control.png.

    python src/synthetic_control.py                       # runs on data/sample/gsc_daily.parquet
    python src/synthetic_control.py file.parquet --treated "Specialty Barbershop" \
        --donors "Insulation Contractor" "Dental Practice" "Fractional-CTO Consultancy" "Latin Dance Studio" \
        --change 2026-09-20 --post-end 2026-09-29 --pre-start 2026-07-16
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample" / "gsc_daily.parquet"
ALPHAS = [0.3, 1, 3, 10, 30, 100]


def scan_mask(g):
    return (g.device == "DESKTOP") & (g.clicks == 0) & (g.impressions >= 20) & (g.position <= 2.5)


def load(path):
    g = pd.read_parquet(path)
    g["date"] = pd.to_datetime(g["date"])
    g["is_scan"] = scan_mask(g)
    daily = (g[~g.is_scan].pivot_table(index="date", columns="client", values="impressions", aggfunc="sum")
             .asfreq("D").fillna(0))
    return g, daily


def design(df, donors):
    X = np.log1p(df[donors].to_numpy())
    dow = pd.get_dummies(df.index.dayofweek).reindex(columns=range(7), fill_value=0).to_numpy()[:, 1:]
    return np.hstack([X, dow])


def fit_predict(daily, treated, donors, pre_start, change, post_end):
    pre = daily.loc[pre_start:change - pd.Timedelta(days=1)]
    post = daily.loc[change:post_end]
    Xp, yp = design(pre, donors), np.log1p(pre[treated].to_numpy())

    # pick alpha by time-series cross validation on the pre period only
    best, best_err = None, np.inf
    for a in ALPHAS:
        errs = []
        for tr, te in TimeSeriesSplit(n_splits=4).split(Xp):
            m = Ridge(alpha=a).fit(Xp[tr], yp[tr])
            errs.append(np.mean((m.predict(Xp[te]) - yp[te]) ** 2))
        if np.mean(errs) < best_err:
            best, best_err = a, np.mean(errs)

    m = Ridge(alpha=best).fit(Xp, yp)
    pre_hat = np.expm1(m.predict(Xp)).clip(0)
    post_hat = np.expm1(m.predict(design(post, donors))).clip(0)
    actual = post[treated].to_numpy()
    effect = float(actual.sum() - post_hat.sum())
    return dict(alpha=best, pre_r2_log=round(float(r2_score(yp, m.predict(Xp))), 3),
                pre_days=len(pre), post_days=len(post),
                post_actual=int(actual.sum()), post_model=round(float(post_hat.sum()), 1),
                effect=round(effect, 1), effect_pct=round(100 * effect / max(post_hat.sum(), 1e-9), 1),
                series=(pre.index, pre[treated].to_numpy(), pre_hat, post.index, actual, post_hat))


def placebos(daily, treated, donors, pre_start, change, post_days):
    out = []
    for weeks_back in range(2, 7):
        fake = change - pd.Timedelta(weeks=weeks_back)
        fake_end = fake + pd.Timedelta(days=post_days - 1)
        if fake_end >= change or (fake - pre_start).days < 28:
            continue
        r = fit_predict(daily, treated, donors, pre_start, fake, fake_end)
        out.append(dict(kind="earlier date", label=fake.strftime("%b %d"), effect_pct=r["effect_pct"]))
    for d in donors:
        others = [x for x in donors if x != d]
        r = fit_predict(daily, d, others, pre_start, change, change + pd.Timedelta(days=post_days - 1))
        out.append(dict(kind="untouched business", label=d, effect_pct=r["effect_pct"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=str(SAMPLE))
    ap.add_argument("--treated", default="Sample barbershop")
    ap.add_argument("--donors", nargs="+",
                    default=["Sample insulation contractor", "Sample dental practice", "Sample consultancy", "Sample dance studio"])
    ap.add_argument("--change", default="2026-09-20", help="first day the change was live")
    ap.add_argument("--post-end", default="2026-09-29", help="last day to compare (a donor shipped its own change on 09-30)")
    ap.add_argument("--pre-start", default="2026-07-16", help="first day every donor has data")
    a = ap.parse_args()

    if a.path == str(SAMPLE) and not SAMPLE.exists():
        sys.path.insert(0, str(ROOT / "src"))
        import sample_data
        sample_data.make()

    g, daily = load(a.path)
    missing = [c for c in [a.treated] + a.donors if c not in daily.columns]
    if missing:
        sys.exit(f"not in the data: {missing}. Clients present: {list(daily.columns)}")

    change, post_end, pre_start = pd.Timestamp(a.change), pd.Timestamp(a.post_end), pd.Timestamp(a.pre_start)

    tot = g.groupby("client").impressions.sum()
    ours = g[g.is_scan].groupby("client").impressions.sum().reindex(tot.index, fill_value=0)
    print("Rank-tracker share of impressions (removed before modelling)")
    for c in tot.index:
        print(f"  {c}: {ours[c]:,} of {tot[c]:,} ({100 * ours[c] / max(tot[c], 1):.1f}%)")

    r = fit_predict(daily, a.treated, a.donors, pre_start, change, post_end)
    print(f"\n{a.treated}, change on {change.date()}")
    print(f"  pre period {pre_start.date()} to {(change - pd.Timedelta(days=1)).date()}: {r['pre_days']} days, "
          f"ridge alpha {r['alpha']}, fit on log scale R2 {r['pre_r2_log']}")
    print(f"  after the change, {r['post_days']} days: {r['post_actual']} impressions, model expected {r['post_model']}")
    print(f"  difference {r['effect']:+.1f} ({r['effect_pct']:+.1f}%)")

    pl = placebos(daily, a.treated, a.donors, pre_start, change, r["post_days"])
    bigger = sum(abs(p["effect_pct"]) >= abs(r["effect_pct"]) for p in pl)
    print(f"\nPlacebos ({len(pl)}), {bigger} as big as the real difference or bigger")
    for p in pl:
        print(f"  {p['kind']:<20} {p['label']:<30} {p['effect_pct']:+.1f}%")
    print("\nIf several placebos are as big as the real difference, the real one is inside normal noise.")
    draw(r, a.treated, change)


def draw(r, treated, change):
    """Actual vs what the model expected, 7-day averages, change day marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = ROOT / "results" / "run"
    out.mkdir(parents=True, exist_ok=True)
    pre_idx, pre_y, pre_hat, post_idx, post_y, post_hat = r["series"]
    idx = pre_idx.append(post_idx)
    actual = pd.Series(np.concatenate([pre_y, post_y]), index=idx).rolling(7, min_periods=1).mean()
    model = pd.Series(np.concatenate([pre_hat, post_hat]), index=idx).rolling(7, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(model.index, model.values, "--", color="#6b7280", lw=2, label="expected (from the untouched businesses)")
    ax.plot(actual.index, actual.values, color="#2563eb", lw=2.5, label="actual, rank-tracker rows removed")
    ax.axvline(change, color="#111827", ls=":", lw=1.5)
    ax.axvspan(post_idx[0], post_idx[-1], color="#2563eb", alpha=0.08)
    ax.text(change, ax.get_ylim()[1] * 0.95, f" change {change.date()}", va="top", fontsize=9)
    ax.set_title(f"{treated}: {r['post_actual']} actual vs {r['post_model']:.0f} expected in the {r['post_days']} days after ({r['effect_pct']:+.1f}%)")
    ax.set_ylabel("impressions per day, 7-day average")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out / "synthetic_control.png"
    fig.savefig(path, dpi=150)
    print(f"chart: {path}")


if __name__ == "__main__":
    main()
