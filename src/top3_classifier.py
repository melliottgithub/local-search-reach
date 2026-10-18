#!/usr/bin/env python3
"""Does distance alone predict whether the business shows up in the Maps top 3?

Same scan data and the same unit as reach.py: one logistic regression of top 3 (yes/no)
on distance from the storefront, fitted inside each scan. Here it is scored out of
sample (5 folds within each scan) and the predictions are pooled per business, so you
get the confusion matrix and the usual stats at a 0.5 cutoff, plus AUC.

Scans where the business is top 3 everywhere or nowhere are skipped: there is nothing
to classify there, and reach.py already reports them as ceiling or floor.

One thing I checked and left out: a single model pooled across all of a business's scans
(all keywords, all dates) is close to useless (AUC 0.1 to 0.7 on the real data). Whether
you are top 3 depends on the keyword and the date first, distance second. That is why
the fit is per scan.

    python src/top3_classifier.py                       # data/sample/geogrids.parquet
    python src/top3_classifier.py file.parquet --until 2026-09-20
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reach import ROOT, SAMPLE, load  # noqa: E402


def score(df):
    """Out-of-sample predictions, scan by scan: a logistic fit of top 3 on distance inside
    each scan, 5 stratified folds within that scan. Scans that are top 3 everywhere or
    nowhere have nothing to classify and are skipped. Predictions are then pooled."""
    y_all, p_all, scans = [], [], 0
    for _, s in df.groupby("heatmap_id"):
        X = s[["distance_miles"]].to_numpy()
        y = s["is_top3"].to_numpy().astype(int)
        if y.sum() < 5 or (len(y) - y.sum()) < 5:
            continue
        proba = np.zeros(len(y))
        for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=692).split(X, y):
            m = LogisticRegression().fit(X[tr], y[tr])
            proba[te] = m.predict_proba(X[te])[:, 1]
        y_all.append(y)
        p_all.append(proba)
        scans += 1
    if scans == 0:
        return None
    y = np.concatenate(y_all)
    proba = np.concatenate(p_all)
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(points=len(y), scans=scans, folds=5, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
                auc=roc_auc_score(y, proba))


def show(name, s):
    print(f"\n{name}  ({s['points']} grid points, {s['scans']} scans, {s['folds']} folds)")
    print("                   predicted no   predicted top 3")
    print(f"  actually no      {s['tn']:>10}   {s['fp']:>13}")
    print(f"  actually top 3   {s['fn']:>10}   {s['tp']:>13}")
    print(f"  accuracy {s['accuracy']:.2f}   precision {s['precision']:.2f}   recall {s['recall']:.2f}"
          f"   F1 {s['f1']:.2f}   AUC {s['auc']:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=str(SAMPLE))
    ap.add_argument("--until", help="only scans on or before this date")
    a = ap.parse_args()
    if a.path == str(SAMPLE) and not SAMPLE.exists():
        import sample_data
        sample_data.make()
    df = load(a.path, a.until)
    for b, g in df.groupby("business_name"):
        s = score(g)
        if s is None:
            print(f"\n{b}: not enough variation to score")
        else:
            show(b, s)
    s = score(df)
    if s is not None:
        show("All businesses pooled", s)


if __name__ == "__main__":
    main()
