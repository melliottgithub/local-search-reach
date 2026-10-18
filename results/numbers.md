# The numbers behind the slides

What the scripts print on the real files, run on 2026-10-18 against the full pull of that day
(Search Console August 25, 2025 to October 18, 2026; Maps scans June 16 to October 14, 2026; GA4 to October 18). Where a slide number
differs from today's run, I say why.

## Slide 4: top 3 at home

Latest scan for the main service keyword that names the home city.

| | scan date | grid points in top 3 | on the slide |
|---|---|---|---|
| barbershop | Oct 11 | 169 of 169, 100% | 100% |
| dance studio | Sep 7 | 169 of 169, 100% | 100% |
| clinic | Sep 29 | 131 of 169, 78% | 78% |

`python src/reach.py data/private/geogrids.parquet` prints every scan with its points and top-3
count, so these rows are in that table.

## Slide 5: a rank tracker in the numbers

Rule: a desktop row with zero clicks, 20 or more impressions that day for that query, and
average position 2.5 or better. Those are scans, not people.

Clinic: 10,930 of 14,824 impressions over the whole history to October 18, 73.7%. The slide says
70.4%, which is the same calculation through September 20 (7,325 of 10,401). The others over the
whole history: barbershop 53%, insulation contractor 11%, dance studio 8%, consultancy 3%, dental
practice 0%. `src/synthetic_control.py` prints this block first; `notebooks/01_data_vetting_and_eda.ipynb`
section 3 shows where the rule comes from.

The dance studio's climb from #53 to #9 on the right of slide 5 uses 14 months of Search Console
history. The extract behind this repo starts July 1, 2026, so that chart
(`figures/slide5_dance_studio_web_climb.png`) comes from the longer private pull and cannot be
rebuilt from here.

## Slide 6: did the September change move impressions

```
python src/synthetic_control.py data/private/gsc_daily.parquet --treated "Specialty Barbershop" \
  --donors "Insulation Contractor" "Dental Practice" "Fractional-CTO Consultancy" "Latin Dance Studio" \
  --change 2026-09-20 --post-end 2026-09-29 --pre-start 2026-07-16
```

- Pre period July 16 to September 19, 66 days. Ridge alpha 3. Fit on the log scale, R2 0.35.
- September 20 to 29: 98 impressions, model expected 110.5. Difference -12.5, which is -11.3%.
- Placebos, 8 of them, 4 as big as the real difference or bigger. Earlier fake dates: Sep 6 -2.6%,
  Aug 30 +23.0%, Aug 23 +0.3%, Aug 16 +6.4%. Each donor treated as if it had the change:
  +32.8%, -42.7%, +44.5%, -9.7%.

The slide's 98 vs 110 matches. The slide also shows "+39%" for October 1 to 14 (16.6 a day against
11.9). That window runs past September 29, when one of the donors changed its own site, so the
model is not clean there, and there is no script for that number in this repo. Treat it as a hint
about where things were heading, not as a result.

## Slide 7: how far the reach goes

`python src/reach.py data/private/geogrids.parquet --until 2026-09-20` reproduces the slide.
Without `--until` you get today's numbers on the full October extract.

| | on the slide | scans to Sep 20, kriging | scans to Sep 20, logistic | all scans to Oct 14, kriging | all scans, logistic |
|---|---|---|---|---|---|
| barbershop | 4.2 mi | 1.1 mi (2 scans) | 1.8 mi (2) | 1.2 mi (3); Oct 11 scan top 3 at all 169 points, farthest point 4.2 mi | 1.5 mi (3) |
| dance studio | 4.1 mi | 4.1 mi (2 scans) | 4.9 mi (3) | 4.3 mi (3) | 6.1 mi (4) |
| clinic | 2.9 mi | 2.9 mi (14 scans) | 4.3 mi (20) | 5.1 mi (20) | 4.7 mi (33) |

So the dance studio's 4.1 and the clinic's 2.9 on the slide are the kriging medians over the
scans that existed on September 20, and the code here gives the same numbers on the same scans.
The scan set grew in October (five more fitted clinic scans, wider ones), which is why the full
run says 5.1 for the clinic today. If I had to give the clinic one number now I would say "3 to
5 miles, moving", not 2.9.

The barbershop's 4.2 is not a fitted radius. On October 11 it was top 3 at every point of the
grid, so the radius is somewhere past the grid; 4.2 miles is the farthest point scanned. The
fitted radius from the September scans was about 1 mile. Both are true: it got much bigger.

The logistic fit runs wider than kriging on these scans, by half a mile to a mile and a half.
The slides use kriging. Both are printed per scan so you can see where they disagree.

Scan states in the full run, three businesses: barbershop 13 scans (fitted 3, ceiling 1, floor 2,
sparse 6, no fit 1), dance studio 12 (fitted 4, ceiling 1, floor 4, sparse 1, no fit 2), clinic 52
(fitted 33, ceiling 2, floor 10, sparse 6, no fit 1). "Fitted" counts scans where at least one
estimator fit; the kriging medians above use only the scans where kriging fit.

The charts the two model scripts drew on the real data on 2026-10-17 are in `figures/run_*.png`.

## Classifier stats (asked in Q&A, not on a slide)

`python src/top3_classifier.py data/private/geogrids.parquet`. Logistic fit of top 3 on distance
inside each scan, scored out of sample with 5 folds per scan, predictions pooled. Scans that are
top 3 everywhere or nowhere are skipped.

The three businesses pooled, 41 scans, 6,513 grid points:

```
                   predicted no   predicted top 3
  actually no            2471               795
  actually top 3          478              2769
  accuracy 0.80   precision 0.78   recall 0.85   F1 0.81   AUC 0.88
```

Per business: barbershop accuracy 0.86, AUC 0.92 (4 scans). Clinic 0.82, AUC 0.89 (32 scans).
Dance studio 0.65, AUC 0.72 (5 scans).

For contrast, one model pooled across all of a business's scans, all keywords and dates together,
is close to useless (AUC between 0.1 and 0.7). Keyword and date decide most of it, distance decides
the rest. That is why everything is fitted per scan.
