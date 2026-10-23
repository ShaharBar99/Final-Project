# Results summary - research questions and hypotheses

Data: PubMed-Diabetes after cleaning (19,717 papers, 44,324 citations).
Experiment: 3 kinds of damage (remove / add_random / add_dag) x 3 levels (5 / 10 / 20 % of edges) x 30 runs = 270 perturbed graphs.
Methods: Random Forest **without** PageRank as input (the real test of the Phase A idea), Random Forest **with** PageRank as input (Phase A design as written), and the five single centralities.
All numbers come from `results/tables/`; notebooks 07 and 08 rebuild them.

## Short answer

Combining centralities with a Random Forest does **not** give the most stable ranking overall. It does give a ranking that is clearly more stable than **PageRank - the measure it was trained to imitate** - when citations are *added*, and the gain is largest at the top of the ranking. When citations are *removed*, the forest is slightly less stable than PageRank. Simple local measures (degree, k-core) are the most stable of all, partly because they are coarse (many ties).

## Research questions (Phase A, Section 1)

| # | Question | Answer |
|---|---|---|
| Q1 | Does combining centralities give more stable rankings than single measures? | **Only against PageRank and only under edge addition.** Global Spearman, add_dag 20 %: forest 0.72 vs PageRank 0.67; share of runs a top-10 % paper stays in the top: 61 % vs 42 %. Under removal the forest is the least stable method (Spearman 0.83 vs 0.86 for PageRank at 20 %). Degree and k-core beat it everywhere. |
| Q2 | Do ML rankings stay consistent under perturbation? | **Yes, reasonably.** Mean Spearman with the base ranking is 0.83-0.94 under removal, 0.72-0.87 under add_dag and 0.58-0.82 under add_random. The 30 runs of one setting agree closely (std / mean of Spearman <= 0.011). |
| Q3 | Are the most important papers also the most stable? | **Among the important papers, mostly yes, but weakly** (Spearman between importance and CV inside the top 10 %: -0.04 to -0.48 for the forest). Over *all* papers it is the opposite under removal (+0.71 to +0.82): the bottom of the ranking never moves because its papers have almost no citations to lose. |
| Q4 | Are there papers that look important but are not stable? | **Yes, many.** Papers in the forest's top 10 % that drop out of it in more than half of the runs: 51-253 under removal, 422-968 under addition. For PageRank under addition it is 1,269-1,642 of 1,971 - most of its top 10 % is fragile. |
| Q5 | How consistent is the network core across kinds of damage? | **Moderately.** Jaccard overlap of the forest's core between different settings: mean 0.43 (min 0.25). PageRank 0.40, degree 0.61. The core depends on what kind of damage is assumed. |

## Hypotheses (Phase A, Section 7.1.3)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H1 - ML stability:** ML rankings are more robust than individual metrics | **Partly supported** | Better than PageRank under edge addition (global and especially at the top); worse than PageRank under removal; worse than degree / k-core / closeness in almost every setting. |
| **H2 - Ensemble robustness:** the forest absorbs a shock to one feature by relying on the others | **Not supported as stated** | The forest *with* PageRank puts 99.8 % of its importance on PageRank; when edge addition pushes ~60 % of PageRank values below the training range, it is *less* stable than PageRank itself (add_dag 20 %: 0.54 vs 0.67). Feature randomness does not help when one feature dominates. The forest *without* PageRank is protected only when the damage hits features it does not depend on. |
| **H3 - Core preservation:** key papers keep a high rank | **Supported with the adjusted core rule; the Phase A rule is degenerate** | Forest core papers stay in the top 10 % in 65-94 % of runs, more often than the other important papers (e.g. add_random 10 %: 90 % vs 40 %). The Phase A rule (CV <= 10th percentile of all papers) finds **no core** under removal, because the papers with CV = 0 are at the bottom of the ranking (mean base rank ~18,700). |

## Things that limit or explain the results

1. **Saturation of the trees.** A Random Forest cannot extrapolate. After edge addition many scaled inputs leave the training range (PageRank: 42-60 % of papers below the original minimum; closeness under add_random 20 %: 60 % above the maximum). The forest then gives all of them the same edge value. This can make scores look *more* stable (forest top-10 % CV under add_random falls from 0.16 to 0.06 as damage grows) - stability from blindness, not robustness. The frozen-scaler protocol of Phase A causes this by design.
2. **Ties inflate rank correlation.** k-core has 10 distinct values, degree 82. Coarse measures change little because they cannot change much. Betweenness is 0 for 91 % of papers, which makes it look like the most stable method under removal (Spearman 0.99) while its top 10 % is the least stable of all (CV 0.29-2.7).
3. **The target is PageRank.** The forest learns to imitate PageRank (hold-out Spearman 0.80 without PageRank as input). Its stability is therefore best compared with PageRank; comparing it with degree is comparing different notions of importance.
4. **Random edges are not citations.** add_random creates citation cycles (up to 2,498 papers that all reach each other) which cannot happen in reality. add_dag avoids this; the conclusions are the same for both, with add_random being harsher.
5. **One dataset.** All results are for PubMed-Diabetes only.

## Figures
- `results/spearman_vs_level.png` - global Spearman vs damage level, per kind
- `results/top10_retention.png` - share of the top 10 % that stays in the top 10 %
- `results/importance_vs_cv.png` - importance vs instability, forest, add_dag 10 %
