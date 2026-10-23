# To review before sharing

The points the team (and the advisor) should agree on before the work is shown to others. Ordered by importance. Each item names where it is documented.

## A. Changes to the Phase A plan - need the advisor's agreement

| # | What changed | Why | Where |
|---|---|---|---|
| A1 | Dataset is **PubMed-Diabetes**, not Cora | the dataset given for the project | D1 |
| A2 | Two forests: **with** and **without** PageRank as an input; the "without" one is treated as the main model | with PageRank as input the forest copies it (99.8 % importance, hold-out R^2 = 0.999), so the Phase A design tests nothing | D2, step 4 |
| A3 | **Exact** betweenness (igraph), not sampled | sampled version was inaccurate (Spearman 0.55 with exact) and its noise would look like instability; exact takes < 1 s | D4 |
| A4 | Edge addition run in two versions; **add_dag** (only newer -> older, no cycles) added next to Phase A's random addition | random edges create citation cycles that cannot exist (up to 2,498 papers) | D10 |
| A5 | **Adjusted network-core rule** (stability judged among the top 10 %) next to the Phase A rule | the Phase A rule gives an empty core under removal - its threshold is 0 and only bottom-ranked papers pass it | D16 |
| A6 | CV computed on scores in **natural units**, not on Min-Max values | CV changes with a shift; Min-Max values can be 0 or negative after damage | D12 |
| A7 | Extra metrics: Spearman over the **top 10 %**, **top-k kept**, **stay in top** | ties make coarse measures look stable over all papers | D15 |
| A8 | Temporal validation done **structurally** (no cycles), not by year | the dataset has no dates | D3 |
| A9 | 11 mutual citations resolved with "higher PubMed ID = newer" | no dates; checked afterwards - holds for 89.6 % of citations only | D5 |

## B. Findings - decide how to present them

1. **H1 is only partly supported.** The forest is more stable than PageRank under edge *addition* (especially at the top), slightly *less* stable under *removal*, and never more stable than degree or k-core. Agree on one sentence for the poster/abstract. Suggested: *"Combining centralities makes the ranking more robust than PageRank - the measure it learns - against added noise, but not against missing data, and simple local measures remain the most stable."*
2. **H2 is not supported as stated.** A forest that relies on one feature collapses with it; the with-PageRank forest is *less* stable than PageRank under addition. This contradicts the Phase A text on ensemble robustness (Theoretical Background - "Machine Learning Models", and hypothesis "Ensemble Robustness" in the evaluation chapter) - the final report should say so openly.
3. **Saturation.** Part of the forest's apparent stability comes from inputs leaving the training range (the frozen scaler + trees that cannot extrapolate). This is a direct consequence of the Phase A protocol. Decide whether to present it as a limitation or as a finding (we suggest: finding).
4. **The forest only imitates PageRank moderately** (hold-out Spearman 0.80 without PageRank input). Fine for the stability question, but someone may ask "is the forest a good importance measure at all?".
5. **Phase A's core rule is degenerate on this data** - a methodological finding about the original plan, worth one paragraph.
6. **One dataset, one target (PageRank), one model type.** State this as the main limitation.

## C. Numbers to double-check by a person

All numbers in `docs/results_summary.md` were checked against `results/tables/` once. Re-check at least the ones that go on the poster:
- Spearman, add_dag 20 %: forest 0.72 vs PageRank 0.67 (`ranking_stability.csv`)
- stay-in-top, add_dag 20 %: forest 0.61 vs PageRank 0.42 (`core_summary.csv`, `important_stay_in_top`)
- removal 20 %: forest 0.83 vs PageRank 0.86

## D. Text and metadata

- Phase A document (`docs/PhaseA.docx`) still says Cora and sampled betweenness - it is the Phase A submission and should stay as it was; the Phase B report must describe the changes in section A.
- Names, course code (26-2-R-2), advisor name - check spelling in `app/app.py` (tab "How it works"), `README.md` and the guides.
- Cite the dataset in the README, the report and the poster: Namata, G., London, B., Getoor, L. and Huang, B. (2012). *Query-driven active surveying for collective classification.* Workshop on Mining and Learning with Graphs (MLG). Distributed by the LINQS group. Please verify the reference details before using them.
- The GitHub repository is **public** (checked: GitHub answers without login) and contains the dataset files - check that this is allowed (the dataset is freely distributed for research) and that nothing private is in the history.
- Check the college's rules on AI-assisted work and how it must be acknowledged in the report.

## E. Technical things to know (not decisions, but people may ask)

- Results are fully reproducible: same seed -> bit-identical scores; a fresh clone rebuilds the models identically (checked).
- The raw experiment output (`results/runs/`, ~150 MB) is not on GitHub; notebook 06 rebuilds it in ~6 minutes.
- Two floating-point bugs were found and fixed during the work (tie-breaking noise, multi-threaded prediction). Neither changed any reported result - but they explain why ranking always goes through `to_ranking()`.
