# P-1: a shared network against borrowing on THINGS-EEG1

Implemented in `notebooks/08e_eeg1_confirmation.ipynb` and `src/cross_subject_encoding/eval/eeg1_decision.py`, committed before any reserved
person was scored. Fixed values: margin δ = 0.003548 (measured in a pilot on four development people, see below);
k-draws 13–15 = the first k images of each person's stored order in `eeg1_k_subsets.json` (08a, sha256 3f3ede66…);
network seeds 13–15, 30 epochs, all 22 development people; upper bounds use the same Bonferroni t as the lower bounds.

**People.** THINGS-EEG1's 46 eligible people (`eeg1_pilot_manifest.json`) were split score-blind into 22 development
people and 24 reserved people. Model development used only the development people; reserved-data access was limited to preprocessing and reliability summaries.

**Claim.** With k ∈ {250, 500} labelled image–response pairs from a new person, *with noise normalization fitted on
all 22,248 of that person's training responses*, the shared network trained on the 22 development people with the
`prior_both` read-out (as implemented in 08e) predicts that person's responses to new images better than each named
comparator, by more than δ, on average over the two budgets and 100 ≤ t < 200 ms. Setting: new study, new people,
new images, familiar concepts (all 200 THINGS-EEG1 validation concepts occur in training). Not claimed: recording
savings, a win at each budget or time point, or a mechanism. 08a's reliability summaries touched the reserved data;
"reserved" means no model selection or performance scoring on them before this run.

**Data each method may use from the new person (identical for all methods).**

| Operation | Data of the new person used |
|---|---|
| Noise normalization (08a, `things_eeg1.mvnn_whitener`) | all 22,248 training responses, no labels |
| Fitting the correction, own ridge, converter, Gu's weights | the k labelled pairs only (draws 13–15) |
| Penalty, epoch and source selection | within the k pairs only |
| Network training | none (development people only) |
| Scoring | the 200 validation images × 12 repetitions, once |

**Comparators.** Exactly five: borrowing (the average of the 22 people's encoders, `avg_others`), Gu's window
ensemble (`B2_gu_window`), the linear converter (`B3_converter`), own ridge, and own ridge with the output map.

**Endpoint.** Per person p and comparator c: d_pc = mean over k ∈ {250, 500}, over the 10 samples with
100 ≤ t < 200 ms, of [score(network) − score(c)], where each score is `paper_correlation` (6-vs-6 repetition halves,
100 splits, channel-averaged), averaged over draws 13–15 and, for the network, over the three seeds (score
averaging, not prediction ensembling).

**Decision.** One-sided 95% lower bounds for mean_p d_pc, Bonferroni over the five comparators (t at 1 − 0.05/5,
23 df), paired over the 24 people. *Supported* if all five lower bounds exceed δ; *rejected* if any upper one-sided
bound is below δ; otherwise *inconclusive*. A person whose conversion or run fails is excluded and reported, never
imputed; the claim needs at least 20 of 24. The budget and time averaging is fixed, so multiplicity covers the
comparators only. A sign-flip test is reported as description, not as part of the decision.

**Margin.** δ = 5% of THINGS-EEG1's own room, the rule that gives 0.011 on THINGS-EEG2: the room is own ridge on all
images minus borrowing at k = 0, in the same window, measured on four development people treated as new
before any reserved person was scored. THINGS-EEG1's 12-repetition scores are lower
than THINGS-EEG2's, so a fixed absolute margin would demand a larger relative effect.

**Pilot.** One network on 18 development people, with development people 26, 29, 30 and 48 treated as new, at
k = 250 and 500, to measure the baselines and the margin; nothing was tuned towards this test.

**Failure handling.** An infrastructure failure (a crash, a failed export) may be rerun from the fixed inputs and is
recorded; a scientific result is never rerun to change it.

**Precision (planning assumptions).** THINGS-EEG2's ten people give paired standard deviations of 0.011–0.022 and mean gains of 0.047–0.073 over the five comparators. With 24 participants the simultaneous half-width is 0.006–0.011. The smallest expected lower bound clears 0.011 if THINGS-EEG1's gains are as large as THINGS-EEG2's (even with twice the standard deviation) or half as large with the same standard deviation, and fails at half the gain with 1.5 times the standard deviation. Four pilot participants cannot estimate the paired standard deviation.
