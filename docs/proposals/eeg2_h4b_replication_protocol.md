# P-4: the personal map on THINGS-EEG2

Implemented in `notebooks/09a_eeg2_h4b_replication.ipynb`, committed before any score of the personal map on
THINGS-EEG2 existed.

**Why.** THINGS-EEG2 was never used to develop or test the personal map, so it is a second-dataset test: a different
lab, recording and preprocessing, 4-repetition training averages and held-out test concepts. It opens a new family
at α = 0.05.

**Claim.** For each of the 10 THINGS-EEG2 people as the new person, borrowing from the other 9: borrowing plus the
personal map (`eval/eeg1_h4b.py`, unchanged: per 100 ms window a 17×17 map of the residual to borrowing, ridge
towards no change, penalty by image-wise cross-validation, fitted on the k labelled images) beats each of six
comparators by more than δ = 0.011: borrowing, Gu's window ensemble, the linear converter, own ridge, own ridge with
the output map, and the generic map (the mean of the other 9 people's maps, each fitted with its own leave-one-out
average and never using the new person's data).

**Endpoint and decision.** Per person, the mean over k ∈ {250, 500} and 100 ≤ t < 200 ms of the personal map minus
the comparator, averaged over draws 13–15 (the stored k-subsets, 01's outputs). Scores: the paper's split-half method
on all 80 test repetitions. One-sided bounds with Bonferroni over six at α = 0.05 (t with 9 df, about 3.1). Supported
if all six lower bounds exceed δ; rejected if any upper bound is below δ; otherwise inconclusive. All 10 people are
required; a failure is reported, not imputed. δ = 0.011 is THINGS-EEG2's margin (5% of the room between borrowing at
k = 0 and a person's own full model in the paper's reproduced numbers), fixed before the personal map existed.

**Known exposure.** The THINGS-EEG2 people were used to develop the baselines, so their baseline scores were known;
the personal map's scores on them were not.
