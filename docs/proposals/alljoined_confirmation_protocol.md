# P-6: the personal map on consumer-grade EEG (Alljoined-1.6M)

Implemented in `notebooks/09l_alljoined_confirmation.ipynb`, committed before any Alljoined score existed. Scored
once.

**Why.** Alljoined-1.6M (Xu et al. 2025) recorded 20 people never used here, with a consumer 32-channel headset, on
the THINGS-EEG2 images in the same paradigm. This is a new family on fresh people, at α = 0.05. Only outcome-free
facts were looked at before the protocol was fixed: the image mapping, repetition counts, release windows and each
person's test reliability (09k).

**People (fixed from reliability only).** The 16 people whose test split-half reliability at 100–200 ms is at least
0.05 (09k manifest): sub-01, 02, 03, 04, 05, 06, 08, 10, 11, 12, 13, 14, 15, 17, 18, 20. Excluded: sub-07 (0.039),
sub-09 (0.008), sub-16 (0.023), sub-19 (0.021). Each included person is the new person in turn and borrows from the
other 15. A secondary, descriptive analysis repeats everything with all 20.

**Data.** The 09k files: the 17 THINGS-EEG2 channels, 100 Hz. All people cropped to the common 0–790 ms window (80
samples: sub-08's last 80, everyone else's first 80; sub-08's release starts at −200 ms). Training responses: the mean
over each image's available repetitions. Test: the first R repetitions of every test image, R = the smallest
per-person minimum among the included people (79). Image features: the paper's AlexNet kernel-PCA features (03b),
unchanged, in THINGS-EEG2's image order (the mapping proved in 09k). Labelled subsets: for person p (its number) and
draw d ∈ {13, 14, 15}, `np.random.default_rng([d, p]).permutation(16540)[:k]`, k ∈ {250, 500}.

**Methods (unchanged code).** Sources: each other included person's paper OLS encoder on all their training images.
*Borrowing*: the mean of the 15 source predictions. *Personal map*: borrowing plus a per-100 ms 17×17 residual map
(`eval/eeg1_h4b.py`). *Gu*: Gu's window ensemble (`gu_by_group`, pooled, 100 ms groups). *Gu + map*: the map fitted
on Gu's 5-fold out-of-fold predictions on the k images and applied to Gu's test prediction (`eval/plugin_map.py`).
*Procrustes*: an orthogonal map per 100 ms window (`eval/alignment_controls.py`). Own ridge is computed for
description only.

**Claims (Bonferroni over three at α = 0.05).**
- **C1** personal map minus borrowing > δ.
- **C2** Gu + map minus Gu > δ (the map improves a different borrowed predictor).
- **C3** personal map minus Procrustes per window > δ (beyond hyperalignment).

**Endpoint and decision.** Per person, the mean over k ∈ {250, 500} of the claim's difference in r at
100 ≤ t < 200 ms, each score averaged over draws 13–15; scores with the paper's split-half method on R test
repetitions. One-sided bounds with t at 1 − 0.05/3 (15 df). Per claim: supported if the lower bound exceeds δ,
rejected if the upper bound is below δ, otherwise inconclusive. At least 14 of the 16 people must complete; a failure
is reported and excluded, never imputed.

**Margin.** δ = 0.011 × (median Alljoined split-half ÷ median THINGS-EEG2 split-half), both computed in the scoring
notebook before any prediction is scored, with the same function (`noise_ceiling`, 20 splits, seed 13, 100–200 ms
mean): Alljoined over the 16 included people on their R test repetitions, THINGS-EEG2 over its 10 people on their 80.
0.011 is THINGS-EEG2's margin; scaling keeps the same share of signal.

**Exposure and limits.** No Alljoined prediction or score existed before this run. The methods were developed on
THINGS-EEG1 and THINGS-EEG2. Consumer-grade data are about five times less reliable than THINGS-EEG2, so an
inconclusive verdict is a plausible outcome. Licence: non-commercial research; the converted data stay private.
