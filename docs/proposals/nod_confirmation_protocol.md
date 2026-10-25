# P-8: borrowing and the low-rank map in 3T single-trial fMRI (NOD)

Implemented in `notebooks/11d_nod_confirmation.ipynb`, committed and run on fake data before any target response
existed. Scored once. Until this run nothing of the 21 targets had been read beyond structure (11a: run counts,
shapes, finite checks).

**Why.** P-7 tested the method on NSD (7T, averaged repeats, high headroom). The development analysis on NOD's nine
donors (11c) shows a different regime: with one session of single 3T trials, borrowing from others beats the
person's own model at every budget, and the low-rank map adds nothing. The headroom account (the map recovers part of
the headroom, the person's own ceiling minus borrowing) predicts exactly this. P-8 tests that prediction in advance,
on 21 people never used. A new family at α = 0.05.

**People.** Targets: all 21 single-session people, sub-10 to sub-30. Donors: the nine four-session people, sub-01 to
sub-09, all nine for every target. No exclusions; all 21 must complete; a failure is reported, never imputed.

**Data.** 11a's output: the released single-trial betas (ridge GLM, NOD's `GLM.py`), 59,412 fsLR-32k grayordinates,
float16, no normalisation added. Features: 11b's AlexNet (primary) and DINOv2 (mixed system) kernel-PCA features;
kernel PCA fitted on donors' images only. Test images per target: 200 of their 1,000, drawn with
`np.random.default_rng([374, p]).permutation(1000)[:200]` (p = 10..30, the person's number). Pool: the other 800.
Labelled subsets: draws d ∈ {13, 14, 15}, `np.random.default_rng([d, p]).permutation(pool)[:k]`, k ∈ {250, 500}.
Vertices: those where the nine donors' own ridge models, fitted on 80% of their images and scored on 20% (seed 7),
have mean r above 0.05; fixed from the donors before any target is read (about 15,000 vertices).

**Methods (code from 11c).** *Sources:* each donor's ridge encoder (per-vertex penalty, 5-fold CV) on all 4,000
images. *Borrowing:* the mean of the nine donors' predictions. *Low-rank map:* K = 200, basis from borrowing's
predictions over the target's pool images. *Own ridge:* `RidgePerOutput` on the k labelled images. *Combination:* map
plus own ridge, one weight pair per vertex from 5-fold out-of-fold predictions, the labelled mean as intercept.
*Mixed system:* the map on DINOv2 borrowing plus own ridge on AlexNet. *Own full model:* ridge on all 800 pool images
(reference).

**Claims (Bonferroni over three, one-sided, t with 20 df).**
- **E1 (encoding)** borrowing minus own ridge > δ_r. Predicted: supported.
- **E2 (decoding)** mixed system minus own ridge > δ_top1. Predicted: supported.
- **E3 (encoding, P-7's C1 repeated)** low-rank map minus borrowing > δ_r. Predicted: rejected (upper bound below
  δ_r), because the headroom is near zero here. If it comes out supported, the headroom account is wrong for NOD.

**Endpoints.** *Encoding:* per person, the mean over the selected vertices of the Pearson correlation across the 200
test images. *Decoding:* 200-way identification over the test images, raw Pearson over vertices, the labelled-mean
intercept for every method with k > 0 (as P-7). Each endpoint per person is averaged over draws and over
k ∈ {250, 500}. Verdicts as P-6 and P-7: supported if the lower bound exceeds the margin, rejected if the upper bound
is below it, otherwise inconclusive.

**Margins.** δ_r = 0.011: 5% of the room, taking the median over donors of the 99th-percentile held-out r (0.221,
11c) as the reachable correlation, since NOD has no repeats and no noise ceiling. δ_top1 = 0.5 points: one chance
level among 200 (at 3–4% top-1, P-7's 2 points would be half the whole signal). Both from donors only.

**Power (from the nine donors, 11c).** Projected bounds with 21 people at the donors' means and SDs: E1 lower bound
+0.022 (margin 0.011), E2 +1.39 points (margin 0.5); E3 upper bound about +0.006 (margin 0.011). Risk: targets had
one session only and may differ in data quality from the four-session donors.

**Descriptive (not in the family).** P-7's C2 (combination minus borrowing) and C3 (combination minus own ridge);
mixed minus combination; the own full model; each budget separately; vertices at 0.1; the person-level headroom rule
with the built-in correlation removed (headroom measured on one half of the test images, map gain on the other half,
Spearman over 21 people).

**Exposure and limits.** The methods were developed on THINGS-EEG1 and THINGS-EEG2 and tested on NSD (P-7); the ridge
borrowing and the vertex rule were set on NOD's donors (11c), the claims and margins after seeing the donors' analysis.
Decoding is near chance at 3T single trials, so E2's absolute rates are small. Licence: NOD is CC0; data stay in
private Kaggle outputs.
