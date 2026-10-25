# P-7: borrowing and the low-rank map in 7T fMRI (NSD)

Implemented in `notebooks/10c_nsd_confirmation.ipynb` (helpers in `src/cross_subject_encoding/eval/nsd_confirm.py`), committed before any NSD
score existed. Scored once. It was fixed after 09q set K for the low-rank map and 10b's features existed; until then no
NSD response had been read (10a reported structure and NSD's own noise ceilings only).

**Why.** NSD (Allen et al. 2022), through the Algonauts 2023 release, gives 8 people never used here, in another
modality (7T fMRI), on images never used here. This is a new family at α = 0.05.

**People.** All 8 (median noise ceilings 23–47%, 10a). Each person is the new person in turn and borrows from the
other 7. No exclusions: none falls below the level at which EEG people were excluded in P-6.

**Data.** The 10a output: z-scored training responses averaged over repeats, float16, both hemispheres. Shared space:
the fsaverage vertices present in all 8 people (from each person's `*.all-vertices_fsaverage_space.npy`). Test images:
the 872 images all 8 people saw, removed from every training and labelled set. Training pool per person: their own
images minus the 872. Labelled subsets: for person p and draw d ∈ {13, 14, 15},
`np.random.default_rng([d, p]).permutation(pool)[:k]`, k ∈ {250, 500} primary; 100 and 1,000 descriptive. Features:
10b's AlexNet kernel-PCA features (primary) and DINOv2 (the mixed system only).

**Methods.** *Sources:* each donor's paper OLS encoder (AlexNet features) on all their pool images. *Borrowing:* the
mean of the 7 donors' predictions. *Low-rank map:* `src/cross_subject_encoding/eval/lowrank_map.py` with K = 200 (09q's rule), basis from
borrowing's predictions over the person's pool images (images only). *Own ridge:* `RidgePerOutput` on the k labelled
images. *Combination:* the low-rank map and own ridge combined with one weight pair per vertex, fitted on 5-fold
out-of-fold predictions of the k images and applied as fitted, with the person's mean response over the k images as
intercept. *Own full model:* OLS on the person's whole pool (reference). *Mixed system* (secondary): the map on DINOv2
borrowing, own ridge on AlexNet.

**Claims (Bonferroni over three at α = 0.05, one-sided, t with 7 df).**
- **C1 (encoding)** low-rank map minus borrowing > δ_r.
- **C2 (decoding)** combination minus borrowing > δ_top1.
- **C3 (decoding)** combination minus own ridge > δ_top1 (combining beats the person's own model at the same budget).

**Endpoints.** *Encoding:* per person, the mean over all common vertices of the Pearson correlation across the 872
test images between prediction and response. *Decoding:* 200-way identification, raw (uncentred) Pearson over
vertices, with the labelled-mean intercept for every method with k > 0; borrowing at k = 0 is scored with the centred
score and stated as transductive; galleries: the 872 images split into 4 fixed disjoint sets of 200 (seed 346), top-1
averaged over the 4 (72 images unused). Each endpoint per person is averaged over draws and over k ∈ {250, 500}.
Verdicts as in P-6: supported if the lower bound exceeds the margin, rejected if the upper bound is below it,
otherwise inconclusive. All 8 must complete; a failure is reported, never imputed.

**Margins (outcome-free).** δ_r = 5% of the median theoretical maximum correlation, 0.05 × median over vertices of
√(noise ceiling / 100), from NSD's own ceilings (10a), the same 5%-of-room rule as THINGS-EEG2's 0.011. δ_top1 = 2
points of top-1 (four times chance, 0.5%).

**Descriptive (not in the family).** k = 100 and 1,000; per-region encoding gains (early visual, faces, places,
bodies, words); a forecast of who benefits (Spearman over 8 people); the mixed system; the own full model with and
without feature weighting; a learned-adapter comparator only if 09p shows it is competitive on EEG.

**Exposure and limits.** No NSD prediction or score existed before this run. The methods were developed on
THINGS-EEG1 and THINGS-EEG2; the low-rank form and K were fixed on THINGS-EEG2 (09q). With 8 people the bounds are
wide: a true effect can come out inconclusive. Algonauts responses are averaged over up to three repeats per image
(fewer for the four people who did not finish all sessions). Licence: NSD terms; data private.
