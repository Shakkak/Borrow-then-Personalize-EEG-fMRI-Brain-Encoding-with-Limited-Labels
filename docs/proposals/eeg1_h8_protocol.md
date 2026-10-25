# P-2: does borrowing still win with a short recording? (THINGS-EEG1)

Implemented in `notebooks/08m_eeg1_h8_confirmation.ipynb`, committed before any reserved person was scored under this protocol. The
second of three follow-up tests on the 24 reserved people (`eeg1_followup_family.md`).

**Why.** In P-1 each person's EEG is whitened with a channel matrix estimated from all 22,248 of their training
trials. A new person who records only k trials would not have that. If borrowing only wins with the full-recording
whitening, few labels save no recording time.

**Claim (primary condition m = k).** The new person records only their k labelled trials; their whitening matrix is
estimated from those k trials alone (the same Ledoit-Wolf "time" mode as 08a) and applied to their k training trials
and their validation trials. Under that condition, averaged over k ∈ {250, 500} and 100 ≤ t < 200 ms, each of the
two borrowing methods (borrowing, Gu's window ensemble) beats each of the two own-data methods (own ridge, own ridge
with the output map) by more than δ = 0.0035. Four contrasts.

**Decision.** As P-1, at family-wise α = 0.05/3: per-person differences averaged over draws 13–15, one-sided bounds
with Bonferroni over the four contrasts within that α; supported if all four lower bounds exceed δ, rejected if any
upper bound is below δ, otherwise inconclusive; at least 20 of 24 complete. Descriptive only: m ∈ {k, 2,000, all},
and the change in each contrast from m = all to m = k.

**Scoring coordinates.** Each condition is scored in its own coordinates (validation trials whitened with that
condition's matrix), as a deployed system would face. The claim compares methods within one condition; comparisons
across m are descriptive.

**Development people stay fully calibrated.** They are the existing pool, recorded in full; only the new person's
access is bounded. Evaluation EEG never enters any whitening fit.

**Data.** 08a's arrays are already whitened, so 08g converts unwhitened epochs for the 46 eligible people from the
authors' files, split into two datasets: the 22 development people and the 24 reserved people (never attached during
development).

**Exposure.** The method was checked on the development people first. This is a declared later look at the 24
reserved people; their baseline scores at m = all were known from P-1.

**Prediction.** Supported: 250 trials × 100 samples give a stable 17-channel covariance, so whitening from k trials changes scores by under 0.005, and borrowing keeps its +0.03 to +0.06 lead over own-data methods seen with whitening fitted on all training trials.
