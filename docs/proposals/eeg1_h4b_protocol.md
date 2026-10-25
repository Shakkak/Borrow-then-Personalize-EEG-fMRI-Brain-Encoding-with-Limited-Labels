# P-3: the personal map on THINGS-EEG1

Implemented in `notebooks/08n_eeg1_h4b_confirmation.ipynb`, committed before any reserved person was scored with
it. The third of three follow-up tests on the 24 reserved people (`eeg1_followup_family.md`).

**Claim.** On the 24 reserved THINGS-EEG1 people, borrowing from the 22 development people plus a per-person
channel map (the personal map: per 100 ms window, a 17×17 map of the residual to the borrowed prediction, fitted on
the person's k labelled images by ridge towards no change, penalty by image-wise cross-validation;
`eval/eeg1_h4b.py`) beats each of six comparators by more than δ = 0.0035: borrowing, Gu's window ensemble, the
linear converter, own ridge, own ridge with the output map, and the generic map (the mean of the maps fitted the same
way on each development person, so beating it means the gain is person-specific).

**Endpoint and decision.** As P-1 (`eval/eeg1_decision.py`): per person, the mean over k ∈ {250, 500} and
100 ≤ t < 200 ms of the personal map minus the comparator, averaged over draws 13–15; one-sided bounds with
Bonferroni over the six comparators at family-wise α = 0.05/3. Supported if all six lower bounds exceed δ; rejected
if any upper bound is below δ; otherwise inconclusive; at least 20 of 24 complete, failures excluded and reported.

**Data access.** The whitened data of 08a; the reserved people's k labelled pairs only for the map, own ridge, the
converter and Gu; validation trials scored once. Developed on the development people only; the reserved people had
been scored before in 08h and 08m for other methods.
