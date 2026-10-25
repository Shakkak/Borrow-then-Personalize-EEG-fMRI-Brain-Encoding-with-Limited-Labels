# Provenance of the prespecified tests

Each prespecified test was written as a protocol and committed to the author's version-controlled research
repository no later than the code that scored it, either in an earlier commit or in the same commit. The table
gives, for each test, the commit that holds its protocol and the commit of the code the scoring notebook ran
(printed by the notebook as its code version). Those commits belong to the author's private repository, whose history is not public, so this is a version-controlled record, not a public registration.
The protocols in `docs/proposals/` give each test's decision rules as they were fixed; the follow-up family on
THINGS-EEG1 is described in `docs/proposals/eeg1_followup_family.md`. Each test was scored once; the Alljoined
notebook's first run stopped while loading data, before any score was computed, and was run again.

| Test | Protocol | Scoring notebook | Protocol commit | Code commit | Order |
|---|---|---|---|---|---|
| P-1 (THINGS-EEG1, shared network) | `docs/proposals/eeg1_confirmation_protocol.md` | 08e | `c9503c4` | `c9503c4` | same commit |
| P-1 follow-up 1 (THINGS-EEG1, latent correction) | `docs/proposals/eeg1_confirmation_protocol.md` with the declaration in `docs/proposals/eeg1_followup_family.md` | 08h | `c9503c4`, declaration `be99032` | `6ccb177` | protocol first |
| P-2 (THINGS-EEG1, short recording) | `docs/proposals/eeg1_h8_protocol.md` | 08m | `029290c` | `029290c` | same commit |
| P-3 (THINGS-EEG1, personal map) | `docs/proposals/eeg1_h4b_protocol.md` | 08n | `e4ea90d` | `e4ea90d` | same commit |
| P-4 (THINGS-EEG2, personal map) | `docs/proposals/eeg2_h4b_replication_protocol.md` | 09a | `7f754b2` | `7f754b2` | same commit |
| P-6 (Alljoined-1.6M) | `docs/proposals/alljoined_confirmation_protocol.md` | 09l | `6392e74` | `659a2ec` | protocol first |
| P-7 (NSD) | `docs/proposals/nsd_confirmation_protocol.md` | 10c | `c9a3069` | `c9a3069` | same commit |
| P-8 (NOD) | `docs/proposals/nod_confirmation_protocol.md` | 11d | `0152ac7` | `39a7640` | protocol first |
