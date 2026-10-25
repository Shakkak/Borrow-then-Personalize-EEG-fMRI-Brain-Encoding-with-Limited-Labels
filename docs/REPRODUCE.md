# Reproducing the paper

Three levels, from cheapest to most complete:

1. **Check the code** (CPU): install and run the tests.
2. **Redraw the figures and tables** from the result files in `results/` (CPU, mostly offline).
3. **Rerun the analyses** from the raw datasets (Kaggle CPU or GPU sessions, your own data access).

## 1. Check the code

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU torch; several tests import it
pip install -e ".[test]"
python -m pytest -q
```

The exact versions the suite passed with are in `environment/requirements-tested.txt` (Python 3.12, CPU torch). To
use them in a container:

```bash
docker build -t xsub-torch:local -f environment/Dockerfile .
docker run --rm -v "$PWD":/repo -w /repo xsub-torch:local sh -c 'pip install -q --no-deps -e . && python -m pytest -q'
```

All tests run on synthetic data or on the small files in `results/`; none downloads a dataset. Two of them run the
code of the THINGS-EEG1 test notebooks (08e, 08h) end to end on tiny synthetic releases.

## 2. Redraw the figures and tables

Run from the repository root (`pip install -e .` first). Figures are written to `results/figures/v2/`, `v3/` and
`v5/`; the scripts also draw a few figures the paper does not use.

```bash
python scripts/plot_paper_v2.py                    # offline
python scripts/plot_paper_v3.py                    # offline
python scripts/plot_paper_v5.py --skip-examples    # offline; drop the flag to also draw Figure S5 (needs internet)
python scripts/paper_contrast_table.py             # offline; prints Table S2's rows
python scripts/selection_check.py                  # offline; the selection check of the Limitations
python scripts/plot_paper_v3_fmri.py               # needs nilearn and NSD-derived files you make yourself (below)
```

| Paper item | Drawn by | From the notebooks | Offline |
|---|---|---|---|
| Figure 1 (schematic) | the paper's LaTeX source | none | |
| Figure 2 (borrowing, THINGS-EEG1) | `plot_paper_v3.py`, `fig2_borrow` | 08e, 08m | yes |
| Figure 3 (map minus borrowing, every test) | `plot_paper_v5.py`, `fig_map_strips` | 08n, 09a, 09l, 10c, 11d | yes |
| Figure 4 (fMRI regions and images) | `plot_paper_v3_fmri.py`, `fig_regions` | 10e, 10f (not redistributed) | no: needs 10f |
| Figure 5 (cortex) | `plot_paper_v3_fmri.py`, `fig_cortex` | 10f (not redistributed) | no: needs 10f, nilearn, the fsaverage mesh |
| Figure 6 (label repetitions) | `plot_paper_v5.py`, `fig_label_reps` | 09t | yes |
| Figure 7 (NSD budget curve) | `plot_paper_v5.py`, `fig_nsd_budget` | 10k | yes |
| Figure S1 (NOD) | `plot_paper_v3.py`, `fig_nod` | 11d | yes |
| Figure S2 (what the map is) | `plot_paper_v2.py`, `fig5` | 09b, 09c, 09h | yes |
| Figure S3 (EEG per-person curves) | `plot_paper_v3.py`, `fig_curves` | 16a, 17a, 18j | yes |
| Figure S4 (alignment controls) | `plot_paper_v2.py`, `fig4` | 09i, 09j | yes |
| Figure S5 (NSD decoding examples) | `plot_paper_v5.py`, `fig_examples` | 10g | no: fetches COCO photographs once |
| Figure S6 (EEG decoding) | `plot_paper_v3.py`, `fig_decoding` | 18l, 18m, 18n | yes |
| Table 2 (prespecified tests) | the decision files | 08e, 08h, 08m, 08n, 09a, 09l, 10c, 11d | yes |
| Table S2 (every prespecified contrast) | `scripts/paper_contrast_table.py` | the same eight notebooks | yes |
| Table S3 (EEG by window) | result files | 14c, 14d | yes |
| Table S4 (what the fMRI map draws on: NSD subspace control, Gu's ensemble) | result files | 14a, 14b | yes |
| Table S5 (NSD identification) | result files | 10e (raw columns), 10k (centred column) | yes |
| Table S6 (NSD one-session benchmark) | result files | 12b, 13a, 13b, 14e | yes |
| Table S7 (photograph credits) | `plot_paper_v5.py` writes them to `v5_figures_summary.json` | 10g | no |
| Supplement S2 (margin sensitivity) | `results/analysis_margin_sensitivity.json` | the eight prespecified notebooks | yes |
| K = 200 for the low-rank map (Methods) | result files | 09q | yes |
| EEG headroom relation (Results 3.4) | result files | 09s, 09t | yes |
| Selection check (Limitations) | `scripts/selection_check.py` → `results/selection_check/summary.json` | 18h, 18j | yes |

Tables 1 and S1 describe the datasets and the information each analysis used, and need no code.

**Figure S5.** The examples are drawn from licensed COCO photographs fetched from COCO's own servers (cached under
`COCO_CACHE`, default `~/.cache/xsub_coco`), never from NSD's stimulus files. Notebook 10g chose them and saved the
guesses in `results/10g_nsd_licensed_examples/`; a guess whose licence does not allow reuse is drawn as a grey tile.

**18m's overlap check.** Its file-name check strips the first part of each file name before comparing, so
`results/18m_eeg2_decoding_controls/18m_summary.json` lists shared names such as `01b.jpg` and reports
`no_concept_or_image_overlap: false`. Training and test share no concept (`shared_concepts` is empty), so no test
image is among the training images; the file-name part of that check is not informative.

**Figures 4 and 5.** Both read `10f_vertex.npz` and `10f_images.npz` (per-vertex accuracy and per-image hit rates of
the eight NSD people). They are derived from NSD and are not redistributed here under NSD's terms. Run notebook 10f
with your own NSD access and copy its `results/metrics/` folder to `results/10f_nsd_rich_figures/results/metrics/`.
The cortex figure also needs nilearn and the fsaverage mesh that nilearn downloads; `environment/Dockerfile.cortex`
builds that image.

## 3. Rerun the analyses

Every analysis is a notebook in `notebooks/`. The notebooks are written for Kaggle: most stop at once elsewhere
(their first code cell checks for `/kaggle`) because they read Kaggle's input paths. On Kaggle the first code cell
installs this repository from GitHub at its release tag (Internet on); elsewhere it installs the checkout it sits in.

**Outputs and inputs.** Notebooks 01 to 09o, 16a, 17a, 17b and the 18 series end with a cell
that saves everything they wrote as a private Kaggle dataset in your account, named
`cross-subject-brain-encoding-<id>-outputs`; this needs the Kaggle Secrets `KAGGLE_USERNAME` and `KAGGLE_KEY`. Notebooks 09p to 09t, the 10 to 13 series, and 14a to 14e keep their outputs as the notebook's own Kaggle output
instead (Save Version, then Add Input, Notebook Outputs in a later notebook). Either way, attach the outputs to the
notebooks that need them: the code finds the files it needs in every attached input. The THINGS-EEG2 release saved by
09g must keep its name, `cross-subject-brain-encoding-eeg`, and so must 08g's two datasets,
`cross-subject-brain-encoding-08g-sources` and `cross-subject-brain-encoding-08g-targets`, which 08m finds by name.
Each notebook's first cell names its Kaggle settings (CPU
or GPU) and inputs.

**Data you obtain yourself.** No data are in this repository. The data-preparation notebooks below read each dataset
from its official source (README, section Data); request access where the terms require it.

**NSD.** Notebook 10a reads the Algonauts 2023 release. After you accept NSD's terms at
[algonauts.csail.mit.edu](http://algonauts.csail.mit.edu/), the organisers give you a Google Drive folder link. Put it
in the environment variable `ALGONAUTS_DRIVE_URL`, or on Kaggle in a Secret of that name (Add-ons, Secrets). Do not
commit it. Notebooks 10h, 10i and 12a stream other parts of NSD from its public bucket under the same terms.

### Run order

Each line lists the notebooks that prepare a dataset, then those that use it. A notebook needs the outputs of the
preparation notebooks on its line. GPU notebooks: 03b, 08b, 08e, 08h, 09m, 10b, 10d, 10i, 11b, 13a; all others run on
CPU.

| Dataset | Prepare (in order) | Then |
|---|---|---|
| THINGS-EEG2 | 01 (audit, folds and label budgets), 03b (the published baseline's image features), 09g (the THINGS-EEG2 release as a Kaggle dataset) | 09a (prespecified), 09c and 14d download the release if it is not attached; 09h, 09j, 09m, 09p, 09q, 09r, 09s, 09t, 17b, 18h, 18j, 18l, 18m, 18n, 18o, 18q attach 09g's dataset; 09m (modern features) before 18h, 18j and 18n |
| THINGS-EEG1 | 08a (conversion to THINGS-EEG2's layout), 08b (features, with THINGS-EEG2's 03b); 08g (without noise normalisation) for 08m | 08e, 08h, 08m, 08n (prespecified), 09b, 09i, 09o, 14c, 16a, 17a |
| Alljoined-1.6M | 09k (conversion, with THINGS-EEG2's 09g for the image order) | 09l (prespecified; also attaches 03b and 09g) |
| NSD | 10a (data, needs `ALGONAUTS_DRIVE_URL`), 10b (features); 10d (SigLIP 2 features) for 10e; 12a (one-session data, one notebook per person 1, 2, 5, 7) for 12b, 12c, 13b, 14e | 10c (prespecified), 10e, 10f, 10g, 10h, 10i, 10j, 10k, 12b, 12c, 14a, 14b, 14e |
| NSD, MindEye2 | 13a (MindEye2's released one-session models from Hugging Face, NSD terms) | 13b (with 12a's outputs) |
| NOD | 11a (data from OpenNeuro), 11b (features) | 11c, 11d (prespecified), 11e |

The prespecified tests are 08e, 08h, 08m, 08n, 09a, 09l, 10c and 11d. Their protocols are in `docs/proposals/`, and
`PROVENANCE.md` gives the commits that hold each protocol and the code that scored it. THINGS-EEG1's first test (08e)
and its three follow-ups (08h, 08m, 08n) were scored on the same 24 people, the follow-ups declared after the first
verdict. THINGS-EEG2's ten people (09a) had been used to build the baselines and set the margin; the Alljoined people
(09l) were selected on test-split reliability; no model had been scored on NSD's eight (10c) or NOD's 21 (11d) before their tests.
