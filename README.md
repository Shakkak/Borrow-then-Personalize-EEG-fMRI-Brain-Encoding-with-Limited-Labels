# Borrow, then personalize

Code, prespecified protocols and result tables for the paper *Borrow, then personalize: calibrating visual EEG and
fMRI encoding models with limited labels* (Shakib Hosseinzadeh, 2026).

A new person's visual brain responses are predicted by averaging encoders trained on other people (*borrowing*);
a small linear map, learned from a few hundred of the new person's labelled images, then corrects that average
across EEG sensors or fMRI vertices. The main comparisons were written into version-controlled protocols before
scoring, each scored once: lab EEG (THINGS-EEG1, THINGS-EEG2), consumer EEG (Alljoined-1.6M), 7-tesla fMRI (NSD) and
single-trial 3-tesla fMRI (NOD). Not every test used people new to the analysis. THINGS-EEG1's first test and its
three follow-ups, declared after the first verdict, were scored on the same 24 people. THINGS-EEG2's ten people had
been used to build the baselines and set the margin, and the Alljoined people were selected on test-split
reliability. No model had been scored on NSD's and NOD's people before their tests. Everything else in the paper is development work.

## Start here

- **Run the tests:** install CPU torch (`pip install torch --index-url https://download.pytorch.org/whl/cpu`), then
  `pip install -e ".[test]" && python -m pytest -q`.
- **Redraw the paper's figures and tables** from the result files in `results/`: see
  [`docs/REPRODUCE.md`](docs/REPRODUCE.md). Most need no internet and no data.
- **Rerun an analysis:** every analysis is a notebook in `notebooks/`; `docs/REPRODUCE.md` gives the order and the
  data each one needs.
- **Check the prespecified tests:** the protocols in `docs/proposals/` and the commit order in `PROVENANCE.md`.

## What is here

| Path | Contents |
|---|---|
| `src/cross_subject_encoding/` | The package (installed as `cross-subject-encoding`): data readers, features, ridge and least-squares encoders, the personal maps, scoring and the prespecified decision rules. |
| `notebooks/` | The analysis notebooks behind the paper's results, with their data preparation (no outputs). |
| `results/<notebook>/` | Result tables, summaries and decision files of each notebook the paper reports. |
| `scripts/` | The paper's figures (`plot_paper_v2.py`, `plot_paper_v3.py`, `plot_paper_v3_fmri.py`, `plot_paper_v5.py`, with helpers in `plot_paper_figures.py` and `fingerprint_maps.py`), Table S2 (`paper_contrast_table.py`) and the selection check in the Limitations (`selection_check.py`). |
| `docs/proposals/` | The protocols of the prespecified tests, the THINGS-EEG1 follow-up family, and the THINGS-EEG1 people assignment. |
| `docs/REPRODUCE.md` | Figures and tables to scripts and notebooks; the order in which to rerun the notebooks. |
| `PROVENANCE.md` | For each prespecified test: the commit that holds its protocol and the commit of the code that scored it. |
| `environment/` | The exact package versions the tests passed with, and Dockerfiles for the CPU and cortex-plotting images. |
| `tests/` | Unit and small integration tests on synthetic data. |

## Install and test

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # several tests import torch
pip install -e ".[test]"          # numpy, pandas, scipy, scikit-learn, matplotlib, pytest
pip install -e ".[cortex]"        # nilearn and nibabel, for the two cortical figures
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu   # full environment for feature extraction
python -m pytest -q
```

`environment/requirements-tested.txt` lists the exact versions the tests passed with (Python 3.12, CPU torch), and
`environment/Dockerfile` builds that environment.

## Reproducing the results

The analyses are notebooks written for Kaggle CPU and GPU sessions, and most read Kaggle's input paths. On Kaggle
each notebook installs this repository at its release tag and prints the code commit it runs. `docs/REPRODUCE.md`
says which notebooks save a Kaggle dataset and which keep notebook outputs, what each one needs, and the order. The
prespecified tests are 08e, 08h, 08m and 08n (THINGS-EEG1), 09a (THINGS-EEG2), 09l (Alljoined-1.6M), 10c (NSD) and
11d (NOD).

Most figures need only the files in `results/`. Two exceptions:

- The two cortical figures need per-vertex NSD summaries that are not redistributed under NSD's terms; rerun notebook
  10f with your own NSD access.
- The NSD decoding examples fetch licensed COCO photographs from COCO's servers once; pass `--skip-examples` to work
  offline.

## Data

No raw datasets are included; `results/` holds only small result tables and summaries. Each dataset is public under
its own terms:

- THINGS-EEG1: [OpenNeuro ds003825](https://openneuro.org/datasets/ds003825)
- THINGS-EEG2: [OSF 3jk45](https://osf.io/3jk45/)
- Alljoined-1.6M: [Hugging Face](https://huggingface.co/datasets/Alljoined/Alljoined-1.6M) (gated, non-commercial)
- NSD: [naturalscenesdataset.org](https://naturalscenesdataset.org/) (data use agreement), used here through the
  [Algonauts 2023 challenge](http://algonauts.csail.mit.edu/challenge.html) release. After you accept the terms, the
  organisers send a Google Drive link; notebook 10a reads it from the `ALGONAUTS_DRIVE_URL` environment variable or
  Kaggle Secret. Do not commit it.
- NOD: [OpenNeuro ds004496](https://openneuro.org/datasets/ds004496)
- MindEye2 one-session models (notebook 13a): [MedARC-AI/MindEyeV2](https://github.com/MedARC-AI/MindEyeV2)

Collection of the NSD dataset was supported by NSF IIS-1822683 and NSF IIS-1822929.

## Citation

See `CITATION.cff`. Please also cite the datasets you use.

## Licence

MIT (see `LICENSE`). The datasets keep their own licences.
