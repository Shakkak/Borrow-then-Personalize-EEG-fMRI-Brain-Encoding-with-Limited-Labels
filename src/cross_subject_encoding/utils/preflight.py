"""Checks notebook inputs and output access before computation.

A run that finds out at its end that the export credentials are missing, an input is not attached or
the disk is too small for its checkpoints has spent its compute for nothing. `preflight` checks all of
that first and stops with one message listing every problem.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from cross_subject_encoding.utils import kaggle_export as ke

IS_KAGGLE = os.path.exists("/kaggle")


def preflight(slug: str, inputs=(), need_gpu: bool = False, min_gpu_gb: float = 0.0, expect_output_gb: float = 0.0,
              work_dir: Path | None = None, find=None) -> dict:
    """Stop now if the run could not finish and keep its results.

    slug: the Kaggle dataset the export will write; inputs: file names earlier notebooks wrote (found with
    `config.find_artifact`); expect_output_gb: outputs plus checkpoints this run will write.
    """
    from cross_subject_encoding.utils import config as C

    find = find or C.find_artifact
    problems, info = [], {"slug": slug}
    if ke._load_credentials():
        out = ke._kaggle("datasets", "list", "--mine")
        bad = [w for w in ("unauthorized", "forbidden", "401", "403") if w in (out.stdout + out.stderr).lower()]
        if out.returncode != 0 or bad:
            problems.append(f"the Kaggle API key does not work (exit {out.returncode}{', ' + bad[0] if bad else ''}): the export would fail")
        info["export"] = "credentials work"
        ref = f"{os.environ.get('KAGGLE_USERNAME', 'me')}/{slug}"
        status = ke._kaggle("datasets", "status", ref)
        st = (status.stdout + status.stderr).strip().lower()
        if status.returncode == 0 and st and "ready" not in st and not any(w in st for w in ke.ERROR_WORDS):
            print(f"preflight | WARNING: {ref} is still processing ({st[:80]}): if another session of this notebook "
                  "is exporting, stop one of them; two runs must never write to the same dataset at once")
        info["dataset"] = "exists" if status.returncode == 0 and "ready" in st else "will be created or is processing"
    elif IS_KAGGLE:
        problems.append("Kaggle Secrets KAGGLE_USERNAME and KAGGLE_KEY are not attached: the results could not be exported")
    else:
        info["export"] = "off Kaggle: export will be skipped"
    found = {}
    for name in inputs:
        try:
            p = Path(find(name))
            found[name] = f"{p} ({p.stat().st_size / 2**20:.1f} MB)"
        except Exception as e:  # noqa: BLE001 - every missing input is reported together
            problems.append(f"input {name} not found ({type(e).__name__}): attach the dataset that holds it")
    info["inputs"] = found
    wd = Path(work_dir or C.PROJECT_ROOT)
    free_gb = shutil.disk_usage(wd if wd.exists() else wd.anchor).free / 2**30
    info["free disk GB"] = round(free_gb, 1)
    if free_gb < 1.2 * expect_output_gb:
        problems.append(f"{free_gb:.1f} GB free where outputs and checkpoints need about {expect_output_gb:.1f} GB")
    if need_gpu:
        try:
            import torch
            ok = torch.cuda.is_available()
            mem = torch.cuda.get_device_properties(0).total_memory / 2**30 if ok else 0.0
            info["gpu"] = f"{torch.cuda.get_device_name(0)}, {mem:.1f} GB" if ok else "none"
            if not ok:
                problems.append("no GPU: set the accelerator to GPU before running")
            elif mem < min_gpu_gb:
                problems.append(f"the GPU has {mem:.1f} GB, this run needs about {min_gpu_gb:.1f} GB")
        except ImportError:
            problems.append("torch is not installed")
    for k, v in info.items():
        print(f"preflight | {k}: {v}")
    if problems:
        raise RuntimeError("preflight stopped the run before it spent any compute:\n- " + "\n- ".join(problems))
    print("preflight | all checks passed")
    return info
