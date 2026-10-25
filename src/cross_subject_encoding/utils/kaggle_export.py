"""Publish a notebook's weights and outputs as a private Kaggle Dataset.

Kaggle already keeps each committed notebook's /kaggle/working as its
output, but that output belongs to one notebook version. A Dataset has its
own version history, can be attached to any notebook, and survives deleting
or rerunning the notebook that made it.

Needs Internet on and a Kaggle API token stored as two Kaggle Secrets
(Add-ons -> Secrets): ``KAGGLE_USERNAME`` and ``KAGGLE_KEY``, from
kaggle.com -> Settings -> API -> Create New Token. Without them the export
is skipped with a message, so a missing token never fails a committed run.
"""
from __future__ import annotations

import csv
import io
import json
import re
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

STAGING_ROOT = Path("/tmp/kaggle-export")


def _load_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    try:
        from kaggle_secrets import UserSecretsClient

        secrets = UserSecretsClient()
        os.environ["KAGGLE_USERNAME"] = secrets.get_secret("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = secrets.get_secret("KAGGLE_KEY")
        return True
    except Exception as exc:  # not on Kaggle, secret missing, or not attached to this notebook
        print(f"Kaggle API credentials not available ({type(exc).__name__}: {exc})")
        return False


def _kaggle(*args: str) -> subprocess.CompletedProcess:
    if shutil.which("kaggle") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
    return subprocess.run(["kaggle", *args], capture_output=True, text=True)


def _dataset_exists(ref: str) -> bool:
    """Ask for the dataset itself. A search of `datasets list --mine` missed an existing private dataset, so the
    export tried to create it again, which Kaggle refused with exit code 0."""
    out = _kaggle("datasets", "status", ref)
    text = (out.stdout + out.stderr).lower()
    return out.returncode == 0 and bool(text.strip()) and not any(w in text for w in ERROR_WORDS)


STAMP = "export_stamp.json"  # uploaded with every export, then read back to prove the new version is live
ERROR_WORDS = ("error", "failed", "invalid", "not found", "forbidden", "unauthorized", "too large")


def _check_cli(out: subprocess.CompletedProcess, what: str) -> list[str]:
    """What looks wrong in a CLI call: its exit code and every printed line with an error word.

    The CLI can exit 0 after a failed upload and print the error instead, so its text is read.
    It can also look wrong after an upload that worked, so this only reports; the caller decides after reading the stamp back from the dataset.
    """
    text = (out.stdout + "\n" + out.stderr).strip()
    lines = [ln.strip() for ln in re.split(r"[\r\n]+", text) if ln.strip()]
    print(f"kaggle {what}: exit code {out.returncode}; last line: {lines[-1] if lines else '(no output)'}")
    bad = [ln for ln in lines if any(w in ln.lower() for w in ERROR_WORDS)]
    if out.returncode != 0:
        bad.insert(0, f"exit code {out.returncode}")
    for ln in bad:
        print(f"  looks wrong: {ln[:300]}")
    return bad


def _wait_ready(ref: str, timeout_s: int = 1800) -> bool:
    """Wait until the dataset has finished processing its last version.

    A run can export several times (milestones, then the end). A new version uploaded while the previous
    one is still processing can be refused or interleave with it, so every export waits here first.
    """
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if "ready" in _kaggle("datasets", "status", ref).stdout.strip().lower():
            return True
        time.sleep(30)
    print(f"{ref} was still processing after {timeout_s // 60} minutes; uploading anyway")
    return False


def _confirm_live(ref: str, stamp: str, timeout_s: int = 1200) -> bool:
    """Download the stamp until it is this run's, polling every 30 s; False if it never is.

    Right after ``datasets version`` the status can still read "ready" for the previous version, so an
    old stamp at first is normal and is waited out, not treated as a failure. Only the caller raises, and only when the CLI
    also printed an error.
    """
    for _ in range(max(1, timeout_s // 30)):
        if "ready" in _kaggle("datasets", "status", ref).stdout.strip().lower():
            with tempfile.TemporaryDirectory() as d:
                _kaggle("datasets", "download", ref, "-f", STAMP, "-p", d, "--force")
                got = next(Path(d).rglob(STAMP + "*"), None)
                if got is not None and got.suffix == ".zip":
                    import zipfile
                    with zipfile.ZipFile(got) as z:
                        z.extractall(d)
                    got = Path(d) / STAMP
                if got is not None and got.exists() and json.loads(got.read_text()).get("stamp") == stamp:
                    return True
        time.sleep(30)
    print(f"Could not confirm within {timeout_s // 60} minutes that {ref} holds this run's files; "
          "check its page before attaching it to the next notebook.")
    return False


SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints", ".virtual_documents"}
SKIP_NAMES = {".gitkeep"}  # placeholders that keep empty output folders in git


def collect_files(dirs: list[Path]) -> list[Path]:
    """Every file under ``dirs`` (recursively), skipping editor and cache folders."""
    out = []
    for d in map(Path, dirs):
        if d.is_dir():
            out += [f for f in sorted(d.rglob("*")) if f.is_file() and f.name not in SKIP_NAMES and not SKIP_PARTS & set(f.parts)]
    return out


def check_title(title: str) -> None:
    """Kaggle refuses dataset titles outside 6-50 characters ("The dataset title must be between 6 and 50 characters");
    fail before staging or uploading anything."""
    if not 6 <= len(title) <= 50:
        raise ValueError(f"Kaggle dataset titles must be 6-50 characters; this one has {len(title)}: {title!r}")


def export_to_dataset(
    slug: str,
    title: str,
    files: list[Path],
    include_dirs: list[Path] = (),
    message: str = "notebook run",
) -> str | None:
    """Create the private Dataset ``<user>/<slug>``, or add a new version.

    ``files`` are the notebook's required outputs and must all exist (a
    missing one means the notebook did not finish, so nothing is uploaded).
    Every other file found under ``include_dirs`` is uploaded too, so
    optional outputs and anything added later are never left out. Files are
    uploaded flat, by file name, which is how ``find_artifact`` looks them
    up. Returns the Dataset URL, or None if the export was skipped.

    Called more than once in a run (milestone exports): each call waits until the dataset has
    finished processing the previous version, and each version is a full snapshot, so every call should
    pass the same ``include_dirs`` and the latest version always holds everything written so far.
    """
    check_title(title)
    files = [Path(f) for f in files]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"not exported, required outputs missing: {missing}")
    seen = {f.resolve() for f in files}
    for f in collect_files(include_dirs):
        if f.resolve() not in seen:
            seen.add(f.resolve())
            files.append(f)
    names = [f.name for f in files]
    if len(set(names)) != len(names):
        raise ValueError(f"two outputs share a file name: {sorted(n for n in names if names.count(n) > 1)}")

    if not _load_credentials():
        print("Export skipped. To enable it: Add-ons -> Secrets -> add KAGGLE_USERNAME and KAGGLE_KEY, "
              "tick both for this notebook, turn Internet on, and rerun this cell.")
        return None

    ref = f"{os.environ['KAGGLE_USERNAME']}/{slug}"
    stage = STAGING_ROOT / slug
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    for f in files:
        shutil.copy2(f, stage / f.name)
    stamp = f"{time.strftime('%Y-%m-%dT%H:%M:%S')}-{uuid.uuid4().hex[:8]}"
    (stage / STAMP).write_text(json.dumps({"stamp": stamp, "files": sorted(names)}))
    meta = {"title": title, "id": ref, "licenses": [{"name": "unknown"}]}
    (stage / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))

    if _dataset_exists(ref):
        _wait_ready(ref)
        bad = _check_cli(_kaggle("datasets", "version", "-p", str(stage), "-m", message), "datasets version")
    else:
        out = _kaggle("datasets", "create", "-p", str(stage))  # private unless --public is passed
        if "already in use" in (out.stdout + out.stderr).lower():  # it exists after all: add a version instead
            print(f"{ref} already exists; adding a version")
            _wait_ready(ref)
            out = _kaggle("datasets", "version", "-p", str(stage), "-m", message)
            bad = _check_cli(out, "datasets version")
        else:
            bad = _check_cli(out, "datasets create")
    confirmed = _confirm_live(ref, stamp)  # the stamp read back from the dataset decides, not the CLI's words
    if bad and not confirmed:
        raise RuntimeError("Kaggle upload failed and the dataset does not hold this run's files: " + " | ".join(b[:300] for b in bad))
    if bad:
        print("the CLI printed the lines above, but the dataset holds this run's files (stamp read back), so the export stands")

    size_mb = sum(f.stat().st_size for f in files) / 1e6
    url = f"https://www.kaggle.com/datasets/{ref}"
    print(f"{len(files)} files ({size_mb:.1f} MB) -> {url} (private; "
          + ("confirmed: the dataset now holds this run's files, stamp " + stamp if confirmed else "NOT confirmed") + ")")
    for n in sorted(names):
        print("  ", n)
    return url


def export_folder(slug: str, title: str, folder: Path, message: str = "dataset build") -> str | None:
    """Upload every file in ``folder`` (flat, no subfolders) as the private Dataset ``<user>/<slug>``, in place:
    nothing is copied, so a multi-GB dataset does not need twice its size on disk (09g). The stamp and
    metadata files are written into the folder, and the stamp read back decides success, as in
    ``export_to_dataset``."""
    check_title(title)
    folder = Path(folder)
    if any(p.is_dir() for p in folder.iterdir()):
        raise ValueError(f"{folder} has subfolders; the Kaggle CLI would skip them")
    if not _load_credentials():
        print("Export skipped: add KAGGLE_USERNAME and KAGGLE_KEY as Secrets, tick both, turn Internet on.")
        return None
    ref = f"{os.environ['KAGGLE_USERNAME']}/{slug}"
    names = sorted(p.name for p in folder.iterdir() if p.name not in (STAMP, "dataset-metadata.json"))
    stamp = f"{time.strftime('%Y-%m-%dT%H:%M:%S')}-{uuid.uuid4().hex[:8]}"
    (folder / STAMP).write_text(json.dumps({"stamp": stamp, "files": names}))
    (folder / "dataset-metadata.json").write_text(json.dumps({"title": title, "id": ref, "licenses": [{"name": "unknown"}]}, indent=1))
    if _dataset_exists(ref):
        _wait_ready(ref)
        bad = _check_cli(_kaggle("datasets", "version", "-p", str(folder), "-m", message), "datasets version")
    else:
        bad = _check_cli(_kaggle("datasets", "create", "-p", str(folder)), "datasets create")
        if bad and not _dataset_exists(ref):  # creation refused outright: nothing to wait for
            raise RuntimeError("Kaggle refused to create the dataset: " + " | ".join(b[:300] for b in bad))
    confirmed = _confirm_live(ref, stamp, timeout_s=3600)
    if bad and not confirmed:
        raise RuntimeError("Kaggle upload failed and the dataset does not hold this run's files: " + " | ".join(b[:300] for b in bad))
    size_gb = sum((folder / n).stat().st_size for n in names) / 1e9
    url = f"https://www.kaggle.com/datasets/{ref}"
    print(f"{len(names)} files ({size_gb:.2f} GB) -> {url} (private; " + ("confirmed, stamp " + stamp if confirmed else "NOT confirmed") + ")")
    return url
