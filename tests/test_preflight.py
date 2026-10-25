"""Preflight rejects missing inputs and inaccessible output destinations."""
import subprocess

import pytest

from cross_subject_encoding.utils import kaggle_export as ke
from cross_subject_encoding.utils import preflight as pf


def _find(present):
    def f(name):
        if name not in present:
            raise FileNotFoundError(name)
        return present[name]
    return f


def test_everything_present_passes(tmp_path, monkeypatch):
    a = tmp_path / "a.npz"
    a.write_bytes(b"x" * 10)
    monkeypatch.setattr(ke, "_load_credentials", lambda: True)
    monkeypatch.setattr(ke, "_kaggle", lambda *a: subprocess.CompletedProcess(a, 0, "ref title size", ""))
    info = pf.preflight("x-04-outputs", inputs=["a.npz"], expect_output_gb=0.0, work_dir=tmp_path, find=_find({"a.npz": a}))
    assert "a.npz" in info["inputs"]


def test_every_problem_is_listed_together(tmp_path, monkeypatch):
    monkeypatch.setattr(ke, "_load_credentials", lambda: False)
    monkeypatch.setattr(pf, "IS_KAGGLE", True)
    with pytest.raises(RuntimeError) as e:
        pf.preflight("x-04-outputs", inputs=["a.npz", "b.json"], expect_output_gb=1e6, work_dir=tmp_path, find=_find({}))
    msg = str(e.value)
    assert "Secrets" in msg and "a.npz" in msg and "b.json" in msg and "GB free" in msg


def test_a_key_that_does_not_work_stops_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(ke, "_load_credentials", lambda: True)
    monkeypatch.setattr(ke, "_kaggle", lambda *a: subprocess.CompletedProcess(a, 1, "", "401 - Unauthorized"))
    with pytest.raises(RuntimeError, match="API key"):
        pf.preflight("x-04-outputs", work_dir=tmp_path, find=_find({}))
