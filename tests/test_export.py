"""The export refuses a failed upload even when the Kaggle CLI exits 0, and confirms a good one by its stamp."""
import json
import subprocess

import pytest

from cross_subject_encoding.utils import kaggle_export as ke


def fake_cli(version_reply, live_stamp, status_first="ready", calls=None, old_downloads=0):
    seen = {"status": 0, "download": 0}

    def run(*args):
        cmd = args[:2]
        if calls is not None:
            calls.append(cmd[1])
        if cmd == ("datasets", "create"):  # what Kaggle said when the search missed the existing dataset
            return subprocess.CompletedProcess(args, 0, 'Dataset creation error: The requested title "t" is already in use by a dataset.', "")
        if cmd == ("datasets", "version"):
            return subprocess.CompletedProcess(args, 0, version_reply, "")
        if cmd == ("datasets", "status"):
            seen["status"] += 1
            return subprocess.CompletedProcess(args, 0, status_first if seen["status"] == 1 else "ready", "")
        if cmd == ("datasets", "download"):
            d = args[args.index("-p") + 1]
            seen["download"] += 1
            live = live_stamp and seen["download"] > old_downloads
            stage_stamp = json.loads(next(ke.STAGING_ROOT.rglob(ke.STAMP)).read_text())["stamp"]
            (ke.Path(d) / ke.STAMP).write_text(json.dumps({"stamp": stage_stamp if live else "old"}))
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(args)
    return run


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "me")
    monkeypatch.setenv("KAGGLE_KEY", "k")
    monkeypatch.setattr(ke, "STAGING_ROOT", tmp_path / "stage")
    monkeypatch.setattr(ke.time, "sleep", lambda s: None)
    f = tmp_path / "a.csv"
    f.write_text("x\n1\n")
    return f, monkeypatch


def test_error_text_and_an_old_stamp_is_a_failure(setup):
    f, mp = setup
    mp.setattr(ke, "_kaggle", fake_cli("Dataset version creation error: something", False))
    with pytest.raises(RuntimeError):
        ke.export_to_dataset("x-02-outputs", "XSUB test outputs", [f])


def test_error_text_but_the_new_stamp_is_live_stands(setup, capsys):
    """The CLI's output can trip the check while the files and the new stamp are already in the dataset."""
    f, mp = setup
    mp.setattr(ke, "_kaggle", fake_cli("100%|##########| 52.4M/52.4M\nsome error the upload survived", True))
    ke.export_to_dataset("x-02-outputs", "XSUB test outputs", [f])
    out = capsys.readouterr().out
    assert "looks wrong: some error the upload survived" in out and "the export stands" in out


def test_old_stamp_at_first_is_waited_out(setup, capsys):
    """Right after the version upload the dataset can read "ready" with the old stamp; that is waited out."""
    f, mp = setup
    mp.setattr(ke, "_kaggle", fake_cli("Dataset version is being created.", True, old_downloads=3))
    ke.export_to_dataset("x-02-outputs", "XSUB test outputs", [f])
    assert "confirmed: the dataset now holds" in capsys.readouterr().out


def test_old_stamp_without_cli_error_warns_and_goes_on(setup, capsys):
    """No error from the CLI and no proof either: a milestone export must not end a long run."""
    f, mp = setup
    mp.setattr(ke, "_kaggle", fake_cli("Dataset version is being created.", False))
    assert ke.export_to_dataset("x-02-outputs", "XSUB test outputs", [f]).endswith("me/x-02-outputs")
    out = capsys.readouterr().out
    assert "Could not confirm" in out and "NOT confirmed" in out


def test_good_upload_is_confirmed(setup, capsys):
    f, mp = setup
    mp.setattr(ke, "_kaggle", fake_cli("Dataset version is being created.", True))
    assert ke.export_to_dataset("x-02-outputs", "XSUB test outputs", [f]).endswith("me/x-02-outputs")
    assert "confirmed" in capsys.readouterr().out


def test_existing_dataset_that_the_check_misses_still_gets_a_version(setup):
    f, mp = setup
    calls = []
    mp.setattr(ke, "_kaggle", fake_cli("Dataset version is being created.", True, status_first="404 - Not Found", calls=calls))
    ke.export_to_dataset("x-02-outputs", "XSUB test outputs", [f])
    assert calls[:4] == ["status", "create", "status", "version"]  # it waits for "ready" before the version


def test_two_exports_in_one_run_create_once_then_add_a_version_after_ready(setup):
    """A milestone export and the final one in the same run; Kaggle lags after creating the dataset."""
    f, mp = setup
    state = {"exists": False, "processing": 0, "calls": []}

    def run(*args):
        cmd = args[:2]
        state["calls"].append(cmd[1])
        if cmd == ("datasets", "status"):
            if not state["exists"]:
                return subprocess.CompletedProcess(args, 1, "", "404 Not Found")
            if state["processing"] > 0:
                state["processing"] -= 1
                return subprocess.CompletedProcess(args, 0, "pending", "")
            return subprocess.CompletedProcess(args, 0, "ready", "")
        if cmd in (("datasets", "create"), ("datasets", "version")):
            state["exists"], state["processing"] = True, 2
            return subprocess.CompletedProcess(args, 0, "Your dataset is being created" if cmd[1] == "create" else "version created", "")
        if cmd == ("datasets", "download"):
            d = args[args.index("-p") + 1]
            stage_stamp = json.loads(next(ke.STAGING_ROOT.rglob(ke.STAMP)).read_text())["stamp"]
            (ke.Path(d) / ke.STAMP).write_text(json.dumps({"stamp": stage_stamp}))
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(args)

    mp.setattr(ke, "_kaggle", run)
    ke.export_to_dataset("x-04-outputs", "XSUB test outputs", [f], message="milestone")
    state["processing"] = 3  # the first version is still processing when the final export starts
    ke.export_to_dataset("x-04-outputs", "XSUB test outputs", [f], message="final")
    assert state["calls"].count("create") == 1 and state["calls"].count("version") == 1
    v = state["calls"].index("version")
    assert state["calls"][v - 1] == "status"  # it waited for "ready" right before the new version
