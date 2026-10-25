"""Tiny actual-directory/shared-device storage checks, no data downloads."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("storage", Path(__file__).resolve().parents[1] / "src/cross_subject_encoding/utils/storage.py")
storage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage)


class StorageTests(unittest.TestCase):
    def test_actual_destinations_created_and_shared_budget_summed(self):
        with tempfile.TemporaryDirectory() as directory:
            scratch, output = Path(directory) / "scratch", Path(directory) / "output"
            with patch.object(storage.shutil, "disk_usage", return_value=SimpleNamespace(free=100)) as usage:
                plan = storage.storage_plan({"scratch": (scratch, 70), "output": (output, 40)})
            self.assertTrue(scratch.is_dir() and output.is_dir())
            self.assertEqual([call.args[0] for call in usage.call_args_list], [scratch.resolve(), output.resolve()])
            self.assertEqual(len(plan["devices"]), 1)
            self.assertEqual(next(iter(plan["devices"].values()))["required_bytes"], 110)
            self.assertFalse(plan["sufficient"])

    def test_device_budget_uses_smallest_reported_free_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(storage.shutil, "disk_usage", side_effect=[SimpleNamespace(free=200), SimpleNamespace(free=100)]):
                plan = storage.storage_plan({"first": (root, 70), "second": (root, 40)})
            self.assertFalse(plan["sufficient"])

    def test_separate_devices_are_not_double_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / name for name in ("scratch", "output")]
            for root in roots:
                root.mkdir()
            original = Path.stat

            def stat(path, *args, **kwargs):
                if path in roots:
                    return SimpleNamespace(st_dev=roots.index(path) + 1)
                return original(path, *args, **kwargs)

            with patch.object(Path, "mkdir"), patch.object(Path, "resolve", lambda path: path), patch.object(Path, "stat", stat), patch.object(storage.shutil, "disk_usage", return_value=SimpleNamespace(free=100)):
                plan = storage.storage_plan({"scratch": (roots[0], 70), "output": (roots[1], 40)})
            self.assertEqual(len(plan["devices"]), 2)
            self.assertTrue(plan["sufficient"])

    def test_exact_capacity_and_invalid_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(storage.shutil, "disk_usage", return_value=SimpleNamespace(free=100)):
                self.assertTrue(storage.storage_plan({"scratch": (root, 100)})["sufficient"])
            for budget in (-1, 1.5):
                with self.assertRaises(ValueError):
                    storage.storage_plan({"scratch": (root, budget)})


if __name__ == "__main__":
    unittest.main()
