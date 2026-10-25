"""Inspect actual storage destinations and aggregate budgets on shared devices."""
from __future__ import annotations

import shutil
from pathlib import Path


def storage_plan(destinations: dict[str, tuple[Path, int]]) -> dict:
    locations, devices = {}, {}
    for name, (directory, required_bytes) in destinations.items():
        if not isinstance(required_bytes, int) or required_bytes < 0:
            raise ValueError("Storage budgets must be nonnegative integer bytes")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        directory = directory.resolve()
        device = str(directory.stat().st_dev)
        free = shutil.disk_usage(directory).free
        locations[name] = {"path": str(directory), "device": device,
                           "free_bytes": free, "required_bytes": required_bytes}
        group = devices.setdefault(device, {"destinations": [], "required_bytes": 0, "free_bytes": free})
        group["destinations"].append(name)
        group["required_bytes"] += required_bytes
        group["free_bytes"] = min(group["free_bytes"], free)
    for group in devices.values():
        group["sufficient"] = group["free_bytes"] >= group["required_bytes"]
    return {"locations": locations, "devices": devices,
            "sufficient": all(group["sufficient"] for group in devices.values()),
            "note": "Conservative simultaneous budgets; free space is not a guarantee against quota changes or later exhaustion."}
