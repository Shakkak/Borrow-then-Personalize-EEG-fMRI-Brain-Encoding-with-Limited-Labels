"""Score-independent access and scheduling guards for the EEG1 people assignment and its pilot."""
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path


def validate_assignment(manifest: dict) -> tuple[list[str], list[str], list[str]]:
    if manifest.get("assignment_approved") is not True:
        raise ValueError("assignment_approved must be explicitly true before preparing source-only inputs")
    if manifest.get("protocol") != "N188-source-pilot-draft-v1":
        raise ValueError("Wrong pilot protocol version")
    if not isinstance(manifest.get("approval_reference"), str) or not manifest["approval_reference"].strip():
        raise ValueError("The manifest needs a non-empty approval_reference")
    sources, targets, pseudo = [manifest.get(field) for field in (
        "primary_sources", "confirmation_targets", "pseudo_targets")]
    for people, count in ((sources, 22), (targets, 24), (pseudo, 4)):
        if not isinstance(people, list) or len(people) != count:
            raise ValueError(f"Expected {count} explicitly assigned participants")
        if any(not isinstance(person, str) for person in people) or len(set(people)) != count:
            raise ValueError("Participant IDs must be unique strings")
    eligible = {f"sub-{number:02d}" for number in range(1, 51)} - {"sub-01", "sub-06", "sub-18", "sub-23"}
    if set(sources) & set(targets) or set(sources) | set(targets) != eligible:
        raise ValueError("The assignment requires disjoint roles covering the 46 strict candidates")
    if not set(pseudo) <= set(sources):
        raise ValueError("Pseudo-targets must come exclusively from the source pool")
    return [person for person in sources if person not in pseudo], pseudo, targets


def validate_manifest(manifest: dict) -> tuple[list[str], list[str], list[str]]:
    roles = validate_assignment(manifest)
    for field in ("channel_mapping_checked", "pilot_spending_approved"):
        if manifest.get(field) is not True:
            raise ValueError(f"{field} must be explicitly true before the pilot")
    return roles


def source_file(root: Path, person: str, part: str, allowed: list[str]) -> Path:
    if person not in allowed or part not in ("training", "test"):
        raise ValueError("EEG access outside the explicit pilot allowlist")
    hits = list(root.rglob(f"{person}_preprocessed_eeg_{part}.npy"))
    if len(hits) != 1:
        raise ValueError(f"Expected one unambiguous {person}/{part} file, found {len(hits)}")
    return hits[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


class PilotClock:
    """Cooperative phase/epoch guard, not a preemptive wall-clock guarantee."""

    def __init__(self, minutes: float = 30, reserve_minutes: float = 2, now=time.monotonic):
        if not all(math.isfinite(value) for value in (minutes, reserve_minutes)) or not 0 < reserve_minutes < minutes:
            raise ValueError("Require finite positive budget and a smaller export reserve")
        self.now, self.start = now, now()
        self.limit, self.reserve = minutes * 60, reserve_minutes * 60

    def elapsed(self) -> float:
        return self.now() - self.start

    def require(self, phase: str, projected_seconds: float = 0) -> None:
        if not math.isfinite(projected_seconds) or projected_seconds < 0:
            raise ValueError("Invalid phase projection")
        if self.elapsed() + projected_seconds >= self.limit - self.reserve:
            raise TimeoutError(f"Incomplete pilot: no safe time for {phase}")
