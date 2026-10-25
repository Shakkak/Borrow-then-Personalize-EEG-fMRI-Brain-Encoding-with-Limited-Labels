"""The last cells of a notebook: measured values against the expected-results table, and headline numbers.

`check` takes rows of expected results, each with the value this run
measured and the range it should fall in. It
prints one line per row, OK or LOOK AGAIN, so a run can be judged on the
Kaggle page before anything is downloaded, and writes `<NN>_selfcheck.csv`.
A row whose range is None is only reported (a number with no published
anchor yet).

`headline` writes `<NN>_headline.json`: the few numbers that sum the run
up.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def _fmt(v) -> str:
    if v is None:
        return "not measured"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 1000 else f"{v:.0f}"
    return str(v)


def check(nb: str, rows: list[dict], out_dir: Path) -> list[dict]:
    """rows: {"measure", "measured", "expected" (text), "ok" ((lo, hi), a callable, or None), "look_again" (text)}.

    Returns the rows with a "status" added: OK, LOOK AGAIN, NOT MEASURED or REPORTED (no range).
    """
    done = []
    for r in rows:
        v, ok = r.get("measured"), r.get("ok")
        if v is None or (isinstance(v, float) and math.isnan(v)):
            status = "NOT MEASURED"
        elif ok is None:
            status = "REPORTED"
        elif callable(ok):
            status = "OK" if ok(v) else "LOOK AGAIN"
        else:
            lo, hi = ok
            status = "OK" if (lo is None or v >= lo) and (hi is None or v <= hi) else "LOOK AGAIN"
        done.append({**{k: r.get(k, "") for k in ("measure", "expected", "look_again")},
                     "measured": _fmt(v), "status": status})
    width = max(len(r["measure"]) for r in done)
    print(f"Self-check of notebook {nb} against its expected results:\n")
    for r in done:
        print(f"  {r['status']:<12} {r['measure']:<{width}}  measured {r['measured']}  (expected {r['expected']})")
        if r["status"] == "LOOK AGAIN":
            print(f"  {'':<12} {'':<{width}}  -> {r['look_again']}")
    counts = {s: sum(r["status"] == s for r in done) for s in ("OK", "LOOK AGAIN", "REPORTED", "NOT MEASURED")}
    print("\n" + ", ".join(f"{n} {s}" for s, n in counts.items() if n))
    with open(Path(out_dir) / f"{nb}_selfcheck.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["measure", "measured", "status", "expected", "look_again"])
        w.writeheader()
        w.writerows(done)
    return done


def headline(nb: str, values: dict, out_dir: Path) -> Path:
    """Write {name: number or short text} as <nb>_headline.json; floats rounded to 4 decimals."""
    clean = {k: round(v, 4) if isinstance(v, float) else v for k, v in values.items()}
    path = Path(out_dir) / f"{nb}_headline.json"
    path.write_text(json.dumps(clean, indent=1))
    print(json.dumps(clean, indent=1))
    return path
