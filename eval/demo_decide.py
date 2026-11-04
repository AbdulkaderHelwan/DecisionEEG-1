#!/usr/bin/env python3
"""Run DecisionEEG-1 on real EDF files and print a compact summary (full JSON is saved).

    python eval/demo_decide.py file1.edf [file2.edf ...] [--max-seconds 600]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decisioneeg.model import DecisionEEG     # noqa: E402
from decisioneeg.paths import RESULTS          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edf", nargs="+")
    ap.add_argument("--max-seconds", type=float, default=600)
    ap.add_argument("--tasks", default=None)
    a = ap.parse_args()
    m = DecisionEEG.load()
    out_dir = RESULTS / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in a.edf:
        r = m.decide_edf(f, a.tasks.split(",") if a.tasks else None, a.max_seconds)
        name = Path(f).stem.replace(" ", "_")
        (out_dir / f"{name}.json").write_text(json.dumps(r, indent=1))
        print(f"\n=== {f}  ({r['input']['duration_s']:.0f} s, {len(r['input'].get('channels_used', []))} channels used, "
              f"{r['seconds']} s)")
        for w in r["warnings"]:
            print(f"  warning: {w}")
        for t, d in r["decisions"].items():
            if d["abstain"]:
                print(f"  {t:<14} ABSTAIN ({d['reason']})"); continue
            if d["scope"] == "recording":
                print(f"  {t:<14} {d['decision']:<18} p={d['probabilities']}  [{d['tier']}]")
            else:
                print(f"  {t:<14} summary {d['summary']}  [{d['tier']}]")
        print(f"  full output -> {out_dir / (name + '.json')}")


if __name__ == "__main__":
    main()
