#!/usr/bin/env python3
"""Step 4 (GPU): how often does a decision change when the input changes in ways that should
not matter?  (Decision-1 reports the same quantity as its "flip rate".)

For every task, up to --n test windows of its first source go through the FULL inference path
(DecisionEEG.predict_windows: units, resampling, scaling, backbone, calibrated head), once as
recorded and once per perturbation:

  reorder      electrodes listed in a random order
  drop25/50    25% / 50% of electrodes removed (3 random subsets each)
  fs256/fs500  the same signal delivered at 256 / 500 Hz (resampled back by the pipeline)
  gain0.8/1.25 amplifier gain differences
  volts        the same signal in volts, units detected automatically
  noise5uV     white noise, 5 uV rms
  line50       50 Hz mains interference, 10 uV

Reports the flip rate (share of decisions that change), the change in the primary metric and
the mean absolute change in calibrated probability.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decisioneeg.model import DecisionEEG     # noqa: E402
from decisioneeg.paths import RESULTS          # noqa: E402
from decisioneeg.sources import Source         # noqa: E402


def primary(y, p, d):
    if p.shape[1] == 2:
        return float(roc_auc_score(y, p[:, 1])) if len(np.unique(y)) == 2 else float("nan")
    return float(balanced_accuracy_score(y, d))


def perturbations(E, rng):
    P = [("reorder", None)]
    if E >= 4:
        P += [(f"drop{k}_s{s}", k) for k in (25, 50) for s in range(3)]
    elif E >= 2:
        P += [(f"drop50_s{s}", 50) for s in range(2)]
    P += [("fs256", None), ("fs500", None), ("gain0.8", None), ("gain1.25", None),
          ("volts", None), ("noise5uV", None), ("line50", None)]
    return P


def apply(name, k, x, ch, rng):
    fs, units = 200, "uV"
    if name == "reorder":
        perm = rng.permutation(len(ch)); x, ch = x[:, perm], [ch[i] for i in perm]
    elif name.startswith("drop"):
        keep = np.sort(rng.choice(len(ch), size=max(1, int(round(len(ch) * (1 - k / 100)))), replace=False))
        x, ch = x[:, keep], [ch[i] for i in keep]
    elif name == "fs256":
        x, fs = resample_poly(x, 32, 25, axis=-1), 256
    elif name == "fs500":
        x, fs = resample_poly(x, 5, 2, axis=-1), 500
    elif name.startswith("gain"):
        x = x * float(name[4:])
    elif name == "volts":
        x, units = x * 1e-6, "auto"
    elif name == "noise5uV":
        x = x + rng.normal(0, 5, x.shape)
    elif name == "line50":
        t = np.arange(x.shape[-1]) / 200
        x = x + 10 * np.sin(2 * np.pi * 50 * t)
    return x.astype(np.float32), ch, fs, units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--task", default="all")
    a = ap.parse_args()
    m = DecisionEEG.load()
    out = {}
    for tid, task in m.tasks.items():
        if a.task != "all" and tid not in a.task.split(","):
            continue
        src = Source(task.sources[0])
        rng = np.random.default_rng(0)
        n = src.n("test")
        idx = np.sort(rng.choice(n, size=min(a.n, n), replace=False))
        x = np.concatenate([src.uv("test", i, i + 1) for i in idx])      # [N, E, T] microvolts
        y = src.labels("test")[idx]
        base, info = m.predict_windows(x, src.channels, 200, [tid], "uV")
        p0, d0, _ = base[tid]
        r = {"source": src.name, "n": int(len(y)), "electrodes": len(src.channels),
             "baseline_primary": primary(y, p0, d0), "perturbations": {}}
        print(f"\n=== {tid} ({src.name}, {len(y)} windows, {len(src.channels)} electrodes): "
              f"baseline {r['baseline_primary']:.4f}", flush=True)
        for name, k in perturbations(len(src.channels), rng):
            xp, ch, fs, units = apply(name, k, x, list(src.channels), rng)
            o, inf = m.predict_windows(xp, ch, fs, [tid], units)
            if inf["abstain"]:
                r["perturbations"][name] = {"abstain": inf["abstain"]}
                print(f"  {name:<12} ABSTAIN: {inf['abstain']}", flush=True)
                continue
            p, d, _ = o[tid]
            r["perturbations"][name] = {"electrodes": len(inf.get("channels_used", ch)),
                                        "flip_rate": float((d != d0).mean()),
                                        "primary": primary(y, p, d),
                                        "mean_abs_prob_change": float(np.abs(p - p0).sum(1).mean() / 2),
                                        "warnings": inf["warnings"]}
            q = r["perturbations"][name]
            print(f"  {name:<12} flips {100 * q['flip_rate']:5.2f}%  primary {q['primary']:.4f}  "
                  f"|dp| {q['mean_abs_prob_change']:.4f}", flush=True)
        flips = [v["flip_rate"] for v in r["perturbations"].values() if "flip_rate" in v]
        r["mean_flip_rate"] = float(np.mean(flips))
        out[tid] = r
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "robustness.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {RESULTS / 'robustness.json'}")


if __name__ == "__main__":
    main()
