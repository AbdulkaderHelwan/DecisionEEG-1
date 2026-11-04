#!/usr/bin/env python3
"""Step 0 (CPU): Siena and Bonn windows in microvolts, and the reference input scale.

Siena and Bonn have no NOMAD-EEG cache; the paper read them through the benchmark's loader.
This writes them with the SAME windows and splits (asserted against splits_<ds>.json):
  drop ROC -> 21 channels, 256 -> 200 Hz, convert to microvolts (Siena is stored in volts),
  keep only channels that carry signal (Bonn: 1), float16.
It also writes reference.json: the median per-channel IQR of TUEV in NOMAD-EEG-Large's
pre-training pool, the one input scale DecisionEEG-1 uses for every recording.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decisioneeg.paths import BENCH, CACHE, DATA, LARGE_TUEV_STATS, LINAH, LINAH_ORIG, REFERENCE, V3   # noqa: E402

sys.path.insert(0, str(BENCH / "code"))
sys.path.insert(0, str(V3 / "eval"))
import data_adapter as A                                   # noqa: E402
from extract_features import get_splits                    # noqa: E402
from extract_external import load_raw, to_model_input      # noqa: E402

# When the raw datasets live elsewhere (DECISIONEEG_LINAH), the benchmark's window folders are re-rooted there.
if str(LINAH) != LINAH_ORIG:
    for _ds in ("siena", "bonn"):
        A.DATASETS[_ds]["splits"] = {sp: [(Path(str(d).replace(LINAH_ORIG, str(LINAH))), lab) for d, lab in v]
                                     for sp, v in A.DATASETS[_ds]["splits"].items()}

CH21 = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
        "F7", "F8", "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ"]


def reference():
    """NOMAD-EEG-Large was trained with TUEV's per-channel IQR pooled over its build shards
    (weighted by sample count, as in pool_dataset.py); the reference is their median."""
    assert LARGE_TUEV_STATS, "no NOMAD-EEG-Large TUEV statistics found"
    acc = {}
    for f in LARGE_TUEV_STATS:
        for nm, v in json.loads(f.read_text())["channels"].items():
            acc.setdefault(nm, []).append((v["iqr"], v["n"]))
    iqr_uv = {nm: float(np.average([i for i, _ in v], weights=[n for _, n in v])) for nm, v in acc.items()}
    ref = {"reference_scale_uV": float(np.median(list(iqr_uv.values()))),
           "per_channel_iqr_uV": {k: round(v, 3) for k, v in iqr_uv.items()},
           "source": [str(f) for f in LARGE_TUEV_STATS],
           "definition": "median over channels of the per-channel IQR of TUEV (NOMAD-EEG-Large pool, uV)"}
    DATA.mkdir(parents=True, exist_ok=True)
    REFERENCE.write_text(json.dumps(ref, indent=2))
    print(f"reference scale {ref['reference_scale_uV']:.3f} uV  -> {REFERENCE}", flush=True)


def build(ds, limit=0):
    out = CACHE / ds
    out.mkdir(parents=True, exist_ok=True)
    splits = get_splits(ds, A)
    ref = json.loads((BENCH / "results" / f"splits_{ds}.json").read_text())["n"]
    got = {k: len(v) for k, v in splits.items()}
    assert got == ref, f"{ds}: split sizes {got} differ from the benchmark's {ref}"
    to_uv = float(A.DATASETS[ds].get("to_uv", 1.0))
    print(f"{ds}: splits {got}, to_uv {to_uv:g}", flush=True)
    live = None
    for sp in ("train", "val", "test"):
        idx = splits[sp]
        if limit:                                                # smoke test: a few windows of every class
            labs = np.array([l for _, _, l in idx])
            pick = np.sort(np.concatenate([np.where(labs == c)[0][:max(1, limit // 2)] for c in np.unique(labs)]))
            idx = [idx[i] for i in pick]
        X, y = load_raw(idx)
        X = to_model_input(X) * np.float32(to_uv)                # [N, 21, 1000] microvolts
        if live is None:                                         # channels that carry signal, from train
            q75, q25 = np.percentile(X[:3000].transpose(1, 0, 2).reshape(21, -1), [75, 25], axis=1)
            live = np.where((q75 - q25) > 0)[0]
            print(f"  live channels ({len(live)}): {[CH21[i] for i in live]} | "
                  f"median IQR {np.median((q75 - q25)[live]):.2f} uV", flush=True)
        np.save(out / f"{sp}_X.npy", np.clip(X[:, live], -6e4, 6e4).astype(np.float16))
        np.save(out / f"{sp}_y.npy", y.astype(np.int64))
        print(f"  {sp}: {X.shape[0]:,} windows, labels {np.bincount(y)}", flush=True)
        del X
    meta = {"channels": [CH21[i] for i in live], "classes": A.DATASETS[ds]["classes"], "fs": 200,
            "n_samples": 1000, "units": "uV", "built_from": "benchmarking2 splits + extract_external.load_raw"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="windows per split (smoke test)")
    a = ap.parse_args()
    reference()
    for ds in ("bonn", "siena"):
        build(ds, a.limit)
