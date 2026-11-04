#!/usr/bin/env python3
"""Step 1 (GPU): frozen NOMAD-EEG features of one dataset under the deployable input transform.

    python build/extract_features.py --source tusz

Microvolt windows (sources.py) -> per-window DC removal, reference scale, clip (normalize.py)
-> NOMAD-EEG-Large features (5 windows x mean+std x 4 gate blocks x 512 = 20,480-d, bf16 on GPU).
Writes <DATA>/features/<source>/{train,val,test}_X.npy (float16), _y.npy, _subjects.npy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decisioneeg.normalize import to_model_input   # noqa: E402
from decisioneeg.paths import BACKBONE, FEATURES, REFERENCE   # noqa: E402
from decisioneeg.sources import Source               # noqa: E402
import nomad_eeg                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=2048, help="windows read from disk at a time")
    ap.add_argument("--limit", type=int, default=0, help="windows per split (smoke test)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    scale = json.loads(REFERENCE.read_text())["reference_scale_uV"]
    model = nomad_eeg.load_pretrained(BACKBONE, device=dev)
    dim = 5 * len(model.cfg.feature_blocks) * 2 * model.cfg.d_model
    src = Source(a.source)
    out = FEATURES / a.source
    out.mkdir(parents=True, exist_ok=True)
    print(f"{a.source}: {len(src.channels)} channels {src.channels[:6]}..., classes {src.classes}, "
          f"reference scale {scale:.3f} uV, device {dev}", flush=True)
    for sp in ("train", "val", "test"):
        rows = src.stratified_rows(sp, a.limit) if a.limit else np.arange(src.n(sp))
        n = len(rows)
        F = np.lib.format.open_memmap(out / f"{sp}_X.npy.tmp", mode="w+", dtype=np.float16, shape=(n, dim))
        t0 = time.time()
        for i in range(0, n, a.chunk):
            r = rows[i:i + a.chunk]
            uv = src.uv(sp, int(r[0]), int(r[-1]) + 1) if not a.limit else src.uv_rows(sp, r)
            if i == 0:
                print(f"  {sp}: {n:,} windows, input {uv.shape}, median |x| {np.median(np.abs(uv)):.2f} uV, "
                      f"p95 |x| {np.percentile(np.abs(uv), 95):.1f} uV", flush=True)
            z = to_model_input(uv, scale)
            f = nomad_eeg.extract_features(model, z, src.channels, batch_size=a.bs, device=dev)
            assert f.shape[1] == dim, (f.shape, dim)
            assert np.isfinite(f).all(), "non-finite features"
            F[i:i + len(f)] = f.astype(np.float16)
            if (i // a.chunk) % 10 == 0:
                print(f"    {i + len(f):>8,}/{n:,}  ({time.time() - t0:.0f}s)", flush=True)
        F.flush(); del F
        (out / f"{sp}_X.npy.tmp").replace(out / f"{sp}_X.npy")
        np.save(out / f"{sp}_y.npy", src.labels(sp)[rows])
        subj = src.subjects(sp)
        np.save(out / f"{sp}_subjects.npy", (subj[rows] if subj is not None else np.array(["unknown"] * n)))
        print(f"  {sp} done in {time.time() - t0:.0f}s", flush=True)
    (out / "meta.json").write_text(json.dumps({"channels": src.channels, "classes": src.classes,
                                               "reference_scale_uV": scale, "backbone": str(BACKBONE),
                                               "feature_dim": dim}, indent=2))


if __name__ == "__main__":
    main()
