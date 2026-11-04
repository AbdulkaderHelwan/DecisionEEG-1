#!/usr/bin/env python3
"""Step 5: throughput of the full decision path (units, resampling, scaling, backbone, heads).

Times DecisionEEG.decide() on a synthetic one-hour recording (19 and 64 electrodes, 256 Hz)
for every task, on the GPU and on --cpu-threads CPU cores. The signal content does not affect
speed; it is random noise at EEG amplitude.
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
from decisioneeg.model import DecisionEEG     # noqa: E402
from decisioneeg.paths import RESULTS          # noqa: E402

C19 = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ", "C4", "T4", "T5", "P3", "PZ", "P4", "T6", "O1", "O2"]
C64 = ["FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPZ",
       "CP2", "CP4", "CP6", "FP1", "FPZ", "FP2", "AF7", "AF3", "AFZ", "AF4", "AF8", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6",
       "F8", "FT7", "FT8", "T7", "T8", "T9", "T10", "TP7", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7",
       "PO3", "POZ", "PO4", "PO8", "O1", "OZ", "O2", "IZ"]


def bench(m, ch, seconds, fs=256):
    x = np.random.default_rng(0).normal(0, 20, (len(ch), int(seconds * fs))).astype(np.float32)
    m.decide(x[:, :fs * 60], ch, fs)                                  # warm-up
    if m.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    r = m.decide(x, ch, fs)
    if m.device == "cuda":
        torch.cuda.synchronize()
    return time.time() - t0, len(r["decisions"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--cpu-seconds", type=int, default=300, help="recording length timed on CPU (extrapolated to 1 h)")
    a = ap.parse_args()
    out = {}
    if torch.cuda.is_available():
        m = DecisionEEG.load(device="cuda")
        for name, ch in (("19 electrodes", C19), ("64 electrodes", C64)):
            s, k = bench(m, ch, 3600)
            out[f"gpu {name}"] = {"seconds_per_hour_of_eeg": round(s, 2), "tasks": k, "device": torch.cuda.get_device_name()}
            print(f"GPU {name}: 1 h of EEG, {k} tasks -> {s:.1f} s", flush=True)
        del m; torch.cuda.empty_cache()
    torch.set_num_threads(a.cpu_threads)
    m = DecisionEEG.load(device="cpu")
    for name, ch in (("19 electrodes", C19),):
        s, k = bench(m, ch, a.cpu_seconds)
        out[f"cpu{a.cpu_threads} {name}"] = {"seconds_per_hour_of_eeg": round(s * 3600 / a.cpu_seconds, 1), "tasks": k,
                                            "measured_on_seconds": a.cpu_seconds}
        print(f"CPU x{a.cpu_threads} {name}: {a.cpu_seconds} s of EEG -> {s:.1f} s "
              f"(~{s * 3600 / a.cpu_seconds:.0f} s per hour)", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "speed.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
