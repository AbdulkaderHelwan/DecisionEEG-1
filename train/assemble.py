#!/usr/bin/env python3
"""Step 3 (CPU): bundle the reference scale, the task catalogue and every head into one file.

The bundle does not contain the backbone; DecisionEEG.load() loads the NOMAD-EEG-Large weights
separately. This step also writes those weights as a small release file (EMA weights + config,
float32, ~270 MB instead of the 1.08 GB training checkpoint) for running the chat demo on a
laptop: copy decisioneeg1.pt and backbone_large_ema.pt.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decisioneeg.paths import BACKBONE, BACKBONE_EXPORT, BUNDLE, HEADS, REFERENCE, RESULTS   # noqa: E402
from decisioneeg.tasks import TASKS                                                           # noqa: E402


def export_backbone():
    ck = torch.load(BACKBONE, map_location="cpu", weights_only=False)
    sd = {k: v.float() for k, v in ck["ema"].items()}
    cfg = dict(ck["cfg"])
    n = sum(v.numel() for k, v in sd.items() if "alpha_bar" not in k)
    torch.save({"cfg": cfg, "state_dict": sd, "step": ck.get("step"), "source": str(BACKBONE)}, BACKBONE_EXPORT)
    info = {"name": "NOMAD-EEG-Large", "file": str(BACKBONE), "release_file": str(BACKBONE_EXPORT),
            "training_step": ck.get("step"), "parameters": int(n), "d_model": cfg["d_model"],
            "n_blocks": cfg["n_blocks"], "feature_dim": 5 * len(cfg["feature_blocks"]) * 2 * cfg["d_model"]}
    print(f"backbone {BACKBONE.name}: step {info['training_step']}, {n / 1e6:.1f} M parameters "
          f"-> {BACKBONE_EXPORT}", flush=True)
    return info


def main():
    heads, missing = {}, []
    for t in TASKS:
        f = HEADS / f"{t}.pt"
        if f.exists():
            heads[t] = torch.load(f, map_location="cpu", weights_only=False)
        else:
            missing.append(t)
    assert heads, "no heads found: run train/fit_heads.py first"
    bundle = {"name": "DecisionEEG-1", "version": "1.1-large",
              "reference_scale_uV": json.loads(REFERENCE.read_text())["reference_scale_uV"],
              "tasks": {t: dataclasses.asdict(TASKS[t]) for t in heads}, "heads": heads,
              "backbone": export_backbone(),
              "results": {t: json.loads((RESULTS / f"{t}.json").read_text())["test"] for t in heads
                          if (RESULTS / f"{t}.json").exists()}}
    torch.save(bundle, BUNDLE)
    print(f"bundle -> {BUNDLE}  ({len(heads)} heads: {', '.join(heads)})"
          + (f"  | MISSING: {', '.join(missing)}" if missing else ""))


if __name__ == "__main__":
    main()
