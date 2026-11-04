#!/usr/bin/env python3
"""Step 2 (GPU): one calibrated linear head per task, tested on every source.

    python train/fit_heads.py                 # all tasks
    python train/fit_heads.py --task seizure

For each task: pool the train (and val) features of its sources -> fit the head, C chosen on
validation (AUROC for 2 classes, balanced accuracy otherwise) -> fit temperature + class bias
on validation -> score the test split of every source and every external source, at window
level and, where subjects are known, at subject level (mean probability per subject).
Reports the paper's metrics plus calibration (ECE, Brier, reliability bins) before and after
calibration, with 1,000-sample bootstrap CIs.
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
from decisioneeg.heads import (bootstrap, brier, by_subject, ece, fit_calibration, fit_head,   # noqa: E402
                               reliability, scores)
from decisioneeg.paths import BACKBONE, FEATURES, HEADS, RESULTS   # noqa: E402
from decisioneeg.tasks import TASKS                       # noqa: E402


def load(source, split):
    d = FEATURES / source
    return (np.load(d / f"{split}_X.npy", mmap_mode="r"), np.load(d / f"{split}_y.npy"),
            np.load(d / f"{split}_subjects.npy").astype(str))


def to_dev(X, dev, chunk=20000):
    out = torch.empty(X.shape, dtype=torch.float32, device=dev)
    for i in range(0, len(X), chunk):
        out[i:i + chunk] = torch.from_numpy(np.asarray(X[i:i + chunk], np.float32)).to(dev)
    return out


def evaluate(head, X, y, subj, dev, n_boot):
    p, d, q = head.outputs(to_dev(X, dev))
    raw = head.raw_probs(to_dev(X, dev))
    r = {"n": int(len(y)), "class_counts": np.bincount(y, minlength=len(head.classes)).tolist(),
         "window": scores(y, p, d), "ci95": bootstrap(y, p, d, n=n_boot),
         "uncalibrated": {"ece": ece(y, raw), "brier": brier(y, raw)},
         "reliability": reliability(y, p)}
    if subj is not None and not (subj == "unknown").all():
        Y, P = by_subject(y, p, subj)
        _, Q = by_subject(y, q, subj)
        D = Q.argmax(1)
        r["subject"] = {"n_subjects": int(len(Y)), **scores(Y, P, D), "ci95": bootstrap(Y, P, D, n=n_boot)}
    return r


def run(task_id, dev, n_boot):
    t = TASKS[task_id]
    t0 = time.time()
    print(f"\n=== {task_id}: {t.question}  sources {t.sources}  external {t.external}", flush=True)
    for s in (*t.sources, *t.external):
        meta = json.loads((FEATURES / s / "meta.json").read_text())
        assert len(meta["classes"]) == len(t.options), (s, meta["classes"], t.options)
        assert meta["backbone"] == str(BACKBONE), f"{s}: features from {meta['backbone']}, backbone is {BACKBONE}"
        print(f"  {s}: classes {meta['classes']} -> options {list(t.options)}", flush=True)
    tr = [load(s, "train") for s in t.sources]
    va = [load(s, "val") for s in t.sources]
    Ftr = torch.cat([to_dev(X, dev) for X, _, _ in tr])
    ytr = np.concatenate([y for _, y, _ in tr])
    Fva = torch.cat([to_dev(X, dev) for X, _, _ in va])
    yva = np.concatenate([y for _, y, _ in va])
    print(f"  train {len(ytr):,} {np.bincount(ytr).tolist()} | val {len(yva):,} {np.bincount(yva).tolist()}", flush=True)
    head, curve = fit_head(Ftr, ytr, Fva, yva, list(t.options), device=dev)
    del Ftr
    z_val = head.logits(Fva)
    head.T, head.bias = fit_calibration(z_val, yva)
    print(f"  C={head.C}  temperature {head.T:.3f}  bias {head.bias.cpu().numpy().round(3).tolist()}", flush=True)
    res = {"task": task_id, "question": t.question, "options": list(t.options), "tier": t.tier,
           "sources": list(t.sources), "external": list(t.external), "best_C": head.C, "val_curve": curve,
           "temperature": head.T, "calibration_bias": head.bias.cpu().tolist(),
           "val": scores(yva, *head.outputs(Fva)[:2]), "paper_reference": t.paper, "note": t.note, "test": {}}
    for s in (*t.sources, *t.external):
        X, y, subj = load(s, "test")
        res["test"][s] = evaluate(head, X, y, subj, dev, n_boot)
        w = res["test"][s]["window"]
        print(f"  TEST {s:<9} " + "  ".join(f"{k} {w[k]:.4f}" for k in
              ("auroc", "pr_auc", "balanced_accuracy", "cohen_kappa", "f1_macro", "ece") if k in w)
              + (f" | subject AUROC {res['test'][s]['subject'].get('auroc', float('nan')):.4f}"
                 if "subject" in res["test"][s] else ""), flush=True)
    HEADS.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(parents=True, exist_ok=True)
    torch.save(head.state(), HEADS / f"{task_id}.pt")
    (RESULTS / f"{task_id}.json").write_text(json.dumps(res, indent=2))
    print(f"  saved head and results ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for tid in (TASKS if a.task == "all" else a.task.split(",")):
        run(tid, dev, a.n_boot)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
