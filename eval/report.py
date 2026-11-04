#!/usr/bin/env python3
"""Step 6 (CPU): MODEL_CARD.md from the result files. Every number is read from results/*.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decisioneeg.paths import BACKBONE, DATA, REFERENCE, RESULTS, ROOT   # noqa: E402
from decisioneeg.tasks import TASKS                             # noqa: E402


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def ci(r, k):
    c = r.get("ci95", {}).get(k)
    return f"{r['window'][k]:.3f} [{c[0]:.3f}, {c[1]:.3f}]" if c else f3(r["window"].get(k))


def main():
    L = ["# DecisionEEG-1 model card", "",
         "Calibrated answers to a fixed set of EEG questions, from any electrode montage, on the frozen "
         "NOMAD-EEG-Large backbone (67.6 M parameters, " + BACKBONE.name + ") with one linear head per question. Research use only; "
         "not a medical device.", ""]
    ref = json.loads(REFERENCE.read_text())
    L += [f"Input transform: microvolts, 200 Hz, per-window median removed, divided by "
          f"{ref['reference_scale_uV']:.2f} uV ({ref['definition']}), clipped at +-20.", ""]
    L += ["## Questions and test results", "",
          "Primary metric: AUROC (2 options) or balanced accuracy. Decisions are class-balanced; probabilities "
          "are calibrated on validation (temperature + class bias). ECE: expected calibration error (lower is "
          "better). Paper: NOMAD-EEG BASE frozen probe in the paper (different backbone and protocol: per-dataset scaling, one "
          "dataset per head).", "",
          "| Question | Tier | Test set | n | Primary [95% CI] | Bal. acc. | Macro F1 | ECE before → after | Paper |",
          "|---|---|---|---|---|---|---|---|---|"]
    for tid, t in TASKS.items():
        f = RESULTS / f"{tid}.json"
        if not f.exists():
            L.append(f"| {t.question} | {t.tier} | not run | | | | | | |"); continue
        res = json.loads(f.read_text())
        for s, r in res["test"].items():
            k = "auroc" if "auroc" in r["window"] else "balanced_accuracy"
            paper = t.paper.get(s, {})
            ext = " (external)" if s in t.external else ""
            L.append(f"| {t.question} | {t.tier} | {s}{ext} | {r['n']:,} | {ci(r, k)} | {f3(r['window']['balanced_accuracy'])} "
                     f"| {f3(r['window']['f1_macro'])} | {f3(r['uncalibrated']['ece'])} → {f3(r['window']['ece'])} "
                     f"| {', '.join(f'{a} {b:.3f}' for a, b in paper.items()) or '-'} |")
            if "subject" in r:
                sr = r["subject"]
                L.append(f"| ↳ per subject | | {s} | {sr['n_subjects']} subjects | "
                         f"{f3(sr.get('auroc', sr.get('balanced_accuracy')))} | {f3(sr['balanced_accuracy'])} | "
                         f"{f3(sr['f1_macro'])} | {f3(sr['ece'])} | |")
    rob = RESULTS / "robustness.json"
    if rob.exists():
        R = json.loads(rob.read_text())
        names = sorted({p for r in R.values() for p in r["perturbations"]})
        L += ["", "## Robustness: share of decisions that change", "",
              "| Question | " + " | ".join(names) + " | mean |", "|---|" + "---|" * (len(names) + 1)]
        for tid, r in R.items():
            cells = []
            for p in names:
                v = r["perturbations"].get(p)
                cells.append("-" if v is None else "abstain" if "abstain" in v else f"{100 * v['flip_rate']:.1f}%")
            L.append(f"| {tid} | " + " | ".join(cells) + f" | {100 * r['mean_flip_rate']:.1f}% |")
    sp = RESULTS / "speed.json"
    if sp.exists():
        S = json.loads(sp.read_text())
        L += ["", "## Speed (all questions, one hour of EEG)", "", "| Setting | Seconds |", "|---|---|"]
        L += [f"| {k} | {v['seconds_per_hour_of_eeg']} |" for k, v in S.items()]
    L += ["", "## Limits", "",
          "- Answers only the questions above; each was trained on one or two public datasets.",
          "- Seizure detection, not prediction. Window-level metrics; event-level scoring (sensitivity, false "
          "alarms per 24 h) is not yet done.",
          "- Probabilities are calibrated at the class mix of the validation data; at a different prevalence, "
          "rank (AUROC) holds but stated probabilities shift.",
          "- Experimental tier: small test sets or modest accuracy; do not rely on them."]
    text = "\n".join(L) + "\n"
    (DATA / "MODEL_CARD.md").write_text(text)
    (ROOT / "MODEL_CARD.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
