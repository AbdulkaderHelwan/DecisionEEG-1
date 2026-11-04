"""One pass of DecisionEEG-1 over a whole recording, saved so a chat can query it later.

    python -m decisioneeg.chat.analysis recording.edf --out analyses/rec1 [--max-seconds 3600]

This is the only part of "Ask your EEG" that needs torch and the backbone. It runs every
question once (one backbone pass per window length), keeps the per-window calibrated
probabilities and decisions, the cleaned 200 Hz microvolt signal (for descriptive statistics
and plots) and the recording's annotations. Stimulus- or cue-locked questions (P300, motor
imagery) are answered at every annotation onset, since fixed windows do not answer them.

The saved folder holds:
  analysis.json   recording facts, warnings, abstentions, task cards with test results
  arrays.npz      per question: start times, calibrated probabilities, decisions, artefact flags
  signal.npz      cleaned signal [channels, samples] in microvolts at 200 Hz (float16)
The chat side (tools.py) reads only these files: it needs numpy and matplotlib, not torch.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .. import normalize as N
from ..tasks import TASKS

MAX_EPOCHS_PER_LABEL = 500


class Analysis:
    def __init__(self, meta: dict, arrays: dict, signal: np.ndarray):
        self.meta, self.arrays, self.signal = meta, arrays, signal     # signal: [E, T] uV at 200 Hz
        self.fs = N.FS

    # ------------------------------------------------------------------ build
    @classmethod
    def from_edf(cls, path, model, max_seconds=None):
        import mne
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        fs = float(raw.info["sfreq"])
        stop = None if max_seconds is None else min(raw.n_times, int(max_seconds * fs))
        x = raw.get_data(stop=stop) * 1e6                                # MNE returns volts
        ann = [{"onset_s": float(a["onset"]), "duration_s": float(a["duration"]), "label": str(a["description"])}
               for a in raw.annotations if stop is None or a["onset"] * fs < stop]
        a = cls.from_array(x, raw.ch_names, fs, model, annotations=ann, units="uV", name=Path(path).name)
        a.meta["recording"]["full_duration_s"] = raw.n_times / fs
        return a

    @classmethod
    def from_array(cls, x, channels, fs, model, annotations=(), units="auto", name="array"):
        t0 = time.time()
        x = np.asarray(x)
        rec = {"name": name, "fs_original": fs, "duration_s": x.shape[-1] / fs,
               "channels_given": list(channels), "n_channels_given": len(channels)}
        meta = {"model": "DecisionEEG-1", "backbone": model.describe_backbone(),
                "reference_scale_uV": model.scale, "recording": rec, "warnings": [],
                "annotations": list(annotations), "questions": {}, "cards": task_cards(model)}
        arrays = {}
        keep, names, w = N.resolve_channels(channels)
        meta["warnings"] += w
        if not keep:
            return cls._abstain_all(meta, model, "no channel with a standard 10-20 / 10-05 name")
        x = x[keep].astype(np.float64)
        if units == "auto":
            f, warn, stop = N.detect_units(x[None])
            if stop:
                return cls._abstain_all(meta, model, stop)
            if warn:
                meta["warnings"].append(warn)
        else:
            f = {"uV": 1.0, "mV": 1e3, "V": 1e6}[units]
        rec["unit_factor_to_uV"] = f
        x = N.resample((x * f).astype(np.float32), fs)                      # [E, T] uV at 200 Hz
        flat = N.flat_channels(x[None])
        if len(flat) == x.shape[0]:
            return cls._abstain_all(meta, model, "all channels are flat")
        if len(flat):
            meta["warnings"].append(f"dropped {len(flat)} flat channel(s): {', '.join(names[i] for i in flat)}")
            good = [i for i in range(x.shape[0]) if i not in set(flat)]
            x, names = x[good], [names[i] for i in good]
        rec["channels_used"] = names
        # continuous questions: consecutive windows of each window length
        by_len = {}
        for t in model.heads:
            if not model.tasks[t].locked:
                by_len.setdefault(model.tasks[t].window_s, []).append(t)
        for win_s, group in sorted(by_len.items()):
            L = int(round(win_s * N.FS))
            n = x.shape[-1] // L
            if n == 0:
                for t in group:
                    meta["questions"][t] = {"abstain": f"recording shorter than this question's {win_s:g} s window"}
                continue
            W = np.stack([x[:, i * L:(i + 1) * L] for i in range(n)])
            out, info = model.predict_windows(W, names, N.FS, group, units="uV")
            meta["warnings"] += [m for m in info["warnings"] if m not in meta["warnings"]]
            for t in group:
                if info["abstain"]:
                    meta["questions"][t] = {"abstain": info["abstain"]}; continue
                p, d, q = out[t]
                cls._store(arrays, t, np.arange(n) * win_s, p, d, q, info)
                meta["questions"][t] = {"abstain": None, "kind": "continuous", "window_s": win_s, "n_windows": n}
        # locked questions: one window at each onset of each annotation label
        labels = {}
        for an in annotations:
            labels.setdefault(an["label"], []).append(an["onset_s"])
        for t in model.heads:
            if not model.tasks[t].locked:
                continue
            if not labels:
                meta["questions"][t] = {"abstain": "this question needs stimulus or cue onsets and the recording "
                                                   "has no annotations"}
                continue
            done = {}
            for lab, onsets in labels.items():
                kept, res, info = model.predict_epochs(x, names, N.FS, t, onsets[:MAX_EPOCHS_PER_LABEL], units="uV")
                if res is None or info.get("abstain"):
                    continue
                cls._store(arrays, f"{t}@{lab}", np.asarray(kept), *res, info)
                done[lab] = len(kept)
            meta["questions"][t] = ({"abstain": None, "kind": "epochs", "window_s": model.tasks[t].window_s,
                                     "epochs_per_label": done} if done else
                                    {"abstain": "no annotation onset leaves a full window inside the recording"})
        meta["seconds"] = round(time.time() - t0, 2)
        return cls(meta, arrays, x.astype(np.float16))

    @staticmethod
    def _store(arrays, key, starts, p, d, q, info):
        arrays[f"{key}/start_s"] = np.asarray(starts, np.float32)
        arrays[f"{key}/p"] = p.astype(np.float32)                  # calibrated probabilities
        arrays[f"{key}/q"] = q.astype(np.float32)                  # class-balanced probabilities
        arrays[f"{key}/decision"] = d.astype(np.int16)             # class-balanced decision
        arrays[f"{key}/artefact"] = np.asarray(info["artefact_abstain"], bool)

    @classmethod
    def _abstain_all(cls, meta, model, reason):
        meta["questions"] = {t: {"abstain": reason} for t in model.heads}
        return cls(meta, {}, np.zeros((0, 0), np.float16))

    # --------------------------------------------------------------- storage
    def save(self, out):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "analysis.json").write_text(json.dumps(self.meta, indent=1))
        np.savez_compressed(out / "arrays.npz", **self.arrays)
        np.savez_compressed(out / "signal.npz", signal=self.signal)
        return out

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        meta = json.loads((folder / "analysis.json").read_text())
        with np.load(folder / "arrays.npz") as z:
            arrays = {k: z[k] for k in z.files}
        with np.load(folder / "signal.npz") as z:
            signal = z["signal"]
        return cls(meta, arrays, signal)

    # ---------------------------------------------------------------- access
    def has(self, key):
        return f"{key}/p" in self.arrays

    def get(self, key):
        """(start_s, calibrated p, balanced q, decision, artefact) for a question or 'task@label'."""
        a = self.arrays
        return (a[f"{key}/start_s"], a[f"{key}/p"], a[f"{key}/q"], a[f"{key}/decision"].astype(int),
                a[f"{key}/artefact"])


def copy_annotations(edf, folder):
    """Keep expert seizure annotations (TUSZ .csv_bi next to the EDF) with the analysis, for the
    report only: the chat tools never read them."""
    src = Path(edf).with_suffix(".csv_bi")
    if src.exists():
        (Path(folder) / "annotations.csv_bi").write_text(src.read_text())


def task_cards(model) -> dict:
    """What each question is, how it was tested, and how well it did (test splits, from the bundle)."""
    cards = {}
    res = model.bundle.get("results", {})
    for t, task in model.tasks.items():
        tests = {}
        for src, r in res.get(t, {}).items():
            w = r["window"]
            keep = {k: round(w[k], 4) for k in ("auroc", "pr_auc", "balanced_accuracy", "cohen_kappa", "f1_macro", "ece")
                    if k in w and w[k] == w[k]}
            entry = {"n_windows": r["n"], "class_counts": r["class_counts"], "window_level": keep,
                     "ci95": {k: [round(v, 4) for v in r["ci95"][k]] for k in keep if k in r.get("ci95", {})},
                     "role": "external (never trained on)" if src in task.external else "held-out test split"}
            if "subject" in r:
                entry["subject_level"] = {k: round(v, 4) for k, v in r["subject"].items()
                                          if isinstance(v, float) and v == v}
            tests[src] = entry
        cards[t] = {"question": task.question, "options": list(task.options), "tier": task.tier,
                    "scope": task.scope, "window_s": task.window_s, "locked": task.locked,
                    "trained_on": list(task.sources), "external_tests": list(task.external),
                    "note": task.note, "test_results": tests,
                    "reference_base_paper": task.paper}
    return cards


def main():
    import argparse
    from ..model import DecisionEEG
    ap = argparse.ArgumentParser()
    ap.add_argument("edf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seconds", type=float, default=3600)
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    kw = {"bundle_path": a.bundle} if a.bundle else {}
    m = DecisionEEG.load(device=a.device, **kw)
    an = Analysis.from_edf(a.edf, m, a.max_seconds)
    out = an.save(a.out)
    copy_annotations(a.edf, out)
    q = an.meta["questions"]
    print(f"{a.edf}: {an.meta['recording']['duration_s']:.0f} s, {len(an.meta['recording'].get('channels_used', []))} "
          f"channels, {an.meta.get('seconds')} s | answered: {[t for t, v in q.items() if not v['abstain']]} | "
          f"abstained: {[t for t, v in q.items() if v['abstain']]} -> {out}", flush=True)


if __name__ == "__main__":
    main()
