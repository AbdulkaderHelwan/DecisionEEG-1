"""DecisionEEG-1: EEG in, calibrated answers to a fixed set of questions out.

    from decisioneeg import DecisionEEG
    m = DecisionEEG.load()                               # bundle + NOMAD-EEG backbone
    out = m.decide_edf("recording.edf")                  # every task
    out = m.decide(x, channels, fs=256, tasks=["seizure", "abnormal"])

One backbone pass per window length serves every task with that length (all 5 s tasks share
it). Long recordings are cut into consecutive windows; window tasks return a timeline,
recording tasks the mean over windows.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from . import normalize as N
from .heads import LinearHead
from .paths import BACKBONE, BACKBONE_EXPORT, BUNDLE
from .tasks import TASKS

ARTEFACT_WARN, ARTEFACT_ABSTAIN = 0.2, 0.5      # fraction of samples at the clip limit
BATCH_BUDGET = 32 * 21 * 1000                    # electrodes x samples x windows per backbone pass
LOCKED_REASON = ("this question needs windows that start at a stimulus or cue; give the event onsets "
                 "(predict_epochs) instead of a continuous recording")


RELEASE_URL = "https://github.com/AbdulkaderHelwan/DecisionEEG-1/releases/download/v1.0/"


def download_model_files():
    """Fetch decisioneeg1.pt (3 MB) and backbone_large_ema.pt (270 MB) into paths.DATA once."""
    for f in (BUNDLE, BACKBONE_EXPORT):
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading {f.name} from {RELEASE_URL} ...", flush=True)
            torch.hub.download_url_to_file(RELEASE_URL + f.name, str(f), progress=True)


class DecisionEEG:
    def __init__(self, bundle: dict, backbone, device):
        self.bundle, self.backbone, self.device = bundle, backbone, device
        self.scale = float(bundle["reference_scale_uV"])
        self.heads = {t: LinearHead.from_state(s).to(device) for t, s in bundle["heads"].items()}
        self.tasks = {t: TASKS[t] for t in self.heads}

    @classmethod
    def load(cls, bundle_path=BUNDLE, backbone_path=None, device=None):
        """backbone_path: default is backbone_large_ema.pt next to the bundle, else paths.BACKBONE.
        With the default paths, missing model files are downloaded once from the GitHub release."""
        import nomad_eeg
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if Path(bundle_path) == BUNDLE and backbone_path is None:
            download_model_files()
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        if backbone_path is None:
            local = Path(bundle_path).with_name(BACKBONE_EXPORT.name)
            backbone_path = local if local.exists() else BACKBONE
        return cls(bundle, nomad_eeg.load_pretrained(backbone_path, device=device), device)

    def describe_backbone(self) -> str:
        b = self.bundle.get("backbone")
        if not isinstance(b, dict):
            return str(b)
        return f"{b['name']} ({b['parameters'] / 1e6:.1f} M parameters, training step {b['training_step']})"

    # ------------------------------------------------------------------ core
    @torch.no_grad()
    def features(self, z: np.ndarray, channels, batch=None) -> np.ndarray:
        """batch default: windows per pass so that electrodes x samples x batch stays within what
        NOMAD-EEG-Large fits on this GPU (on 20 GB: 32 windows of 21 x 1000; 64 electrodes x 30 s -> 1).
        Out of GPU memory: the batch is halved and retried; at batch 1 the model moves to the CPU."""
        from nomad_eeg import extract_features
        if batch is None:
            budget = BATCH_BUDGET
            if str(self.device).startswith("cuda"):
                budget *= torch.cuda.get_device_properties(self.device).total_memory / 20e9
            batch = int(np.clip(budget // (z.shape[-2] * z.shape[-1]), 1, 32))
        while True:
            try:
                return extract_features(self.backbone, z, channels, batch_size=batch, device=self.device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch > 1:
                    batch //= 2
                    continue
                print("GPU memory too small even for one window: continuing on the CPU (slower)", flush=True)
                self.device = "cpu"
                self.backbone.to("cpu")
                self.heads = {t: h.to("cpu") for t, h in self.heads.items()}

    def predict_windows(self, x, channels, fs, tasks=None, units="auto"):
        """x: [N, E, T] windows cut to one task window length (at fs).

        Returns ({task: (calibrated probs [N, K], decisions [N])}, info) for the given tasks,
        which must all share that window length."""
        info = {"warnings": [], "abstain": None}
        keep, names, w = N.resolve_channels(channels)
        info["warnings"] += w
        if not keep:
            info["abstain"] = "no channel with a standard 10-20 / 10-05 name"
            return {}, info
        x = np.asarray(x)[:, keep]
        if units == "auto":
            f, warn, stop = N.detect_units(x)
            if stop:
                info["abstain"] = stop
                return {}, info
            if warn:
                info["warnings"].append(warn)
        else:
            f = {"uV": 1.0, "mV": 1e3, "V": 1e6}[units]
        x = x * f
        x = np.stack([N.resample(w_, fs) for w_ in x]) if fs != N.FS else x.astype(np.float32)
        flat = N.flat_channels(x)
        if len(flat) == x.shape[1]:
            info["abstain"] = "all channels are flat"
            return {}, info
        if len(flat):
            info["warnings"].append(f"dropped {len(flat)} flat channel(s): {', '.join(names[i] for i in flat)}")
            good = [i for i in range(x.shape[1]) if i not in set(flat)]
            x, names = x[:, good], [names[i] for i in good]
        z = N.to_model_input(x, self.scale)
        clip = N.clipped_fraction(z)
        info["artefact_fraction"] = clip.tolist()
        if (clip > ARTEFACT_WARN).mean() > 0:
            info["warnings"].append(f"{int((clip > ARTEFACT_WARN).sum())} of {len(clip)} window(s) have heavy artefacts")
        info["channels_used"] = names
        F_ = self.features(z, names)
        out = {}
        for t in (tasks or list(self.heads)):
            out[t] = self.heads[t].outputs(F_)                  # (calibrated p, decision, balanced q)
        info["artefact_abstain"] = (clip > ARTEFACT_ABSTAIN).tolist()
        return out, info

    def decide(self, x, channels, fs, tasks=None, units="auto", start_s=0.0):
        """x: [E, T] one continuous recording. Returns a JSON-serialisable dict."""
        t0 = time.time()
        x = np.asarray(x)
        tasks = [t for t in (tasks or list(self.heads)) if t in self.heads]
        res = {"model": "DecisionEEG-1", "backbone": self.describe_backbone(),
               "input": {"channels_given": len(channels), "fs": fs, "duration_s": x.shape[-1] / fs},
               "warnings": [], "decisions": {}}
        by_len = {}
        for t in tasks:
            if self.tasks[t].locked:
                res["decisions"][t] = self._abstain(t, LOCKED_REASON)
                continue
            by_len.setdefault(self.tasks[t].window_s, []).append(t)
        for win_s, group in by_len.items():
            L = int(round(win_s * fs))
            n = x.shape[-1] // L
            if n == 0:
                for t in group:
                    res["decisions"][t] = self._abstain(t, f"recording shorter than this task's {win_s:g} s window")
                continue
            W = np.stack([x[:, i * L:(i + 1) * L] for i in range(n)])
            out, info = self.predict_windows(W, channels, fs, group, units)
            res["warnings"] += [w for w in info["warnings"] if w not in res["warnings"]]
            res["input"]["channels_used"] = info.get("channels_used", [])
            for t in group:
                if info["abstain"]:
                    res["decisions"][t] = self._abstain(t, info["abstain"]); continue
                res["decisions"][t] = self._summarise(t, *out[t], win_s, start_s, info["artefact_abstain"])
        res["seconds"] = round(time.time() - t0, 3)
        return res

    def predict_epochs(self, x, channels, fs, task, onsets_s, units="auto"):
        """Stimulus- or cue-locked questions: one window of the task's length starting at each
        onset (seconds). Onsets whose window runs past the end are dropped.

        Returns (onsets kept, (calibrated probs, decisions, balanced probs), info)."""
        L = int(round(self.tasks[task].window_s * fs))
        x = np.asarray(x)
        keep = [float(o) for o in onsets_s if 0 <= int(round(o * fs)) and int(round(o * fs)) + L <= x.shape[-1]]
        if not keep:
            return [], None, {"abstain": "no onset leaves a full window inside the recording", "warnings": []}
        W = np.stack([x[:, int(round(o * fs)):int(round(o * fs)) + L] for o in keep])
        out, info = self.predict_windows(W, channels, fs, [task], units)
        return keep, out.get(task), info

    def decide_edf(self, path, tasks=None, max_seconds=None):
        import mne
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        stop = None if max_seconds is None else min(raw.n_times, int(max_seconds * raw.info["sfreq"]))
        x = raw.get_data(stop=stop) * 1e6                       # MNE returns volts
        r = self.decide(x, raw.ch_names, raw.info["sfreq"], tasks, units="uV")
        r["input"]["file"] = str(path)
        return r

    # --------------------------------------------------------------- output
    def _abstain(self, t, reason):
        task = self.tasks[t]
        return {"question": task.question, "tier": task.tier, "abstain": True, "reason": reason}

    def _summarise(self, t, p, d, q, win_s, start_s, art_abstain):
        task = self.tasks[t]
        opts = list(task.options)
        ok = ~np.asarray(art_abstain, bool)
        r = {"question": task.question, "options": opts, "tier": task.tier, "scope": task.scope,
             "abstain": False, "windows": int(len(d)), "windows_skipped_artefact": int((~ok).sum())}
        if not ok.any():
            r.update(abstain=True, reason="every window is dominated by artefact")
            return r
        if task.scope == "recording":
            pm = p[ok].mean(0)
            r["probabilities"] = dict(zip(opts, np.round(pm, 4).tolist()))
            r["decision"] = opts[int(np.argmax(q[ok].mean(0)))]          # balanced, as in evaluation
            votes = np.bincount(d[ok], minlength=len(opts)) / ok.sum()
            r["window_votes"] = dict(zip(opts, np.round(votes, 4).tolist()))
        else:
            r["timeline"] = [{"start_s": round(start_s + i * win_s, 2),
                              "probabilities": dict(zip(opts, np.round(p[i], 4).tolist())),
                              "decision": opts[int(d[i])], "artefact": bool(not ok[i])}
                             for i in range(len(d))]
            r["summary"] = dict(zip(opts, (np.bincount(d[ok], minlength=len(opts)) / ok.sum()).round(4).tolist()))
            if len(opts) == 2:
                r["max_probability_yes"] = float(p[ok, 1].max())
        return r
