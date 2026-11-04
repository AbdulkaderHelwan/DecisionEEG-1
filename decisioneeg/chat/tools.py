"""The tools the chat model calls. Every answer about the EEG comes from one of these.

Model tools (calibrated DecisionEEG-1 outputs, read from a saved Analysis):
  recording_info, list_questions, ask_model, find_segments, get_timeline, epoch_results
Descriptive tools (plain signal processing, NOT model decisions):
  signal_stats, view_signal, view_timeline

Each tool returns JSON text (and, for the view tools, a PNG the chat model can look at). The
tools only read numpy arrays, so they run anywhere numpy and matplotlib do.
"""
from __future__ import annotations

import base64
import io
import json

import numpy as np

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
POSTERIOR = ("O1", "O2", "OZ", "P3", "P4", "PZ", "PO3", "PO4", "POZ", "T5", "T6", "P7", "P8")
# options that mean "nothing found" (segments are reported for the other options)
NEGATIVE = {"seizure": "no", "eeg_event": "background", "abnormal": "normal", "epilepsy": "no",
            "depression": "no", "p300": "no"}

# what each training / test dataset is (so answers about reliability can state facts, not guesses)
DATASETS = {
    "tusz": "TUH EEG Seizure Corpus: clinical scalp EEG from Temple University Hospital, 19-21 channels, seizures annotated by experts",
    "siena": "Siena Scalp EEG: 14 adults with epilepsy, long clinical recordings, 19 channels",
    "bonn": "Bonn University EEG: short single-channel segments; here 1 channel, unlike the 19-21 channels the seizure question was trained on",
    "tuev": "TUH EEG Events: clinical EEG with expert-annotated spikes, periodic discharges, eye movements and artefacts",
    "tuab5s": "TUH Abnormal EEG: clinical recordings each labelled normal or abnormal by a neurologist, cut into 5 s windows",
    "sleepedf": "Sleep-EDF: overnight recordings with 2 EEG channels (Fpz-Cz, Pz-Oz), scored in 30 s epochs",
    "bciaut": "BCIAUT-P300: P300 speller sessions of 15 people with autism, 8 channels",
    "tuep": "TUH EEG Epilepsy: clinical EEG of patients with and without an epilepsy diagnosis",
    "mumtaz": "Mumtaz 2016: resting EEG, 19 channels, people with major depressive disorder and healthy controls",
    "eegmmidb": "PhysioNet EEG Motor Movement/Imagery: 109 volunteers, 64 channels, cued imagined movements",
}
# measured problems (spot checks of DecisionEEG-1 on NOMAD-EEG-Large, 2026-11-02)
_FA = ("on healthy volunteers recorded outside the training datasets this question has given false positives: ")
KNOWN_ISSUES = {
    "seizure": _FA + "seizure decided in 100% of windows of one EEGMMIDB recording and 42% of one Mumtaz healthy control; "
               "and it was trained on adults: in 5 Helsinki newborn recordings it flagged a window in only 9 of the 101 "
               "seizures experts marked (2026-11-04)",
    "abnormal": _FA + "one EEGMMIDB recording of a healthy volunteer was called abnormal with probability 1.0",
    "epilepsy": _FA + "healthy volunteers (EEGMMIDB, Mumtaz) were called epileptic with probability 0.84-0.92",
}

_Q = {"type": "string", "description": "Question id from list_questions, e.g. 'seizure', 'eeg_event', 'sleep_stage'."}
_T0 = {"type": "number", "description": "Start of the time range in seconds from the recording start (default 0)."}
_T1 = {"type": "number", "description": "End of the time range in seconds (default: end of recording)."}

TOOLS = [
    {"name": "recording_info",
     "description": "Facts about the loaded recording: file, duration, sampling rate, channels given and used, "
                    "unit conversion, data-quality warnings, annotations (counts per label), and which questions "
                    "were answered or abstained (with the reason). Call this first in a conversation.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "list_questions",
     "description": "The fixed questions DecisionEEG-1 can answer, with their options, window length, tier "
                    "(ready / experimental), training datasets and measured test performance (AUROC, balanced "
                    "accuracy, calibration error, 95% CIs). Use it to say what the model can and cannot answer "
                    "and how much to trust an answer.",
     "input_schema": {"type": "object", "properties": {"question_id": {**_Q, "description": "Optional: one question only."}},
                      "additionalProperties": False}},
    {"name": "ask_model",
     "description": "The model's answer to one question over the whole recording or a time range. Recording-level "
                    "questions (abnormal, epilepsy, depression) return mean calibrated probabilities and the "
                    "decision. Window-level questions (seizure, eeg_event, sleep_stage) return how many windows "
                    "got each option, total time per option, the longest runs, and the highest-probability windows.",
     "input_schema": {"type": "object", "properties": {"question_id": _Q, "start_s": _T0, "end_s": _T1},
                      "required": ["question_id"], "additionalProperties": False}},
    {"name": "find_segments",
     "description": "Contiguous time segments where a window-level question gives a chosen option, either by the "
                    "model's decision (default) or by calibrated probability >= min_probability. Returns start, "
                    "end, duration, peak and mean probability per segment.",
     "input_schema": {"type": "object", "properties": {
         "question_id": _Q,
         "option": {"type": "string", "description": "One of the question's options, e.g. 'yes', 'PLED', 'N2'."},
         "min_probability": {"type": "number", "description": "Optional threshold on the calibrated probability (0-1)."},
         "min_duration_s": {"type": "number", "description": "Optional: drop segments shorter than this (seconds)."},
         "max_gap_s": {"type": "number", "description": "Optional: merge segments separated by at most this gap (seconds, default 0)."}},
         "required": ["question_id", "option"], "additionalProperties": False}},
    {"name": "get_timeline",
     "description": "Window-by-window output of one question in a time range: start time, decision, calibrated "
                    "probabilities, artefact flag. At most max_rows rows (evenly spaced if the range has more).",
     "input_schema": {"type": "object", "properties": {"question_id": _Q, "start_s": _T0, "end_s": _T1,
                                                       "max_rows": {"type": "integer", "description": "Default 40, at most 200."}},
                      "required": ["question_id"], "additionalProperties": False}},
    {"name": "epoch_results",
     "description": "Stimulus- or cue-locked questions (p300, motor_imagery): the model's answer in the window "
                    "starting at each onset of one annotation label. Without a label, lists the labels available.",
     "input_schema": {"type": "object", "properties": {"question_id": _Q,
                                                       "label": {"type": "string", "description": "Annotation label, e.g. 'T1'."}},
                      "required": ["question_id"], "additionalProperties": False}},
    {"name": "signal_stats",
     "description": "Descriptive signal statistics in a time range (NOT a model decision): per channel amplitude "
                    "(RMS, inter-quartile range in microvolts), relative band power (delta, theta, alpha, beta, "
                    "gamma), peak frequency 1-30 Hz; overall: posterior peak frequency, mains-noise share, and the "
                    "fraction of samples at extreme amplitude.",
     "input_schema": {"type": "object", "properties": {
         "start_s": _T0, "end_s": _T1,
         "channels": {"type": "array", "items": {"type": "string"}, "description": "Optional channel names (default: all)."}},
         "additionalProperties": False}},
    {"name": "view_signal",
     "description": "Plot the raw EEG traces of a short stretch (at most 30 s) so you and the user can see it. "
                    "Optionally shade the windows where a question's decision is not the 'nothing found' option. "
                    "Visual impressions are qualitative and never replace the model's calibrated output.",
     "input_schema": {"type": "object", "properties": {
         "start_s": {"type": "number", "description": "Start in seconds."},
         "duration_s": {"type": "number", "description": "Length in seconds (default 10, at most 30)."},
         "channels": {"type": "array", "items": {"type": "string"}, "description": "Optional channel names (default: up to 21)."},
         "highlight_question": {**_Q, "description": "Optional window-level question whose positive windows are shaded."}},
         "required": ["start_s"], "additionalProperties": False}},
    {"name": "view_timeline",
     "description": "Plot one window-level question over the whole recording (or a range): the calibrated "
                    "probability of each option per window and the decisions.",
     "input_schema": {"type": "object", "properties": {"question_id": _Q, "start_s": _T0, "end_s": _T1},
                      "required": ["question_id"], "additionalProperties": False}},
]
SCHEMAS = {t["name"]: t["input_schema"] for t in TOOLS}


class ToolError(Exception):
    pass


def validate(name, args):
    """Check a tool call against its schema (names, required keys, types). Raises ToolError."""
    if name not in SCHEMAS:
        raise ToolError(f"unknown tool {name!r}")
    if not isinstance(args, dict):
        raise ToolError("tool input must be a JSON object")
    sch = SCHEMAS[name]
    props = sch.get("properties", {})
    for k in sch.get("required", []):
        if k not in args:
            raise ToolError(f"missing required field {k!r}")
    py = {"string": str, "number": (int, float), "integer": int, "array": list, "object": dict, "boolean": bool}
    for k, v in args.items():
        if k not in props:
            raise ToolError(f"unknown field {k!r}; allowed: {sorted(props)}")
        want = props[k]["type"]
        if isinstance(v, bool) and want in ("number", "integer") or not isinstance(v, py[want]):
            raise ToolError(f"field {k!r} must be of type {want}")
        if want == "array" and not all(isinstance(i, str) for i in v):
            raise ToolError(f"field {k!r} must be a list of strings")


def _r(x, n=3):
    return None if x is None else round(float(x), n)


def _runs(mask, starts, win_s, max_gap_s=0.0):
    """Contiguous True runs of a per-window mask -> [(first index, last index)], merging small gaps."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    runs, a, b = [], idx[0], idx[0]
    for i in idx[1:]:
        if starts[i] - (starts[b] + win_s) <= max_gap_s + 1e-6:
            b = i
        else:
            runs.append((a, b)); a = b = i
    runs.append((a, b))
    return runs


class EEGTools:
    def __init__(self, analysis):
        self.a = analysis
        self.figures = []                 # PNG bytes produced in this session (the UI shows them)

    # ------------------------------------------------------------------ helpers
    def _card(self, q):
        cards = self.a.meta["cards"]
        if q not in cards:
            raise ToolError(f"unknown question_id {q!r}; known: {sorted(cards)}")
        return cards[q]

    def _continuous(self, q):
        card = self._card(q)
        st = self.a.meta["questions"].get(q, {})
        if st.get("abstain"):
            raise ToolError(f"the model abstained on {q!r}: {st['abstain']}")
        if card["locked"]:
            raise ToolError(f"{q!r} is stimulus/cue-locked: use epoch_results")
        return card

    def _range(self, starts, win_s, start_s, end_s):
        dur = self.a.meta["recording"]["duration_s"]
        t0 = 0.0 if start_s is None else max(0.0, float(start_s))
        t1 = dur if end_s is None else min(dur, float(end_s))
        if t1 <= t0:
            raise ToolError(f"empty time range [{t0}, {t1}] (recording is {dur:.1f} s)")
        m = (starts + win_s > t0) & (starts < t1)        # windows overlapping the range
        if not m.any():
            raise ToolError(f"no complete window of {win_s:g} s in [{t0}, {t1}]")
        return m, t0, t1

    def _channels(self, channels):
        used = self.a.meta["recording"].get("channels_used", [])
        if not channels:
            return list(range(len(used))), used
        up = [c.upper().strip() for c in channels]
        missing = [c for c in up if c not in used]
        if missing:
            raise ToolError(f"channels not in the recording: {missing}; available: {used}")
        return [used.index(c) for c in up], up

    # -------------------------------------------------------------------- tools
    def recording_info(self):
        m = self.a.meta
        r = m["recording"]
        labels = {}
        for an in m.get("annotations", []):
            d = labels.setdefault(an["label"], {"count": 0, "first_onset_s": an["onset_s"]})
            d["count"] += 1
        return {"file": r["name"], "duration_s": _r(r["duration_s"], 1),
                "analysed_duration_note": (f"only the first {r['duration_s']:.0f} s of {r['full_duration_s']:.0f} s were analysed"
                                           if r.get("full_duration_s", r["duration_s"]) > r["duration_s"] + 1 else None),
                "sampling_rate_hz": r["fs_original"], "model_rate_hz": 200,
                "channels_given": r["n_channels_given"], "channels_used": r.get("channels_used", []),
                "unit_factor_to_uV": r.get("unit_factor_to_uV"),
                "warnings": m["warnings"],
                "annotations": dict(list(labels.items())[:40]),
                "questions": {q: ("answered" if not v["abstain"] else f"abstained: {v['abstain']}")
                              for q, v in m["questions"].items()},
                "model": m["model"], "backbone": m["backbone"]}

    def list_questions(self, question_id=None):
        cards = self.a.meta["cards"]

        def extra(q, c):
            used = [*c["trained_on"], *c["external_tests"]]
            return {"datasets": {s: DATASETS[s] for s in used if s in DATASETS},
                    **({"known_issue": KNOWN_ISSUES[q]} if q in KNOWN_ISSUES else {})}
        if question_id:
            return {question_id: self._card(question_id) | extra(question_id, cards[question_id])}
        return {q: {k: c[k] for k in ("question", "options", "tier", "scope", "window_s", "locked", "trained_on",
                                      "external_tests", "note")} |
                   {"test_results": {s: {"role": v["role"], **v["window_level"],
                                         **({"subject_level_auroc": v["subject_level"].get("auroc")}
                                            if "subject_level" in v else {})}
                                     for s, v in c["test_results"].items()}} | extra(q, c)
                for q, c in cards.items()}

    def ask_model(self, question_id, start_s=None, end_s=None):
        card = self._continuous(question_id)
        starts, p, qb, d, art = self.a.get(question_id)
        win = card["window_s"]
        m, t0, t1 = self._range(starts, win, start_s, end_s)
        ok = m & ~art
        opts = card["options"]
        out = {"question": card["question"], "tier": card["tier"], "range_s": [_r(t0, 1), _r(t1, 1)],
               "windows": int(m.sum()), "window_s": win, "windows_skipped_for_artefact": int((m & art).sum())}
        if not ok.any():
            out["abstain"] = "every window in this range is dominated by artefact"
            return out
        if card["scope"] == "recording":
            out["probabilities"] = {o: _r(v, 4) for o, v in zip(opts, p[ok].mean(0))}
            out["decision"] = opts[int(np.argmax(qb[ok].mean(0)))]
            out["share_of_windows_per_decision"] = {o: _r(v, 4) for o, v in
                                                    zip(opts, np.bincount(d[ok], minlength=len(opts)) / ok.sum())}
            out["how_to_read"] = ("probabilities: mean calibrated probability over windows; decision: class-balanced "
                                  "rule (argmax of the mean balanced probability), as scored in testing")
            return out
        counts = np.bincount(d[ok], minlength=len(opts))
        out["per_option"] = {}
        for k, o in enumerate(opts):
            e = {"windows": int(counts[k]), "share": _r(counts[k] / ok.sum(), 4), "seconds": _r(counts[k] * win, 1)}
            if counts[k] and o != NEGATIVE.get(question_id):
                runs = sorted(_runs(ok & (d == k), starts, win), key=lambda ab: ab[0] - ab[1])[:5]
                e["longest_runs"] = [{"start_s": _r(starts[a], 1), "end_s": _r(starts[b] + win, 1)} for a, b in runs]
                top = np.argsort(-np.where(ok, p[:, k], -1))[:3]
                e["highest_probability_windows"] = [{"start_s": _r(starts[i], 1), "p": _r(p[i, k], 4)} for i in top]
            out["per_option"][o] = e
        out["how_to_read"] = ("windows are counted by the class-balanced decision; p is the calibrated probability "
                              "of that option in a window")
        return out

    def find_segments(self, question_id, option, min_probability=None, min_duration_s=0.0, max_gap_s=0.0):
        card = self._continuous(question_id)
        if option not in card["options"]:
            raise ToolError(f"{option!r} is not an option of {question_id}: {card['options']}")
        k = card["options"].index(option)
        starts, p, _, d, art = self.a.get(question_id)
        win = card["window_s"]
        mask = ~art & ((p[:, k] >= min_probability) if min_probability is not None else (d == k))
        segs = []
        for a, b in _runs(mask, starts, win, max_gap_s or 0.0):
            dur = starts[b] + win - starts[a]
            if dur + 1e-6 < (min_duration_s or 0.0):
                continue
            sel = slice(a, b + 1)
            segs.append({"start_s": _r(starts[a], 1), "end_s": _r(starts[b] + win, 1), "duration_s": _r(dur, 1),
                         "peak_p": _r(p[sel, k].max(), 4), "mean_p": _r(p[sel, k].mean(), 4)})
        return {"question_id": question_id, "option": option,
                "criterion": (f"calibrated p >= {min_probability}" if min_probability is not None else "model decision"),
                "n_segments": len(segs), "total_seconds": _r(sum(s["duration_s"] for s in segs), 1),
                "segments": segs[:50], "truncated": len(segs) > 50}

    def get_timeline(self, question_id, start_s=None, end_s=None, max_rows=40):
        card = self._continuous(question_id)
        starts, p, _, d, art = self.a.get(question_id)
        m, _, _ = self._range(starts, card["window_s"], start_s, end_s)
        idx = np.flatnonzero(m)
        n = min(max(int(max_rows or 40), 1), 200)
        note = None
        if len(idx) > n:
            note = f"{len(idx)} windows in range; showing {n} evenly spaced"
            idx = idx[np.linspace(0, len(idx) - 1, n).round().astype(int)]
        opts = card["options"]
        rows = [{"start_s": _r(starts[i], 1), "decision": opts[d[i]], "artefact": bool(art[i]),
                 "p": {o: _r(v, 3) for o, v in zip(opts, p[i])}} for i in idx]
        return {"question_id": question_id, "window_s": card["window_s"], "note": note, "rows": rows}

    def epoch_results(self, question_id, label=None):
        card = self._card(question_id)
        st = self.a.meta["questions"].get(question_id, {})
        if not card["locked"]:
            raise ToolError(f"{question_id!r} is not stimulus/cue-locked: use ask_model")
        if st.get("abstain"):
            raise ToolError(f"the model abstained on {question_id!r}: {st['abstain']}")
        labels = st.get("epochs_per_label", {})
        if label is None or label not in labels:
            return {"question_id": question_id, "labels_available": labels,
                    "note": None if label is None else f"label {label!r} not found"}
        onsets, p, qb, d, art = self.a.get(f"{question_id}@{label}")
        opts = card["options"]
        ok = ~art
        return {"question_id": question_id, "label": label, "epochs": int(len(onsets)),
                "epochs_skipped_for_artefact": int(art.sum()), "window_s": card["window_s"],
                "decisions": {o: int(v) for o, v in zip(opts, np.bincount(d[ok], minlength=len(opts)))},
                "mean_probabilities": {o: _r(v, 4) for o, v in zip(opts, p[ok].mean(0))} if ok.any() else None,
                "caution": ("windows start at every onset of this annotation label; the model cannot tell "
                            "whether these onsets are the stimuli or cues the question assumes"),
                "first_epochs": [{"onset_s": _r(onsets[i], 2), "decision": opts[d[i]],
                                  "p": {o: _r(v, 3) for o, v in zip(opts, p[i])}} for i in range(min(20, len(onsets)))]}

    def signal_stats(self, start_s=None, end_s=None, channels=None):
        x, fs = self.a.signal, self.a.fs
        if x.size == 0:
            raise ToolError("no usable signal in this recording")
        dur = x.shape[1] / fs
        t0 = 0.0 if start_s is None else max(0.0, float(start_s))
        t1 = dur if end_s is None else min(dur, float(end_s))
        if t1 - t0 < 2:
            raise ToolError("the range must be at least 2 s long")
        ci, names = self._channels(channels)
        seg = x[ci, int(t0 * fs):int(t1 * fs)].astype(np.float64)
        seg = seg - np.median(seg, axis=1, keepdims=True)
        f, P = _psd(seg, fs)
        tot = P[:, (f >= 1) & (f <= 45)].sum(1) + 1e-12
        rel = {b: P[:, (f >= lo) & (f < hi)].sum(1) / tot for b, (lo, hi) in BANDS.items()}
        lowf = (f >= 1) & (f <= 30)
        peak = f[lowf][np.argmax(P[:, lowf], axis=1)]
        q75, q25 = np.percentile(seg, [75, 25], axis=1)
        per = {n: {"rms_uV": _r(np.sqrt((seg[i] ** 2).mean()), 2), "iqr_uV": _r(q75[i] - q25[i], 2),
                   "relative_power": {b: _r(rel[b][i], 3) for b in BANDS}, "peak_hz_1_30": _r(peak[i], 2)}
               for i, n in enumerate(names)}
        post = [i for i, n in enumerate(names) if n in POSTERIOR]
        out = {"range_s": [_r(t0, 1), _r(t1, 1)], "per_channel": per,
               "note": "descriptive statistics of the signal, not a model decision"}
        if post:
            Pp = P[post].mean(0)
            a = (f >= 6) & (f <= 14)
            out["posterior_peak_hz_6_14"] = _r(f[a][np.argmax(Pp[a])], 2)
        for mains in (50, 60):
            band = (f >= mains - 1) & (f <= mains + 1)
            if band.any():
                out[f"mains_{mains}hz_share_of_power"] = _r(P[:, band].sum() / (P.sum() + 1e-12), 4)
        scale = self.a.meta["reference_scale_uV"]
        out["fraction_of_samples_beyond_model_clip"] = _r((np.abs(seg) >= 20 * scale).mean(), 4)
        return out

    def view_signal(self, start_s, duration_s=10.0, channels=None, highlight_question=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x, fs = self.a.signal, self.a.fs
        if x.size == 0:
            raise ToolError("no usable signal in this recording")
        dur = min(max(float(duration_s or 10.0), 1.0), 30.0)
        t0 = max(0.0, float(start_s))
        if t0 >= x.shape[1] / fs:
            raise ToolError("start_s is past the end of the recording")
        ci, names = self._channels(channels)
        ci, names = ci[:21], names[:21]
        seg = x[ci, int(t0 * fs):int((t0 + dur) * fs)].astype(np.float32)
        seg = seg - np.median(seg, axis=1, keepdims=True)
        t = t0 + np.arange(seg.shape[1]) / fs
        gap = max(float(np.percentile(np.abs(seg), 99)) * 2, 20.0)
        fig, ax = plt.subplots(figsize=(12, 0.45 * len(ci) + 1.2), dpi=90)
        if highlight_question:
            card = self._continuous(highlight_question)
            starts, _, _, d, _ = self.a.get(highlight_question)
            neg = card["options"].index(NEGATIVE[highlight_question]) if highlight_question in NEGATIVE else None
            for s, k in zip(starts, d):
                if s + card["window_s"] > t0 and s < t0 + dur and (neg is None or k != neg):
                    ax.axvspan(s, s + card["window_s"], color="tab:red", alpha=0.12, lw=0)
            ax.set_title(f"shaded: windows where '{highlight_question}' decision is not '{NEGATIVE.get(highlight_question)}'",
                         fontsize=9)
        for i in range(len(ci)):
            ax.plot(t, seg[i] - i * gap, lw=0.6, color="k")
        ax.set_yticks([-i * gap for i in range(len(ci))], names, fontsize=8)
        ax.set_xlabel("time (s)")
        ax.plot([t[-1] + 0.2] * 2, [0, -100], color="tab:blue", lw=2, clip_on=False)
        ax.text(t[-1] + 0.35, -50, "100 µV", fontsize=8, va="center", color="tab:blue")
        ax.set_xlim(t[0], t[-1] + 0.2)
        fig.tight_layout()
        info = {"shown": {"start_s": _r(t0, 1), "duration_s": _r(dur, 1), "channels": names,
                          "vertical_spacing_uV": _r(gap, 1)},
                "measurements_of_shown_stretch": self._measure(seg, names),
                "note": "the image is for the user; describe the stretch with these measurements, not visual impressions"}
        if highlight_question:
            opts = card["options"]
            info["model_decisions_in_view"] = [
                {"start_s": _r(s, 1), "end_s": _r(s + card["window_s"], 1), "decision": opts[k],
                 "p": {o: _r(v, 3) for o, v in zip(opts, pp)}}
                for s, k, pp in zip(starts, d, self.a.get(highlight_question)[1])
                if s + card["window_s"] > t0 and s < t0 + dur]
        return info, self._png(fig)

    def _measure(self, seg, names):
        """Per channel: peak-to-peak and RMS amplitude (uV), peak frequency 1-30 Hz."""
        f, P = _psd(seg.astype(np.float64), self.a.fs)
        lowf = (f >= 1) & (f <= 30)
        peak = f[lowf][np.argmax(P[:, lowf], axis=1)]
        return {n: {"peak_to_peak_uV": _r(np.ptp(seg[i]), 1), "rms_uV": _r(np.sqrt((seg[i].astype(np.float64) ** 2).mean()), 1),
                    "peak_hz": _r(peak[i], 1)} for i, n in enumerate(names)}

    def view_timeline(self, question_id, start_s=None, end_s=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        card = self._continuous(question_id)
        starts, p, _, d, art = self.a.get(question_id)
        win = card["window_s"]
        m, t0, t1 = self._range(starts, win, start_s, end_s)
        opts = card["options"]
        unit, div = ("min", 60.0) if t1 - t0 > 600 else ("s", 1.0)
        tt = starts[m] / div
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 4.2), dpi=90, sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
        show = [k for k, o in enumerate(opts) if o != NEGATIVE.get(question_id)]
        for k in show:
            a1.plot(tt, p[m, k], lw=0.9, label=opts[k])
        a1.set_ylim(0, 1); a1.set_ylabel("calibrated p"); a1.legend(fontsize=7, ncol=min(6, len(show)), loc="upper right")
        a1.set_title(card["question"], fontsize=10)
        a2.scatter(tt, d[m], s=4, c=np.where(art[m], "lightgrey", "k"))
        a2.set_yticks(range(len(opts)), opts, fontsize=7); a2.set_xlabel(f"time ({unit})"); a2.set_ylabel("decision")
        fig.tight_layout()
        return {"shown": {"question_id": question_id, "range_s": [_r(t0, 1), _r(t1, 1)], "windows": int(m.sum()),
                          "grey_dots": "windows skipped for artefact"}}, self._png(fig)

    # --------------------------------------------------------------- dispatch
    def _png(self, fig):
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        self.figures.append(buf.getvalue())
        return buf.getvalue()

    def run(self, name, args):
        """-> (tool_result content blocks, is_error). Never raises for bad calls."""
        try:
            validate(name, args)
            out = getattr(self, name)(**args)
        except ToolError as e:
            return [{"type": "text", "text": f"Error: {e}"}], True
        except Exception as e:                                       # a bug in a tool: report, do not crash the chat
            return [{"type": "text", "text": f"Error: tool failed ({type(e).__name__}: {e})"}], True
        if isinstance(out, tuple):
            info, png = out
            return [{"type": "text", "text": json.dumps(info)},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                 "data": base64.b64encode(png).decode()}}], False
        return [{"type": "text", "text": json.dumps(out)}], False


def _psd(x, fs, seg_s=2.0):
    """Welch PSD with 2 s Hann segments and 50% overlap. x [E, T] -> (f, P [E, F])."""
    n = int(seg_s * fs)
    if x.shape[1] < n:
        n = x.shape[1]
    w = np.hanning(n)
    step = n // 2
    segs = [x[:, i:i + n] for i in range(0, x.shape[1] - n + 1, step)]
    P = np.mean([np.abs(np.fft.rfft((s - s.mean(1, keepdims=True)) * w, axis=1)) ** 2 for s in segs], axis=0)
    return np.fft.rfftfreq(n, 1 / fs), P
