"""Visual report of an analysed recording, in the design of the "Ask your EEG" walkthrough page.

    python -m decisioneeg.chat.report --analysis analyses/rec [--truth rec.csv_bi] [--open]

Writes <analysis>/report.html: one self-contained page with the whole-recording answers, a
timeline for every window-level question (seizure probability with flagged windows, event strips,
a hypnogram), seizure segments and the EEG around the strongest one, cue-locked results, signal
measurements, the test performance, and the conversation when the chat has run (cli.py rewrites
the page after every answer). Everything comes from the same tools the chat calls.

Expert seizure annotations are drawn when given with --truth or found in the analysis folder as
annotations.csv_bi (TUSZ format; analysis.py copies it there when it sits next to the EDF).
"""
from __future__ import annotations

import argparse
import base64
import json
import webbrowser
from pathlib import Path

import numpy as np

from .analysis import Analysis
from .tools import KNOWN_ISSUES, NEGATIVE, EEGTools, ToolError

TEMPLATE = Path(__file__).with_name("report_template.html")
TRUTH_FILE = "annotations.csv_bi"


def _png(b):
    return "data:image/png;base64," + base64.b64encode(b).decode()


# ------------------------------------------------------------------ expert annotations
def read_csv_bi(path):
    """TUSZ .csv_bi -> [(start_s, end_s)] of 'seiz' rows."""
    out = []
    for line in Path(path).read_text().splitlines():
        p = line.split(",")
        if len(p) >= 4 and p[3] == "seiz":
            out.append((float(p[1]), float(p[2])))
    return out


def truth_overlap(an, seiz):
    """Window-level comparison of the seizure decisions with annotated seizures. Annotations that
    start after the analysed part are listed separately (TUSZ sometimes annotates past the file end)."""
    starts, prob, _, d, art = an.get("seizure")
    win = float(an.meta["cards"]["seizure"]["window_s"])
    end = float(starts[-1] + win)
    outside = [(a, b) for a, b in seiz if a >= end]
    seiz = [(a, min(b, end)) for a, b in seiz if a < end]
    in_seiz = np.array([any(s < b and s + win > a for a, b in seiz) for s in starts])
    flag = (d == 1) & ~art
    events = []
    for a, b in seiz:
        m = (starts < b) & (starts + win > a)
        events.append({"start_s": a, "end_s": b, "windows": int(m.sum()), "windows_flagged": int((flag & m).sum()),
                       "peak_p": round(float(prob[m, 1].max()), 4) if m.any() else None})
    return {"events": events, "annotated_after_analysed_end": outside, "analysed_end_s": end,
            "windows_total": int(len(starts)), "windows_in_seizure": int(in_seiz.sum()),
            "windows_flagged": int(flag.sum()),
            "flagged_inside_seizure": int((flag & in_seiz).sum()),
            "flagged_outside_seizure": int((flag & ~in_seiz).sum()),
            "window_sensitivity": round(float((flag & in_seiz).sum() / max(in_seiz.sum(), 1)), 4),
            "window_specificity": round(float((~flag & ~in_seiz).sum() / max((~in_seiz).sum(), 1)), 4),
            "seizures_detected": int(sum(e["windows_flagged"] > 0 for e in events))}


# ------------------------------------------------------------------------------ data
def collect(an: Analysis, truth=None, chat=None) -> dict:
    t = EEGTools(an)
    info = t.recording_info()
    qs, cards = an.meta["questions"], an.meta["cards"]
    data = {"recording": info, "backbone": an.meta["backbone"], "recording_answers": [], "windows": {},
            "seizure_segments": [], "seizure_snippet": None, "locked": [], "signal": None,
            "tests": [], "known_issues": {q: s for q, s in KNOWN_ISSUES.items() if q in cards},
            "truth": None, "chat": chat}
    for q, c in cards.items():
        st = qs.get(q, {})
        base = {"id": q, "question": c["question"], "tier": c["tier"], "options": c["options"],
                "known_issue": KNOWN_ISSUES.get(q)}
        if c["locked"]:
            if st.get("abstain"):
                data["locked"].append({**base, "abstain": st["abstain"]})
            else:
                data["locked"].append({**base, "labels": [{"label": lab, **{k: v for k, v in t.epoch_results(q, lab).items()
                                                                           if k in ("epochs", "decisions")}}
                                                          for lab in st.get("epochs_per_label", {})]})
        elif c["scope"] == "recording":
            if st.get("abstain"):
                data["recording_answers"].append({**base, "abstain": st["abstain"]})
            else:
                r = t.ask_model(q)
                pos = c["options"][-1]
                data["recording_answers"].append({**base, "decision": r["decision"], "pos": pos,
                                                  "p_pos": r["probabilities"][pos], "abstain": None})
        else:
            if st.get("abstain"):
                data["windows"][q] = {**base, "abstain": st["abstain"]}
                continue
            starts, p, _, d, art = an.get(q)
            data["windows"][q] = {**base, "abstain": None, "window_s": c["window_s"], "negative": NEGATIVE.get(q),
                                  "start_s": starts.round(2).tolist(), "p": p.round(3).tolist(),
                                  "decision": d.tolist(), "artefact": art.astype(int).tolist(),
                                  "per_option": t.ask_model(q)["per_option"]}
    if "seizure" in data["windows"] and not data["windows"]["seizure"]["abstain"]:
        data["seizure_segments"] = t.find_segments("seizure", "yes", max_gap_s=5.0)["segments"]
        top = data["windows"]["seizure"]["per_option"]["yes"].get("highest_probability_windows")
        if top:
            _, png = t.view_signal(max(0.0, top[0]["start_s"] - 10), 25.0, highlight_question="seizure")
            data["seizure_snippet"] = {"top_s": top[0]["start_s"], "p": top[0]["p"], "png": _png(png)}
        if truth:
            seiz = read_csv_bi(truth)
            data["truth"] = {"file": Path(truth).name, **truth_overlap(an, seiz)}
            for g in data["seizure_segments"]:
                g["annotated"] = any(g["start_s"] < b and g["end_s"] > a for a, b in seiz)
    try:
        s = t.signal_stats()
        data["signal"] = {"posterior_peak_hz_6_14": s.get("posterior_peak_hz_6_14"),
                          "mains_50": s.get("mains_50hz_share_of_power"), "mains_60": s.get("mains_60hz_share_of_power"),
                          "clip": s["fraction_of_samples_beyond_model_clip"], "per_channel": s["per_channel"]}
    except ToolError:
        pass
    for q, c in cards.items():
        for src, v in c["test_results"].items():
            w = v["window_level"]
            key = "auroc" if "auroc" in w else "balanced_accuracy"
            data["tests"].append({"question": c["question"], "tier": c["tier"], "source": src,
                                  "external": src in c["external_tests"], "n": v["n_windows"],
                                  "metric": "AUROC" if key == "auroc" else "bal. acc.", "value": w[key],
                                  "ece": w.get("ece"), "bad": key == "auroc" and w[key] < 0.6})
    return data


def build(an: Analysis, truth=None, chat=None) -> str:
    data = json.dumps(collect(an, truth, chat)).replace("</", "<\\/")      # "</" never closes the script tag
    title = f"{an.meta['recording']['name']} · DecisionEEG-1"
    return TEMPLATE.read_text(encoding="utf-8").replace("__TITLE__", title.replace("<", "&lt;")).replace("__DATA__", data)


def write(folder, truth=None, chat=None, an=None) -> Path:
    folder = Path(folder)
    if truth is None and (folder / TRUTH_FILE).exists():
        truth = folder / TRUTH_FILE
    f = folder / "report.html"
    f.write_text(build(an or Analysis.load(folder), truth, chat), encoding="utf-8")
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--truth", default=None, help="TUSZ .csv_bi seizure annotations to draw (optional)")
    ap.add_argument("--open", action="store_true", help="open the report in the default browser")
    a = ap.parse_args()
    f = write(a.analysis, a.truth)
    print(f"report -> {f}")
    if a.open:
        webbrowser.open(f.resolve().as_uri())


if __name__ == "__main__":
    main()
