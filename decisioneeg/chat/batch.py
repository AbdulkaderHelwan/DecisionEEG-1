"""Analyse every EDF in a folder, one by one, and list them on one index page.

    python -m decisioneeg.chat.batch /path/to/edfs --out analyses/my_edfs [--max-seconds 3600]

For each EDF (searched recursively): one DecisionEEG-1 pass (analysis.py), saved to
<out>/<name>/ with its visual report (report.html; TUSZ .csv_bi annotations next to the EDF are
used). The model is loaded once. A file that cannot be read or analysed is listed with the
error and the batch carries on. <out>/index.html lists every recording with its main results and
the command to chat with it; <out>/summary.json holds the same rows. Files already analysed are
skipped unless --redo is given, so an interrupted batch can be restarted.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
import traceback
from pathlib import Path

from .analysis import Analysis, copy_annotations
from .report import write as write_report
from .tools import EEGTools


def _name(edf: Path, root: Path) -> str:
    rel = edf.relative_to(root).with_suffix("")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", "__".join(rel.parts)) or "recording"


def summarise(an: Analysis) -> dict:
    t = EEGTools(an)
    r = an.meta["recording"]
    row = {"file": r["name"], "duration_s": round(r["duration_s"], 1), "channels_used": len(r.get("channels_used", [])),
           "channels_given": r["n_channels_given"], "warnings": an.meta["warnings"], "answers": {}}
    for q, st in an.meta["questions"].items():
        c = an.meta["cards"][q]
        if st.get("abstain"):
            row["answers"][q] = {"abstain": st["abstain"]}
        elif c["locked"]:
            row["answers"][q] = {"epochs_per_label": st.get("epochs_per_label", {})}
        elif c["scope"] == "recording":
            a = t.ask_model(q)
            pos = c["options"][-1]
            row["answers"][q] = {"decision": a["decision"], f"p_{pos}": a["probabilities"][pos]}
        else:
            per = t.ask_model(q)["per_option"]
            row["answers"][q] = {o: round(v["seconds"] / 60, 2) for o, v in per.items() if v["windows"]}    # minutes
    return row


def index_html(rows, out: Path) -> str:
    def cell(r, q, key=None):
        a = r.get("answers", {}).get(q)
        if not a:
            return "<td class='m'>–</td>"
        if a.get("abstain"):
            return "<td class='m' title='" + html.escape(a["abstain"]) + "'>abstained</td>"
        if key:
            return f"<td>{a.get(key, 0):.1f} min</td>"
        p = next((v for k, v in a.items() if k.startswith("p_")), None)
        return f"<td><b>{html.escape(str(a['decision']))}</b> <span class='m'>{100 * p:.0f}%</span></td>"
    body = []
    for r in rows:
        if r.get("error"):
            body.append(f"<tr><td>{html.escape(r['edf'])}</td><td colspan=6 class='bad'>{html.escape(r['error'])}</td></tr>")
            continue
        link = f"{html.escape(r['folder'])}/report.html"
        body.append(f"<tr><td><a href='{link}'>{html.escape(r['file'])}</a><div class='m'>{html.escape(r['folder'])}</div></td>"
                    f"<td>{r['duration_s'] / 60:.1f} min</td><td>{r['channels_used']} / {r['channels_given']}</td>"
                    + cell(r, "seizure", "yes") + cell(r, "abnormal") + cell(r, "epilepsy")
                    + f"<td class='m'>{len(r['warnings'])}</td></tr>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(out.name)} · DecisionEEG-1</title><style>
:root {{ --bg:#f3f6f7; --panel:#fff; --ink:#13212a; --muted:#5a6b75; --rule:#d5dee2; --accent:#0d7480; --bad:#c03568; color-scheme: light }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0e1519; --panel:#151f25; --ink:#e2eaee; --muted:#93a5af; --rule:#2a3a43; --accent:#4cc2cc; --bad:#ef6f9d; color-scheme: dark }} }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 "IBM Plex Sans",system-ui,sans-serif; padding:32px max(16px,env(safe-area-inset-left)) 64px max(16px,env(safe-area-inset-right)) }}
main {{ max-width:1000px; margin:0 auto; display:grid; gap:16px }} h1 {{ margin:0; font-size:30px }} a {{ color:var(--accent) }}
.m {{ color:var(--muted); font-size:13px }} .bad {{ color:var(--bad) }} .s {{ overflow-x:auto }}
table {{ border-collapse:collapse; width:100%; font-size:14px; font-variant-numeric:tabular-nums; background:var(--panel) }}
th {{ text-align:left; color:var(--muted); border-bottom:1px solid var(--ink); padding:8px 12px 8px 8px }}
td {{ border-bottom:1px solid var(--rule); padding:8px 12px 8px 8px; vertical-align:top }}
code {{ font-family:ui-monospace,Menlo,monospace; font-size:13px }}
</style></head><body><main><h1>{html.escape(out.name)}</h1>
<p class="m">{sum(1 for r in rows if not r.get('error'))} recordings analysed by DecisionEEG-1, {sum(1 for r in rows if r.get('error'))} failed.
Open a file for its full report. Seizure and abnormal answers can be false alarms on healthy recordings from outside the training
datasets (see each report). Research use only.</p>
<div class="s"><table><tr><th>Recording</th><th>Length</th><th>Channels</th><th>Seizure flagged</th><th>Abnormal</th><th>Epilepsy</th><th>Warnings</th></tr>
{''.join(body)}</table></div>
<p>Chat with one recording: <code>python -m decisioneeg.chat.cli --analysis {html.escape(str(out))}/&lt;folder&gt;</code></p>
</main></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder with .edf files (searched recursively)")
    ap.add_argument("--out", required=True, help="where the analyses, reports and index.html go")
    ap.add_argument("--max-seconds", type=float, default=3600, help="analyse at most this much of each file")
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--redo", action="store_true", help="analyse files that already have an analysis again")
    a = ap.parse_args()
    root, out = Path(a.folder).resolve(), Path(a.out)
    edfs = sorted(p for p in root.rglob("*") if p.suffix.lower() == ".edf")
    if not edfs:
        raise SystemExit(f"no .edf files under {root}")
    out.mkdir(parents=True, exist_ok=True)
    from ..model import DecisionEEG                    # torch is needed only here
    m = DecisionEEG.load(device=a.device, **({"bundle_path": a.bundle} if a.bundle else {}))
    print(f"{len(edfs)} EDF files under {root} | {m.describe_backbone()} on {m.device}", flush=True)
    rows = []
    for i, edf in enumerate(edfs, 1):
        name = _name(edf, root)
        dest = out / name
        t0 = time.time()
        try:
            if (dest / "analysis.json").exists() and not a.redo:
                an = Analysis.load(dest)
                note = "already analysed"
            else:
                an = Analysis.from_edf(edf, m, a.max_seconds)
                an.save(dest)
                copy_annotations(edf, dest)
                note = f"{time.time() - t0:.1f} s"
            write_report(dest, an=an)
            row = {"edf": str(edf), "folder": name, **summarise(an)}
            sz = row["answers"].get("seizure", {})
            print(f"[{i}/{len(edfs)}] {edf.name}: {row['duration_s'] / 60:.1f} min, {row['channels_used']} channels, "
                  f"seizure {sz.get('yes', 0) if not sz.get('abstain') else 'abstained'} min | {note}", flush=True)
        except Exception as e:                          # one bad file must not stop the batch
            row = {"edf": str(edf), "folder": name, "error": f"{type(e).__name__}: {e}"}
            print(f"[{i}/{len(edfs)}] {edf.name}: FAILED {row['error']}", flush=True)
            traceback.print_exc()
        rows.append(row)
        (out / "summary.json").write_text(json.dumps(rows, indent=1))
        (out / "index.html").write_text(index_html(rows, out), encoding="utf-8")
    print(f"done: {sum(1 for r in rows if not r.get('error'))} analysed, {sum(1 for r in rows if r.get('error'))} failed "
          f"-> {out / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
