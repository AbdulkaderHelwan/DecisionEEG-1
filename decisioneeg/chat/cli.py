"""Terminal chat with one EEG recording.

    python -m decisioneeg.chat.cli --analysis analyses/rec1          # a saved analysis (no torch needed)
    python -m decisioneeg.chat.cli --edf recording.edf               # analyse first (needs the model files)

Default: a free local model through Ollama (`ollama serve`, `ollama pull qwen3:8b`); no API key,
no internet once the model is pulled. `--backend claude` uses the Claude API instead (needs
ANTHROPIC_API_KEY and internet). Plots are saved next to the analysis.
"""
from __future__ import annotations

import argparse
import base64
import sys
import time
import webbrowser
from pathlib import Path

from . import make_chat
from .analysis import Analysis, copy_annotations
from .report import write
from .tools import EEGTools


def load_analysis(a):
    if a.analysis:
        return Analysis.load(a.analysis), Path(a.analysis)
    from ..model import DecisionEEG
    kw = {"bundle_path": a.bundle} if a.bundle else {}
    m = DecisionEEG.load(device=a.device, **kw)
    print(f"analysing {a.edf} with {m.describe_backbone()} ...", flush=True)
    an = Analysis.from_edf(a.edf, m, a.max_seconds)
    out = an.save(Path(a.out) if a.out else Path(a.edf).parent / (Path(a.edf).stem + "_analysis"))
    copy_annotations(a.edf, out)
    print(f"analysis saved to {out}", flush=True)
    return an, out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--analysis")
    g.add_argument("--edf")
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--device", default=None, help="cuda or cpu for analysing an EDF (default: the GPU if there is one)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-seconds", type=float, default=3600)
    ap.add_argument("--backend", default="ollama", choices=["ollama", "claude"],
                    help="ollama: free local open model (default); claude: Claude API (paid)")
    ap.add_argument("--model", default=None, help="default: qwen3:8b (ollama) or claude-opus-5-5 (claude)")
    ap.add_argument("--ollama-url", default=None, help="default: $OLLAMA_HOST or http://127.0.0.1:11434")
    ap.add_argument("--think", action="store_true", help="ollama: let the model reason first (slower)")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--ask", default=None, help="one question, then exit")
    ap.add_argument("--no-report", action="store_true", help="do not write / open the visual report")
    ap.add_argument("--no-open", action="store_true", help="write the report without opening the browser")
    ap.add_argument("--truth", default=None, help="TUSZ .csv_bi seizure annotations to draw in the report")
    a = ap.parse_args()
    an, folder = load_analysis(a)
    tools = EEGTools(an)
    chat = make_chat(tools, a.backend, a.model, effort=a.effort, url=a.ollama_url, think=a.think)
    figdir = folder / "figures"
    turns, calls = [], []                       # the conversation, shown in the report

    def on_tool(name, args, content, err):
        shown = ', '.join(f'{k}={v}' for k, v in args.items()) if isinstance(args, dict) else str(args)
        print(f"\n  · {name}({shown}){'  [error]' if err else ''}", flush=True)
        entry = {"tool": name, "input": args, "error": err,
                 "output": next((c["text"] for c in content if c["type"] == "text"), "")}
        if any(c["type"] == "image" for c in content):
            figdir.mkdir(exist_ok=True)
            f = figdir / f"fig{len(tools.figures):02d}_{name}.png"
            f.write_bytes(tools.figures[-1])
            entry["png"] = "data:image/png;base64," + base64.b64encode(tools.figures[-1]).decode()
            print(f"    plot -> {f}" + ("" if a.no_report else " (also in the report)"), flush=True)
            if a.no_report and not a.no_open:
                webbrowser.open(f.resolve().as_uri())
        calls.append(entry)

    def report():
        rep = write(folder, a.truth, {"model": chat.model, "turns": turns}, an=an)
        return rep

    def turn(q):
        calls.clear()
        t0 = time.time()
        ans = chat.ask(q, on_text=lambda t: print(t, end="", flush=True), on_tool=on_tool)
        turns.append({"question": q, "tool_calls": list(calls), "answer": ans, "seconds": round(time.time() - t0, 1)})
        print("\n")
        if not a.no_report:
            report()
            print("  (report updated: refresh the browser page)\n", flush=True)

    if not a.no_report:                         # every answer as a picture, in the browser
        rep = report()
        print(f"visual report -> {rep}", flush=True)
        if not a.no_open:
            webbrowser.open(rep.resolve().as_uri())
    r = an.meta["recording"]
    print(f"Ask your EEG — {r['name']} ({r['duration_s']:.0f} s). Model: {an.meta['backbone']}. "
          f"Research use only. Ctrl-D to quit.\n")
    if a.ask:
        turn(a.ask)
        return
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q:
            turn(q)


if __name__ == "__main__":
    sys.exit(main())
