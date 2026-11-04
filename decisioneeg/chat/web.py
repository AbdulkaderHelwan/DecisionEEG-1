"""Browser chat: load a recording, see its timelines, ask questions.

    pip install -r requirements-chat.txt
    python -m decisioneeg.chat.web --analysis analyses/rec1        # or start empty and upload an EDF
    # open http://127.0.0.1:7860

Uploading an EDF runs the model locally (needs decisioneeg1.pt and backbone_large_ema.pt, see
README); a saved analysis folder needs neither. Default chat model: qwen3:8b through Ollama (free, local).
"""
from __future__ import annotations

import argparse
import queue
import tempfile
import threading
from pathlib import Path

from . import make_chat
from .analysis import Analysis
from .tools import EEGTools

TMP = Path(tempfile.mkdtemp(prefix="ask_eeg_"))
EXAMPLES = ["Give me an overview of this recording.",
            "Is there any seizure activity? Show me the strongest one.",
            "Is this EEG normal or abnormal, and how sure is the model?",
            "Which sleep stages does the model see, and for how long?",
            "What is the posterior dominant rhythm?",
            "What can you not tell me about this recording?"]


def build(analysis=None, backend="ollama", model=None, bundle=None, **chat_kw):
    import gradio as gr
    state = {"chat": None, "n_fig": 0}

    def start(an):
        tools = EEGTools(an)
        state["chat"], state["n_fig"] = make_chat(tools, backend, model, **chat_kw), 0
        r = an.meta["recording"]
        q = an.meta["questions"]
        card = (f"**{r['name']}** · {r['duration_s']:.0f} s · {len(r.get('channels_used', []))} channels used · "
                f"{an.meta['backbone']}\n\n"
                + "\n".join(f"- `{t}`: " + ("answered" if not v["abstain"] else f"abstained ({v['abstain']})")
                            for t, v in q.items())
                + ("\n\n**Warnings:** " + "; ".join(an.meta["warnings"]) if an.meta["warnings"] else ""))
        figs = []
        for t in ("seizure", "sleep_stage", "eeg_event"):
            if t in q and not q[t]["abstain"]:
                _, png = tools.view_timeline(t)
                f = TMP / f"timeline_{t}.png"
                f.write_bytes(png)
                figs.append(str(f))
        tools.figures.clear()
        return card, figs, []

    def load_edf(file):
        from ..model import DecisionEEG
        if file is None:
            return "No file.", [], []
        kw = {"bundle_path": bundle} if bundle else {}
        an = Analysis.from_edf(file if isinstance(file, str) else file.name, DecisionEEG.load(**kw), 3600)
        return start(an)

    def respond(msg, history):
        if not msg.strip():
            yield history, ""
            return
        if state["chat"] is None:
            yield history + [{"role": "user", "content": msg},
                             {"role": "assistant", "content": "Load a recording first."}], ""
            return
        history = history + [{"role": "user", "content": msg}, {"role": "assistant", "content": ""}]
        q, chat = queue.Queue(), state["chat"]

        def on_tool(name, args, content, err):
            q.put(("tool", name, err))

        def work():
            try:
                chat.ask(msg, on_text=lambda t: q.put(("text", t)), on_tool=on_tool)
            except Exception as e:                                   # API errors are shown in the chat
                q.put(("text", f"\n\n[error: {type(e).__name__}: {e}]"))
            q.put(("done",))

        threading.Thread(target=work, daemon=True).start()
        while True:
            ev = q.get()
            if ev[0] == "done":
                break
            if ev[0] == "text":
                history[-1]["content"] += ev[1]
            else:
                history[-1]["content"] += f"\n\n*· {ev[1]}{' (error)' if ev[2] else ''}*\n\n"
                figs = chat.tools.figures
                while state["n_fig"] < len(figs):                    # a new plot: show it as its own message
                    f = TMP / f"fig{state['n_fig']:03d}.png"
                    f.write_bytes(figs[state["n_fig"]])
                    state["n_fig"] += 1
                    history += [{"role": "assistant", "content": {"path": str(f)}},
                                {"role": "assistant", "content": ""}]
            yield history, ""
        yield [h for h in history if h["content"] != ""], ""

    with gr.Blocks(title="Ask your EEG") as demo:
        gr.Markdown("## Ask your EEG\nDecisionEEG-1 on NOMAD-EEG-Large; a language model answers only from the model's "
                    "calibrated outputs. Research use only; not a medical device.")
        with gr.Row():
            with gr.Column(scale=2):
                edf = gr.File(label="EEG recording (.edf)", file_types=[".edf"])
                card = gr.Markdown("No recording loaded.")
                gallery = gr.Gallery(label="Timelines", columns=1, height="auto")
            with gr.Column(scale=3):
                bot = gr.Chatbot(type="messages", height=620, label="Conversation")
                box = gr.Textbox(placeholder="Ask about this recording ...", show_label=False)
                gr.Examples(EXAMPLES, inputs=box)
        edf.upload(load_edf, edf, [card, gallery, bot])
        box.submit(respond, [box, bot], [bot, box])
        if analysis is not None:
            demo.load(lambda: start(analysis), None, [card, gallery, bot])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default=None)
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--backend", default="ollama", choices=["ollama", "claude"],
                    help="ollama: free local open model (default); claude: Claude API (paid)")
    ap.add_argument("--model", default=None, help="default: qwen3:8b (ollama) or claude-opus-5-5 (claude)")
    ap.add_argument("--ollama-url", default=None, help="default: $OLLAMA_HOST or http://127.0.0.1:11434")
    ap.add_argument("--think", action="store_true", help="ollama: let the model reason first (slower)")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    an = Analysis.load(a.analysis) if a.analysis else None
    build(an, a.backend, a.model, a.bundle, effort=a.effort, url=a.ollama_url, think=a.think).launch(server_port=a.port)


if __name__ == "__main__":
    main()
