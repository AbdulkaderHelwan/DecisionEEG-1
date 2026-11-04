"""The chat loop: Claude answers questions about one EEG recording by calling EEGTools.

    chat = EEGChat(EEGTools(Analysis.load("analyses/rec1")))
    chat.ask("Is there a seizure in this recording?", on_text=print)

Claude never sees the raw signal as numbers; it sees what the tools return (calibrated model
outputs, descriptive statistics, plots), so every claim it makes can be traced to a tool call.
Each ask() runs the manual tool-use loop until Claude stops calling tools. The history keeps
every assistant turn unchanged (thinking and tool-use blocks included) and the whole request
prefix is cached, so follow-up questions are cheap.
"""
from __future__ import annotations

import json

from .tools import TOOLS, EEGTools

MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TOOL_ROUNDS = 12

SYSTEM = """You are "Ask your EEG", the conversational interface of DecisionEEG-1, a research model that \
answers a fixed set of questions about an EEG recording with calibrated probabilities. DecisionEEG-1 is a \
frozen NOMAD-EEG-Large foundation model with one calibrated linear head per question, trained and tested on \
public EEG datasets. One recording is loaded per conversation; the user may be a clinician, a researcher or a \
curious non-specialist, so match their level.

How to answer
- Everything you say about this recording must come from your tools. Never invent or estimate a probability, \
time, count or measurement; quote tool outputs (rounded sensibly) and say which question or statistic they come from.
- Start a conversation with recording_info. Use list_questions to know what the model can answer and how well \
it did on held-out test data; mention the tier and the relevant test performance when it matters for trust \
(for example, an "experimental" question, or a dataset unlike the training data).
- Map the user's question onto the model's questions. If none fits, say plainly that DecisionEEG-1 was not \
trained to answer it, say what it can answer instead, and offer descriptive signal statistics where they help, \
labelled as descriptive and not as a model decision.
- Two numbers come out of each question: the calibrated probability (fitted on the validation split of the \
training datasets, so it reflects their class mix) and the class-balanced decision (the rule scored in testing, \
which treats every option as equally likely). For rare events such as seizures a window can be flagged by the \
decision while its calibrated probability is modest; explain this when it comes up instead of hiding either number.
- Window-level questions give a timeline; find_segments and get_timeline locate events, view_timeline and \
view_signal show them. Use the plots when a picture helps the user, not by default. What you see in a plot is \
a qualitative impression and never overrides the model's calibrated output.
- When the model abstained (unknown channel names, implausible units, flat signal, too short, heavy \
artefact, or a cue-locked question on a continuous recording), pass the reason on and suggest what would fix it.
- When a result is poor or surprising (a weak test score, a confident answer on unusual data), say so and \
give the dataset facts and known issues that list_questions returns; do not guess other causes.
- Seizure detection is detection within a window, not prediction: earlier work found no usable pre-ictal \
signal, so do not suggest the model can forecast seizures.

Limits
DecisionEEG-1 is a research tool, not a medical device. It does not diagnose, and its outputs must not drive \
treatment. When a user asks for a diagnosis or what to do clinically, give the model's findings with their \
uncertainty and recommend review by a qualified clinician or EEG specialist. Keep answers short and concrete; \
use a small table when comparing several numbers."""


def _intro(tools: EEGTools) -> str:
    r = tools.a.meta["recording"]
    return (f"[Recording loaded: {r['name']}, {r['duration_s']:.0f} s, {len(r.get('channels_used', []))} usable "
            f"channels. Model: {tools.a.meta['model']} on {tools.a.meta['backbone']}.]")


class EEGChat:
    def __init__(self, tools: EEGTools, model=MODEL, effort="medium", client=None, call=None,
                 fallbacks=True, max_tokens=16000):
        """call: optional function(messages, on_text) -> message, replacing the API call (offline tests)."""
        self.tools, self.model, self.effort = tools, model, effort
        self.fallbacks, self.max_tokens = fallbacks, max_tokens
        self.messages = []
        self.trace = []                     # (tool name, input, is_error) per call, for the UI and tests
        self._call = call or self._api_call
        self._client = client
        # eager input streaming: tool inputs stream as generated; tools.validate() checks them before use
        self.tool_defs = [{**t, "eager_input_streaming": True} for t in TOOLS]

    # ------------------------------------------------------------------ API
    def _api_call(self, messages, on_text=None):
        import anthropic
        if self._client is None:
            self._client = anthropic.Anthropic()
        kw = dict(model=self.model, max_tokens=self.max_tokens, system=SYSTEM, tools=self.tool_defs,
                  messages=messages, thinking={"type": "adaptive"},
                  output_config={"effort": self.effort}, cache_control={"type": "ephemeral"})
        if self.fallbacks:
            kw.update(betas=[FALLBACK_BETA], fallbacks="default")
        for attempt in range(2):
            try:
                with self._client.beta.messages.stream(**kw) as stream:
                    for text in stream.text_stream:
                        if on_text:
                            on_text(text)
                    return stream.get_final_message()
            except ValueError:              # a tool input that is not parseable JSON: ask again once
                if attempt:
                    raise

    # ----------------------------------------------------------------- loop
    def ask(self, question: str, on_text=None, on_tool=None) -> str:
        """Run one user turn to completion; returns the final text. on_text(str) gets streamed text,
        on_tool(name, input, content, is_error) each tool call (content may hold a PNG)."""
        content = [{"type": "text", "text": question}]
        if not self.messages:
            content.insert(0, {"type": "text", "text": _intro(self.tools)})
        turn_start = len(self.messages)
        self.messages.append({"role": "user", "content": content})
        texts = []
        for _ in range(MAX_TOOL_ROUNDS):
            msg = self._call(self.messages, on_text)
            if msg.stop_reason == "refusal":       # drop this whole turn from the history (tail only)
                del self.messages[turn_start:]
                return "\n".join(texts + ["[The model declined to answer this request.]"])
            self.messages.append({"role": "assistant", "content": msg.content})
            texts += [b.text for b in msg.content if b.type == "text"]
            if msg.stop_reason == "pause_turn":
                continue
            calls = [b for b in msg.content if b.type == "tool_use"]
            if msg.stop_reason == "max_tokens":
                if calls:                    # a truncated tool call: do not run it
                    self.messages.append({"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                         "content": "Error: the tool input was cut off; call it again with a shorter input."}
                        for b in calls]})
                    continue
                return "\n".join(texts + ["[answer cut off at the length limit]"])
            if not calls:
                return "\n".join(texts)
            results = []
            for b in calls:
                args = b.input if isinstance(b.input, dict) else _parse(b.input)
                out, err = self.tools.run(b.name, args)
                self.trace.append((b.name, args, err))
                if on_tool:
                    on_tool(b.name, args, out, err)
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out, **({"is_error": True} if err else {})})
            self.messages.append({"role": "user", "content": results})    # all results in one message
        return "\n".join(texts + ["[stopped after too many tool calls]"])


def _parse(x):
    try:
        return json.loads(x)
    except Exception:
        return x
