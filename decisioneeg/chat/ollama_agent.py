"""The chat loop on a free, local open model served by Ollama (default qwen3:8b).

    ollama pull qwen3:8b          # once (~5 GB); then `ollama serve` runs in the background
    chat = OllamaChat(EEGTools(Analysis.load("analyses/rec1")))
    chat.ask("Is there a seizure in this recording?", on_text=print)

Same tools, same rules and the same ask() interface as the Claude version (agent.py), through
Ollama's /api/chat endpoint (standard library only, no extra package). Differences: the local
model does not see the plots (the user does; the model is told a plot was shown), and Ollama's
default context of a few thousand tokens is raised, since tool results need room.
"""
from __future__ import annotations

import json
import os
import urllib.request

from .agent import MAX_TOOL_ROUNDS, SYSTEM, _intro
from .tools import TOOLS, EEGTools

MODEL = "qwen3:8b"
_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")       # as the ollama CLI reads it
URL = _HOST if _HOST.startswith("http") else f"http://{_HOST}"
SYSTEM_LOCAL = SYSTEM + """

Tool use on this system: call a tool before stating any number about the recording, and copy numbers \
exactly from the tool output. You cannot see images: plots you request are shown to the user, and you \
receive only their text output. After view_signal, report the measurements and model decisions it returns; \
never describe waveforms, spikes or patterns, and never claim to see the plot."""


def _post_stream(url, body):
    """POST JSON to Ollama and yield the streamed JSON lines."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if line.strip():
                yield json.loads(line)


class OllamaChat:
    def __init__(self, tools: EEGTools, model=MODEL, url=None, think=False, num_ctx=32768, post=None):
        """post: optional function(url, body) -> iterator of response chunks (offline tests)."""
        self.tools, self.model, self.url = tools, model, (url or URL).rstrip("/")
        self.think, self.num_ctx = think, num_ctx
        self.messages = [{"role": "system", "content": SYSTEM_LOCAL}]
        self.trace = []
        self._post = post or _post_stream
        self.tool_defs = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                            "parameters": t["input_schema"]}} for t in TOOLS]

    def _call(self, on_text=None):
        body = {"model": self.model, "messages": self.messages, "tools": self.tool_defs, "stream": True,
                "think": self.think, "options": {"num_ctx": self.num_ctx}}
        content, calls, reason = "", [], None
        for ch in self._post(f"{self.url}/api/chat", body):
            if "error" in ch:
                raise RuntimeError(f"Ollama: {ch['error']}")
            m = ch.get("message", {})
            if m.get("content"):
                content += m["content"]
                if on_text:
                    on_text(m["content"])
            calls += m.get("tool_calls") or []
            if ch.get("done"):
                reason = ch.get("done_reason")
        return content, calls, reason

    def ask(self, question: str, on_text=None, on_tool=None) -> str:
        intro = _intro(self.tools) + "\n\n" if len(self.messages) == 1 else ""
        self.messages.append({"role": "user", "content": intro + question})
        texts = []
        for _ in range(MAX_TOOL_ROUNDS):
            content, calls, reason = self._call(on_text)
            msg = {"role": "assistant", "content": content}
            if calls:
                msg["tool_calls"] = calls
            self.messages.append(msg)
            if content.strip():
                texts.append(content)
            if not calls:
                if reason == "length":
                    texts.append("[answer cut off at the length limit]")
                return "\n".join(texts)
            for c in calls:
                f = c.get("function", {})
                name, args = f.get("name", ""), f.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        pass                                   # validate() reports it as an error
                out, err = self.tools.run(name, args)
                self.trace.append((name, args, err))
                if on_tool:
                    on_tool(name, args, out, err)
                text = "\n".join(b["text"] if b["type"] == "text" else "[a plot of this was shown to the user]"
                                 for b in out)
                self.messages.append({"role": "tool", "tool_name": name, "content": text})
        return "\n".join(texts + ["[stopped after too many tool calls]"])
