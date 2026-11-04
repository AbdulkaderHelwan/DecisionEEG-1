"""Ask your EEG: a chat interface over DecisionEEG-1.

analysis.py  one model pass over a recording, saved to a folder (needs torch + backbone)
tools.py     what the chat model can call: calibrated model outputs, descriptive statistics, plots
agent.py     the Claude tool-use loop (paid API)
ollama_agent.py  the same loop on a free local open model via Ollama (default)
cli.py, web.py  terminal and browser front ends
"""


def make_chat(tools, backend="ollama", model=None, **kw):
    """backend "ollama": free local open model (default qwen3:8b); "claude": Claude API (paid)."""
    if backend == "ollama":
        from .ollama_agent import MODEL, OllamaChat
        return OllamaChat(tools, model=model or MODEL, **{k: v for k, v in kw.items() if k in ("url", "think", "num_ctx")})
    if backend == "claude":
        from .agent import MODEL, EEGChat
        return EEGChat(tools, model=model or MODEL, **{k: v for k, v in kw.items() if k in ("effort",)})
    raise ValueError(f"unknown backend {backend!r}")
