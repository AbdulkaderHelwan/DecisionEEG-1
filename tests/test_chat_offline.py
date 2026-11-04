#!/usr/bin/env python3
"""Offline tests of "Ask your EEG" (no API calls, no torch): tools, validation, and the agent loop
driven by a scripted stand-in for Claude.

    python tests/test_chat_offline.py                        # synthetic recording
    python tests/test_chat_offline.py --analysis DIR         # also run every tool on a real saved analysis
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decisioneeg.chat.agent import EEGChat                  # noqa: E402
from decisioneeg.chat.analysis import Analysis              # noqa: E402
from decisioneeg.chat.tools import TOOLS, EEGTools, ToolError, validate   # noqa: E402

CH = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]


def synthetic(dur=300):
    rng = np.random.default_rng(0)
    fs, n = 200, 200 * dur
    t = np.arange(n) / fs
    x = rng.normal(0, 8, (len(CH), n))
    for c in ("O1", "O2", "P3", "P4", "PZ"):                   # 10 Hz posterior rhythm
        x[CH.index(c)] += 30 * np.sin(2 * np.pi * 10 * t)
    cards = {
        "seizure": {"question": "Is there a seizure in this window?", "options": ["no", "yes"], "tier": "ready",
                    "scope": "window", "window_s": 5.0, "locked": False, "trained_on": ["tusz", "siena"],
                    "external_tests": ["bonn"], "note": "", "reference_base_paper": {},
                    "test_results": {"tusz": {"role": "held-out test split", "n_windows": 10, "class_counts": [5, 5],
                                              "window_level": {"auroc": 0.9}, "ci95": {},
                                              "subject_level": {"auroc": 0.95}}}},
        "abnormal": {"question": "Is this EEG abnormal?", "options": ["normal", "abnormal"], "tier": "ready",
                     "scope": "recording", "window_s": 5.0, "locked": False, "trained_on": ["tuab5s"],
                     "external_tests": [], "note": "", "reference_base_paper": {}, "test_results": {}},
        "sleep_stage": {"question": "Which sleep stage is this 30 s epoch?", "options": ["W", "N1", "N2", "N3", "REM"],
                        "tier": "ready", "scope": "window", "window_s": 30.0, "locked": False, "trained_on": ["sleepedf"],
                        "external_tests": [], "note": "", "reference_base_paper": {}, "test_results": {}},
        "motor_imagery": {"question": "Which movement is being imagined?", "locked": True, "tier": "experimental",
                          "options": ["left fist", "right fist", "both fists", "feet"], "scope": "window",
                          "window_s": 4.0, "trained_on": ["eegmmidb"], "external_tests": [], "note": "",
                          "reference_base_paper": {}, "test_results": {}},
    }
    arrays = {}

    def put(key, starts, K, pos=None):
        p = rng.dirichlet(np.ones(K) * 3, len(starts)).astype(np.float32)
        if pos is not None:
            p[:, 0], p[:, 1] = 0.95, 0.05
            p[pos, 0], p[pos, 1] = 0.3, 0.7
        arrays[f"{key}/start_s"] = np.asarray(starts, np.float32)
        arrays[f"{key}/p"], arrays[f"{key}/q"] = p, p
        arrays[f"{key}/decision"] = p.argmax(1).astype(np.int16)
        art = np.zeros(len(starts), bool)
        art[-1] = True
        arrays[f"{key}/artefact"] = art

    put("seizure", np.arange(60) * 5.0, 2, pos=np.arange(20, 26))     # a 30 s "seizure" at 100-130 s
    put("abnormal", np.arange(60) * 5.0, 2)
    put("sleep_stage", np.arange(10) * 30.0, 5)
    put("motor_imagery@T1", np.arange(5) * 20.0 + 3, 4)
    meta = {"model": "DecisionEEG-1", "backbone": "NOMAD-EEG-Large (test)", "reference_scale_uV": 20.0,
            "recording": {"name": "synthetic.edf", "fs_original": 256.0, "duration_s": float(dur),
                          "channels_given": CH, "n_channels_given": len(CH), "channels_used": CH, "unit_factor_to_uV": 1.0},
            "warnings": [], "annotations": [{"onset_s": 3.0 + 20 * i, "duration_s": 4.0, "label": "T1"} for i in range(5)],
            "cards": cards,
            "questions": {"seizure": {"abstain": None}, "abnormal": {"abstain": None}, "sleep_stage": {"abstain": None},
                          "motor_imagery": {"abstain": None, "kind": "epochs", "epochs_per_label": {"T1": 5}}}}
    return Analysis(meta, arrays, x.astype(np.float16))


def check_tools(tools, synthetic_checks=True):
    ok = lambda name, args: tools.run(name, args)                          # noqa: E731
    out, err = ok("recording_info", {})
    assert not err, out
    info = json.loads(out[0]["text"])
    print("recording_info:", {k: info[k] for k in ("file", "duration_s", "questions")})
    out, err = ok("list_questions", {})
    assert not err, out
    for q, st in tools.a.meta["questions"].items():
        if st["abstain"]:
            continue
        card = tools.a.meta["cards"][q]
        if card["locked"]:
            out, err = ok("epoch_results", {"question_id": q})
            assert not err, out
            labels = json.loads(out[0]["text"])["labels_available"]
            for lab in list(labels)[:2]:
                out, err = ok("epoch_results", {"question_id": q, "label": lab})
                assert not err, out
                print(f"epoch_results {q}@{lab}:", json.loads(out[0]["text"])["decisions"])
            continue
        out, err = ok("ask_model", {"question_id": q})
        assert not err, out
        r = json.loads(out[0]["text"])
        print(f"ask_model {q}:", r.get("decision") or {o: v["windows"] for o, v in r["per_option"].items()})
        if card["scope"] == "window":
            for name, args in [("find_segments", {"question_id": q, "option": card["options"][-1]}),
                               ("find_segments", {"question_id": q, "option": card["options"][0], "min_probability": 0.5,
                                                  "min_duration_s": 10, "max_gap_s": 5}),
                               ("get_timeline", {"question_id": q, "start_s": 0, "end_s": 120, "max_rows": 7}),
                               ("view_timeline", {"question_id": q})]:
                out, err = ok(name, args)
                assert not err, (name, out)
    out, err = ok("signal_stats", {"start_s": 10, "end_s": 70})
    assert not err, out
    st = json.loads(out[0]["text"])
    print("signal_stats: posterior peak", st.get("posterior_peak_hz_6_14"), "Hz")
    out, err = ok("view_signal", {"start_s": 95, "duration_s": 20, "highlight_question": "seizure"})
    assert not err and out[1]["type"] == "image", out
    if synthetic_checks:
        assert abs(st["posterior_peak_hz_6_14"] - 10) < 0.6, st["posterior_peak_hz_6_14"]
        seg = json.loads(ok("find_segments", {"question_id": "seizure", "option": "yes"})[0][0]["text"])
        assert seg["segments"][0]["start_s"] == 100 and seg["segments"][0]["end_s"] == 130, seg
        tl = json.loads(ok("get_timeline", {"question_id": "seizure", "max_rows": 7})[0][0]["text"])
        assert len(tl["rows"]) == 7 and tl["note"], tl
    # bad calls come back as errors, never exceptions
    for name, args in [("ask_model", {}), ("ask_model", {"question_id": "nope"}), ("ask_model", {"question_id": 3}),
                       ("ask_model", {"question_id": "seizure", "extra": 1}), ("nope", {}),
                       ("find_segments", {"question_id": "seizure", "option": "maybe"}),
                       ("signal_stats", {"channels": ["XX9"]}), ("ask_model", {"question_id": "seizure", "start_s": 1e6}),
                       ("view_signal", {"start_s": True})]:
        out, err = tools.run(name, args)
        assert err and out[0]["text"].startswith("Error:"), (name, args, out)
    print(f"tools ok ({len(tools.figures)} plots)")


def tool_use(i, name, args):
    return NS(type="tool_use", id=f"tu_{i}", name=name, input=args)


def text(s):
    return NS(type="text", text=s)


def check_agent(tools):
    script = [
        NS(stop_reason="tool_use", content=[text("Let me look."), tool_use(1, "recording_info", {}),
                                            tool_use(2, "ask_model", {"question_id": "seizure"})]),
        NS(stop_reason="tool_use", content=[tool_use(3, "view_signal", {"start_s": 100, "duration_s": 10}),
                                            tool_use(4, "ask_model", {"question_id": "bogus"})]),
        NS(stop_reason="end_turn", content=[text("One seizure-like run at 100-130 s.")]),
        NS(stop_reason="refusal", content=[]),
        NS(stop_reason="max_tokens", content=[tool_use(5, "ask_model", {"question_id": "sei"})]),
        NS(stop_reason="end_turn", content=[text("Done.")]),
    ]
    seen = []

    def fake(messages, on_text=None):
        seen.append(json.loads(json.dumps(messages, default=lambda o: o.__dict__)))
        return script[len(seen) - 1]

    chat = EEGChat(tools, call=fake)
    ans = chat.ask("Is there a seizure?")
    assert "100-130" in ans, ans
    assert [t[0] for t in chat.trace] == ["recording_info", "ask_model", "view_signal", "ask_model"], chat.trace
    assert chat.trace[-1][2] is True                                    # the bogus id came back as an error
    m = chat.messages
    assert m[0]["role"] == "user" and m[0]["content"][0]["text"].startswith("[Recording loaded")
    res = m[2]["content"]                                               # both results of round 1 in ONE message
    assert [r["tool_use_id"] for r in res] == ["tu_1", "tu_2"]
    img = m[4]["content"][0]["content"]
    assert img[1]["type"] == "image" and m[4]["content"][1].get("is_error") is True
    n = len(chat.messages)
    ans = chat.ask("Tell me something you must refuse.")
    assert "declined" in ans and len(chat.messages) == n, (ans, len(chat.messages))
    ans = chat.ask("Again.")
    assert ans == "Done." and chat.messages[n + 2]["content"][0]["is_error"], chat.messages[n + 2]
    assert len(seen[1]) == 3 and not chat.messages[-1]["role"] == "user"
    tools_ok = all(t.get("eager_input_streaming") for t in chat.tool_defs) and len(chat.tool_defs) == len(TOOLS)
    assert tools_ok
    for t in TOOLS:                                                      # every schema is self-consistent
        validate(t["name"], {k: ("x" if v["type"] == "string" else 1 if v["type"] in ("number", "integer") else ["O1"])
                             for k, v in t["input_schema"]["properties"].items()
                             if k in t["input_schema"].get("required", [])})
    try:
        validate("ask_model", {"question_id": "seizure", "zzz": 1})
        raise AssertionError("unknown field accepted")
    except ToolError:
        pass
    print("agent loop ok")


def check_ollama(tools):
    from decisioneeg.chat import make_chat
    script = [
        [{"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "find_segments", "arguments": {"question_id": "seizure", "option": "yes"}}},
            {"function": {"name": "view_signal", "arguments": {"start_s": 100}}}]}, "done": False},
         {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"}],
        [{"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "ask_model", "arguments": "{\"question_id\": 7"}}]}, "done": True}],
        [{"message": {"role": "assistant", "content": "Seizure-like "}, "done": False},
         {"message": {"role": "assistant", "content": "run at 100-130 s."}, "done": True, "done_reason": "stop"}],
    ]
    bodies = []

    def fake_post(url, body):
        assert url.endswith("/api/chat") and body["options"]["num_ctx"] >= 16384 and body["stream"]
        bodies.append(json.loads(json.dumps(body)))
        return iter(script[len(bodies) - 1])

    chat = make_chat(tools, "ollama")
    chat._post = fake_post
    streamed = []
    ans = chat.ask("Is there a seizure?", on_text=streamed.append)
    assert ans == "Seizure-like run at 100-130 s." and "".join(streamed) == ans, ans
    assert [t[0] for t in chat.trace] == ["find_segments", "view_signal", "ask_model"] and chat.trace[2][2], chat.trace
    tool_msgs = [m for m in chat.messages if m["role"] == "tool"]
    assert '"start_s": 100.0' in tool_msgs[0]["content"] and "plot of this was shown" in tool_msgs[1]["content"]
    assert tool_msgs[2]["content"].startswith("Error:")
    assert bodies[0]["messages"][0]["role"] == "system" and bodies[0]["messages"][1]["content"].startswith("[Recording")
    assert {t["function"]["name"] for t in bodies[0]["tools"]} == {t["name"] for t in TOOLS}
    assert len(tools.figures) == 1
    print("ollama loop ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default=None)
    a = ap.parse_args()
    tools = EEGTools(synthetic())
    check_tools(tools)
    check_agent(EEGTools(synthetic()))
    check_ollama(EEGTools(synthetic()))
    if a.analysis:
        print(f"\n--- real analysis {a.analysis}")
        check_tools(EEGTools(Analysis.load(a.analysis)), synthetic_checks=False)
    print("ALL CHAT TESTS PASSED")


if __name__ == "__main__":
    main()
