"""The question catalogue: every decision DecisionEEG-1 can make.

Each task is a fixed question with fixed options (as in a multiple-choice decision model),
answered by one linear head on the frozen NOMAD-EEG-Large embedding, then calibrated.

  sources   datasets whose train/val/test splits train, select and test the head
  external  datasets used only as unseen test sets
  scope     "window": one answer per window (and a timeline over a recording)
            "recording": one answer per recording (mean over its windows)
  tier      "ready" or "experimental" (shown in every output)
  locked    True when the window must start at a stimulus or cue (P300, motor imagery): a
            continuous recording cut into fixed windows does not answer these questions
  paper     frozen-probe numbers of NOMAD-EEG BASE in the paper, for reference only: the paper used
            per-corpus scaling and one dataset per head, so they are not the same protocol.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Task:
    id: str
    question: str
    options: tuple
    sources: tuple
    window_s: float
    scope: str
    tier: str
    external: tuple = ()
    locked: bool = False
    paper: dict = field(default_factory=dict)
    note: str = ""


TASKS = {t.id: t for t in [
    Task("seizure", "Is there a seizure in this window?", ("no", "yes"),
         sources=("tusz", "siena"), external=("bonn",), window_s=5.0, scope="window", tier="ready",
         paper={"tusz": {"auroc": 0.866}, "siena": {"auroc": 0.882}, "bonn": {"auroc": 0.999}},
         note="Detection, not prediction: no pre-ictal signal was found in earlier tests."),
    Task("eeg_event", "Which EEG event is in the middle second of this window?",
         ("spike/sharp wave", "GPED", "PLED", "eye movement", "artefact", "background"),
         sources=("tuev",), window_s=5.0, scope="window", tier="ready",
         paper={"tuev": {"balanced_accuracy": 0.542}},
         note="TUEV was in NOMAD-EEG's (unlabelled) pre-training data."),
    Task("abnormal", "Is this EEG abnormal?", ("normal", "abnormal"),
         sources=("tuab5s",), window_s=5.0, scope="recording", tier="ready",
         paper={"tuab5s": {"auroc": 0.838}}),
    Task("sleep_stage", "Which sleep stage is this 30 s epoch?", ("W", "N1", "N2", "N3", "REM"),
         sources=("sleepedf",), window_s=30.0, scope="window", tier="ready",
         paper={"sleepedf": {"balanced_accuracy": 0.757}}),
    Task("p300", "Does this epoch contain a P300 response to the attended stimulus?", ("no", "yes"),
         sources=("bciaut",), window_s=1.4, scope="window", tier="ready", locked=True,
         paper={"bciaut": {"auroc": 0.718}},
         note="Paper number is 5-fold subject cross-validation; here one subject-disjoint split."),
    Task("epilepsy", "Does this person have epilepsy?", ("no", "yes"),
         sources=("tuep",), window_s=5.0, scope="recording", tier="experimental",
         paper={"tuep": {"auroc_patient": 0.679}},
         note="Paper number is 5-fold patient cross-validation; here one patient-disjoint split."),
    Task("depression", "Does this person have major depressive disorder?", ("no", "yes"),
         sources=("mumtaz",), window_s=5.0, scope="recording", tier="experimental",
         paper={"mumtaz": {"auroc": 0.942}},
         note="Only 10 test subjects."),
    Task("motor_imagery", "Which movement is being imagined?", ("left fist", "right fist", "both fists", "feet"),
         sources=("eegmmidb",), window_s=4.0, scope="window", tier="experimental", locked=True,
         paper={"eegmmidb": {"balanced_accuracy": 0.442}}),
]}

ALL_SOURCES = sorted({s for t in TASKS.values() for s in (*t.sources, *t.external)})
