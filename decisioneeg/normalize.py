"""Input pipeline that can run on ANY recording at inference time.

The NOMAD-EEG paper scaled each dataset with statistics of its own training split. A
deployed model has no such statistics for a new hospital, so DecisionEEG-1 uses one fixed
input transform, identical in training and at inference:

  microvolts -> resample to 200 Hz -> per window and channel, subtract the median
  -> divide by ONE reference scale (the median channel IQR of the pre-training corpus,
     reference.json) -> clip at +-20.

Absolute amplitude is kept (it matters for seizures), and only the units must be right,
which `detect_units` checks. Channel names are cleaned and channels without a standard
10-20 / 10-05 position are dropped.
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

FS = 200
CLIP = 20.0


# ----------------------------------------------------------------------------- channels
@lru_cache(maxsize=1)
def _position_table():
    import mne
    table = set()
    for m in ("standard_1005", "standard_1020"):
        table.update(k.upper() for k in mne.channels.make_standard_montage(m).ch_names)
    return table


def clean_name(name: str) -> str:
    """'EEG Fp1-REF' -> 'FP1', 'Fc5.' -> 'FC5', 'EEG Fpz-Cz' -> 'FPZ-CZ'."""
    s = name.upper().strip().replace(".", "")
    s = re.sub(r"^EEG[\s_:]*", "", s)
    s = re.sub(r"[-_ ](REF|LE|AVG|AR|M1|M2|A1|A2)$", "", s)     # referential suffix
    return s.strip()


def has_position(name: str) -> bool:
    t = _position_table()
    if name in t:
        return True
    if "-" in name:
        a, b = name.split("-", 1)
        return a in t and b in t
    return False


def resolve_channels(names):
    """Return (indices kept, cleaned names, warnings)."""
    keep, clean, dropped = [], [], []
    for i, n in enumerate(names):
        up = n.upper()
        if re.match(r"^(ECG|EKG|EMG|EOG|RESP|PHOTIC|SPO2|PULSE)", up.replace("EEG ", "")):
            dropped.append(n); continue
        c = clean_name(n)
        if has_position(c) and c not in clean:
            keep.append(i); clean.append(c)
        else:
            dropped.append(n)
    warn = [f"dropped {len(dropped)} channel(s) without a standard scalp position: {', '.join(dropped[:8])}"
            + (" ..." if len(dropped) > 8 else "")] if dropped else []
    return keep, clean, warn


# -------------------------------------------------------------------------------- units
def detect_units(x: np.ndarray):
    """Factor that converts x to microvolts, judged from its typical amplitude.

    EEG has a per-channel inter-quartile range of roughly 2-500 uV. Returns (factor, warning,
    abstain_reason)."""
    flat = np.moveaxis(np.asarray(x, np.float64), -2, 0).reshape(x.shape[-2], -1)     # [E, all samples]
    q75, q25 = np.percentile(flat, [75, 25], axis=1)
    a = float(np.median(q75 - q25))
    if not np.isfinite(a) or a == 0:
        return 1.0, None, "signal is flat or not finite"
    if a < 1e-3:
        return 1e6, f"input looks like volts (typical IQR {a:.2g}); converted to microvolts", None
    if a < 0.5:
        return 1e3, f"input looks like millivolts (typical IQR {a:.2g}); converted to microvolts", None
    if a > 5e3:
        return 1.0, None, f"amplitude implausible for EEG in microvolts (typical IQR {a:.3g})"
    return 1.0, None, None


# --------------------------------------------------------------------------- transform
def resample(x: np.ndarray, fs: float) -> np.ndarray:
    from nomad_eeg import preprocess
    return x.astype(np.float32) if fs == FS else preprocess(x, fs)


def to_model_input(x_uv: np.ndarray, scale_uv: float) -> np.ndarray:
    """x_uv: [..., E, T] microvolts at 200 Hz -> model input (float32)."""
    x = np.asarray(x_uv, np.float32)
    z = (x - np.median(x, axis=-1, keepdims=True)) / np.float32(scale_uv)
    return np.clip(z, -CLIP, CLIP)


def flat_channels(x_uv: np.ndarray, tol_uv: float = 0.05):
    """Indices of channels whose IQR over all windows is below tol (dead electrodes)."""
    x = np.moveaxis(np.asarray(x_uv, np.float32), -2, 0).reshape(x_uv.shape[-2], -1)
    q75, q25 = np.percentile(x, [75, 25], axis=1)
    return np.where((q75 - q25) < tol_uv)[0]


def clipped_fraction(z: np.ndarray) -> np.ndarray:
    """Per window: fraction of samples at the clip limit (artefact indicator)."""
    return (np.abs(z) >= CLIP).reshape(z.shape[0], -1).mean(1)
