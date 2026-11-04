"""Every training dataset as microvolt windows at 200 Hz, with the paper's splits.

Most datasets are the NOMAD-EEG caches (diff-eeg-V3/cache/<name>), stored robust-scaled; the
scaling is undone with the median / IQR saved in their meta.json (values that were clipped at
+-20 IQR stay clipped). TUEV is undone with the pre-training corpus statistics (in volts).
Siena and Bonn are built in microvolts by build/build_external.py.

    src = Source("tusz")
    src.n("train"); src.uv("train", 0, 256)     # [256, 21, 1000] microvolts
"""
from __future__ import annotations

import json

import numpy as np

from .paths import CACHE, TUEV_STATS, V3

TUEV_CLASSES = ["spsw", "gped", "pled", "eyem", "artf", "bckg"]
CHANNELS_21 = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
               "F7", "F8", "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ"]
V3_CACHES = ("tusz", "tuab5s", "sleepedf", "tuep", "mumtaz", "eegmmidb")


def _subject_split(subjects, frac_val, frac_test, seed=0):
    """Split row indices by subject (no subject in two splits)."""
    ids = np.array(sorted(set(subjects)))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_te, n_va = int(round(frac_test * len(ids))), int(round(frac_val * len(ids)))
    te, va = set(ids[:n_te]), set(ids[n_te:n_te + n_va])
    s = np.asarray(subjects)
    return (np.where(~np.isin(s, list(te | va)))[0], np.where(np.isin(s, list(va)))[0],
            np.where(np.isin(s, list(te)))[0])


class Source:
    def __init__(self, name: str):
        self.name = name
        self.splits = {}                      # split -> (X memmap, row index or None, y, subjects)
        if name in V3_CACHES:
            root = V3 / "cache" / name
            meta = json.loads((root / "meta.json").read_text())
            self.channels, self.classes = meta["channels"], meta["classes"]
            self.med = np.asarray(meta["median_uV"], np.float32)
            self.iqr = np.asarray(meta["iqr_uV"], np.float32)
            for sp in ("train", "val", "test"):
                subj = root / f"{sp}_subjects.npy"
                self.splits[sp] = (np.load(root / f"{sp}_X.npy", mmap_mode="r"), None,
                                   np.load(root / f"{sp}_y.npy").astype(np.int64),
                                   np.load(subj).astype(str) if subj.exists() else None)
        elif name == "bciaut":
            root = V3 / "cache" / "bciaut"
            meta = json.loads((root / "meta.json").read_text())
            self.channels, self.classes = meta["channels"], meta["classes"]
            to_uv = float(meta.get("to_uv", 1.0))
            self.med = np.asarray(meta["median_uV"], np.float32) * to_uv
            self.iqr = np.asarray(meta["iqr_uV"], np.float32) * to_uv
            X = np.load(root / "all_X.npy", mmap_mode="r")
            y = np.load(root / "all_y.npy").astype(np.int64)
            subj = np.load(root / "all_subjects.npy").astype(str)
            # 15 participants: 9 train / 3 validation / 3 test, seeded (the paper used 5-fold CV)
            for sp, idx in zip(("train", "val", "test"), _subject_split(subj, 0.2, 0.2)):
                self.splits[sp] = (X, idx, y[idx], subj[idx])
        elif name == "tuev":
            root = V3 / "cache" / "tuev_events_200hz_f16"
            st = json.loads(TUEV_STATS.read_text())
            self.channels, self.classes = CHANNELS_21, TUEV_CLASSES
            self.med = np.asarray(st["median"], np.float32) * 1e6          # stats are in volts
            self.iqr = np.asarray(st["iqr"], np.float32) * 1e6
            Xtr = np.load(root / "train_X.npy", mmap_mode="r")
            ytr = np.load(root / "train_y.npy").astype(np.int64)
            str_ = np.load(root / "train_subjects.npy").astype(str)
            tr, va, _ = _subject_split(str_, 0.1, 0.0)                     # validation patients from train
            self.splits["train"] = (Xtr, tr, ytr[tr], str_[tr])
            self.splits["val"] = (Xtr, va, ytr[va], str_[va])
            self.splits["test"] = (np.load(root / "test_X.npy", mmap_mode="r"), None,
                                   np.load(root / "test_y.npy").astype(np.int64),
                                   np.load(root / "test_subjects.npy").astype(str))
        elif name in ("siena", "bonn"):
            root = CACHE / name
            meta = json.loads((root / "meta.json").read_text())
            self.channels, self.classes = meta["channels"], meta["classes"]
            self.med = np.zeros(len(self.channels), np.float32)               # stored in microvolts
            self.iqr = np.ones(len(self.channels), np.float32)
            for sp in ("train", "val", "test"):
                self.splits[sp] = (np.load(root / f"{sp}_X.npy", mmap_mode="r"), None,
                                   np.load(root / f"{sp}_y.npy").astype(np.int64), None)
        else:
            raise KeyError(name)
        assert len(self.med) == len(self.channels) == len(self.iqr), (name, len(self.med), len(self.channels))

    def n(self, split):
        return len(self.splits[split][2])

    def labels(self, split):
        return self.splits[split][2]

    def subjects(self, split):
        return self.splits[split][3]

    def uv(self, split, a, b):
        X, idx, _, _ = self.splits[split]
        rows = np.asarray(X[a:b] if idx is None else X[idx[a:b]], np.float32)     # idx is sorted
        return rows * self.iqr[:, None] + self.med[:, None]

    def uv_rows(self, split, rows):
        """Windows at positions `rows` (sorted) of the split."""
        X, idx, _, _ = self.splits[split]
        r = np.asarray(rows) if idx is None else idx[np.asarray(rows)]
        return np.asarray(X[r], np.float32) * self.iqr[:, None] + self.med[:, None]

    def stratified_rows(self, split, limit, seed=0):
        """Up to `limit` rows with every class represented (smoke tests)."""
        y = self.labels(split)
        K = int(y.max()) + 1
        rng = np.random.default_rng(seed)
        per = max(1, limit // K)
        rows = [rng.choice(np.where(y == c)[0], size=min(per, int((y == c).sum())), replace=False)
                for c in range(K) if (y == c).any()]
        return np.sort(np.concatenate(rows))
