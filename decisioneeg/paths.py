"""Where everything lives.

Using the model needs only DATA, which holds the two model files (decisioneeg1.pt and
backbone_large_ema.pt; downloaded there on first use). The other paths are for re-training the
heads and point to preprocessed dataset caches; set them with environment variables.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("DECISIONEEG_DATA", str(ROOT / "data")))
BUNDLE = DATA / "decisioneeg1.pt"                 # calibrated heads, task catalogue, test results
BACKBONE_EXPORT = DATA / "backbone_large_ema.pt"  # NOMAD-EEG-Large EMA weights + config
CACHE = DATA / "cache"            # siena / bonn windows in microvolts (built by build/build_external.py)
FEATURES = DATA / "features"      # <source>/<split>_X.npy (float16), _y.npy, _subjects.npy
HEADS = DATA / "heads"            # <task>.pt
RESULTS = DATA / "results"        # <task>.json, robustness.json, speed.json
REFERENCE = DATA / "reference.json"

# a source checkout of NOMAD-EEG instead of `pip install nomad-eeg` (optional)
if os.environ.get("NOMAD_EEG_REPO") and os.environ["NOMAD_EEG_REPO"] not in sys.path:
    sys.path.insert(0, os.environ["NOMAD_EEG_REPO"])

# ---------------------------------------------------------------- re-training only
_TRAIN = Path(os.environ.get("DECISIONEEG_TRAIN_ROOT", str(ROOT / "datasets")))
V3 = Path(os.environ.get("DECISIONEEG_V3", str(_TRAIN / "nomad-eeg")))            # NOMAD-EEG caches: V3/cache/<dataset>
LARGE = Path(os.environ.get("DECISIONEEG_LARGE", str(_TRAIN / "nomad-eeg-large")))  # LARGE/pool/tuev/stats_shard*.json
# the backbone the heads are fitted on; every training step must use the same file
BACKBONE = Path(os.environ.get("DECISIONEEG_BACKBONE", str(BACKBONE_EXPORT)))
# per-corpus, per-channel robust statistics NOMAD-EEG-Large was trained with (microvolts, computed
# after per-window DC removal); TUEV's give the one deployable input scale
LARGE_TUEV_STATS = sorted((LARGE / "pool" / "tuev").glob("stats_shard*.json"))
BENCH = Path(os.environ.get("DECISIONEEG_BENCH", str(_TRAIN / "benchmark")))     # benchmark loaders and split files
LINAH_ORIG = str(_TRAIN / "raw")                                                   # raw dataset folder
LINAH = Path(os.environ.get("DECISIONEEG_LINAH", LINAH_ORIG))
TUEV_STATS = LINAH / "V3_pool_30s_200hz" / "tuev" / "stats.json"                  # undoes the TUEV cache scaling
RAW = LINAH / "external_raw"
