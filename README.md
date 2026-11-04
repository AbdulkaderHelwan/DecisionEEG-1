# DecisionEEG-1

**Calibrated decisions from any EEG recording.**

DecisionEEG-1 answers a fixed set of clinical and research questions about an EEG recording, with a
calibrated probability for every answer. It runs on the frozen
[NOMAD-EEG-Large](https://github.com/AbdulkaderHelwan/NOMAD-EEG) foundation model (67.6 M parameters),
so it reads any electrode montage with standard 10-20 names, at any sampling rate. When the input is
unusable it abstains and says why, instead of guessing.

```python
from decisioneeg import DecisionEEG

model = DecisionEEG.load()                         # downloads the model files once (~273 MB)
result = model.decide_edf("recording.edf")
result["decisions"]["abnormal"]                    # {'decision': 'abnormal', 'probabilities': {...}, ...}
result["decisions"]["seizure"]["timeline"][:3]     # one calibrated answer per 5 s window
```

An optional chat interface lets you ask about a recording in plain words, with a free local
language model that can only quote DecisionEEG-1's outputs ([below](#chat-with-a-recording)).

Research use only. Not a medical device.

## Questions

| Question | Options | Window | Tier |
|---|---|---|---|
| Is there a seizure in this window? | no, yes | 5 s | ready |
| Which EEG event is in the middle second? | spike/sharp wave, GPED, PLED, eye movement, artefact, background | 5 s | ready |
| Is this EEG abnormal? (per recording) | normal, abnormal | 5 s | ready |
| Which sleep stage is this epoch? | W, N1, N2, N3, REM | 30 s | ready |
| P300 response to the attended stimulus? (at stimulus onsets) | no, yes | 1.4 s | ready |
| Does this person have epilepsy? (per recording) | no, yes | 5 s | experimental |
| Major depressive disorder? (per recording) | no, yes | 5 s | experimental |
| Which movement is imagined? (at cue onsets) | left fist, right fist, both fists, feet | 4 s | experimental |

Window questions return a timeline over the recording; per-recording questions return one answer
from the mean over windows. P300 and motor imagery are answered only at stimulus or cue onsets
marked in the file.

## Test results

Held-out test splits (patients not used for training); Bonn was never trained on. AUROC for two
options, balanced accuracy otherwise; ECE is the calibration error of the stated probabilities.
Full table with 95% bootstrap intervals, robustness and speed: [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

| Question | Test set | Windows | Result | ECE |
|---|---|---|---|---|
| Seizure | TUSZ | 50,000 | AUROC 0.872 | 0.008 |
| Seizure | Siena | 18,729 | AUROC 0.860 | 0.004 |
| Seizure | Bonn (external, single channel) | 100 | AUROC 0.275 | 0.829 |
| EEG event | TUEV | 29,421 | balanced accuracy 0.546 | 0.034 |
| Abnormal | TUAB | 50,000 | AUROC 0.853 | 0.039 |
| Sleep stage | Sleep-EDF | 9,137 | balanced accuracy 0.720 | 0.066 |
| P300 | BCIAUT | 33,600 | AUROC 0.649 | 0.007 |
| Epilepsy | TUEP | 1,800 (30 patients) | AUROC 0.605 (0.644 per patient) | 0.083 |
| Depression | Mumtaz | 1,179 (10 people) | AUROC 0.936 (0.960 per person) | 0.157 |
| Motor imagery | EEGMMIDB | 1,803 | balanced accuracy 0.372 | 0.025 |

Speed: one hour of 19-electrode EEG, all questions, takes 58 s on one A100 GPU slice and about an
hour on 6 CPU threads.

## How it is built

1. **Input**: microvolts, resampled to 200 Hz, per-window and per-channel median removed, divided by
   one fixed reference scale (22.59 µV, the median channel IQR of TUEV in NOMAD-EEG-Large's
   pre-training data), clipped at ±20. The same transform is used in training and use, so it needs no
   statistics from the recording's site. Channels without a standard 10-20 / 10-05 position are dropped;
   the units are checked from the signal's amplitude.
2. **Backbone**: frozen NOMAD-EEG-Large; one pass per window length serves every question with that
   length. Features: 5 time windows × mean and std × 4 gated blocks × 512 = 20,480 per window.
3. **Heads**: one class-balanced L2 logistic regression per question, regularisation chosen on
   validation, then temperature and class-bias calibration on validation.
4. **Outputs**: calibrated probabilities (right at the validation class mix) and a class-balanced
   decision (the rule the test metrics score).
5. **Abstention**: no usable channel names, implausible units, flat signal, recording shorter than the
   window, or more than half of a window at the clip limit (artefact). Stimulus- or cue-locked
   questions abstain on continuous recordings without onsets.

## Known limits

- The seizure, abnormal and epilepsy questions were trained only on hospital recordings, mostly from
  Temple University Hospital. On some healthy volunteers recorded elsewhere they gave false positives
  (seizure decided in 100% of one EEGMMIDB recording and 42% of one Mumtaz control).
- The seizure question was trained on adults. In 5 Helsinki neonatal recordings it flagged a window
  in only 9 of the 101 seizures experts marked.
- It detects seizures within a window; it does not predict them.
- Results are window-level; event-level scoring (false alarms per 24 h) is not done yet.

## Installation

Python 3.9 or newer. A GPU is optional.

```bash
git clone https://github.com/AbdulkaderHelwan/DecisionEEG-1.git
cd DecisionEEG-1
pip install -e .              # installs NOMAD-EEG, torch, mne, scipy, scikit-learn, matplotlib
```

The model files (`decisioneeg1.pt`, 3 MB: calibrated heads and test results; `backbone_large_ema.pt`,
270 MB: NOMAD-EEG-Large) are downloaded into `data/` from the
[v1.0 release](https://github.com/AbdulkaderHelwan/DecisionEEG-1/releases/tag/v1.0) the first time the
model loads. Set `DECISIONEEG_DATA` to keep them elsewhere.

## Use

```python
from decisioneeg import DecisionEEG

m = DecisionEEG.load(device="cpu")               # or "cuda"
r = m.decide_edf("recording.edf", tasks=["seizure", "abnormal"], max_seconds=3600)
r["warnings"]                                     # dropped channels, unit conversion, artefact
r["decisions"]["seizure"]["summary"]              # share of windows per option
r["decisions"]["abnormal"]["probabilities"]       # calibrated

# arrays instead of a file: x is [channels, samples] in microvolts
r = m.decide(x, ["Fp1", "Fp2", "C3", "C4", "O1", "O2"], fs=256)
```

## Chat with a recording

The chat turns a plain-language question into calls to DecisionEEG-1's outputs (the seizure
timeline, events, sleep stages, per-recording answers, test results, signal measurements and plots)
and answers only with what those return. It also opens a visual report of the recording.

The default chat model is free and runs locally: install [Ollama](https://ollama.com), then

```bash
ollama pull qwen3:8b
python -m decisioneeg.chat.cli --edf recording.edf
```

The recording is analysed once and saved as `recording_analysis/`; the report is
`recording_analysis/report.html`. Next time, start from the saved analysis with
`--analysis recording_analysis`.

| Option | Effect |
|---|---|
| `--model qwen3:14b` | any Ollama model with tool calling |
| `--backend claude` | Claude API instead (needs `pip install -e .[claude]` and `ANTHROPIC_API_KEY`) |
| `--device cpu` | analyse on the CPU |
| `--max-seconds 7200` | analyse two hours (default: the first hour) |
| `--no-open` | do not open the browser |

A whole folder, with one report per recording and an index page:

```bash
python -m decisioneeg.chat.batch /path/to/edfs --out analyses/my_edfs
```

Browser chat: `pip install -e .[web]`, then `python -m decisioneeg.chat.web --analysis recording_analysis`.

Offline test of the chat layer (no model, no API): `python tests/test_chat_offline.py`.

## Re-training the heads

`build/`, `train/` and `eval/` are the pipeline that produced the released model. They read
preprocessed caches of the ten datasets (built with the NOMAD-EEG benchmark code; see
`decisioneeg/paths.py` for the environment variables) and run in order:

```bash
python build/build_external.py                   # reference scale; Siena and Bonn windows
python build/extract_features.py --source tusz    # frozen features, one run per dataset
python train/fit_heads.py                         # fit, calibrate and test every head
python train/assemble.py                          # bundle + backbone export
python eval/robustness.py && python eval/speed.py && python eval/report.py
```

## Citation

DecisionEEG-1 is built on NOMAD-EEG:

```bibtex
@article{helwan2026nomadeeg,
  title   = {{NOMAD-EEG}: Native Omni-Montage Adaptive Diffusion},
  author  = {Helwan, Abdulkader and Henni, Khadidja and Abou-Abbas, Lina},
  journal = {arXiv preprint arXiv:2610.35246},
  year    = {2026}
}
```

## License

Apache-2.0. The datasets used for training and testing keep their own licences.
