# DecisionEEG-1 model card

Calibrated answers to a fixed set of EEG questions, from any electrode montage, on the frozen NOMAD-EEG-Large backbone (67.6 M parameters, pre-training step 16,000; released as backbone_large_ema.pt) with one linear head per question. Research use only; not a medical device.

Input transform: microvolts, 200 Hz, per-window median removed, divided by 22.59 uV (median over channels of the per-channel IQR of TUEV (NOMAD-EEG-Large pool, uV)), clipped at +-20.

## Questions and test results

Per-subject rows (mean probability per person) are given for the two per-person questions only.
† The backbone's unlabelled pre-training included recordings from this corpus (no labels were used).

Primary metric: AUROC (2 options) or balanced accuracy. Decisions are class-balanced; probabilities are calibrated on validation (temperature + class bias). ECE: expected calibration error (lower is better). Paper: NOMAD-EEG BASE frozen probe in the paper (different backbone and protocol: per-dataset scaling, one dataset per head).

| Question | Tier | Test set | n | Primary [95% CI] | Bal. acc. | Macro F1 | ECE before → after | Paper |
|---|---|---|---|---|---|---|---|---|
| Is there a seizure in this window? | ready | tusz† | 50,000 | 0.872 [0.865, 0.878] | 0.782 | 0.557 | 0.109 → 0.008 | auroc 0.866 |
| Is there a seizure in this window? | ready | siena | 18,729 | 0.860 [0.820, 0.898] | 0.654 | 0.607 | 0.007 → 0.004 | auroc 0.882 |
| Is there a seizure in this window? | ready | bonn (external) | 100 | 0.275 [0.158, 0.385] | 0.425 | 0.145 | 0.828 → 0.829 | auroc 0.999 |
| Which EEG event is in the middle second of this window? | ready | tuev† | 29,421 | 0.546 [0.533, 0.558] | 0.546 | 0.456 | 0.068 → 0.034 | balanced_accuracy 0.542 |
| Is this EEG abnormal? | ready | tuab5s† | 50,000 | 0.853 [0.849, 0.856] | 0.777 | 0.778 | 0.041 → 0.039 | auroc 0.838 |
| Which sleep stage is this 30 s epoch? | ready | sleepedf† | 9,137 | 0.720 [0.710, 0.730] | 0.720 | 0.706 | 0.055 → 0.066 | balanced_accuracy 0.757 |
| Does this epoch contain a P300 response to the attended stimulus? | ready | bciaut | 33,600 | 0.649 [0.640, 0.658] | 0.606 | 0.535 | 0.035 → 0.007 | auroc 0.718 |
| Does this person have epilepsy? | experimental | tuep† | 1,800 | 0.605 [0.580, 0.632] | 0.583 | 0.582 | 0.080 → 0.083 | auroc_patient 0.679 |
| ↳ per subject | | tuep† | 30 subjects | 0.644 | 0.600 | 0.600 | 0.052 | |
| Does this person have major depressive disorder? | experimental | mumtaz | 1,179 | 0.936 [0.923, 0.949] | 0.835 | 0.835 | 0.165 → 0.157 | auroc 0.942 |
| ↳ per subject | | mumtaz | 10 subjects | 0.960 | 0.800 | 0.800 | 0.156 | |
| Which movement is being imagined? | experimental | eegmmidb | 1,803 | 0.372 [0.351, 0.395] | 0.372 | 0.372 | 0.125 → 0.025 | balanced_accuracy 0.442 |

## Robustness: share of decisions that change

| Question | drop25_s0 | drop25_s1 | drop25_s2 | drop50_s0 | drop50_s1 | drop50_s2 | fs256 | fs500 | gain0.8 | gain1.25 | line50 | noise5uV | reorder | volts | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seizure | 24.1% | 57.0% | 24.0% | 66.2% | 62.2% | 23.7% | 3.9% | 4.7% | 12.8% | 12.7% | 24.5% | 28.9% | 3.2% | 0.1% | 24.9% |
| eeg_event | 15.0% | 23.5% | 11.9% | 53.0% | 42.3% | 27.9% | 7.7% | 7.8% | 11.3% | 12.7% | 74.0% | 78.8% | 0.4% | 0.0% | 26.2% |
| abnormal | 17.9% | 9.3% | 46.6% | 17.3% | 28.3% | 38.4% | 3.7% | 3.8% | 6.2% | 5.9% | 28.2% | 19.1% | 1.2% | 0.0% | 16.1% |
| sleep_stage | - | - | - | 82.6% | 62.4% | - | 2.7% | 3.1% | 18.2% | 15.5% | 90.6% | 64.0% | 0.0% | 0.0% | 33.9% |
| p300 | 25.9% | 17.2% | 14.3% | 23.9% | 23.4% | 27.2% | 49.6% | 10.0% | 8.2% | 8.8% | 66.7% | 33.9% | 3.7% | 0.0% | 22.4% |
| epilepsy | 2.7% | 2.7% | 2.9% | 6.6% | 6.8% | 4.6% | 6.4% | 6.3% | 10.4% | 12.8% | 23.5% | 21.5% | 0.0% | 0.0% | 7.7% |
| depression | 0.8% | 7.0% | 1.3% | 3.2% | 3.7% | 6.3% | 0.0% | 0.0% | 6.1% | 1.4% | 51.0% | 48.9% | 0.0% | 0.0% | 9.3% |
| motor_imagery | 19.5% | 20.0% | 19.0% | 43.6% | 33.4% | 43.0% | 6.3% | 8.7% | 16.5% | 16.4% | 35.7% | 61.6% | 3.5% | 0.2% | 23.4% |

## Speed (all questions, one hour of EEG)

| Setting | Seconds |
|---|---|
| gpu 19 electrodes | 57.75 |
| gpu 64 electrodes | 185.2 |
| cpu6 19 electrodes | 3619.7 |

## Limits

- Answers only the questions above; each was trained on one or two public datasets.
- Seizure detection, not prediction. Window-level metrics; event-level scoring (sensitivity, false alarms per 24 h) is not yet done.
- Probabilities are calibrated at the class mix of the validation data; at a different prevalence, rank (AUROC) holds but stated probabilities shift.
- Experimental tier: small test sets or modest accuracy; do not rely on them.

## Spot checks outside the test sets

| Recording | What happened |
|---|---|
| 30-min TUSZ evaluation recording (aaaaasoc_s001_t000, 8 annotated seizures) | every seizure had flagged windows; 66% of seizure windows flagged, 94% of the others left unflagged |
| 5 Helsinki neonatal recordings (101 seizures marked by at least 2 of 3 experts) | only 9 seizures had a flagged window; at most 1.7% of seizure windows flagged per recording. The seizure question was trained on adults |
| EEGMMIDB S001R04, healthy volunteer | seizure decided in 100% of windows; abnormal with probability 1.0 |
| Mumtaz healthy control S1 (eyes closed) | seizure decided in 42% of windows; epilepsy with probability 0.84 |

The seizure, abnormal and epilepsy questions were trained only on hospital recordings, mostly from
Temple University Hospital. Treat their answers on research or non-hospital recordings, and on
newborns, with caution.
