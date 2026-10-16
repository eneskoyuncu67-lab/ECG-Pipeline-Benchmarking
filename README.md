# ECG Pipeline Comparison

A config-driven Python tool that compares three ECG preprocessing pipelines on the same recordings. It measures how well their cardiac landmarks and HRV metrics agree. It works with any project whose pipelines export landmarks as JSON or MAT files.

## Contents

- [What it computes](#what-it-computes)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Adapting it to your project](#adapting-it-to-your-project)
- [Outputs](#outputs)
- [Notes](#notes)
- [References](#references)

## What it computes

The three pipelines have fixed roles. You choose their display names in the config.

| Role key | Role |
|---|---|
| `bbsig` | Reference pipeline (ground truth in scenario A) |
| `lslh` | Second pipeline (candidate in A and ground truth in scenario B) |
| `cp` | Third pipeline (candidate in A and B) |

Supported landmarks are R-peaks, T-peaks and T-offsets. R-peaks are required for every pipeline. A landmark that a pipeline does not provide is skipped for that pipeline.

### 1. Peak-timing agreement (Jaccard and F1)

Candidate landmarks are matched one-to-one to reference landmarks. Matching is greedy. The closest pairs within the tolerance window are matched first. Unmatched candidates count as false positives. Unmatched references count as false negatives.

Two ground-truth scenarios are run:

- **Scenario A.** `lslh` and `cp` are compared against `bbsig`.
- **Scenario B.** `cp` is compared against `lslh`.

Both scenarios are computed at ±50 ms and ±70 ms. The output also includes the mean signed timing error (bias) and its SD for matched pairs.

### 2. Landmark timing precision (MAE and SD)

This is the continuous counterpart to Jaccard and F1. For pairs matched within ±50 ms it reports the mean absolute timing error. It also reports the within-subject SD of the absolute errors. `bbsig` is the reference.

### 3. Detection Error Rate vs. Detector Temporal Tolerance

This follows the approach of Heryan et al. [1]. The detector temporal tolerance (DTT) is swept from 20 to 160 ms in 20 ms steps. TP, FP and FN are pooled across subjects at each step. The tool then computes sensitivity, positive predictivity, detection error rate (DER), Jaccard and F1. This analysis uses R-peaks only.

### 4. Bland-Altman agreement of HRV metrics

Bland-Altman statistics [2] are computed for `n_peaks`, `MeanBPM`, `SDNN`, `RMSSD` and `LF_HF`. This covers bias, SD and 95% limits of agreement. All three pipeline pairs are compared.

### 5. Raw signal quality indices

These are computed on the raw ECG. This is the shared input of all three pipelines.

- **kSQI:** kurtosis of the signal [3]. The Pearson definition is used (not excess kurtosis).
- **pSQI:** power in 5 to 15 Hz divided by power in 5 to 40 Hz (Welch PSD).

## Installation

Python 3 is required.

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
pip install -r requirements.txt
```

## Quick start

```bash
cp config.example.yaml config.yaml    # then edit config.yaml
python recompute_all.py               # reads ./config.yaml
python recompute_all.py --config other.yaml
```

`config.yaml` is git-ignored. Your local paths stay out of the repository. Results are written to `out_dir`.

## Adapting it to your project

All project-specific settings live in `config.yaml`. The script itself needs no changes. `config.example.yaml` documents every option.

### File patterns

File paths accept these placeholders:

| Placeholder | Example |
|---|---|
| `{subject}` | `sub-09` |
| `{num}` | `09` |
| `{task}` | `MyTask` |
| `{task_lower}` | `mytask` |

Subjects are numbered from `subjects.first` to `subjects.last`. A subject with a missing file is reported and skipped.

### Landmark files

Each pipeline needs a `file` pattern and a `format` (`json` or `mat`). The sampling rate is either a fixed number (`sampling_rate`) or a field in the file (`sampling_rate_field`).

Each landmark points to a field. Nested fields use dots, e.g. `results.peaks.r`. Two field types are supported:

| `type` | The field holds | Extra keys |
|---|---|---|
| `indices` | Sample indices of the landmarks (NaNs are dropped) | none |
| `equals` | One value per sample. Landmarks are the samples equal to `value` | `value`, optional `row` (0-based) for 2-D arrays |

Example:

```yaml
pipelines:
  cp:
    label: Pipeline C
    file: 'data/pipeline_c/{subject}_{task_lower}_landmarks.mat'
    format: mat
    sampling_rate_field: fs
    landmarks:
      rpeaks:   {field: R_peaks, type: indices}
      toffsets: {field: T_offsets, type: indices}
```

### Summary tables

Four tab-separated tables provide per-subject HRV metrics.

| Config key | Required columns | Optional columns |
|---|---|---|
| `bbsig_artifacts` | `subject_id`, `task`, `n_peaks`, `pct_artifacts` | |
| `bbsig_hrv` | `subject_id`, `task`, `timestamp` | `MeanBPM`, `SDNN`, `RMSSD`, `LF_HF` |
| `lslh_hrv` | `subject_id`, `task` | `timestamp`, `n_peaks`, HRV metrics |
| `cp_hrv` | `subject_id`, `task`, `timestamp` | `n_peaks`, HRV metrics |

`subject_id` must use the form `sub-<NN>`. When a subject appears more than once the row with the latest `timestamp` is kept. For `bbsig_artifacts` the row with the lowest `pct_artifacts` is kept.

### Raw ECG

Raw recordings must be EDF or BDF. All files matching `raw_ecg.files` are concatenated in file name order. The ECG is `channel_pos` minus `channel_neg`. Leave `channel_neg` empty to use a single channel.

## Outputs

File names use the role keys (`BBSIG`, `LSLH`, `CP`). Figure titles and axis labels use your configured labels.

### Tables

| File | Content |
|---|---|
| `peak_agreement_<ref>_ground_truth_<task>_<tol>.tsv` | Per-subject TP, FP, FN, Jaccard, F1, timing bias and SD |
| `landmark_timing_precision_BBSIG_ground_truth_<task>.tsv` | Per-subject MAE and SD |
| `bland_altman_recomputed_<pair>.csv` | Bland-Altman statistics per metric |
| `raw_signal_quality_<task>.tsv` | Per-subject kSQI and pSQI |

### Figures

| File | Content |
|---|---|
| `jaccard_f1_<ref>_ground_truth_combined.png` | Jaccard and F1 at ±50 ms (top) and ±70 ms (bottom) |
| `f1_only_<ref>_ground_truth_50ms.png` | F1 at ±50 ms |
| `jaccard_only_<ref>_ground_truth_50ms.png` | Jaccard at ±50 ms |
| `landmark_timing_precision_BBSIG_ground_truth.png` | MAE and SD |
| `der_vs_dtt_combined.png` | DER vs DTT for both scenarios |
| `outcome_measures_vs_dtt_<ref>_ground_truth.png` | All five outcome measures vs DTT |
| `timing_error_<ref>_ground_truth*.png` | Timing bias and jitter at ±70 ms |
| `bland_altman_recomputed_<pair>.png` | Bland-Altman plots |
| `variable_comparison_by_pipeline.png` | Each HRV metric per pipeline |
| `raw_signal_quality_indices.png` | kSQI and pSQI per subject |

Figures use Times New Roman at 12 pt. Matplotlib falls back to a default font if it is not installed.

## Notes

- `SDNN` and `RMSSD` in the `bbsig_hrv` and `cp_hrv` tables are expected in seconds. They are converted to ms. The `lslh_hrv` table is used as it is, so give it in ms.
- Sensitivity, positive predictivity, DER, Jaccard and F1 are all derived from the same pooled TP, FP and FN. They are not independent pieces of evidence.
- Jaccard and F1 are linked by F1 = 2J / (1 + J).
- The pSQI band limits here (5 to 15 Hz over 5 to 40 Hz) differ from the spectral index in Li et al. [3], which uses 5 to 14 Hz over 5 to 50 Hz.

## Data

No participant data is included in this repository. The `.gitignore` blocks common data and output formats.

## References

[1] Heryan, K., Reklewski, W., Szaflarski, A., Ordowski, M., Augustyniak, P., & Miśkowicz, M. (2021). Sensitivity of QRS detection accuracy to detector temporal resolution. In *Computing in Cardiology 2021* (Vol. 48, pp. 1–4). Brno, Czech Republic. https://www.cinc.org/2021/Program/accepted/267.html

[2] Bland, J. M., & Altman, D. G. (1986). Statistical methods for assessing agreement between two methods of clinical measurement. *The Lancet, 327*(8476), 307–310. https://doi.org/10.1016/S0140-6736(86)90837-8

[3] Li, Q., Mark, R. G., & Clifford, G. D. (2008). Robust heart rate estimation from multiple asynchronous noisy sources using signal quality indices and a Kalman filter. *Physiological Measurement, 29*(1), 15–32. https://doi.org/10.1088/0967-3334/29/1/002
