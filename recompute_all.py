# Copy this file to config.yaml and adapt it to your project.
# config.yaml is git-ignored so local paths never end up in the repository.
# Single-quoted strings keep Windows backslashes as they are.
#
# File patterns accept these placeholders:
#   {subject}     subject ID, e.g. sub-01
#   {num}         subject number only, e.g. 01
#   {task}        value of `task` below
#   {task_lower}  value of `task` in lower case

task: MyTask

# Inclusive subject number range. 1 and 20 give sub-01 ... sub-20.
subjects:
  first: 1
  last: 20

out_dir: 'path/to/output'

# ---------------------------------------------------------------------------
# Pipelines
# The keys pipe_a, pipe_b and pipe_c are fixed roles:
#   pipe_a  reference pipeline (ground truth in scenario A)
#   pipe_b  second pipeline (candidate in A and ground truth in scenario B)
#   pipe_c  third pipeline (candidate in A and B)
# `label` sets the name shown in figures.
#
# format: json or mat
# Sampling rate: give either `sampling_rate` (Hz) or `sampling_rate_field`
# (a field in the file that holds it).
#
# Landmarks: rpeaks is required for every pipeline. tpeaks and toffsets are
# optional. A landmark left out is simply not compared.
# Field paths use dots for nesting (e.g. results.peaks.r).
#   type: indices  the field holds sample indices (default)
#   type: equals   the field is a per-sample array and landmarks are the
#                  samples equal to `value`. Add `row` (0-based) for 2-D arrays.
# ---------------------------------------------------------------------------
pipelines:
  pipe_a:
    label: Pipeline A
    file: 'path/to/pipeline_a/{subject}/{subject}_task-{task}_landmarks.json'
    format: json
    sampling_rate: 1000
    landmarks:
      rpeaks:   {field: rpeaks, type: indices}
      tpeaks:   {field: tpeaks, type: indices}
      toffsets: {field: toffsets, type: indices}

  pipe_b:
    label: Pipeline B
    file: 'path/to/pipeline_b/{subject}_{task}_landmarks.mat'
    format: mat
    sampling_rate_field: fs
    landmarks:
      rpeaks:   {field: events, type: equals, row: 0, value: 1}
      tpeaks:   {field: phase, type: equals, value: 4}
      toffsets: {field: phase, type: equals, value: 2}

  pipe_c:
    label: Pipeline C
    file: 'path/to/pipeline_c/{subject}_{task_lower}_landmarks.mat'
    format: mat
    sampling_rate_field: fs
    landmarks:
      rpeaks:   {field: R_peaks, type: indices}
      toffsets: {field: T_offsets, type: indices}

# ---------------------------------------------------------------------------
# Per-subject summary tables (tab-separated). See README for the columns.
# ---------------------------------------------------------------------------
summary_tables:
  pipe_a_artifacts: 'path/to/pipeline_a_artifacts.tsv'
  pipe_a_hrv: 'path/to/pipeline_a_hrv.tsv'
  pipe_b_hrv: 'path/to/pipeline_b_hrv.tsv'
  pipe_c_hrv: 'path/to/pipeline_c_hrv.tsv'

# Unit of SDNN and RMSSD in each HRV table: s or ms.
# Tables in s are converted to ms before comparison.
hrv_time_units:
  pipe_a: ms
  pipe_b: ms
  pipe_c: ms

# ---------------------------------------------------------------------------
# Raw ECG for the signal quality indices (EDF or BDF).
# All files matching the pattern are concatenated in file name order.
# ECG = channel_pos - channel_neg. Leave channel_neg empty for a single channel.
# ---------------------------------------------------------------------------
raw_ecg:
  files: 'path/to/raw/{subject}/{subject}_{task}_block*.bdf'
  channel_pos: ECG1
  channel_neg: ECG2
  sampling_rate: 1000
