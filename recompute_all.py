"""
Recompute Bland-Altman + Jaccard/F1 across the three ECG/HRV pipelines
(BBSIG, LSLH, CP), using whatever data currently exists
on disk. Reports N explicitly per metric and skips any metric a pipeline
doesn't actually report (rather than silently comparing a column to itself).

Two ground-truth scenarios for peak-timing agreement (Jaccard/F1):
  (A) BBSIG as ground truth -- LSLH vs BBSIG, CP vs BBSIG (R-peaks, T-peaks*, T-offsets)
      * CP has no T-peak marker, so CP T-peaks is always N/A.
  (B) LSLH as ground truth -- CP vs LSLH (R-peaks, T-offsets only; no T-peaks,
      since CP never produces a T-peak marker regardless of reference)

Bland-Altman agreement of aggregate HRV metrics is computed for all three
pipeline pairs (BBSIG vs LSLH, BBSIG vs CP, LSLH vs CP).

Every Jaccard/F1 scatter dot is labeled with its subject ID (e.g. "09").
All figures are saved as PNGs to the output directory (none are shown on screen).

Paths, file layouts and labels are read from a YAML config file (default:
config.yaml next to this script). See config.example.yaml and README.md.

Usage:
    python recompute_all.py                    # uses ./config.yaml
    python recompute_all.py --config my.yaml
"""
import os, json, bisect, glob, string, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless backend: figures are only saved, never displayed
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import kurtosis as sp_kurtosis
from scipy.signal import welch
import pyedflib
import yaml

# ---------------- config ----------------
_parser = argparse.ArgumentParser(description='Compare ECG/HRV pipelines (BBSIG, LSLH, CP).')
_parser.add_argument('--config', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml'),
                     help='Path to YAML config file (default: config.yaml next to this script)')
_args = _parser.parse_args()
if not os.path.exists(_args.config):
    raise SystemExit(f'Config file not found: {_args.config}\n'
                     f'Copy config.example.yaml to config.yaml and fill in your paths.')
with open(_args.config) as _f:
    cfg = yaml.safe_load(_f)

# ---------------- settings from config ----------------
# The three pipelines have fixed internal roles. The keys 'bbsig', 'lslh' and
# 'cp' name those roles in the config and in output file names. Display
# labels are free to choose (config: pipelines.<key>.label).
#   bbsig: reference pipeline (ground truth in scenario A)
#   lslh:  second pipeline (candidate in A, ground truth in scenario B)
#   cp:    third pipeline (candidate in A and B)
PIPES = ('bbsig', 'lslh', 'cp')
pipe_cfg = {p: cfg['pipelines'][p] for p in PIPES}

out_dir = cfg['out_dir']
os.makedirs(out_dir, exist_ok=True)

task = cfg['task']
subjects = [f'sub-{n:02d}' for n in range(cfg['subjects']['first'], cfg['subjects']['last'] + 1)]

landmark_labels = {'rpeaks': 'R-peak', 'tpeaks': 'T-peak', 'toffsets': 'T-offset'}
LANDMARK_ORDER = ['rpeaks', 'toffsets', 'tpeaks']  # panel order in figures
pipeline_labels = {p: pipe_cfg[p].get('label', p.upper()) for p in PIPES}

import re as _re
_ROLE_TOKENS = {'BBSIG': 'bbsig', 'LSLH': 'lslh', 'CP': 'cp'}
def relabel(text):
    """Swap the role names used in figure titles for the configured labels."""
    return _re.sub(r'\b(BBSIG|LSLH|CP)\b', lambda m: pipeline_labels[_ROLE_TOKENS[m.group(1)]], text)

def fill_pattern(pattern, subj_id):
    """Fill {subject}, {num}, {task} and {task_lower} placeholders."""
    return pattern.format(subject=subj_id, num=subj_id.split('-')[1],
                          task=task, task_lower=task.lower())

# ================= LOADERS =================
def _unwrap(x):
    if isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        return x.flat[0]
    return x

def get_field(container, path):
    """Resolve a dotted field path ('a.b.c') in a JSON dict or a loaded
    .mat struct (attribute access)."""
    cur = container
    for part in path.split('.'):
        cur = _unwrap(cur)
        cur = cur[part] if isinstance(cur, dict) else getattr(cur, part)
    return _unwrap(cur)

def _read_file(fpath, fmt):
    if fmt == 'json':
        with open(fpath) as f:
            return json.load(f)
    if fmt == 'mat':
        return loadmat(fpath, squeeze_me=True, struct_as_record=False)
    raise ValueError(f'Unknown file format "{fmt}" (use json or mat)')

def _extract_landmark(data, spec):
    """Return landmark sample indices from one field, following its spec.
      type: indices   -> the field holds sample indices (NaNs dropped)
      type: equals    -> the field is a per-sample array; landmarks are the
                         samples equal to `value` (optional `row` for 2-D)"""
    arr = np.asarray(get_field(data, spec['field']), dtype=float)
    kind = spec.get('type', 'indices')
    if kind == 'indices':
        return arr[~np.isnan(arr)]
    if kind == 'equals':
        if 'row' in spec:
            arr = arr[spec['row'], :]
        return np.where(arr == spec['value'])[0].astype(float)
    raise ValueError(f'Unknown landmark type "{kind}" (use indices or equals)')

def load_landmarks(pipe, subj_id):
    """Load one pipeline's landmarks (in seconds) for one subject, or None
    if its file is missing."""
    c = pipe_cfg[pipe]
    fpath = fill_pattern(c['file'], subj_id)
    if not os.path.exists(fpath):
        return None
    data = _read_file(fpath, c['format'])
    if 'sampling_rate_field' in c:
        fs = float(get_field(data, c['sampling_rate_field']))
    else:
        fs = float(c['sampling_rate'])
    return {lm: _extract_landmark(data, spec) / fs for lm, spec in c['landmarks'].items()}

# Load every pipeline's landmarks once per subject, cache for reuse across
# both ground-truth scenarios (avoids re-reading .mat/.json files twice).
landmarks_by_subject = {}
for subj_id in subjects:
    landmarks_by_subject[subj_id] = {p: load_landmarks(p, subj_id) for p in PIPES}

# Which landmarks each comparison covers follows from the config: a pipeline
# is compared on a landmark only if both it and the reference provide it.
def _landmark_pipes(ref, candidates):
    out = {}
    for lm in LANDMARK_ORDER:
        if lm not in pipe_cfg[ref]['landmarks']:
            continue
        pipes = [p for p in candidates if lm in pipe_cfg[p]['landmarks']]
        if pipes:
            out[lm] = pipes
    return out

# ================= MATCHING =================
def greedy_match(candidate, reference, tol):
    """Returns TP, FP, FN, and the signed timing errors (candidate - reference,
    in seconds) for every matched (TP) pair -- the continuous counterpart to
    the binary TP/FP/FN counts."""
    a = np.sort(np.asarray(candidate, dtype=float))
    b = np.sort(np.asarray(reference, dtype=float))
    b_list = b.tolist()
    cands = []
    for i, ta in enumerate(a):
        lo = bisect.bisect_left(b_list, ta - tol)
        hi = bisect.bisect_right(b_list, ta + tol)
        for j in range(lo, hi):
            cands.append((abs(ta - b[j]), i, j))
    cands.sort(key=lambda x: x[0])
    matched_a, matched_b = set(), set()
    matched_pairs = []
    for _, i, j in cands:
        if i not in matched_a and j not in matched_b:
            matched_a.add(i); matched_b.add(j)
            matched_pairs.append((i, j))
    TP = len(matched_a)
    diffs = np.array([a[i] - b[j] for i, j in matched_pairs])
    return TP, len(a) - TP, len(b) - TP, diffs

def jacc(TP, FP, FN):
    d = TP + FP + FN
    return TP / d if d > 0 else np.nan

def f1(TP, FP, FN):
    d = 2*TP + FP + FN
    return 2*TP / d if d > 0 else np.nan

def compute_peak_agreement(ref_name, landmark_pipes, tol_s):
    """ref_name: 'bbsig' or 'lslh'. landmark_pipes: {landmark: [candidate_names]}."""
    rows = []
    for subj_id in subjects:
        ref_landmarks = landmarks_by_subject[subj_id][ref_name]
        if ref_landmarks is None:
            print(f'SKIP {subj_id}: no {ref_name} reference data')
            continue
        row = {'subject_id': subj_id, 'task': task}
        for landmark, pipes in landmark_pipes.items():
            ref = ref_landmarks[landmark]
            row[f'n_{ref_name}_{landmark}'] = len(ref)
            for pipe in pipes:
                cd = landmarks_by_subject[subj_id][pipe]
                if cd is None or landmark not in cd:
                    for k in ['n', 'TP', 'FP', 'FN', 'jaccard', 'f1', 'timing_bias_ms', 'timing_sd_ms']:
                        row[f'{k}_{pipe}_{landmark}'] = np.nan
                    continue
                cand = cd[landmark]
                TP, FP, FN, diffs = greedy_match(cand, ref, tol_s)
                row[f'n_{pipe}_{landmark}'] = len(cand)
                row[f'TP_{pipe}_{landmark}'] = TP
                row[f'FP_{pipe}_{landmark}'] = FP
                row[f'FN_{pipe}_{landmark}'] = FN
                row[f'jaccard_{pipe}_{landmark}'] = jacc(TP, FP, FN)
                row[f'f1_{pipe}_{landmark}'] = f1(TP, FP, FN)
                diffs_ms = diffs * 1000
                row[f'timing_bias_ms_{pipe}_{landmark}'] = diffs_ms.mean() if len(diffs_ms) > 0 else np.nan
                row[f'timing_sd_ms_{pipe}_{landmark}'] = diffs_ms.std(ddof=1) if len(diffs_ms) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

landmark_pipes_bbsig_gt = _landmark_pipes('bbsig', ['lslh', 'cp'])
landmark_pipes_lslh_gt = _landmark_pipes('lslh', ['cp'])
for _lm in ('rpeaks',):
    if _lm not in landmark_pipes_bbsig_gt or landmark_pipes_bbsig_gt[_lm] != ['lslh', 'cp'] \
            or _lm not in landmark_pipes_lslh_gt:
        raise SystemExit('All three pipelines must provide rpeaks (config: pipelines.<key>.landmarks.rpeaks).')

def compute_scenarios(tol_s):
    """Runs both ground-truth scenarios (BBSIG=GT, LSLH=GT) at the given
    tolerance and returns everything plot_jf_box_figure / plot_timing_grid
    need. Called once per tolerance (50ms, 70ms) so both can be reported."""
    tag = f'{tol_s*1000:.0f}ms'

    # ---- Scenario A: BBSIG as ground truth ----
    peak_df_bbsig_gt = compute_peak_agreement('bbsig', landmark_pipes_bbsig_gt, tol_s)
    peak_df_bbsig_gt.to_csv(os.path.join(out_dir, f'peak_agreement_BBSIG_ground_truth_{task}_{tag}.tsv'), sep='\t', index=False)

    print(f'\n=== JACCARD/F1, BBSIG = ground truth, N={len(peak_df_bbsig_gt)} subjects, tolerance=+/-{tol_s*1000:.0f}ms ===')
    jf_summary_bbsig_gt = {}
    timing_summary_bbsig_gt = {}
    for landmark, pipes in landmark_pipes_bbsig_gt.items():
        jf_summary_bbsig_gt[landmark] = {}
        timing_summary_bbsig_gt[landmark] = {}
        for pipe in pipes:
            jac = peak_df_bbsig_gt[f'jaccard_{pipe}_{landmark}'].dropna()
            f1v = peak_df_bbsig_gt[f'f1_{pipe}_{landmark}'].dropna()
            subj_ids = peak_df_bbsig_gt.loc[jac.index, 'subject_id'].tolist()
            jf_summary_bbsig_gt[landmark][pipe] = {'jaccard': jac, 'f1': f1v, 'subject_ids': subj_ids}
            print(f'{landmark:10s} {pipe:5s} N={len(jac):2d}  Jaccard M={jac.mean():.3f} SD={jac.std():.3f}  F1 M={f1v.mean():.3f} SD={f1v.std():.3f}')

            bias = peak_df_bbsig_gt[f'timing_bias_ms_{pipe}_{landmark}'].dropna()
            jitter = peak_df_bbsig_gt[f'timing_sd_ms_{pipe}_{landmark}'].dropna()
            bias_ids = peak_df_bbsig_gt.loc[bias.index, 'subject_id'].tolist()
            jitter_ids = peak_df_bbsig_gt.loc[jitter.index, 'subject_id'].tolist()
            timing_summary_bbsig_gt[landmark][pipe] = {'bias': bias, 'bias_subject_ids': bias_ids,
                                                         'jitter': jitter, 'jitter_subject_ids': jitter_ids}
            print(f'{landmark:10s} {pipe:5s} N={len(bias):2d}  Timing bias M={bias.mean():.2f}ms SD={bias.std():.2f}ms  Within-subj jitter M={jitter.mean():.2f}ms SD={jitter.std():.2f}ms')

    # ---- Scenario B: LSLH as ground truth (CP vs LSLH only) ----
    peak_df_lslh_gt = compute_peak_agreement('lslh', landmark_pipes_lslh_gt, tol_s)
    peak_df_lslh_gt.to_csv(os.path.join(out_dir, f'peak_agreement_LSLH_ground_truth_{task}_{tag}.tsv'), sep='\t', index=False)

    print(f'\n=== JACCARD/F1, LSLH = ground truth, CP = candidate, N={len(peak_df_lslh_gt)} subjects, tolerance=+/-{tol_s*1000:.0f}ms ===')
    jf_summary_lslh_gt = {}
    timing_summary_lslh_gt = {}
    for landmark, pipes in landmark_pipes_lslh_gt.items():
        jf_summary_lslh_gt[landmark] = {}
        timing_summary_lslh_gt[landmark] = {}
        for pipe in pipes:
            jac = peak_df_lslh_gt[f'jaccard_{pipe}_{landmark}'].dropna()
            f1v = peak_df_lslh_gt[f'f1_{pipe}_{landmark}'].dropna()
            subj_ids = peak_df_lslh_gt.loc[jac.index, 'subject_id'].tolist()
            jf_summary_lslh_gt[landmark][pipe] = {'jaccard': jac, 'f1': f1v, 'subject_ids': subj_ids}
            print(f'{landmark:10s} {pipe:5s} N={len(jac):2d}  Jaccard M={jac.mean():.3f} SD={jac.std():.3f}  F1 M={f1v.mean():.3f} SD={f1v.std():.3f}')

            bias = peak_df_lslh_gt[f'timing_bias_ms_{pipe}_{landmark}'].dropna()
            jitter = peak_df_lslh_gt[f'timing_sd_ms_{pipe}_{landmark}'].dropna()
            bias_ids = peak_df_lslh_gt.loc[bias.index, 'subject_id'].tolist()
            jitter_ids = peak_df_lslh_gt.loc[jitter.index, 'subject_id'].tolist()
            timing_summary_lslh_gt[landmark][pipe] = {'bias': bias, 'bias_subject_ids': bias_ids,
                                                        'jitter': jitter, 'jitter_subject_ids': jitter_ids}
            print(f'{landmark:10s} {pipe:5s} N={len(bias):2d}  Timing bias M={bias.mean():.2f}ms SD={bias.std():.2f}ms  Within-subj jitter M={jitter.mean():.2f}ms SD={jitter.std():.2f}ms')

    return dict(tag=tag, peak_df_bbsig_gt=peak_df_bbsig_gt, jf_summary_bbsig_gt=jf_summary_bbsig_gt,
                timing_summary_bbsig_gt=timing_summary_bbsig_gt, peak_df_lslh_gt=peak_df_lslh_gt,
                jf_summary_lslh_gt=jf_summary_lslh_gt, timing_summary_lslh_gt=timing_summary_lslh_gt)

# 50ms first, 70ms second -- both get their own figures; output order
# the 50ms results ahead of the 70ms ones.
scenarios_50ms = compute_scenarios(0.050)
scenarios_70ms = compute_scenarios(0.070)

# ================= LANDMARK TIMING PRECISION (MAE, SD) =================
# Continuous-error counterpart to the Jaccard/F1 classification metrics
# above: rather than asking "did a candidate peak fall within a tolerance
# window" (a yes/no classification), this asks "how far off was it" for
# pairs matched within a +/-50ms window -- the same matching tolerance used
# elsewhere in this script. Unmatched candidates (no reference within 50ms)
# are excluded here, since that reflects a detection miss already captured
# by the Jaccard/F1 analysis above, not a timing error.
PRECISION_SANITY_CUTOFF_S = 0.050
precision_landmark_pipes = landmark_pipes_bbsig_gt

precision_rows = []
for subj_id in subjects:
    ref_landmarks = landmarks_by_subject[subj_id]['bbsig']
    if ref_landmarks is None:
        continue
    row = {'subject_id': subj_id}
    for landmark, pipes in precision_landmark_pipes.items():
        ref = ref_landmarks[landmark]
        for pipe in pipes:
            cd = landmarks_by_subject[subj_id][pipe]
            if cd is None or landmark not in cd:
                row[f'mae_ms_{pipe}_{landmark}'] = np.nan
                row[f'sd_ms_{pipe}_{landmark}'] = np.nan
                continue
            cand = cd[landmark]
            _, _, _, diffs = greedy_match(cand, ref, PRECISION_SANITY_CUTOFF_S)
            abs_diffs_ms = np.abs(diffs) * 1000
            row[f'mae_ms_{pipe}_{landmark}'] = abs_diffs_ms.mean() if len(abs_diffs_ms) > 0 else np.nan
            row[f'sd_ms_{pipe}_{landmark}'] = abs_diffs_ms.std(ddof=1) if len(abs_diffs_ms) > 1 else np.nan
    precision_rows.append(row)

precision_df = pd.DataFrame(precision_rows)
precision_df.to_csv(os.path.join(out_dir, f'landmark_timing_precision_BBSIG_ground_truth_{task}.tsv'), sep='\t', index=False)

print(f'\n=== LANDMARK TIMING PRECISION (MAE, SD), BBSIG = ground truth, N={len(precision_df)} subjects, '
      f'+/-{PRECISION_SANITY_CUTOFF_S*1000:.0f}ms matching tolerance ===')
precision_summary = {}
for landmark, pipes in precision_landmark_pipes.items():
    precision_summary[landmark] = {}
    for pipe in pipes:
        mae = precision_df[f'mae_ms_{pipe}_{landmark}'].dropna()
        sd = precision_df[f'sd_ms_{pipe}_{landmark}'].dropna()
        subj_ids = precision_df.loc[mae.index, 'subject_id'].tolist()
        precision_summary[landmark][pipe] = {'mae': mae, 'sd': sd, 'subject_ids': subj_ids}
        print(f'{landmark:10s} {pipe:5s} N={len(mae):2d}  MAE M={mae.mean():.2f}ms SD={mae.std():.2f}ms  '
              f'(within-subject SD) M={sd.mean():.2f}ms SD={sd.std():.2f}ms')

# ================= BLAND-ALTMAN (BBSIG vs LSLH, BBSIG vs CP) =================
def load_tsv(p):
    return pd.read_csv(p, sep='\t') if os.path.exists(p) else pd.DataFrame()

def to_ms(df):
    df = df.copy()
    for c in ['SDNN', 'RMSSD']:
        if c in df.columns: df[c] = df[c] * 1000
    for c in ['VLF', 'LF', 'HF']:
        if c in df.columns: df[c] = df[c] * 1e6
    return df

_tables = cfg['summary_tables']
bbsig_art = load_tsv(_tables['bbsig_artifacts'])
bbsig_hrv = load_tsv(_tables['bbsig_hrv'])
lslh_sum = load_tsv(_tables['lslh_hrv'])
cp_sum = load_tsv(_tables['cp_hrv'])

bbsig_art = bbsig_art[bbsig_art['task'] == task].sort_values('pct_artifacts').drop_duplicates('subject_id', keep='first')
bbsig_hrv = bbsig_hrv[bbsig_hrv['task'] == task].sort_values('timestamp').drop_duplicates('subject_id', keep='last')
bbsig_hrv = to_ms(bbsig_hrv)
lslh_sum = lslh_sum[lslh_sum['task'] == task].sort_values('timestamp').drop_duplicates('subject_id', keep='last') if 'timestamp' in lslh_sum.columns else lslh_sum[lslh_sum['task']==task].drop_duplicates('subject_id', keep='last')
cp_sum = cp_sum[cp_sum['task'] == task].sort_values('timestamp').drop_duplicates('subject_id', keep='last')
cp_sum = to_ms(cp_sum)

hrv_cols = [c for c in ['subject_id', 'MeanBPM', 'SDNN', 'RMSSD', 'LF_HF'] if c in bbsig_hrv.columns]
bbsig = pd.merge(bbsig_art[['subject_id', 'n_peaks', 'pct_artifacts']], bbsig_hrv[hrv_cols], on='subject_id', how='outer')
bbsig = bbsig[bbsig['subject_id'].isin(subjects)]

lslh_sum = lslh_sum[lslh_sum['subject_id'].isin(subjects)]
cp_sum = cp_sum[cp_sum['subject_id'].isin(subjects)]

metrics = ['n_peaks', 'MeanBPM', 'SDNN', 'RMSSD', 'LF_HF']
metric_units = {'n_peaks':'count','MeanBPM':'bpm','SDNN':'ms','RMSSD':'ms','LF_HF':'ratio'}

def ba_stats(x, y, subj_ids):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    ids = [s for s, m in zip(subj_ids, mask) if m]
    if len(x) < 2:
        return None
    diff = x - y
    bias = np.mean(diff)
    sd = np.std(diff, ddof=1)
    return {'n': len(x), 'mean': (x+y)/2, 'diff': diff, 'subject_ids': ids, 'bias': bias, 'sd': sd,
            'loa_lower': bias - 1.96*sd, 'loa_upper': bias + 1.96*sd}

def compute_ba(ref_df, other_df, ref_name, other_name, ref_col_check, other_col_check):
    """ref_df/other_df: per-subject metric tables. ref_name/other_name used for
    column suffixing after merge. ref_col_check/other_col_check: RAW (pre-merge)
    column sets, used to decide if a metric is genuinely available on BOTH
    sides -- avoids silently comparing a column to itself when only one side
    actually reports a metric."""
    merged = pd.merge(ref_df, other_df, on='subject_id', suffixes=(f'_{ref_name}', f'_{other_name}'))
    results = {}
    for metric in metrics:
        if metric not in ref_col_check:
            print(f'{metric:15s} NOT AVAILABLE in {ref_name} data -- skipped')
            continue
        if metric not in other_col_check:
            print(f'{metric:15s} NOT AVAILABLE in {other_name} data -- skipped (not compared to itself)')
            continue
        rcol = f'{metric}_{ref_name}' if f'{metric}_{ref_name}' in merged.columns else metric
        ocol = f'{metric}_{other_name}' if f'{metric}_{other_name}' in merged.columns else metric
        if rcol == ocol:
            print(f'{metric:15s} AMBIGUOUS COLUMN after merge -- skipped')
            continue
        s = ba_stats(merged[rcol], merged[ocol], merged['subject_id'].tolist())
        if s is None:
            print(f'{metric:15s} insufficient data'); continue
        print(f'{metric:15s} N={s["n"]:2d}  bias={s["bias"]:.3f} {metric_units[metric]}  SD={s["sd"]:.3f}  LoA=[{s["loa_lower"]:.3f}, {s["loa_upper"]:.3f}]')
        results[metric] = s
    return results

print(f'\n=== BLAND-ALTMAN: BBSIG vs LSLH ===')
ba_ls = compute_ba(bbsig, lslh_sum, 'bbsig', 'ls', set(bbsig.columns), set(lslh_sum.columns))

print(f'\n=== BLAND-ALTMAN: BBSIG vs CP ===')
ba_cp = compute_ba(bbsig, cp_sum, 'bbsig', 'cp', set(bbsig.columns), set(cp_sum.columns))

print(f'\n=== BLAND-ALTMAN: LSLH vs CP ===')
ba_ls_cp = compute_ba(lslh_sum, cp_sum, 'ls', 'cp', set(lslh_sum.columns), set(cp_sum.columns))

pd.DataFrame([{'metric': k, **{kk: vv for kk, vv in v.items() if kk not in ('mean','diff','subject_ids')}} for k, v in ba_ls.items()]) \
    .to_csv(os.path.join(out_dir, 'bland_altman_recomputed_BBSIG_vs_LSLH.csv'), index=False)
pd.DataFrame([{'metric': k, **{kk: vv for kk, vv in v.items() if kk not in ('mean','diff','subject_ids')}} for k, v in ba_cp.items()]) \
    .to_csv(os.path.join(out_dir, 'bland_altman_recomputed_BBSIG_vs_CP.csv'), index=False)
pd.DataFrame([{'metric': k, **{kk: vv for kk, vv in v.items() if kk not in ('mean','diff','subject_ids')}} for k, v in ba_ls_cp.items()]) \
    .to_csv(os.path.join(out_dir, 'bland_altman_recomputed_LSLH_vs_CP.csv'), index=False)

print(f'\nSaved TSV/CSV outputs to {out_dir}')

# ================= PLOTS =================
# All figure text set to 12pt, Times New Roman (APA style); figures sized 30%
# larger than the original layout so text remains legible at that size
# without crowding.
FIGSIZE_SCALE = 1.3
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})
COLOR_JACCARD, COLOR_F1, COLOR_DOT = '#2a78d6', '#008300', '#898781'
COLOR_BIAS, COLOR_JITTER = '#b35806', '#542788'

# Per-panel size kept constant across all grids (whether 1, 2, or 3 columns /
# rows) so figures of different shapes still sit at a consistent visual scale
# when placed side by side -- this is what avoids ever needing an "N/A" cell:
# a landmark that only one pipeline reports (T-peak: LSLH only) simply gets
# its own smaller figure built from the same per-panel unit instead of an
# empty cell in the main grid.
PANEL_W = 11 * FIGSIZE_SCALE / 3
PANEL_H = 3.6 * FIGSIZE_SCALE

def short_id(subj_id):
    return subj_id.split('-')[1]  # 'sub-09' -> '09'

def add_panel_letter(ax, letter):
    """Scientific-paper-style panel label (A, B, C...) in the top-left corner."""
    ax.text(-0.14, 1.14, letter, transform=ax.transAxes, fontsize=12, fontweight='bold',
            fontname='Times New Roman', ha='left', va='bottom')

def _box_group(ax, positions, values_list, colors, label_y=1.22, label_fmt='{:.2f}'):
    """Draws one box per (position, values) pair, colored per `colors`, with
    jittered (unlabeled) dots overlaid and the group mean printed above.
    `label_y` defaults to 1.22, fitting the 0-1.3 Jaccard/F1 scale used by
    plot_jf_box_figure_multi; callers plotting a different scale (e.g.
    millisecond timing errors) must pass their own y position."""
    bp = ax.boxplot(values_list, positions=positions, widths=0.75, patch_artist=True,
                     medianprops=dict(color='#444444', linewidth=1.2),
                     whiskerprops=dict(color='#444444'), capprops=dict(color='#444444'),
                     flierprops=dict(marker='o', markerfacecolor=COLOR_DOT, markersize=4,
                                      markeredgecolor='none', alpha=0.6))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor('none')
    rng = np.random.default_rng(0)
    for pos, vals in zip(positions, values_list):
        jitter = rng.uniform(-0.1, 0.1, size=len(vals))
        xpos = pos + jitter
        ax.scatter(xpos, vals, s=14, color=COLOR_DOT, alpha=0.75, edgecolor='none', zorder=3)
        mean_val = np.mean(vals)
        ax.text(pos, label_y, label_fmt.format(mean_val), ha='center', va='bottom', fontsize=10, fontweight='bold')

def plot_jf_box_figure_multi(jf_summaries_by_tag, landmark_pipes, title, fname, fig_width_inches):
    """Letter-labeled box-plot figure for Jaccard/F1 agreement, one row per
    tolerance (e.g. +/-50ms on top, +/-70ms below) so the two tolerances sit
    directly next to each other for comparison instead of being split across
    separate figures. `landmark_pipes` is an ordered dict {landmark:
    [candidate pipeline names]}, column order = panel order (left to right).
    A landmark that only some pipelines report (e.g. T-peaks: LSLH only)
    simply gets fewer boxes in its panel -- never an empty/N-A cell.

    `fig_width_inches` is set equal to the width the figure will be embedded
    at in a document
    so Word displays the PNG at 1:1 scale -- otherwise Word rescales the
    image to fit its embed width, which shrinks the 12pt matplotlib text
    along with it."""
    from matplotlib.patches import Patch
    import string
    tags = list(jf_summaries_by_tag.keys())
    landmarks = list(landmark_pipes.keys())
    n_rows, n_cols = len(tags), len(landmarks)
    alphabet = string.ascii_uppercase
    letters = [[alphabet[r * n_cols + c] for c in range(n_cols)] for r in range(n_rows)]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width_inches, 3.0 * n_rows), squeeze=False)

    for row_i, tol_tag in enumerate(tags):
        jf_summary = jf_summaries_by_tag[tol_tag]
        for col_i, (landmark, pipes) in enumerate(landmark_pipes.items()):
            ax = axes[row_i][col_i]
            positions, values_list, colors = [], [], []
            group_centers, group_labels = [], []
            pos = 0
            for pipe in pipes:
                s = jf_summary[landmark][pipe]
                positions += [pos, pos + 1.0]
                values_list += [s['jaccard'].values, s['f1'].values]
                colors += [COLOR_JACCARD, COLOR_F1]
                group_centers.append(pos + 0.5)
                group_labels.append(pipeline_labels[pipe])
                pos += 2.3

            _box_group(ax, positions, values_list, colors)
            ax.set_xticks(group_centers)
            ax.set_xticklabels(group_labels)
            # Margin of +/-1.0 around the outermost box centers -- covers the
            # 0.75-wide box plus jitter/outlier dots on both sides, so nothing
            # gets clipped at the panel edge (the previous fixed `pos - 1.3`
            # formula under-covered whenever a panel had 1 or 2 groups).
            ax.set_xlim(min(positions) - 1.0, max(positions) + 1.0)
            ax.set_ylim(0, 1.3)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.grid(axis='y', color='#e1e0d9', linewidth=0.7, zorder=0)
            if row_i == 0:
                ax.set_title(landmark_labels[landmark], fontname='Times New Roman')
            if col_i == 0:
                ax.set_ylabel(f'Score\n(tolerance {tol_tag})')
            add_panel_letter(ax, letters[row_i][col_i])

    legend_handles = [Patch(facecolor=COLOR_JACCARD, alpha=0.55, label='Jaccard'),
                       Patch(facecolor=COLOR_F1, alpha=0.55, label='F1')]
    # Keep suptitle/legend within the figure's own [0, 1] coordinate range (not
    # above y=1) and skip bbox_inches='tight' on save -- both would otherwise
    # let matplotlib silently grow the canvas beyond the declared figsize,
    # which breaks the exact width-inches match to how the figure is embedded
    # in a document (the point of setting fig_width_inches in the first place).
    # subplots_adjust (not tight_layout) gives explicit control over how much
    # top margin the suptitle+legend actually need, avoiding the large blank
    # gap tight_layout's own heuristics left above the panels.
    fig.suptitle(relabel(title), fontname='Times New Roman', wrap=True, fontsize=12, y=0.99)
    # Upper-right (not upper-center) so the legend never collides with a panel
    # letter -- letters always sit at the top-LEFT of their own panel, so the
    # top-right corner of the whole figure is clear regardless of column count.
    fig.legend(handles=legend_handles, loc='upper right', ncol=1, bbox_to_anchor=(0.99, 0.965),
               fontsize=12, frameon=False)
    # Left margin set from an absolute-inches target (not a fixed fraction) so
    # the multi-line y-axis label ("Score\n(tolerance ...)") isn't clipped on
    # the narrower (6in) figure the way a fixed fraction would leave it.
    left_frac = 0.95 / fig_width_inches
    # Narrower figures wrap this title to two lines; reserve more top margin
    # there so the wrapped second line doesn't collide with a row-1 panel
    # letter/title.
    top_frac = 0.80 if fig_width_inches >= 6.0 else 0.72
    fig.subplots_adjust(top=top_frac, bottom=0.09, left=left_frac, right=0.98, hspace=0.55, wspace=0.35)
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    return fig

def plot_precision_box_figure(precision_summary, landmark_pipes, title, fname, fig_width_inches):
    """Single-row, letter-labeled box-plot figure for landmark timing
    precision (MAE, within-subject SD), the continuous counterpart to
    plot_jf_box_figure_multi's classification-based Jaccard/F1. No tolerance
    row split is needed here since MAE/SD don't depend on a tolerance
    parameter (see PRECISION_SANITY_CUTOFF_S)."""
    from matplotlib.patches import Patch
    import string
    landmarks = list(landmark_pipes.keys())
    n_cols = len(landmarks)
    letters = list(string.ascii_uppercase[:n_cols])

    fig, axes = plt.subplots(1, n_cols, figsize=(fig_width_inches, 3.4), squeeze=False)
    axes = axes[0]

    # First pass to find the shared y-scale, so the mean-value label height
    # (label_y) can be set from real data instead of a fixed Jaccard/F1-scale
    # constant (1.22, which would be invisible against millisecond values).
    y_max = 0
    for landmark, pipes in landmark_pipes.items():
        for pipe in pipes:
            s = precision_summary[landmark][pipe]
            y_max = max(y_max, np.nanmax(s['mae'].values), np.nanmax(s['sd'].values))
    label_y = y_max * 1.08

    for col_i, (landmark, pipes) in enumerate(landmark_pipes.items()):
        ax = axes[col_i]
        positions, values_list, colors = [], [], []
        group_centers, group_labels = [], []
        pos = 0
        for pipe in pipes:
            s = precision_summary[landmark][pipe]
            positions += [pos, pos + 1.0]
            values_list += [s['mae'].values, s['sd'].values]
            colors += [COLOR_BIAS, COLOR_JITTER]
            group_centers.append(pos + 0.5)
            group_labels.append(pipeline_labels[pipe])
            pos += 2.3

        _box_group(ax, positions, values_list, colors, label_y=label_y, label_fmt='{:.1f}')
        ax.set_xticks(group_centers)
        ax.set_xticklabels(group_labels)
        ax.set_xlim(min(positions) - 1.0, max(positions) + 1.0)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(axis='y', color='#e1e0d9', linewidth=0.7, zorder=0)
        ax.set_title(landmark_labels[landmark], fontname='Times New Roman')
        if col_i == 0:
            ax.set_ylabel('Timing error (ms)')
        add_panel_letter(ax, letters[col_i])

    # Shared y-limit across panels (with headroom for the mean-value labels
    # _box_group prints above each box) so magnitudes are visually comparable.
    for ax in axes:
        ax.set_ylim(0, y_max * 1.25)

    legend_handles = [Patch(facecolor=COLOR_BIAS, alpha=0.55, label='MAE'),
                       Patch(facecolor=COLOR_JITTER, alpha=0.55, label='SD')]
    fig.suptitle(relabel(title), fontname='Times New Roman', wrap=True, fontsize=12, y=0.99)
    # Centered, horizontal legend in the gap between the suptitle and the
    # panel row -- an "upper right" anchor collides with the rightmost
    # panel's title in a multi-column layout like this one.
    fig.legend(handles=legend_handles, loc='upper center', ncol=2, bbox_to_anchor=(0.55, 0.88),
               fontsize=12, frameon=False, columnspacing=1.2)
    left_frac = 0.75 / fig_width_inches
    fig.subplots_adjust(top=0.68, bottom=0.14, left=left_frac, right=0.98, wspace=0.35)
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    return fig

def plot_single_metric_box_figure(jf_summary, landmark_pipes, metric_key, metric_label, color,
                                    title, fname, fig_width_inches):
    """Single-row, letter-labeled box-plot figure for ONE classification metric
    (F1 alone, or Jaccard alone) -- unlike plot_jf_box_figure_multi, which
    always plots Jaccard and F1 side by side. Jaccard and F1 are related by a
    fixed algebraic identity (F1 = 2J/(1+J)), so plotting both is a
    conventional but redundant choice; this function is for outputs that
    pick one metric deliberately instead."""
    landmarks = list(landmark_pipes.keys())
    n_cols = len(landmarks)
    letters = list(string.ascii_uppercase[:n_cols])

    fig, axes = plt.subplots(1, n_cols, figsize=(fig_width_inches, 3.4), squeeze=False)
    axes = axes[0]

    for col_i, (landmark, pipes) in enumerate(landmark_pipes.items()):
        ax = axes[col_i]
        positions, values_list, colors = [], [], []
        group_labels = []
        for pos, pipe in enumerate(pipes):
            s = jf_summary[landmark][pipe]
            positions.append(pos)
            values_list.append(s[metric_key].values)
            colors.append(color)
            group_labels.append(pipeline_labels[pipe])

        _box_group(ax, positions, values_list, colors)
        ax.set_xticks(positions)
        ax.set_xticklabels(group_labels)
        ax.set_xlim(min(positions) - 1.0, max(positions) + 1.0)
        ax.set_ylim(0, 1.25)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(axis='y', color='#e1e0d9', linewidth=0.7, zorder=0)
        ax.set_title(landmark_labels[landmark], fontname='Times New Roman')
        if col_i == 0:
            ax.set_ylabel(metric_label)
        add_panel_letter(ax, letters[col_i])

    fig.suptitle(relabel(title), fontname='Times New Roman', wrap=True, fontsize=12, y=0.99)
    left_frac = 0.75 / fig_width_inches
    # Narrower figures wrap this title to two lines; reserve more top margin
    # there so the wrapped second line doesn't collide with a panel letter.
    top_frac = 0.80 if fig_width_inches >= 6.0 else 0.66
    fig.subplots_adjust(top=top_frac, bottom=0.14, left=left_frac, right=0.98, wspace=0.35)
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    return fig

def plot_jf_grid(jf_summary, row_pipelines, landmarks, title, fname):
    """Binary detection agreement (Jaccard, F1) grid. `landmarks` must be a
    landmark list every one of `row_pipelines` actually reports -- callers
    split off any landmark not common to all rows into its own grid via a
    separate call, rather than padding this one with N/A cells."""
    rng = np.random.default_rng(0)
    n_rows, n_cols = len(row_pipelines), len(landmarks)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(PANEL_W * n_cols, PANEL_H * n_rows), sharey=True, squeeze=False)

    for row_i, pipe in enumerate(row_pipelines):
        for col_i, landmark in enumerate(landmarks):
            ax = axes[row_i, col_i]
            s = jf_summary[landmark][pipe]
            for mi, (key, color) in enumerate([('jaccard', COLOR_JACCARD), ('f1', COLOR_F1)]):
                vals = s[key]
                mean_val, sd_val = vals.mean(), vals.std()
                ax.bar(mi, mean_val, yerr=sd_val, width=0.55, color=color, edgecolor='none', capsize=3, zorder=2)
                jitter = rng.uniform(-0.1, 0.1, size=len(vals))
                xpos = mi + jitter
                ax.scatter(xpos, vals.values, s=14, color=COLOR_DOT, alpha=0.7, edgecolor='none', zorder=3)
                for xp, yp, sid in zip(xpos, vals.values, s['subject_ids']):
                    ax.annotate(short_id(sid), (xp, yp), textcoords='offset points', xytext=(0, 4),
                                ha='center', fontsize=7.5, color='#444444', rotation=90, zorder=4)
                ax.text(mi, min(mean_val + (sd_val if not np.isnan(sd_val) else 0) + 0.14, 1.20),
                        f'{mean_val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            ax.set_xticks([0, 1]); ax.set_xticklabels(['Jaccard', 'F1'])
            ax.set_ylim(0, 1.3)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.grid(axis='y', color='#e1e0d9', linewidth=0.7, zorder=0)
            if row_i == 0:
                ax.set_title(landmark_labels[landmark], fontname='Times New Roman')
            if col_i == 0:
                ax.set_ylabel(f'{pipeline_labels[pipe]} vs {pipeline_labels[gt_name]}\nScore')

    fig.suptitle(relabel(title), fontname='Times New Roman', wrap=True, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    return fig

def plot_timing_grid(timing_summary, row_pipelines, landmarks, title, fname):
    """Continuous counterpart to plot_jf_grid: per-subject mean timing bias
    (candidate - reference, ms) and per-subject within-subject jitter (SD of
    matched-beat timing errors, ms), each shown as across-subject mean +/- SD."""
    rng = np.random.default_rng(0)
    n_rows, n_cols = len(row_pipelines), len(landmarks)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(PANEL_W * n_cols, PANEL_H * n_rows), squeeze=False)

    for row_i, pipe in enumerate(row_pipelines):
        for col_i, landmark in enumerate(landmarks):
            ax = axes[row_i, col_i]
            s = timing_summary[landmark][pipe]
            for mi, (key, id_key, color) in enumerate([('bias', 'bias_subject_ids', COLOR_BIAS),
                                                         ('jitter', 'jitter_subject_ids', COLOR_JITTER)]):
                vals = s[key]
                mean_val, sd_val = vals.mean(), vals.std()
                ax.bar(mi, mean_val, yerr=sd_val, width=0.55, color=color, edgecolor='none', capsize=3, zorder=2)
                jit = rng.uniform(-0.1, 0.1, size=len(vals))
                xpos = mi + jit
                ax.scatter(xpos, vals.values, s=14, color=COLOR_DOT, alpha=0.7, edgecolor='none', zorder=3)
                for xp, yp, sid in zip(xpos, vals.values, s[id_key]):
                    ax.annotate(short_id(sid), (xp, yp), textcoords='offset points', xytext=(0, 4),
                                ha='center', fontsize=7.5, color='#444444', rotation=90, zorder=4)
                sd_safe = sd_val if not np.isnan(sd_val) else 0
                label_y = mean_val + sd_safe + 2 if mean_val >= 0 else mean_val - sd_safe - 2
                ax.text(mi, label_y, f'{mean_val:.1f}', ha='center',
                        va='bottom' if mean_val >= 0 else 'top', fontsize=12, fontweight='bold')
            ax.axhline(0, color='#999999', linewidth=0.8, zorder=1)
            ax.set_xticks([0, 1]); ax.set_xticklabels(['Bias (ms)', 'Jitter SD (ms)'])
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.grid(axis='y', color='#e1e0d9', linewidth=0.7, zorder=0)
            if row_i == 0:
                ax.set_title(landmark_labels[landmark], fontname='Times New Roman')
            if col_i == 0:
                ax.set_ylabel(f'{pipeline_labels[pipe]} vs {pipeline_labels[gt_name]}\nTiming error (ms)')

    fig.suptitle(relabel(title), fontname='Times New Roman', wrap=True, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    return fig

# One combined, letter-labeled box-plot figure per ground-truth scenario,
# with +/-50ms on top and +/-70ms below (rather than 4 separate figures) so
# the two tolerances sit directly next to each other for comparison. T-peaks
# is placed as the rightmost panel (column C); since CP never produces a
# T-peak marker, that panel simply has fewer boxes in it instead of an
# empty/N-A cell for CP.
fig1 = plot_jf_box_figure_multi(
    {'50ms': scenarios_50ms['jf_summary_bbsig_gt'], '70ms': scenarios_70ms['jf_summary_bbsig_gt']},
    landmark_pipes_bbsig_gt,
    f'Peak-timing agreement, BBSIG as ground truth (N={len(scenarios_70ms["peak_df_bbsig_gt"])})',
    'jaccard_f1_BBSIG_ground_truth_combined.png', fig_width_inches=6.5)

fig2 = plot_jf_box_figure_multi(
    {'50ms': scenarios_50ms['jf_summary_lslh_gt'], '70ms': scenarios_70ms['jf_summary_lslh_gt']},
    landmark_pipes_lslh_gt,
    f'Peak-timing agreement, LSLH as ground truth (N={len(scenarios_70ms["peak_df_lslh_gt"])})',
    'jaccard_f1_LSLH_ground_truth_combined.png', fig_width_inches=4.6)

# Single-metric versions (F1 alone, Jaccard alone) at +/-50ms, since Jaccard and F1 are related by a
# fixed identity (F1 = 2J/(1+J)) and plotting both is redundant.
bbsig_gt_landmarks = landmark_pipes_bbsig_gt
lslh_gt_landmarks = landmark_pipes_lslh_gt

fig_f1_bbsig = plot_single_metric_box_figure(
    scenarios_50ms['jf_summary_bbsig_gt'], bbsig_gt_landmarks, 'f1', 'F1 Score', COLOR_F1,
    f'Peak-timing agreement (F1), BBSIG as ground truth, tolerance=+/-50ms '
    f'(N={len(scenarios_50ms["peak_df_bbsig_gt"])})',
    'f1_only_BBSIG_ground_truth_50ms.png', fig_width_inches=6.5)
fig_f1_lslh = plot_single_metric_box_figure(
    scenarios_50ms['jf_summary_lslh_gt'], lslh_gt_landmarks, 'f1', 'F1 Score', COLOR_F1,
    f'Peak-timing agreement (F1), LSLH as ground truth, tolerance=+/-50ms '
    f'(N={len(scenarios_50ms["peak_df_lslh_gt"])})',
    'f1_only_LSLH_ground_truth_50ms.png', fig_width_inches=4.6)

fig_jac_bbsig = plot_single_metric_box_figure(
    scenarios_50ms['jf_summary_bbsig_gt'], bbsig_gt_landmarks, 'jaccard', 'Jaccard Index', COLOR_JACCARD,
    f'Peak-timing agreement (Jaccard), BBSIG as ground truth, tolerance=+/-50ms '
    f'(N={len(scenarios_50ms["peak_df_bbsig_gt"])})',
    'jaccard_only_BBSIG_ground_truth_50ms.png', fig_width_inches=6.5)
fig_jac_lslh = plot_single_metric_box_figure(
    scenarios_50ms['jf_summary_lslh_gt'], lslh_gt_landmarks, 'jaccard', 'Jaccard Index', COLOR_JACCARD,
    f'Peak-timing agreement (Jaccard), LSLH as ground truth, tolerance=+/-50ms '
    f'(N={len(scenarios_50ms["peak_df_lslh_gt"])})',
    'jaccard_only_LSLH_ground_truth_50ms.png', fig_width_inches=4.6)

fig3_precision = plot_precision_box_figure(
    precision_summary, precision_landmark_pipes,
    f'Landmark timing precision, BBSIG as ground truth (N={len(precision_df)}, '
    f'+/-{PRECISION_SANITY_CUTOFF_S*1000:.0f}ms matching tolerance)',
    'landmark_timing_precision_BBSIG_ground_truth.png', fig_width_inches=6.5)

# ================= DETECTION ERROR RATE vs. DETECTOR TEMPORAL TOLERANCE =================
# Replicates the analysis in Heryan et al. (2021): rather than reporting
# classification metrics at one fixed matching tolerance, sweep the
# tolerance (Detector Temporal Tolerance, DTT) and show how the metrics
# change. TP/FP/FN are pooled across all subjects at each DTT (not averaged
# per-subject), matching Heryan et al.'s approach of pooling over the whole
# dataset before computing the metrics. R-peaks only (as in Heryan et al.).
DTT_VALUES_MS = list(range(20, 161, 20))  # 20, 40, ..., 160 ms
COLOR_REF_A, COLOR_REF_B = '#0072B2', '#D55E00'  # Okabe-Ito colorblind-safe palette

def sweep_dtt(ref_pipe, candidate_pipes, landmark='rpeaks'):
    """Pooled TP/FP/FN -> S, +P, DER, Jaccard, F1 at each DTT, for each
    candidate pipeline against ref_pipe. Returns {candidate: {metric: [values per DTT]}}."""
    TB = sum(len(landmarks_by_subject[s][ref_pipe][landmark]) for s in subjects
             if landmarks_by_subject[s][ref_pipe] is not None)
    out = {c: {m: [] for m in ('sensitivity', 'ppv', 'der', 'jaccard', 'f1')} for c in candidate_pipes}
    for dtt_ms in DTT_VALUES_MS:
        tol = dtt_ms / 1000.0
        for cand in candidate_pipes:
            TP_t, FP_t, FN_t = 0, 0, 0
            for subj_id in subjects:
                ref_lm = landmarks_by_subject[subj_id][ref_pipe]
                cand_lm = landmarks_by_subject[subj_id][cand]
                if ref_lm is None or cand_lm is None or landmark not in ref_lm or landmark not in cand_lm:
                    continue
                TP, FP, FN, _ = greedy_match(cand_lm[landmark], ref_lm[landmark], tol)
                TP_t += TP; FP_t += FP; FN_t += FN
            S = TP_t / (TP_t + FN_t) * 100 if (TP_t + FN_t) > 0 else np.nan
            PPV = TP_t / (TP_t + FP_t) * 100 if (TP_t + FP_t) > 0 else np.nan
            DER = (FP_t + FN_t) / TB * 100
            dj = TP_t + FP_t + FN_t
            Jaccard = TP_t / dj if dj > 0 else np.nan
            df1 = 2 * TP_t + FP_t + FN_t
            F1 = 2 * TP_t / df1 if df1 > 0 else np.nan
            out[cand]['sensitivity'].append(S)
            out[cand]['ppv'].append(PPV)
            out[cand]['der'].append(DER)
            out[cand]['jaccard'].append(Jaccard)
            out[cand]['f1'].append(F1)
    return out, TB

sweep_bbsig_gt, TB_bbsig = sweep_dtt('bbsig', ['lslh', 'cp'])
sweep_lslh_gt, TB_lslh = sweep_dtt('lslh', ['cp'])

print(f'\n=== DER vs DTT sweep, BBSIG = ground truth, R-peaks, TB={TB_bbsig} ===')
for dtt_i, dtt_ms in enumerate(DTT_VALUES_MS):
    for cand in ['lslh', 'cp']:
        print(f'DTT={dtt_ms:3d}ms  {cand:5s}  DER={sweep_bbsig_gt[cand]["der"][dtt_i]:6.2f}%  '
              f'Jaccard={sweep_bbsig_gt[cand]["jaccard"][dtt_i]:.3f}  F1={sweep_bbsig_gt[cand]["f1"][dtt_i]:.3f}')

print(f'\n=== DER vs DTT sweep, LSLH = ground truth, R-peaks, TB={TB_lslh} ===')
for dtt_i, dtt_ms in enumerate(DTT_VALUES_MS):
    print(f'DTT={dtt_ms:3d}ms  cp     DER={sweep_lslh_gt["cp"]["der"][dtt_i]:6.2f}%  '
          f'Jaccard={sweep_lslh_gt["cp"]["jaccard"][dtt_i]:.3f}  F1={sweep_lslh_gt["cp"]["f1"][dtt_i]:.3f}')

def plot_der_vs_dtt_combined(panel_a, panel_b, title, fname, fig_width_inches=6.5):
    """DER vs DTT, two ground-truth scenarios as
    letter-labeled panels (A, B) in one figure rather than two separate
    ones. Each of panel_a/panel_b is (sweep, candidate_colors, subtitle).
    Colorblind-safe colors + redundant marker/linestyle encoding; y-axis
    inverted (0 at top) matching Heryan et al. Fig. 3."""
    fig, axes = plt.subplots(1, 2, figsize=(fig_width_inches, 4.2))
    markers = ['o', 's', '^']
    linestyles = ['-', '--', ':']
    letters = ['A', 'B']

    for ax, (sweep, candidate_colors, subtitle), letter in zip(axes, (panel_a, panel_b), letters):
        for i, (cand, color) in enumerate(candidate_colors.items()):
            ax.plot(DTT_VALUES_MS, sweep[cand]['der'], marker=markers[i % 3], color=color,
                     label=pipeline_labels[cand], linewidth=1.5, linestyle=linestyles[i % 3])
        ax.set_xlabel('DTT [ms]', fontname='Times New Roman')
        ax.set_ylabel('Detection Error Rate [%]', fontname='Times New Roman')
        ax.set_xlim(0, max(DTT_VALUES_MS) + 10)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(True, color='#e1e0d9', linewidth=0.7)
        ax.legend(loc='lower right', frameon=True, fontsize=10)
        ax.set_title(relabel(subtitle), fontname='Times New Roman', fontsize=11)
        add_panel_letter(ax, letter)

    fig.suptitle(relabel(title), fontname='Times New Roman', fontsize=12, wrap=True)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.subplots_adjust(wspace=0.4)
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    return fig

def plot_outcome_measures_vs_dtt(sweep, candidate_colors, title, fname, fig_width_inches=11.0):
    """All five outcome measures (S, +P, DER,
    Jaccard, F1), one panel each, letter-labeled -- included in full for
    completeness/transparency even though S, +P, DER, Jaccard, and F1 are
    all monotonic transforms of the same pooled TP(DTT) and are therefore
    not independent pieces of evidence (see Discussion)."""
    metrics = ['sensitivity', 'ppv', 'der', 'jaccard', 'f1']
    metric_labels = {'sensitivity': 'Sensitivity (S) [%]', 'ppv': 'Positive Predictivity (+P) [%]',
                      'der': 'Detection Error Rate (DER) [%]', 'jaccard': 'Jaccard Index', 'f1': 'F1 Score'}
    letters = ['A', 'B', 'C', 'D', 'E']
    markers = ['o', 's', '^']
    linestyles = ['-', '--', ':']

    fig, axes = plt.subplots(2, 3, figsize=(fig_width_inches, 6.5))
    axes = axes.flatten()
    for i, m in enumerate(metrics):
        ax = axes[i]
        for j, (cand, color) in enumerate(candidate_colors.items()):
            ax.plot(DTT_VALUES_MS, sweep[cand][m], marker=markers[j % 3], color=color,
                     label=pipeline_labels[cand], linewidth=1.5, linestyle=linestyles[j % 3])
        ax.set_xlabel('DTT [ms]', fontname='Times New Roman')
        ax.set_ylabel(metric_labels[m], fontname='Times New Roman')
        ax.set_xlim(0, max(DTT_VALUES_MS) + 10)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(True, color='#e1e0d9', linewidth=0.7)
        if m == 'der':
            ax.invert_yaxis()
        add_panel_letter(ax, letters[i])
    axes[0].legend(loc='best', frameon=True, fontsize=10)
    axes[5].axis('off')
    fig.suptitle(relabel(title), fontname='Times New Roman', fontsize=12, wrap=True)
    # Extra wspace so a panel letter (positioned left of its own axes) never
    # collides with the previous panel's y-axis label.
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.subplots_adjust(wspace=0.45, hspace=0.5)
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    return fig

# DER-only figure. fig_width_inches matches a 6.5in embed width so a
# word processor displays it at 1:1 scale -- otherwise Word
# rescales the image to fit its embed width, shrinking the 12pt text with it.
fig_der_combined = plot_der_vs_dtt_combined(
    (sweep_bbsig_gt, {'lslh': COLOR_REF_A, 'cp': COLOR_REF_B}, 'BBSIG as ground truth'),
    (sweep_lslh_gt, {'cp': COLOR_REF_B}, 'LSLH as ground truth'),
    f'Detection Error Rate vs. Detector Temporal Tolerance, R-peaks (N={len(subjects)})',
    'der_vs_dtt_combined.png', fig_width_inches=6.5)

# All five outcome measures
fig_outcomes_bbsig = plot_outcome_measures_vs_dtt(
    sweep_bbsig_gt, {'lslh': COLOR_REF_A, 'cp': COLOR_REF_B},
    f'Outcome measures vs. Detector Temporal Tolerance, BBSIG as ground truth, R-peaks (N={len(subjects)})',
    'outcome_measures_vs_dtt_BBSIG_ground_truth.png', fig_width_inches=6.5)
fig_outcomes_lslh = plot_outcome_measures_vs_dtt(
    sweep_lslh_gt, {'cp': COLOR_REF_B},
    f'Outcome measures vs. Detector Temporal Tolerance, LSLH as ground truth, R-peaks (N={len(subjects)})',
    'outcome_measures_vs_dtt_LSLH_ground_truth.png', fig_width_inches=6.5)

# Continuous timing-error grids are only generated
# at the 70ms tolerance -- matched-pair count changes with
# tolerance, but the bias/jitter of pairs that DO match is not meaningfully
# different at 50 vs 70ms, so a duplicate set adds no value here.
def plot_timing_groups(timing_summary, landmark_pipes, ref_label, n, fname_stem):
    """One timing grid per group of landmarks that share the same candidate
    pipelines, so no grid ever needs an empty cell."""
    groups = {}
    for lm, pipes in landmark_pipes.items():
        groups.setdefault(tuple(pipes), []).append(lm)
    for i, (pipes, lms) in enumerate(groups.items()):
        suffix = '' if i == 0 else '_' + '_'.join(lms)
        plot_timing_grid(timing_summary, list(pipes), lms,
                         f'Continuous timing error, {ref_label} = ground truth, task={task} (N={n})',
                         f'{fname_stem}{suffix}.png')

gt_name = 'bbsig'
plot_timing_groups(scenarios_70ms['timing_summary_bbsig_gt'], landmark_pipes_bbsig_gt, 'BBSIG',
                   len(scenarios_70ms['peak_df_bbsig_gt']), 'timing_error_BBSIG_ground_truth')
gt_name = 'lslh'
plot_timing_groups(scenarios_70ms['timing_summary_lslh_gt'], landmark_pipes_lslh_gt, 'LSLH',
                   len(scenarios_70ms['peak_df_lslh_gt']), 'timing_error_LSLH_ground_truth')

BA_LETTERS = ['A', 'B', 'C', 'D', 'E']

def plot_ba_grid(results, title, fname):
    # figsize width matches a 6.5in embed width so a word processor displays
    # it at 1:1 scale -- otherwise Word rescales the image to fit, which
    # shrinks the 12pt matplotlib text along with it.
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 6.2))
    axes = axes.flatten()
    for i, metric in enumerate(metrics):
        ax = axes[i]
        letter = BA_LETTERS[i] if i < len(BA_LETTERS) else None
        if metric not in results:
            ax.set_title(f'{metric}\n(N/A)', fontsize=12); ax.axis('off')
            if letter:
                add_panel_letter(ax, letter)
            continue
        s = results[metric]
        ax.scatter(s['mean'], s['diff'], alpha=0.8, color='#2a78d6', s=32)
        for xp, yp, sid in zip(s['mean'], s['diff'], s['subject_ids']):
            ax.annotate(short_id(sid), (xp, yp), textcoords='offset points', xytext=(3, 3),
                        ha='left', fontsize=8, color='#444444', zorder=4)
        ax.axhline(s['bias'], color='#c0392b', linewidth=1.1)
        ax.axhline(s['loa_upper'], color='#7f8c8d', linestyle='--', linewidth=0.9)
        ax.axhline(s['loa_lower'], color='#7f8c8d', linestyle='--', linewidth=0.9)
        ax.set_title(f'{metric} (N={s["n"]})\nbias={s["bias"]:.2f}, SD={s["sd"]:.2f}', fontsize=12)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if letter:
            add_panel_letter(ax, letter)
    for j in range(len(metrics), len(axes)):
        axes[j].axis('off')
    fig.suptitle(relabel(title), fontname='Times New Roman', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    return fig

fig3 = plot_ba_grid(ba_ls, 'Bland-Altman: BBSIG vs LSLH', 'bland_altman_recomputed_BBSIG_vs_LSLH.png')
fig4 = plot_ba_grid(ba_cp, f'Bland-Altman: BBSIG vs CP, task={task}', 'bland_altman_recomputed_BBSIG_vs_CP.png')
fig4b = plot_ba_grid(ba_ls_cp, f'Bland-Altman: LSLH vs CP, task={task}', 'bland_altman_recomputed_LSLH_vs_CP.png')

# ---- Per-variable pipeline comparison: one subplot per metric, one bar per pipeline ----
PIPE_COLORS = {'bbsig': '#2a78d6', 'lslh': '#e08214', 'cp': '#5aae61'}
pipe_source = {
    'bbsig': bbsig,
    'lslh': lslh_sum,
    'cp': cp_sum,
}

def plot_variable_bars(metrics_list, title, fname):
    n = len(metrics_list)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11 * FIGSIZE_SCALE, 3.8 * n_rows * FIGSIZE_SCALE))
    axes = np.atleast_2d(axes).flatten()
    rng = np.random.default_rng(0)

    for i, metric in enumerate(metrics_list):
        ax = axes[i]
        xpos = 0
        xticks, xticklabels = [], []
        for pipe_name in ['bbsig', 'lslh', 'cp']:
            df = pipe_source[pipe_name]
            if metric not in df.columns:
                xticks.append(xpos); xticklabels.append(f'{pipeline_labels[pipe_name]}\n(N/A)')
                xpos += 1
                continue
            sub = df[['subject_id', metric]].dropna()
            vals = sub[metric].values.astype(float)
            ids = sub['subject_id'].tolist()
            if len(vals) == 0:
                xticks.append(xpos); xticklabels.append(f'{pipeline_labels[pipe_name]}\n(N/A)')
                xpos += 1
                continue
            mean_val, sd_val = vals.mean(), vals.std()
            ax.bar(xpos, mean_val, yerr=sd_val, width=0.6, color=PIPE_COLORS[pipe_name],
                   edgecolor='none', capsize=3, zorder=2, alpha=0.85)
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(xpos + jitter, vals, s=16, color='#444444', alpha=0.7, edgecolor='none', zorder=3)
            for xp, yp, sid in zip(xpos + jitter, vals, ids):
                ax.annotate(short_id(sid), (xp, yp), textcoords='offset points', xytext=(0, 4),
                            ha='center', fontsize=7.5, color='#333333', rotation=90, zorder=4)
            xticks.append(xpos); xticklabels.append(f'{pipeline_labels[pipe_name]}\n(N={len(vals)})')
            xpos += 1

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=12)
        unit = metric_units.get(metric, '')
        ax.set_title(f'{metric}' + (f' ({unit})' if unit not in ('', 'count', 'ratio') else ''), fontsize=12)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(axis='y', color='#e1e0d9', linewidth=0.7, zorder=0)

    for j in range(len(metrics_list), len(axes)):
        axes[j].axis('off')

    fig.suptitle(relabel(title))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    return fig

fig5 = plot_variable_bars(metrics, f'Pipeline Comparison by Variable, task={task}', 'variable_comparison_by_pipeline.png')

# ================= RAW SIGNAL QUALITY INDICES (kurtosis, flatline %, pSQI) =================
# Computed on the raw concatenated ECG signal per subject -- this is the
# common input all three pipelines start from, before any pipeline-specific
# filtering. Useful to check whether pipeline disagreement tracks
# poor input signal quality rather than being purely an algorithm difference.

raw_cfg = cfg['raw_ecg']
fs_raw = raw_cfg['sampling_rate']

def load_concat_raw_ecg(subj_id):
    """Reload the raw ECG from the original EDF/BDF recording(s) of one
    subject, so the quality indices do not depend on any pipeline's output.
    Files matching the pattern are concatenated in file name order. The ECG
    is channel_pos minus channel_neg (or channel_pos alone if no
    channel_neg is set)."""
    block_files = sorted(glob.glob(fill_pattern(raw_cfg['files'], subj_id)))
    if not block_files:
        return None
    segments = []
    for bf in block_files:
        f = pyedflib.EdfReader(bf)
        labels = f.getSignalLabels()
        sig = f.readSignal(labels.index(raw_cfg['channel_pos']))
        if raw_cfg.get('channel_neg'):
            sig = sig - f.readSignal(labels.index(raw_cfg['channel_neg']))
        f.close()
        segments.append(sig)
    return np.concatenate(segments)

def psqi(ecg, fs, qrs_band=(5, 15), total_band=(5, 40)):
    """Spectral power ratio: power in the QRS-dominant band (5-15 Hz) over
    power in the broader diagnostic band (5-40 Hz). Higher = signal energy
    concentrated where a clean QRS complex should be."""
    freqs, psd = welch(ecg, fs=fs, nperseg=min(len(ecg), int(fs * 4)))
    qrs_mask = (freqs >= qrs_band[0]) & (freqs <= qrs_band[1])
    total_mask = (freqs >= total_band[0]) & (freqs <= total_band[1])
    total_power = np.sum(psd[total_mask])
    if total_power <= 0:
        return np.nan
    return np.sum(psd[qrs_mask]) / total_power

sqi_rows = []
for subj_id in subjects:
    ecg = load_concat_raw_ecg(subj_id)
    if ecg is None:
        print(f'SQI: SKIP {subj_id}, no raw BDF blocks found')
        continue
    sqi_rows.append({
        'subject_id': subj_id,
        'kurtosis': sp_kurtosis(ecg, fisher=False),
        'psqi': psqi(ecg, fs_raw),
    })

sqi_df = pd.DataFrame(sqi_rows)
sqi_df.to_csv(os.path.join(out_dir, f'raw_signal_quality_{task}.tsv'), sep='\t', index=False)

print(f'\n=== RAW SIGNAL QUALITY INDICES, N={len(sqi_df)} subjects ===')
for col in ['kurtosis', 'psqi']:
    print(f'{col:15s} M={sqi_df[col].mean():.3f}  SD={sqi_df[col].std():.3f}  '
          f'min={sqi_df[col].min():.3f}  max={sqi_df[col].max():.3f}')

def plot_sqi_bars(df, title, fname):
    sqi_cols = ['kurtosis', 'psqi']
    sqi_titles = {'kurtosis': 'Kurtosis (kSQI)', 'psqi': 'Spectral Power Ratio (pSQI)'}
    fig, axes = plt.subplots(1, 2, figsize=(9 * FIGSIZE_SCALE, 4.2 * FIGSIZE_SCALE))
    df_sorted = df.sort_values('subject_id')
    x = np.arange(len(df_sorted))
    for ax, col in zip(axes, sqi_cols):
        vals = df_sorted[col].values
        ax.bar(x, vals, color='#2a78d6', width=0.65, zorder=2)
        for xi, v, sid in zip(x, vals, df_sorted['subject_id']):
            ax.annotate(short_id(sid), (xi, v), textcoords='offset points', xytext=(0, 3),
                        ha='center', fontsize=9, rotation=90, color='#333333')
        ax.set_xticks(x)
        ax.set_xticklabels([short_id(s) for s in df_sorted['subject_id']], fontsize=12, rotation=0)
        ax.set_title(sqi_titles[col], fontsize=12)
        ax.set_xlabel('Subject')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(axis='y', color='#e1e0d9', linewidth=0.7, zorder=0)
    fig.suptitle(relabel(title))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    return fig

fig6 = plot_sqi_bars(sqi_df, f'Raw Signal Quality Indices per Subject, task={task}', 'raw_signal_quality_indices.png')

print(f'Saved plots to {out_dir}')
print('Done. Open the PNG files in the output folder to view them.')
plt.close('all')
