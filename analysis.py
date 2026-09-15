"""
analysis.py
===========
Post-decoding analysis and plotting for the distributed WM project.
Reads prediction pkl files produced by decoder.py and generates all results/plots.

Inputs  (RESULTS_DIR, produced by decoder.py)
----------------------------------------------
  predicted_data.pkl         standard predictions + config fingerprint
  shuffled_data.pkl          null (label-permuted) predictions
  trial_data.pkl             normalised trial DataFrames + time / event metadata
  cross_temp_predictions.pkl  cross-temporal predictions (optional)
  cross_temp_shuf_stats.pkl   shuffle stats for CT z-scoring (optional)

Outputs
-------
  per_session_circcorr.pkl     per-session circular correlation curves
  errors_angular.csv           per-trial decoding errors + behavioral columns
  cross_temp_circcorr.pkl      cross-temporal circular correlation matrices
  cross_temp_errors.pkl        cross-temporal error matrices (+ z-scores)
  neurobeh_baseline.pkl        legacy package (backward-compatible)
  neurobeh_results.pkl         all computed statistics (used by RECOMPUTE=False)

  circ_correlation.svg, prediction_errors.svg
  cross_temporal_<angle>.svg, cross_temporal_error_<angle>.svg
  neurobeh_timeseries.svg, neurobeh_barplot_mean.svg, neurobeh_barplot_slope.svg
  decoding_timeseries.svg, decoding_barplot_mean.svg, decoding_barplot_zscore.svg
  decoding_barplot_slope.svg, neurobeh_vs_decoding.svg

Usage
-----
  RECOMPUTE = True   → load predictions, run all analysis, save results + plots
  RECOMPUTE = False  → load neurobeh_results.pkl, regenerate plots only
"""

import os
import pickle
import warnings
import re
from math import ceil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

from joblib import Parallel, delayed
from scipy.stats import linregress, ttest_1samp, wilcoxon, mannwhitneyu
from scipy.stats import t as t_dist


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  ← edit here
# ══════════════════════════════════════════════════════════════════════════════

from config import (
    AREAS,
    AREA_COLORS,
    BEHAVIOR_CSV,
    DELAY_END,
    DELAY_START,
    EV_RESPONSE,
    EV_TARGET_OFF,
    EV_TARGET_ON,
    FIG_DIR,
    PKL_DIR,
    RESULTS_DIR,
)

RECOMPUTE = True   # False → load cached neurobeh_results.pkl and only replot

# Ordered ASCENDING the cortical hierarchy (sensory → frontal). With this order,
# a positive lag in the inter-area xcorr = lower area leads = feedforward.
# AREAS      = ['V4', 'MT', 'IT', 'LIP', 'Parietal', 'FEF', 'PFC']
# AREAS      = ['LIP', 'FEF', 'PFC']
# AREAS      = ['V4', 'MT', 'IT', 'Parietal']

# AREAS      = ['PFC']

ANGLE_NAME = 'targetAngle'

# ── Event times in absolute seconds (must match decoder.py) ──────────────────

# ── Analysis periods relative to target onset = 0 ────────────────────────────
EV_TARGET_OFF_REL = 0.10
EV_FIXPT_OFF_REL  = 0.85
DELAY_START_REL   = 0.10
DELAY_END_REL     = 0.85
RESPONSE_START_REL = 0.85
RESPONSE_END_REL   = 1.80

# ── Statistics ────────────────────────────────────────────────────────────────
ALPHA         = 0.05
N_PERM        = 1000    # cluster permutation test
N_NULL_PERMS  = 50      # angle permutations for decoding null
MIN_TRIALS    = 10
FRONTAL_AREAS = ['PFC', 'FEF', 'LIP']
SENSORY_AREAS = ['IT', 'MT', 'V4']

# ── Save options ──────────────────────────────────────────────────────────────
SAVE_ERRORS_CSV = True
BEH_ERR_RADIANS = False   # True if 'err' column is already in radians

# ── Plot aesthetics ───────────────────────────────────────────────────────────
YLIM_CIRCCORR = (-0.2, 0.5)
SHOW_NULL     = True
SHOW_SEM      = True


os.makedirs(PKL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
_results_pkl = os.path.join(PKL_DIR, 'neurobeh_results.pkl')


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _circcorr_vec(pred_v, true_v):
    """Vectorised circular correlation for all time bins at once.
    pred_v: (n_valid, n_time), true_v: (n_valid,) → (n_time,)"""
    alpha_bar = np.arctan2(np.nanmean(np.sin(pred_v), axis=0),
                            np.nanmean(np.cos(pred_v), axis=0))
    beta_bar  = np.arctan2(np.nanmean(np.sin(true_v)),
                            np.nanmean(np.cos(true_v)))
    sin_a = np.sin(pred_v - alpha_bar)
    sin_b = np.sin(true_v  - beta_bar)[:, None]
    num   = np.nansum(sin_a * sin_b, axis=0)
    denom = np.sqrt(np.nansum(sin_a ** 2, axis=0) * np.nansum(sin_b ** 2))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(denom > 0, num / denom, np.nan)


def _circcorr_null_vec(pred_v, true_v, n_perms, rng):
    """Vectorised null circular correlation — all perms × all bins at once.
    Returns (n_perms, n_time)."""
    perm_true = np.stack([rng.permutation(true_v) for _ in range(n_perms)])
    alpha_bar = np.arctan2(np.nanmean(np.sin(pred_v), axis=0),
                            np.nanmean(np.cos(pred_v), axis=0))
    beta_bar  = np.arctan2(np.nanmean(np.sin(perm_true), axis=1),
                            np.nanmean(np.cos(perm_true), axis=1))
    sin_a = np.sin(pred_v   - alpha_bar)
    sin_b = np.sin(perm_true - beta_bar[:, None])
    num   = np.einsum('vt,pv->pt', sin_a, sin_b)
    denom = np.sqrt(np.nansum(sin_a ** 2, axis=0)[None]
                    * np.nansum(sin_b ** 2, axis=1)[:, None])
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(denom > 0, num / denom, np.nan)


def _z_from_shuffle(err_real, err_shuf):
    """z = (shuffle_mean − real) / shuffle_std.  Positive → better than chance."""
    mu = np.nanmean(err_shuf, axis=0)
    sd = np.nanstd(err_shuf,  axis=0, ddof=1)
    return (mu - err_real) / np.where(sd > 0, sd, np.nan)


def _mean_sem(arr, axis=0):
    m   = np.nanmean(arr, axis=axis)
    n   = np.sum(~np.isnan(arr), axis=axis)
    sem = np.nanstd(arr, axis=axis) / np.where(n > 0, np.sqrt(n), np.nan)
    return m, sem


def _stars(p):
    if p is None or np.isnan(p): return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < ALPHA: return '*'
    return 'n.s.'


def _test_vs_zero(values):
    clean = values[np.isfinite(values)]
    if len(clean) < 3:  return np.nan
    if len(clean) < 5:
        _, p = ttest_1samp(clean, 0); return float(p)
    try:
        _, p = wilcoxon(clean, alternative='two-sided'); return float(p)
    except ValueError:
        return np.nan


def _cluster_perm_1samp(r_mat, n_perm=N_PERM, thresh_p=0.05, seed=42):
    """Sign-flip cluster permutation test: H0 = mean across sessions is zero.
    Returns (n_time,) bool significant mask."""
    valid_rows = ~np.all(np.isnan(r_mat), axis=1)
    r = r_mat[valid_rows]
    if r.shape[0] < 3:
        return np.zeros(r_mat.shape[1], dtype=bool)
    n_sess, n_t = r.shape
    t_thresh = t_dist.ppf(1 - thresh_p / 2, df=n_sess - 1)

    def _t_stats(mat):
        n = np.sum(np.isfinite(mat), axis=-2)
        m = np.nanmean(mat, axis=-2)
        s = np.nanstd(mat,  axis=-2, ddof=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(n >= 3, m / (s / np.sqrt(n)), np.nan)

    def _max_cluster(t_vals):
        mass = cur = 0.0
        for v in t_vals:
            if np.isfinite(v) and abs(v) > t_thresh:
                cur += abs(v)
            else:
                mass = max(mass, cur); cur = 0.0
        return max(mass, cur)

    t_obs    = _t_stats(r)
    obs_mass = _max_cluster(t_obs)
    if obs_mass == 0:
        return np.zeros(n_t, dtype=bool)

    rng    = np.random.default_rng(seed)
    signs  = rng.choice([-1, 1], size=(n_perm, n_sess, 1)).astype(float)
    t_perm = _t_stats(r[None] * signs)
    null_max = np.array([_max_cluster(t_perm[pi]) for pi in range(n_perm)])

    sig_mask = np.zeros(n_t, dtype=bool)
    i = 0
    while i < n_t:
        if np.isfinite(t_obs[i]) and abs(t_obs[i]) > t_thresh:
            j = i
            while j < n_t and np.isfinite(t_obs[j]) and abs(t_obs[j]) > t_thresh:
                j += 1
            if np.mean(null_max >= np.nansum(np.abs(t_obs[i:j]))) < ALPHA:
                sig_mask[i:j] = True
            i = j
        else:
            i += 1
    return sig_mask


def _slope_per_session(r_mat, mask, times_subset):
    """Linear slope of correlation over time within a period, per session."""
    Y = r_mat[:, mask]
    x = times_subset
    n_sess = Y.shape[0]
    slopes = np.full(n_sess, np.nan)
    pvals  = np.full(n_sess, np.nan)

    all_valid = np.all(np.isfinite(Y), axis=1)
    if all_valid.any():
        Yv  = Y[all_valid]
        xm  = x - x.mean()
        slopes[all_valid] = (Yv * xm).sum(axis=1) / (xm ** 2).sum()
        n   = len(x)
        yhat = slopes[all_valid, None] * xm + np.nanmean(Yv, axis=1, keepdims=True)
        sse  = ((Yv - yhat) ** 2).sum(axis=1)
        se   = np.sqrt(sse / (n - 2) / (xm ** 2).sum())
        with np.errstate(invalid='ignore', divide='ignore'):
            pvals[all_valid] = 2 * t_dist.sf(np.abs(slopes[all_valid] / se), df=n - 2)

    for si in np.where(~all_valid)[0]:
        row   = Y[si]; valid = np.isfinite(row)
        if valid.sum() > 2:
            sl, _, _, pv, _ = linregress(x[valid], row[valid])
            slopes[si] = sl; pvals[si] = pv
    return slopes, pvals


# ══════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _draw_events(ax, ev_off=EV_TARGET_OFF_REL, ev_resp=EV_FIXPT_OFF_REL):
    ax.axvline(0,        color='steelblue', lw=1.5, ls='--', alpha=0.75, label='Target on')
    ax.axvline(ev_off,   color='steelblue', lw=1.0, ls=':',  alpha=0.60, label='Target off')
    ax.axvline(ev_resp,  color='seagreen',  lw=1.5, ls='--', alpha=0.75, label='Fixpoint off')


def _shade_periods(ax):
    ax.axvspan(DELAY_START_REL,    DELAY_END_REL,
               alpha=0.2, color='gray',      label='Delay')
    ax.axvspan(RESPONSE_START_REL, RESPONSE_END_REL,
               alpha=0.2, color='steelblue', label='Response')


def _annotate_bar(ax, x, y_val, y_err, label, offset=0.01):
    if not label or label == 'n.s.': return
    y_err = y_err if (y_err is not None and np.isfinite(y_err)) else 0.0
    y_top = (max(y_val, 0) if np.isfinite(y_val) else 0) + y_err + offset
    ax.text(x, y_top, label, ha='center', va='bottom', fontsize=10, fontweight='bold')


def _draw_sig_ribbon(ax, t, sig_mask, color, y_pos, height=0.012, alpha=0.4):
    ax.fill_between(t, y_pos, y_pos + height, where=sig_mask,
                    color=color, alpha=alpha,
                    transform=ax.get_xaxis_transform(), clip_on=False)


# ══════════════════════════════════════════════════════════════════════════════
# CIRCULAR CORRELATION
# ══════════════════════════════════════════════════════════════════════════════

def _circcorr_session(session, preds_dict, trial_df, angle_pairs, n_null_perms):
    rng = np.random.default_rng(seed=session)
    real_out, null_out = {}, {}
    for angle_name, (varX, varY) in angle_pairs.items():
        if varX not in preds_dict or varY not in preds_dict:
            continue
        pX = np.asarray(preds_dict[varX]['predictions'])
        pY = np.asarray(preds_dict[varY]['predictions'])
        pred_ang = np.arctan2(pY, pX)
        tX = trial_df[varX].values.astype(float)
        tY = trial_df[varY].values.astype(float)
        true_ang = np.arctan2(tY, tX)
        valid    = np.isfinite(true_ang)
        pred_v   = pred_ang[valid]; true_v = true_ang[valid]
        real_out[angle_name] = _circcorr_vec(pred_v, true_v)
        null_mat = _circcorr_null_vec(pred_v, true_v, n_null_perms, rng)
        with np.errstate(all='ignore'):
            null_out[angle_name] = np.nanmean(null_mat, axis=0)
    return session, real_out, null_out


def compute_per_session_circcorr(all_predictions, trial_data, angle_pairs,
                                  n_null_perms=N_NULL_PERMS):
    """Compute per-session circcorr curves for every area and angle."""
    real_curves, null_curves = {}, {}
    for area, sess_preds in all_predictions.items():
        results = Parallel(n_jobs=-1, prefer='threads')(
            delayed(_circcorr_session)(
                session, preds_dict, trial_data[session], angle_pairs, n_null_perms)
            for session, preds_dict in sess_preds.items()
        )
        real_curves[area] = {s: r for s, r, _ in results}
        null_curves[area] = {s: n for s, _, n in results}
    return real_curves, null_curves


def average_circcorr(per_session_real, per_session_null):
    mean_scores, sem_scores, null_scores, null_sem_scores, n_sessions = {}, {}, {}, {}, {}
    for area in per_session_real:
        mean_scores[area] = {}; sem_scores[area] = {}
        null_scores[area] = {}; null_sem_scores[area] = {}
        angle_names = set()
        for sd in per_session_real[area].values():
            angle_names.update(sd.keys())
        sess_count = 0
        for ang in angle_names:
            r_curves = [per_session_real[area][s][ang]
                        for s in per_session_real[area]
                        if ang in per_session_real[area][s]]
            if not r_curves: continue
            sess_count = max(sess_count, len(r_curves))
            r_stack = np.stack(r_curves, axis=0)
            n_eff   = np.sum(~np.isnan(r_stack), axis=0)
            mean_scores[area][ang] = np.nanmean(r_stack, axis=0)
            sem_scores[area][ang]  = np.nanstd(r_stack, axis=0) / np.where(n_eff > 0, np.sqrt(n_eff), np.nan)
            n_curves = [per_session_null[area][s][ang]
                        for s in per_session_null[area]
                        if ang in per_session_null[area].get(s, {})]
            if n_curves:
                n_stack = np.stack(n_curves, axis=0)
                n_eff_n = np.sum(~np.isnan(n_stack), axis=0)
                null_scores[area][ang]     = np.nanmean(n_stack, axis=0)
                null_sem_scores[area][ang] = np.nanstd(n_stack, axis=0) / np.where(n_eff_n > 0, np.sqrt(n_eff_n), np.nan)
        n_sessions[area] = sess_count
    return mean_scores, sem_scores, null_scores, null_sem_scores, n_sessions


def _circcorr_sig(per_session_real, areas, angle_names, n_time, n_perm=N_PERM):
    """Cluster-permutation significance per (area, angle). Returns {area: {angle: bool_mask}}."""
    sig = {}
    for area in areas:
        if area not in per_session_real:
            continue
        sig[area] = {}
        for ang in angle_names:
            curves = [np.asarray(per_session_real[area][s][ang], dtype=float)
                      for s in per_session_real[area]
                      if ang in per_session_real[area][s]]
            curves = [c for c in curves if len(c) == n_time]
            if len(curves) < 3:
                sig[area][ang] = np.zeros(n_time, dtype=bool)
                continue
            r_mat = np.stack(curves)
            sig[area][ang] = _cluster_perm_1samp(r_mat, n_perm=n_perm)
    return sig


# ══════════════════════════════════════════════════════════════════════════════
# ANGULAR ERRORS & Z-SCORES
# ══════════════════════════════════════════════════════════════════════════════

def calculate_angular_errors(all_predictions, trial_data, angle_pairs):
    errors = {}
    for area, sess_preds in all_predictions.items():
        errors[area] = {}
        for session, preds_dict in sess_preds.items():
            trial_df = trial_data[session]
            errors[area][session] = {}
            for angle_name, (varX, varY) in angle_pairs.items():
                if varX not in preds_dict or varY not in preds_dict: continue
                pX = np.asarray(preds_dict[varX]['predictions'])
                pY = np.asarray(preds_dict[varY]['predictions'])
                pred_ang = np.arctan2(pY, pX)
                tX = trial_df[varX].values.astype(float)[:, None]
                tY = trial_df[varY].values.astype(float)[:, None]
                true_ang = np.arctan2(tY, tX)
                errors[area][session][angle_name] = np.arctan2(
                    np.sin(true_ang - pred_ang), np.cos(true_ang - pred_ang))
    return errors


def calculate_shuffle_angular_errors(all_shuffles, trial_data, angle_pairs):
    shuf_errors = {}
    for area, sess_shuf in all_shuffles.items():
        shuf_errors[area] = {}
        for session, shuf_dict in sess_shuf.items():
            if shuf_dict is None: continue
            trial_df = trial_data[session]
            shuf_errors[area][session] = {}
            for angle_name, (varX, varY) in angle_pairs.items():
                if varX not in shuf_dict or varY not in shuf_dict: continue
                pX = np.asarray(shuf_dict[varX]); pY = np.asarray(shuf_dict[varY])
                pred_ang = np.arctan2(pY, pX)
                tX = trial_df[varX].values.astype(float)[None, :, None]
                tY = trial_df[varY].values.astype(float)[None, :, None]
                true_ang = np.arctan2(tY, tX)
                shuf_errors[area][session][angle_name] = np.arctan2(
                    np.sin(true_ang - pred_ang), np.cos(true_ang - pred_ang))
    return shuf_errors


def zscore_errors_against_shuffle(errors_angular, shuf_errors):
    """Positive z → real error below shuffle mean (better than chance)."""
    errors_z = {}
    for area, sess_errs in errors_angular.items():
        errors_z[area] = {}
        if area not in shuf_errors: continue
        for session, ang_errs in sess_errs.items():
            if session not in shuf_errors[area]: continue
            errors_z[area][session] = {}
            for angle_name, real_err in ang_errs.items():
                if angle_name not in shuf_errors[area][session]: continue
                shuf_err   = shuf_errors[area][session][angle_name]
                real_curve = np.nanmean(np.abs(real_err), axis=0)
                shuf_curve = np.nanmean(np.abs(shuf_err), axis=1)
                errors_z[area][session][angle_name] = _z_from_shuffle(real_curve, shuf_curve)
    return errors_z


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-TEMPORAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def _ct_circcorr_session(session, cross_preds, trial_df, angle_pairs):
    result = {}
    for angle_name, (varX, varY) in angle_pairs.items():
        if varX not in cross_preds or varY not in cross_preds: continue
        pX = cross_preds[varX]; pY = cross_preds[varY]
        pred_ang = np.arctan2(pY, pX)
        true_ang = np.arctan2(trial_df[varY].values.astype(float),
                               trial_df[varX].values.astype(float))
        valid    = np.isfinite(true_ang)
        pred_v   = pred_ang[valid]
        n_valid, n_tr, n_te = pred_v.shape
        cc_flat = _circcorr_vec(pred_v.reshape(n_valid, n_tr * n_te), true_ang[valid])
        result[angle_name] = cc_flat.reshape(n_tr, n_te)
    return session, result


def compute_cross_temporal_circcorr(all_cross_preds, trial_data, angle_pairs):
    ct_real = {}
    for area, sess_cross in all_cross_preds.items():
        results = Parallel(n_jobs=-1, prefer='threads')(
            delayed(_ct_circcorr_session)(
                session, cross_preds, trial_data[session], angle_pairs)
            for session, cross_preds in sess_cross.items()
        )
        ct_real[area] = {s: res for s, res in results}
    return ct_real


def _ct_errors_session(session, cross_preds, trial_df, angle_pairs):
    result = {}
    for angle_name, (varX, varY) in angle_pairs.items():
        if varX not in cross_preds or varY not in cross_preds: continue
        pX = cross_preds[varX]; pY = cross_preds[varY]
        pred_ang = np.arctan2(pY, pX)
        tX = trial_df[varX].values.astype(float)[:, None, None]
        tY = trial_df[varY].values.astype(float)[:, None, None]
        true_ang = np.arctan2(tY, tX)
        err = np.arctan2(np.sin(true_ang - pred_ang), np.cos(true_ang - pred_ang))
        result[angle_name] = np.nanmean(np.abs(err), axis=0)
    return session, result


def compute_cross_temporal_errors(all_cross_preds, trial_data, angle_pairs):
    ct_err = {}
    for area, sess_cross in all_cross_preds.items():
        results = Parallel(n_jobs=-1, prefer='threads')(
            delayed(_ct_errors_session)(
                session, cross_preds, trial_data[session], angle_pairs)
            for session, cross_preds in sess_cross.items()
        )
        ct_err[area] = {s: res for s, res in results}
    return ct_err


def _average_ct_mats(ct_dict):
    mean_out, sem_out = {}, {}
    for area in ct_dict:
        mean_out[area] = {}; sem_out[area] = {}
        angle_names = set()
        for sd in ct_dict[area].values():
            angle_names.update(sd.keys())
        for ang in angle_names:
            mats = [ct_dict[area][s][ang] for s in ct_dict[area]
                    if ang in ct_dict[area][s]]
            if not mats: continue
            stack = np.stack(mats, axis=0)
            n_eff = np.sum(~np.isnan(stack), axis=0)
            mean_out[area][ang] = np.nanmean(stack, axis=0)
            sem_out[area][ang]  = np.nanstd(stack, axis=0) / np.where(n_eff > 0, np.sqrt(n_eff), np.nan)
    return mean_out, sem_out


def zscore_cross_temporal_errors(ct_err, all_ct_shuf):
    ct_err_z = {}
    for area, sess_errs in ct_err.items():
        ct_err_z[area] = {}
        if area not in all_ct_shuf: continue
        for session, ang_errs in sess_errs.items():
            if session not in all_ct_shuf[area]: continue
            ct_err_z[area][session] = {}
            for angle_name, real_mat in ang_errs.items():
                if angle_name not in all_ct_shuf[area][session]: continue
                mu_shuf, sd_shuf = all_ct_shuf[area][session][angle_name]
                ct_err_z[area][session][angle_name] = (
                    (mu_shuf - real_mat) / np.where(sd_shuf > 0, sd_shuf, np.nan))
    return ct_err_z


# ══════════════════════════════════════════════════════════════════════════════
# NEURAL–BEHAVIOURAL CORRELATION
# ══════════════════════════════════════════════════════════════════════════════

def _build_errors_df(errors_angular, trial_data, time_abs, angle_name, beh_cols):
    """Build flat long-form DataFrame equivalent to errors_angular.csv."""
    chunks = []
    time_r = np.round(time_abs, 3)

    def _norm_id(x):
        s = str(x).strip().strip("[]'\"")
        return re.sub(r'\.0$', '', s)

    for area, area_errs in errors_angular.items():
        for session, sess_errs in area_errs.items():
            ang_err = sess_errs.get(angle_name)
            if ang_err is None: continue
            trial_df = trial_data[session]
            orig_sess = _norm_id(trial_df['session'].iloc[0]
                                  if 'session' in trial_df.columns else np.nan)
            n_t, n_b = ang_err.shape
            n_t = min(n_t, len(trial_df))
            trial_rep = np.repeat(np.arange(n_t), n_b)
            time_tile = np.tile(time_r[:n_b], n_t)
            err_flat  = ang_err[:n_t].ravel()

            chunk = pd.DataFrame({
                'area':             area,
                'session_idx':      int(session),
                'session_id':       orig_sess,
                'original_session': orig_sess,
                'original_trial':   (np.repeat(
                    pd.to_numeric(trial_df['trial'], errors='coerce').values[:n_t], n_b)
                    if 'trial' in trial_df.columns else np.nan),
                'angle_name':       angle_name,
                'trial_idx':        trial_rep,
                'time':             time_tile,
                'error':            err_flat,
                'error_deg':        np.degrees(err_flat),
            })
            for col in beh_cols:
                chunk[col] = (np.repeat(trial_df[col].values[:n_t], n_b)
                              if col in trial_df.columns else np.nan)
            if 'err' in trial_df.columns:
                beh_err = trial_df['err'].values[:n_t].astype(float)
                chunk['beh_err_rad'] = np.repeat(
                    np.radians(beh_err) if not BEH_ERR_RADIANS else beh_err, n_b)
            else:
                chunk['beh_err_rad'] = np.nan
            chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _circ_corr_series(area_df, time_abs, n_time):
    """Neural–behavioural circular correlation per session and time bin."""
    time_r   = np.round(time_abs, 3)
    sessions = sorted(area_df['session_idx'].dropna().unique().astype(int).tolist())
    r_mat    = np.full((len(sessions), n_time), np.nan)

    for si, sess_idx in enumerate(sessions):
        sess_df = area_df[area_df['session_idx'] == sess_idx]
        piv = sess_df.pivot_table(index='trial_idx', columns='time',
                                   values='error', aggfunc='first')
        piv = piv.reindex(columns=time_r)
        beh = sess_df.groupby('trial_idx')['beh_err_rad'].first().reindex(piv.index)
        valid = np.isfinite(beh.values)
        if valid.sum() < MIN_TRIALS: continue
        neural_mat = piv.values[valid]
        behav_vec  = beh.values[valid]
        r_series   = _circcorr_vec(neural_mat, behav_vec)
        n_valid_per_bin = np.sum(np.isfinite(neural_mat), axis=0)
        r_series[n_valid_per_bin < MIN_TRIALS] = np.nan
        r_mat[si] = r_series
    return sessions, r_mat


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_circcorr(mean_scores, sem_scores, null_scores, time_rel, areas, angle_pairs, n_sessions=None, cc_sig=None, null_sem=None):
    angle_names = list(angle_pairs.keys())
    fig, axes   = plt.subplots(len(angle_names), 1,
                               figsize=(6, 4 * len(angle_names)), sharex=True)
    axes = np.atleast_1d(axes)
    have_null = False
    for ax, ang in zip(axes, angle_names):
        _shade_periods(ax)
        if SHOW_NULL:
            null_curves = [null_scores.get(a, {}).get(ang) for a in areas]
            null_curves = [np.asarray(c) for c in null_curves
                           if c is not None and np.any(np.isfinite(c))]
            if null_curves:
                ns = np.stack(null_curves)
                nm = np.nanmean(ns, axis=0)
                # Band = typical across-session SEM of the shuffle (comparable to
                # each area's SEM). Falls back to across-area spread if unavailable.
                if null_sem:
                    sem_curves = [null_sem.get(a, {}).get(ang) for a in areas]
                    sem_curves = [np.asarray(c) for c in sem_curves
                                  if c is not None and np.any(np.isfinite(c))]
                    nband = np.nanmean(np.stack(sem_curves), axis=0) if sem_curves \
                            else np.nanstd(ns, axis=0)
                else:
                    nband = np.nanstd(ns, axis=0)
                ax.fill_between(time_rel, nm - nband, nm + nband,
                                color='gray', alpha=0.25, zorder=0)
                ax.plot(time_rel, nm, color='gray', lw=1.2, ls='--', alpha=0.7, zorder=0)
                have_null = True
        for area in areas:
            m = mean_scores.get(area, {}).get(ang)
            s = sem_scores.get(area, {}).get(ang)
            if m is None or np.all(np.isnan(m)): continue
            c = AREA_COLORS.get(area, 'black')
            ax.plot(time_rel, m, color=c, lw=3, label=area, zorder=3)
            if SHOW_SEM and s is not None:
                ax.fill_between(time_rel, m - s, m + s, alpha=0.18, color=c, zorder=2)
        if cc_sig:
            ribbon_areas = [a for a in areas
                            if a in cc_sig and ang in cc_sig[a] and cc_sig[a][ang].any()]
            for ri, area in enumerate(ribbon_areas):
                _draw_sig_ribbon(ax, time_rel, cc_sig[area][ang],
                                 AREA_COLORS.get(area, 'k'), -(ri + 1) * 0.025)
        _draw_events(ax); ax.axhline(0, color='k', lw=0.8, alpha=0.35)
        ax.set_xlim(time_rel[0], time_rel[-1])
        ax.set_ylim(*YLIM_CIRCCORR)
        ax.set_ylabel(f'{ang}\nCirc. correlation', fontsize=13)
        ax.tick_params(labelsize=13)
        for sp in ax.spines.values(): sp.set_linewidth(1.5)
    axes[-1].set_xlabel('Time from target onset (s)', fontsize=14)
    handles = [Line2D([0], [0], color=AREA_COLORS.get(a, 'k'), lw=2.5,
                       label=f'{a} (n={n_sessions[a]})' if (n_sessions and a in n_sessions) else a)
               for a in areas if mean_scores.get(a)]
    handles += [
        Patch(facecolor='gray',      alpha=0.2, label='Delay'),
        Patch(facecolor='steelblue', alpha=0.2, label='Response'),
        Line2D([0], [0], color='steelblue', lw=1.5, ls='--', alpha=0.75, label='Target on'),
        Line2D([0], [0], color='seagreen',  lw=1.5, ls='--', alpha=0.75, label='Fixpoint off'),
        Line2D([0], [0], color='steelblue', lw=1.0, ls=':',  alpha=0.60, label='Target off'),
    ]
    if have_null:
        handles.append(
            Line2D([0], [0], color='gray', lw=1.2, ls='--', alpha=0.7, label='Shuffle'))
    fig.legend(handles=handles, loc='lower center',
               ncol=min(len(handles), 3), fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    p = os.path.join(FIG_DIR, '01_circ_correlation.svg')
    fig.savefig(p, format='svg', bbox_inches='tight')
    print(f"Saved → {p}"); plt.close(fig)


def plot_prediction_errors(errors_angular_z, time_rel, areas, angle_pairs, sig_dict=None):
    angle_names = list(angle_pairs.keys())
    fig, axes   = plt.subplots(len(angle_names), 1,
                               figsize=(7, 4), sharex=True)
    axes = np.atleast_1d(axes)
    legend_handles = []
    for ax, angle_name in zip(axes, angle_names):
        _shade_periods(ax)
        for area in areas:
            if area not in errors_angular_z: continue
            curves = [errors_angular_z[area][s][angle_name]
                      for s in errors_angular_z[area]
                      if angle_name in errors_angular_z[area][s]]
            if not curves: continue
            n_sess = len(curves)
            stack = np.stack(curves)
            n_eff = np.sum(~np.isnan(stack), axis=0)
            m     = np.nanmean(stack, axis=0)
            s     = np.nanstd(stack, axis=0) / np.where(n_eff > 0, np.sqrt(n_eff), np.nan)
            c     = AREA_COLORS.get(area, 'black')
            ax.plot(time_rel, m, color=c, lw=2.5)
            ax.fill_between(time_rel, m - s, m + s, alpha=0.2, color=c)
            if angle_name == angle_names[0]:
                legend_handles.append(
                    Line2D([0], [0], color=c, lw=2.5, label=f'{area} (n={n_sess})'))
        ax.axhline(0, color='k', lw=1.0, alpha=0.4)
        _draw_events(ax)
        if sig_dict:
            ribbon_areas = [a for a in areas if a in sig_dict and sig_dict[a].any()]
            for ri, area in enumerate(ribbon_areas):
                _draw_sig_ribbon(ax, time_rel, sig_dict[area],
                                 AREA_COLORS.get(area, 'black'), -(ri + 1) * 0.025)
        ax.set_ylabel('Distance from shuffle (σ)', fontsize=13)
        ax.set_title(f'Decoding accuracy vs. shuffle — {angle_name}',
                     fontsize=14, fontweight='bold')
        ax.set_xlim(time_rel[0], time_rel[-1])
        ax.tick_params(labelsize=12)
        for sp in ax.spines.values(): sp.set_linewidth(1.5)
    axes[-1].set_xlabel('Time from target onset (s)', fontsize=14)
    legend_handles += [
        Patch(facecolor='gray',      alpha=0.2, label='Delay'),
        Patch(facecolor='steelblue', alpha=0.2, label='Response'),
        Line2D([0], [0], color='steelblue', lw=1.5, ls='--', alpha=0.75, label='Target on'),
        Line2D([0], [0], color='seagreen',  lw=1.5, ls='--', alpha=0.75, label='Fixpoint off'),
        Line2D([0], [0], color='steelblue', lw=1.0, ls=':',  alpha=0.60, label='Target off'),
    ]
    ncol = min(len(legend_handles), 3)
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=ncol, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    p = os.path.join(FIG_DIR, '02_prediction_errors.svg')
    fig.savefig(p, format='svg', bbox_inches='tight')
    print(f"Saved → {p}"); plt.close(fig)


def _plot_ct_heatmaps(mean_dict, time_rel, areas, angle_pairs, fname_template,
                      cmap='RdBu_r', zlabel='Circ. corr.', z_sig=None):
    ext = [time_rel[0], time_rel[-1], time_rel[0], time_rel[-1]]
    ev_off  = EV_TARGET_OFF_REL
    ev_resp = EV_FIXPT_OFF_REL

    for angle_name in angle_pairs:
        active = [a for a in areas
                  if mean_dict.get(a, {}).get(angle_name) is not None
                  and not np.all(np.isnan(mean_dict[a][angle_name]))]
        if not active: continue
        ncols = min(4, len(active))
        nrows = ceil(len(active) / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(5.2 * ncols, 4.5 * nrows), squeeze=False)

        all_vals = np.concatenate([mean_dict[a][angle_name].ravel() for a in active])
        finite   = all_vals[np.isfinite(all_vals)]
        vlim     = float(np.percentile(np.abs(finite), 98)) if len(finite) else 1.0
        vlim     = max(vlim, 1e-6)

        if z_sig is not None:
            norm = mcolors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
        else:
            norm = None

        for idx, area in enumerate(active):
            ax  = axes[idx // ncols][idx % ncols]
            mat = mean_dict[area][angle_name]
            kw  = dict(origin='lower', aspect='auto', extent=ext, cmap=cmap)
            im  = ax.imshow(mat, **(kw | (dict(norm=norm) if norm else dict(vmin=-vlim, vmax=vlim))))
            if z_sig is not None and np.any(np.isfinite(mat) & (mat > z_sig)):
                ax.contour(mat, levels=[z_sig], colors='black', lw=1.0, alpha=0.85,
                           extent=ext, origin='lower')
            ax.plot([time_rel[0], time_rel[-1]], [time_rel[0], time_rel[-1]],
                    'k--', lw=1.0, alpha=0.6)
            for ev_t, col in [(0, 'steelblue'), (ev_off, 'steelblue'), (ev_resp, 'seagreen')]:
                ax.axvline(ev_t, color=col, lw=0.8, ls=':', alpha=0.7)
                ax.axhline(ev_t, color=col, lw=0.8, ls=':', alpha=0.7)
            ax.set_title(area, fontsize=13, fontweight='bold',
                         color=AREA_COLORS.get(area, 'black'))
            ax.set_xlabel('Test time from target onset (s)', fontsize=14)
            ax.set_ylabel('Train time from target onset (s)', fontsize=11)
            ax.tick_params(labelsize=10)
            cb = plt.colorbar(im, ax=ax, shrink=0.82, label=zlabel)
            cb.ax.axhline(0, color='k', lw=0.8, alpha=0.6)
            if z_sig is not None:
                cb.ax.axhline(z_sig, color='k', lw=0.8, alpha=0.6, ls='--')
        for idx in range(len(active), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        suptitle = (f'Cross-temporal decoding — {angle_name}'
                    if z_sig is None else
                    f'Cross-temporal decoding accuracy — {angle_name}\n'
                    f'(black contour: z > {z_sig:.2f}, one-sided p < 0.05)')
        fig.suptitle(suptitle, fontsize=14, fontweight='bold')
        plt.tight_layout()
        fname = os.path.join(FIG_DIR, fname_template.format(angle=angle_name))
        fig.savefig(fname, format='svg', bbox_inches='tight')
        print(f"Saved → {fname}"); plt.close(fig)


def plot_neurobeh_timeseries(nb_r, nb_sig, time_rel, areas, title, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for area in areas:
        if area not in nb_r: continue
        sessions, r_mat = nb_r[area]
        m, sem = _mean_sem(r_mat, axis=0)
        c = AREA_COLORS[area]
        ax.plot(time_rel, m, color=c, lw=3.5, label=f'{area} (n={len(sessions)})')
        ax.fill_between(time_rel, m - sem, m + sem, alpha=0.2, color=c)
    ribbon_areas = [a for a in areas if a in nb_sig and nb_sig[a].any()]
    for ri, area in enumerate(ribbon_areas):
        _draw_sig_ribbon(ax, time_rel, nb_sig[area], AREA_COLORS[area], -(ri + 1) * 0.025)
    _shade_periods(ax); _draw_events(ax)
    ax.axhline(0, color='k', lw=0.8, alpha=0.4)
    ax.set_xlabel('Time from target onset (s)', fontsize=14)
    ax.set_ylabel('Circ. correlation', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(time_rel[0], time_rel[-1])
    ax.set_ylim(-0.2, 0.2)
    ax.legend(fontsize=10, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
    fig.savefig(save_path, format='svg', bbox_inches='tight')
    print(f"Saved → {save_path}"); plt.close(fig)


def _r_ci_fisher(r, n, alpha=0.05):
    """95% CI on Pearson r via Fisher z-transform. Returns (lo, hi) or (nan, nan)."""
    if not np.isfinite(r) or n < 4 or abs(r) >= 1:
        return np.nan, np.nan
    z   = np.arctanh(r)
    se  = 1.0 / np.sqrt(n - 3)
    crit = 1.959963984540054  # z_{0.975}
    return float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se))


def plot_neurobeh_vs_decoding(nb_delay, dec_delay, areas, save_path):
    # Per-region: is session-level decoding quality related to neural–behavioural
    # coupling? One faceted panel per region with its own regression + r [CI], p.
    active = [a for a in areas if a in nb_delay and a in dec_delay]
    panels = []
    for area in active:
        nb_map  = dict(zip(nb_delay[area]['sessions'],  nb_delay[area]['values']))
        dec_map = dict(zip(dec_delay[area]['sessions'], dec_delay[area]['values']))
        common  = sorted(set(nb_map) & set(dec_map))
        x = np.array([dec_map[s] for s in common], dtype=float)
        y = np.array([nb_map[s]  for s in common], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        panels.append((area, x[mask], y[mask]))
    panels = [p for p in panels if len(p[1]) >= 2]
    if not panels:
        print("plot_neurobeh_vs_decoding: no regions with ≥2 paired sessions — skipping.")
        return

    ncols = min(3, len(panels))
    nrows = ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows),
                             squeeze=False)
    for idx, (area, x, y) in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        c  = AREA_COLORS.get(area, 'black')
        n  = len(x)
        ax.scatter(x, y, color=c, alpha=0.7, s=42, edgecolors=c, linewidths=0.5, zorder=3)
        if n >= 3 and np.std(x) > 0 and np.std(y) > 0:
            sl, intercept, r, p, _ = linregress(x, y)
            lo, hi  = _r_ci_fisher(r, n)
            x_line  = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, sl * x_line + intercept, color=c, ls='--', lw=1.6, zorder=2)
            ci_txt  = f'  [{lo:.2f}, {hi:.2f}]' if np.isfinite(lo) else ''
            stat_txt = f'r={r:.2f}{ci_txt}\np={p:.3f}   n={n}'
        else:
            stat_txt = f'n={n} (too few)'
        ax.text(0.04, 0.96, stat_txt, transform=ax.transAxes, fontsize=9,
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=c, alpha=0.85))
        ax.axhline(0, color='k', lw=0.6, alpha=0.35)
        ax.axvline(0, color='k', lw=0.6, alpha=0.35)
        ax.set_title(area, fontsize=13, fontweight='bold', color=c)
        ax.tick_params(labelsize=10)
        for sp in ax.spines.values(): sp.set_linewidth(1.2)
    for idx in range(len(panels), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.supxlabel('Decoding circ. corr. (delay)', fontsize=13)
    fig.supylabel('Neural–behavioural circ. corr. (delay)', fontsize=13)
    fig.suptitle('Decoding vs neural–behavioural correlation, per region (delay period)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, format='svg', bbox_inches='tight')
    print(f"Saved → {save_path}"); plt.close(fig)


def plot_decoding_vs_coupling_profiles(areas, dec_means, dec_sems, dec_pvals,
                                       cpl_means, cpl_sems, cpl_pvals, save_path,
                                       dec_label='Target decoding\n(distance from shuffle, σ)',
                                       cpl_label='Behavioural coupling\n(neural–beh error circ. corr.)'):
    """Side-by-side per-region profiles of two DIFFERENT quantities in the delay:
    how well each region represents the target (decoding) vs how strongly its
    trial-by-trial errors drive behaviour (coupling). Divergent profiles — and
    especially divergent *significant-region sets* — show behavioural relevance is
    not just a re-reading of decodability. Same region order in both panels; bar
    colour tracks each region across panels."""
    x = np.arange(len(areas))
    fig, (axd, axc) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, means, sems, pvals, ylabel in [
        (axd, dec_means, dec_sems, dec_pvals, dec_label),
        (axc, cpl_means, cpl_sems, cpl_pvals, cpl_label),
    ]:
        means = np.asarray(means, dtype=float)
        sems  = np.asarray(sems,  dtype=float)
        for i, area in enumerate(areas):
            c = AREA_COLORS.get(area, 'k')
            ax.bar(x[i], means[i], 0.7, yerr=(sems[i] if np.isfinite(sems[i]) else None),
                   capsize=4, color=c, edgecolor=c, linewidth=1.5, alpha=0.85)
            if pvals is not None:
                _annotate_bar(ax, x[i], means[i], sems[i], _stars(pvals[i]))
        ax.axhline(0, color='k', lw=0.8, alpha=0.5)
        ax.set_xticks(x); ax.set_xticklabels(areas, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(labelsize=11)
        for sp in ax.spines.values(): sp.set_linewidth(1.3)
    axd.set_title('Target decoding', fontsize=14, fontweight='bold')
    axc.set_title('Behavioural coupling', fontsize=14, fontweight='bold')
    fig.suptitle('Decoding vs behavioural coupling per region (delay period)\n'
                 'stars = significance vs 0 (* p<.05, ** p<.01, *** p<.001)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, format='svg', bbox_inches='tight')
    print(f"Saved → {save_path}"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — RECOMPUTE or LOAD CACHE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if RECOMPUTE or not os.path.exists(_results_pkl):

        # ── Load prediction files ─────────────────────────────────────────────
        _pred_path    = os.path.join(PKL_DIR, 'predicted_data.pkl')
        _shuf_path    = os.path.join(PKL_DIR, 'shuffled_data.pkl')
        _trial_path   = os.path.join(PKL_DIR, 'trial_data.pkl')
        _ct_pred_path = os.path.join(PKL_DIR, 'cross_temp_predictions.pkl')
        _ct_shuf_path = os.path.join(PKL_DIR, 'cross_temp_shuf_stats.pkl')

        print("Loading predicted_data.pkl ...")
        with open(_pred_path, 'rb') as fh:
            _pkg = pickle.load(fh)
        all_predictions = _pkg['predictions']

        print("Loading shuffled_data.pkl ...")
        with open(_shuf_path, 'rb') as fh:
            all_shuffles = pickle.load(fh)

        print("Loading trial_data.pkl ...")
        with open(_trial_path, 'rb') as fh:
            _td_pkg = pickle.load(fh)
        trial_data   = _td_pkg['trial_data']
        time_abs     = _td_pkg['time']
        n_time       = _td_pkg['n_time']
        angle_pairs  = _td_pkg['angle_pairs']
        step_s       = _td_pkg['step_s']
        beh_cols     = _td_pkg.get('beh_cols', [])
        ev_target_on  = _td_pkg.get('ev_target_on',  EV_TARGET_ON)
        ev_target_off = _td_pkg.get('ev_target_off', EV_TARGET_OFF)
        ev_response   = _td_pkg.get('ev_response',   EV_RESPONSE)
        delay_start   = _td_pkg.get('delay_start',   DELAY_START)
        delay_end     = _td_pkg.get('delay_end',     DELAY_END)
        time_rel      = time_abs - ev_target_on   # aligned: target on = 0

        have_ct = os.path.exists(_ct_pred_path) and os.path.exists(_ct_shuf_path)
        if have_ct:
            print("Loading cross_temp_predictions.pkl ...")
            with open(_ct_pred_path, 'rb') as fh:
                all_cross_preds = pickle.load(fh)
            with open(_ct_shuf_path, 'rb') as fh:
                all_ct_shuf = pickle.load(fh)
        else:
            print("No cross-temporal files found — skipping CT analysis.")
            all_cross_preds = {}; all_ct_shuf = {}

        # ── Circular correlation ──────────────────────────────────────────────
        print("\nComputing circular correlation ...")
        per_session_real, per_session_null = compute_per_session_circcorr(
            all_predictions, trial_data, angle_pairs, n_null_perms=N_NULL_PERMS)
        all_mean_scores, all_sem_scores, all_null_scores, all_null_sem_scores, all_n_sessions = average_circcorr(
            per_session_real, per_session_null)
        print("  Running cluster permutation tests (circcorr per angle) ...")
        all_cc_sig = _circcorr_sig(per_session_real, AREAS, list(angle_pairs.keys()),
                                    n_time, n_perm=N_PERM)
        for area, ang_sigs in all_cc_sig.items():
            for ang, mask in ang_sigs.items():
                print(f"    {area} [{ang}]: {mask.sum()} significant bins")

        _cc_path = os.path.join(PKL_DIR, 'per_session_circcorr.pkl')
        with open(_cc_path, 'wb') as fh:
            pickle.dump(per_session_real, fh)
        print(f"Saved → {_cc_path}")

        # ── Angular errors + z-scores ─────────────────────────────────────────
        print("\nCalculating angular errors ...")
        errors_angular   = calculate_angular_errors(all_predictions, trial_data, angle_pairs)
        shuf_err_angular = calculate_shuffle_angular_errors(all_shuffles, trial_data, angle_pairs)
        errors_angular_z = zscore_errors_against_shuffle(errors_angular, shuf_err_angular)

        # ── errors_angular.csv ────────────────────────────────────────────────
        df_errors = _build_errors_df(errors_angular, trial_data, time_abs, ANGLE_NAME, beh_cols)
        if SAVE_ERRORS_CSV:
            print("\nBuilding errors_angular.csv ...")
            _csv_path = os.path.join(PKL_DIR, 'errors_angular.csv')
            df_errors.to_csv(_csv_path, index=False)
            print(f"Saved → {_csv_path}  ({len(df_errors):,} rows)")

        # ── Cross-temporal analysis ───────────────────────────────────────────
        if have_ct:
            print("\nComputing cross-temporal circcorr ...")
            ct_real         = compute_cross_temporal_circcorr(all_cross_preds, trial_data, angle_pairs)
            mean_ct, sem_ct = _average_ct_mats(ct_real)

            print("Computing cross-temporal errors ...")
            ct_err                = compute_cross_temporal_errors(all_cross_preds, trial_data, angle_pairs)
            mean_cte, sem_cte     = _average_ct_mats(ct_err)
            ct_err_z              = zscore_cross_temporal_errors(ct_err, all_ct_shuf)
            mean_cte_z, sem_cte_z = _average_ct_mats(ct_err_z)

            _ct_cc_path  = os.path.join(PKL_DIR, 'cross_temp_circcorr.pkl')
            _ct_err_path = os.path.join(PKL_DIR, 'cross_temp_errors.pkl')
            with open(_ct_cc_path, 'wb') as fh:
                pickle.dump({'per_session': ct_real, 'mean': mean_ct, 'sem': sem_ct,
                             'time': time_abs, 'angle_pairs': angle_pairs}, fh)
            with open(_ct_err_path, 'wb') as fh:
                pickle.dump({'per_session': ct_err, 'mean': mean_cte, 'sem': sem_cte,
                             'per_session_z': ct_err_z, 'mean_z': mean_cte_z, 'sem_z': sem_cte_z,
                             'time': time_abs, 'angle_pairs': angle_pairs}, fh)
            print(f"Saved → {_ct_cc_path}")
            print(f"Saved → {_ct_err_path}")
        else:
            mean_ct = {}; sem_ct = {}; mean_cte_z = {}; sem_cte_z = {}

        # ── Neural–behavioural correlation ────────────────────────────────────
        print("\n── Neural–behavioural correlation over time ──")
        delay_mask_r    = (time_rel >= DELAY_START_REL)    & (time_rel <= DELAY_END_REL)
        response_mask_r = (time_rel >= RESPONSE_START_REL) & (time_rel <= RESPONSE_END_REL)

        nb_corr: dict = {}
        for area in AREAS:
            area_df = df_errors[df_errors['area'] == area]
            if area_df.empty: continue
            nb_corr[area] = _circ_corr_series(area_df, time_abs, n_time)
            m, _ = _mean_sem(nb_corr[area][1], axis=0)
            print(f"  {area}: {len(nb_corr[area][0])} sessions  |  {np.sum(~np.isnan(m))}/{n_time} bins")

        print("  Running cluster permutation tests (neural–beh) ...")
        nb_sig = {}
        for area in AREAS:
            if area not in nb_corr: continue
            nb_sig[area] = _cluster_perm_1samp(nb_corr[area][1], n_perm=N_PERM)
            print(f"    {area}: {nb_sig[area].sum()} significant bins")

        # ── Decoding circcorr curves ──────────────────────────────────────────
        dec_sessions: dict = {}; dec_r: dict = {}
        for area in AREAS:
            if area not in per_session_real: continue
            curves, keys = [], []
            for s_key, ang_dict in per_session_real[area].items():
                c = ang_dict.get(ANGLE_NAME)
                if c is None: continue
                arr = np.asarray(c, dtype=float)
                if len(arr) != n_time:
                    warnings.warn(f"Skipping {area} sess {s_key}: length {len(arr)} ≠ {n_time}")
                    continue
                curves.append(arr); keys.append(int(s_key))
            if not curves: continue
            order = np.argsort(keys)
            dec_sessions[area] = [keys[i] for i in order]
            dec_r[area]        = np.stack([curves[i] for i in order])

        dec_z_sessions: dict = {}; dec_z: dict = {}
        for area in AREAS:
            if area not in errors_angular_z: continue
            curves, keys = [], []
            for s_key, ang_dict in errors_angular_z[area].items():
                c = ang_dict.get(ANGLE_NAME)
                if c is None: continue
                arr = np.asarray(c, dtype=float)
                if len(arr) != n_time: continue
                curves.append(arr); keys.append(int(s_key))
            if not curves: continue
            order = np.argsort(keys)
            dec_z_sessions[area] = [keys[i] for i in order]
            dec_z[area]          = np.stack([curves[i] for i in order])

        print("\n  Running cluster permutation tests (decoding circcorr) ...")
        dec_sig = {}
        for area in AREAS:
            if area not in dec_r: continue
            dec_sig[area] = _cluster_perm_1samp(dec_r[area], n_perm=N_PERM)
            print(f"    {area}: {dec_sig[area].sum()} significant bins")

        print("  Running cluster permutation tests (decoding z-score) ...")
        dec_z_sig = {}
        for area in AREAS:
            if area not in dec_z: continue
            dec_z_sig[area] = _cluster_perm_1samp(dec_z[area], n_perm=N_PERM)
            print(f"    {area}: {dec_z_sig[area].sum()} significant bins")

        # ── Method A: mean per window ─────────────────────────────────────────
        def _window_stats(r_mat_dict, mask):
            means, sems, pvals, per_sess = [], [], [], {}
            for area in AREAS:
                if area not in r_mat_dict:
                    means.append(np.nan); sems.append(np.nan); pvals.append(np.nan)
                    continue
                sessions_l, r_mat = r_mat_dict[area]
                vals = np.nanmean(r_mat[:, mask], axis=1)
                per_sess[area] = {'sessions': sessions_l, 'values': vals}
                m, s = _mean_sem(vals); p = _test_vs_zero(vals)
                means.append(m); sems.append(s); pvals.append(p)
            return means, sems, pvals, per_sess

        nb_r_dict  = nb_corr
        dec_r_dict = {a: (dec_sessions[a], dec_r[a]) for a in dec_sessions}
        dec_z_dict = {a: (dec_z_sessions[a], dec_z[a]) for a in dec_z_sessions}

        d_means,  d_sems,  d_pvals,  nb_delay_per_sess    = _window_stats(nb_r_dict,  delay_mask_r)
        r_means,  r_sems,  r_pvals,  nb_response_per_sess = _window_stats(nb_r_dict,  response_mask_r)
        dd_means, dd_sems, dd_pvals, dec_delay_per_sess   = _window_stats(dec_r_dict, delay_mask_r)
        dr_means, dr_sems, dr_pvals, dec_response_per_sess = _window_stats(dec_r_dict, response_mask_r)
        zd_means, zd_sems, zd_pvals, _ = _window_stats(dec_z_dict, delay_mask_r)
        zr_means, zr_sems, zr_pvals, _ = _window_stats(dec_z_dict, response_mask_r)

        _f_arrs = [nb_delay_per_sess[a]['values'] for a in FRONTAL_AREAS if a in nb_delay_per_sess]
        _s_arrs = [nb_delay_per_sess[a]['values'] for a in SENSORY_AREAS if a in nb_delay_per_sess]
        if _f_arrs and _s_arrs:
            frontal_d = np.concatenate(_f_arrs); frontal_d = frontal_d[np.isfinite(frontal_d)]
            sensory_d = np.concatenate(_s_arrs); sensory_d = sensory_d[np.isfinite(sensory_d)]
            if len(frontal_d) >= 3 and len(sensory_d) >= 3:
                _, p_grp = mannwhitneyu(frontal_d, sensory_d, alternative='greater')
                group_label = f'Frontal > Sensory (delay): p={p_grp:.3f} {_stars(p_grp)}'
                print(f"\n  Frontal vs Sensory (delay): p={p_grp:.4f} {_stars(p_grp)}")
            else:
                group_label = None
        else:
            group_label = None

        # ── Method B: slope per window ────────────────────────────────────────
        def _slope_stats(r_mat_dict, d_mask, r_mask):
            sd_m, sd_s, sd_p, sr_m, sr_s, sr_p = [], [], [], [], [], []
            for area in AREAS:
                if area not in r_mat_dict:
                    for lst in (sd_m, sd_s, sd_p, sr_m, sr_s, sr_p):
                        lst.append(np.nan)
                    continue
                _, r_mat = r_mat_dict[area]
                sl_d, _ = _slope_per_session(r_mat, d_mask, time_rel[d_mask])
                sl_r, _ = _slope_per_session(r_mat, r_mask, time_rel[r_mask])
                dm, ds = _mean_sem(sl_d); dp = _test_vs_zero(sl_d)
                rm, rs = _mean_sem(sl_r); rp = _test_vs_zero(sl_r)
                sd_m.append(dm); sd_s.append(ds); sd_p.append(dp)
                sr_m.append(rm); sr_s.append(rs); sr_p.append(rp)
            return sd_m, sd_s, sd_p, sr_m, sr_s, sr_p

        sd_means, sd_sems, sd_pvals, sr_means, sr_sems, sr_pvals = _slope_stats(
            nb_r_dict, delay_mask_r, response_mask_r)
        dsd_means, dsd_sems, dsd_pvals, dsr_means, dsr_sems, dsr_pvals = _slope_stats(
            dec_r_dict, delay_mask_r, response_mask_r)

        # ── Save legacy neurobeh_baseline.pkl ─────────────────────────────────
        _bl_path = os.path.join(PKL_DIR, 'neurobeh_baseline.pkl')
        with open(_bl_path, 'wb') as fh:
            pickle.dump({
                'time': time_abs, 'delay_idx': (time_abs >= delay_start) & (time_abs <= delay_end),
                'areas': AREAS, 'angle_pairs': angle_pairs,
                'per_session_real': per_session_real, 'per_session_null': per_session_null,
                'mean_circcorr': all_mean_scores, 'sem_circcorr': all_sem_scores,
                'null_circcorr': all_null_scores, 'null_sem_circcorr': all_null_sem_scores,
                'n_sessions_circcorr': all_n_sessions,
                'errors_angular': errors_angular, 'errors_angular_z': errors_angular_z,
                'trial_data': trial_data,
            }, fh)
        print(f"\nSaved → {_bl_path}")

        # ── Build + save comprehensive results dict ───────────────────────────
        results = {
            'meta': {
                'areas': AREAS, 'angle_pairs': angle_pairs, 'angle_name': ANGLE_NAME,
                'time_abs': time_abs, 'time_rel': time_rel, 'n_time': n_time,
                'step_s': step_s,
                'ev_target_on': ev_target_on, 'ev_target_off': ev_target_off,
                'ev_response': ev_response, 'delay_start': delay_start, 'delay_end': delay_end,
                'delay_start_rel': DELAY_START_REL, 'delay_end_rel': DELAY_END_REL,
                'response_start_rel': RESPONSE_START_REL, 'response_end_rel': RESPONSE_END_REL,
                'ev_target_off_rel': EV_TARGET_OFF_REL, 'ev_fixpt_off_rel': EV_FIXPT_OFF_REL,
                'n_perm': N_PERM, 'alpha': ALPHA,
                'frontal_areas': FRONTAL_AREAS, 'sensory_areas': SENSORY_AREAS,
            },
            'mean_scores': all_mean_scores, 'sem_scores': all_sem_scores, 'null_scores': all_null_scores,
            'null_sem_scores': all_null_sem_scores,
            'n_sessions': all_n_sessions, 'cc_sig': all_cc_sig,
            'per_session_real': per_session_real, 'per_session_null': per_session_null,
            'errors_angular_z': errors_angular_z,
            'ct': {'mean_cc': mean_ct, 'sem_cc': sem_ct, 'mean_z': mean_cte_z, 'sem_z': sem_cte_z},
            'nb_sessions': {a: s for a, (s, _) in nb_corr.items()},
            'nb_r_mat':    {a: r for a, (_, r) in nb_corr.items()},
            'nb_sig': nb_sig,
            'dec_sessions': dec_sessions, 'dec_r': dec_r, 'dec_sig': dec_sig,
            'dec_z_sessions': dec_z_sessions, 'dec_z': dec_z, 'dec_z_sig': dec_z_sig,
            'nb_mean':  {'delay_means': d_means,   'delay_sems': d_sems,   'delay_pvals': d_pvals,
                         'response_means': r_means, 'response_sems': r_sems,'response_pvals': r_pvals,
                         'group_label': group_label},
            'dec_mean': {'delay_means': dd_means,  'delay_sems': dd_sems,  'delay_pvals': dd_pvals,
                         'response_means': dr_means,'response_sems': dr_sems,'response_pvals': dr_pvals},
            'dec_mean_z':{'delay_means': zd_means,  'delay_sems': zd_sems, 'delay_pvals': zd_pvals,
                          'response_means': zr_means,'response_sems': zr_sems,'response_pvals': zr_pvals},
            'nb_slope': {'delay_means': sd_means,  'delay_sems': sd_sems,  'delay_pvals': sd_pvals,
                         'response_means': sr_means,'response_sems': sr_sems,'response_pvals': sr_pvals},
            'dec_slope':{'delay_means': dsd_means, 'delay_sems': dsd_sems, 'delay_pvals': dsd_pvals,
                         'response_means': dsr_means,'response_sems': dsr_sems,'response_pvals': dsr_pvals},
            'nb_delay_per_sess':      nb_delay_per_sess,
            'nb_response_per_sess':   nb_response_per_sess,
            'dec_delay_per_sess':     dec_delay_per_sess,
            'dec_response_per_sess':  dec_response_per_sess,
        }
        with open(_results_pkl, 'wb') as fh:
            pickle.dump(results, fh)
        print(f"Saved → {_results_pkl}")

    else:
        print(f"Loading cached results from:\n  {_results_pkl}")
        with open(_results_pkl, 'rb') as fh:
            results = pickle.load(fh)

    # ══════════════════════════════════════════════════════════════════════════
    # UNPACK RESULTS → plotting variables (same in both paths)
    # ══════════════════════════════════════════════════════════════════════════

    _meta          = results['meta']
    areas          = _meta['areas']
    angle_pairs    = _meta['angle_pairs']
    angle_name     = _meta.get('angle_name', ANGLE_NAME)
    time_abs       = _meta['time_abs']
    time_rel       = _meta['time_rel']
    n_time         = _meta['n_time']
    step_s         = _meta['step_s']
    delay_start_rel    = _meta['delay_start_rel']
    delay_end_rel      = _meta['delay_end_rel']
    response_start_rel = _meta['response_start_rel']
    response_end_rel   = _meta['response_end_rel']

    all_mean_scores  = results['mean_scores']
    all_sem_scores   = results['sem_scores']
    all_null_scores  = results['null_scores']
    all_null_sem_scores = results.get('null_sem_scores', {})
    all_n_sessions   = results.get('n_sessions', {})
    all_cc_sig       = results.get('cc_sig', {})
    errors_angular_z = results['errors_angular_z']

    _ct = results['ct']
    mean_ct    = _ct['mean_cc']
    mean_cte_z = _ct['mean_z']

    nb_corr  = {a: (results['nb_sessions'][a], results['nb_r_mat'][a])
                for a in results['nb_sessions']}
    nb_sig   = results['nb_sig']
    dec_sessions = results['dec_sessions']
    dec_r        = results['dec_r']
    dec_sig      = results['dec_sig']
    dec_z_sig    = results.get('dec_z_sig', {})

    delay_mask_r    = (time_rel >= delay_start_rel)    & (time_rel <= delay_end_rel)
    response_mask_r = (time_rel >= response_start_rel) & (time_rel <= response_end_rel)

    nb_delay_per_sess    = results['nb_delay_per_sess']
    nb_response_per_sess = results['nb_response_per_sess']
    dec_delay_per_sess   = results['dec_delay_per_sess']
    dec_response_per_sess = results['dec_response_per_sess']

    _nbm  = results['nb_mean'];   _decm = results['dec_mean']
    _decz = results['dec_mean_z'];_nbs  = results['nb_slope'];  _decs = results['dec_slope']

    # ══════════════════════════════════════════════════════════════════════════
    # GENERATE ALL PLOTS
    # ══════════════════════════════════════════════════════════════════════════

    print("\n── Generating plots ──")

    # 01. Circular correlation time series
    plot_circcorr(all_mean_scores, all_sem_scores, all_null_scores,
                  time_rel, areas, angle_pairs,
                  n_sessions=all_n_sessions, cc_sig=all_cc_sig,
                  null_sem=all_null_sem_scores)

    # 02. Prediction errors (distance from shuffle)
    plot_prediction_errors(errors_angular_z, time_rel, areas, angle_pairs, sig_dict=dec_z_sig)

    # 04–05. Cross-temporal heatmaps (if computed)
    if mean_ct:
        _plot_ct_heatmaps(mean_ct, time_rel, areas, angle_pairs,
                          '04_cross_temporal_{angle}.svg',
                          cmap='RdBu_r', zlabel='Circ. corr.', z_sig=None)
    if mean_cte_z:
        _plot_ct_heatmaps(mean_cte_z, time_rel, areas, angle_pairs,
                          '05_cross_temporal_error_{angle}.svg',
                          cmap='RdBu_r', zlabel='Distance from shuffle (σ)', z_sig=1.645)

    # 06. Neural–behavioural correlation time series
    plot_neurobeh_timeseries(
        nb_corr, nb_sig, time_rel, areas,
        title='Neural decoding error vs behavioural error',
        save_path=os.path.join(FIG_DIR, '06_neurobeh_timeseries.svg'))

    # 12. Scatter: neural–beh vs decoding
    plot_neurobeh_vs_decoding(
        nb_delay_per_sess, dec_delay_per_sess, areas,
        save_path=os.path.join(FIG_DIR, '12_neurobeh_vs_decoding.svg'))

    # 12b. Regional-profile dissociation: target decoding vs behavioural coupling
    plot_decoding_vs_coupling_profiles(
        areas,
        _decz['delay_means'], _decz['delay_sems'], _decz['delay_pvals'],
        _nbm['delay_means'],  _nbm['delay_sems'],  _nbm['delay_pvals'],
        save_path=os.path.join(FIG_DIR, '12b_decoding_vs_coupling_profiles.svg'))

    print("\n" + "═" * 60)
    print("ANALYSIS COMPLETE")
    print(f"  pkl    : {PKL_DIR}")
    print(f"  figures: {FIG_DIR}")
    print("═" * 60)


if __name__ == '__main__':
    main()
