"""
extra_analysis.py
=================
Exploratory inter-area information-flow analyses, kept separate from the
established pipeline in analysis.py (which holds only results we are confident
about). Structure here is expected to change as the exploration evolves.

  (A) Cross-correlation of trial-by-trial decoding errors between areas,
      summarised as a feedforward ratio (FFR).        → figures 03 / 13 / 14
  (B) CCA of residual population activity (Semedo et al. 2022, Nat Commun,
      Fig 3b/3d): population-correlation-vs-lag → FFR. → figures 15 / 16 / 17

Toggle the two analyses with RUN_XCORR / RUN_CCA in the config below.

Inputs (in PKL_DIR / the decoder's data dir)
  trial_data.pkl          time axis, step_s, angle_pairs, event times  [for (A)]
  neurobeh_baseline.pkl   errors_angular (raw per-trial decoding errors) [for (A)]
  test.pkl                raw spike counts (n_trials, n_neurons, n_time)  [for (B)]
  interarea_xcorr.pkl / cca_popcorr.pkl   caches written here (used when RECOMPUTE=False)

Run
  python extra_analysis.py
Edit aesthetics (figsize, line width, colours, y-limits) directly in the
plotting functions below, then rerun with RECOMPUTE=False to replot from cache.
"""

import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from scipy.stats import wilcoxon

from config import PKL_DIR, FIG_DIR, AREA_COLORS, EV_TARGET_ON

from core.stats import stars


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  ← edit here
# ══════════════════════════════════════════════════════════════════════════════

RECOMPUTE = True   # False → load cached results and only replot

RUN_XCORR = True   # (A) decoding-error cross-correlation  → figs 03 / 13 / 14
RUN_CCA   = True   # (B) CCA of residual population activity → figs 15 / 16 / 17

# Areas ordered ASCENDING the cortical hierarchy (sensory → frontal). With this
# order, a positive lag = lower area leads = feedforward.
from config import AREAS_SENSORY_FIRST as AREAS  # flow analyses read V4 -> PFC

# Sensory / frontal groups for the aggregated flow figure (fig 14). Editable
# here independently of analysis.py so groupings can be explored freely.
SENSORY_AREAS = ['IT', 'MT', 'V4']
FRONTAL_AREAS = ['PFC', 'FEF', 'LIP']

# ── Inter-area cross-correlation windows (s, relative to target onset = 0) ────
INTERAREA_WINDOWS = {
    'encoding': (0.00, 0.45),   # stimulus-evoked sweep — expect feedforward (sensory → frontal)
    'delay':    (0.45, 0.85),   # late delay / maintenance
    'response': (0.85, 1.60),   # saccade execution — expect frontoparietal-led
}
# Max inter-area lag scanned (±s). Kept well below the shortest window so the
# two lagged windows still overlap substantially at the extremes.
INTERAREA_XCORR_MAX_LAG_S = 0.20
# Feedforward-ratio shuffle null: per pair/window/session we break the trial
# correspondence between the two areas N times to build a null FFR distribution.
N_INTERAREA_SHUFFLE    = 200
INTERAREA_SHUFFLE_SEED = 0

# ── (B) CCA of residual population activity (Semedo et al. 2022, Fig 3b/3d) ───
# Reads RAW spike counts (sliding-window 100 ms / 25 ms, NOT exp-smoothed) so the
# heavy 1.5 s smoothing in decoder.py does not erase the lag structure. The same
# epoch windows (INTERAREA_WINDOWS) are reused. Population correlation = first
# canonical correlation of the two areas' residual (per-condition mean-subtracted)
# activity; its asymmetry across inter-area lag gives the feedforward ratio.
CCA_DATA_PATH    = '/home/aarghavan/aslan/data/test.pkl'   # raw spikecounts (= decoder DATA_PATH)
CCA_T_START      = -2.5      # epoch start (s); must match decoder.py
CCA_ORIG_BIN     = 0.025     # raw bin (s)
CCA_WINDOW_S     = 0.10      # sliding-window width (s)
CCA_STEP_S       = 0.025     # step between bins (s)
CCA_EV_TARGET_ON = 1.70      # target onset (abs s); windows are relative to this
CCA_N_PCS        = 10        # top-k PCA per area before CCA (regularisation)
CCA_MAX_LAG_S    = 0.15      # ± inter-area lag scanned (s)
CCA_MIN_NEURONS  = 10        # skip an area in a session below this many units
N_CCA_SHUFFLE    = 200       # trial-correspondence shuffles for the FFR null
CCA_SHUFFLE_SEED = 0

ALPHA = 0.05


# ══════════════════════════════════════════════════════════════════════════════
# (A) INTER-AREA CROSS-CORRELATION OF DECODING ERRORS
# ══════════════════════════════════════════════════════════════════════════════

def _prep_errors_window(err_mat, mask):
    """Slice, residualize per bin, z-score per trial."""
    w  = err_mat[:, mask].astype(float)
    w  = w - np.nanmean(w, axis=0, keepdims=True)
    w  = w - np.nanmean(w, axis=1, keepdims=True)
    sd = np.nanstd(w, axis=1, keepdims=True)
    return w / np.where(sd > 0, sd, np.nan)


def _xcorr_lagged(A, B, max_lag):
    """Per-trial cross-correlation at lags −max_lag…+max_lag.
    Returns (n_trials, 2*max_lag+1)."""
    n_trials, n_bins = A.shape
    out = np.full((n_trials, 2 * max_lag + 1), np.nan)
    for li, lag in enumerate(range(-max_lag, max_lag + 1)):
        if lag > 0:
            a = A[:, :n_bins - lag]; b = B[:, lag:]
        elif lag < 0:
            a = A[:, -lag:];         b = B[:, :n_bins + lag]
        else:
            a, b = A, B
        out[:, li] = np.nansum(a * b, axis=1) / n_bins
    return out


def _ffr_from_curve(curve, lags):
    """Feedforward ratio of a cross-correlation curve, as a normalized lag
    asymmetry bounded in [−1, +1]:
        (Σ C[lag>0] − Σ C[lag<0]) / Σ |C[lag≠0]|.
    +1 → fully feedforward (row area leads column area); −1 → fully feedback;
    0 → symmetric (no net lead). Normalizing by total absolute area keeps the
    ratio stable for the signed, mean-subtracted cross-correlation used here
    (it reduces to the classic Σpos/Σneg ratio when the curve is all-positive).
    NaN if the curve carries no power."""
    c     = np.asarray(curve, dtype=float)
    pos   = np.nansum(c[lags > 0])
    neg   = np.nansum(c[lags < 0])
    total = np.nansum(np.abs(c[lags != 0]))
    if not np.isfinite(total) or total <= 0:
        return np.nan
    return (pos - neg) / total


def _null_ffrs_via_gram(A, B, max_lag, lags, n_shuffle, rng):
    """Null FFR distribution under broken trial correspondence, computed without
    recomputing the cross-correlation per shuffle. Each shuffle only re-pairs
    trials of B with A, so all shuffles read off a single trial×trial Gram matrix
    per lag (G[t1,t2] = Σ_bin A[t1]·B[t2]). Matches the naive per-shuffle loop to
    machine precision but is far faster (one BLAS matmul per lag vs n_shuffle
    nan-reductions). Zero-filling reproduces nansum since 0 contributes 0."""
    if n_shuffle <= 0:
        return np.empty(0)
    A0 = np.nan_to_num(A); B0 = np.nan_to_num(B)
    T, n_bins = A0.shape
    rows = np.arange(T)
    P    = np.stack([rng.permutation(T) for _ in range(n_shuffle)])   # (S, T)
    null_curves = np.empty((n_shuffle, len(lags)))
    for li, lag in enumerate(range(-max_lag, max_lag + 1)):
        if lag > 0:   a = A0[:, :n_bins - lag]; b = B0[:, lag:]
        elif lag < 0: a = A0[:, -lag:];         b = B0[:, :n_bins + lag]
        else:         a, b = A0, B0
        G = (a @ b.T) / n_bins                                        # (T, T)
        null_curves[:, li] = G[rows[None, :], P].mean(axis=1)         # mean_t G[t, perm[t]]
    pos   = null_curves[:, lags > 0].sum(axis=1)
    neg   = null_curves[:, lags < 0].sum(axis=1)
    total = np.abs(null_curves[:, lags != 0]).sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(total > 0, (pos - neg) / total, np.nan)


def _interarea_xcorr_session(session, sess_errs_by_area, areas, period_masks, max_lag,
                             n_shuffle=0, seed=0):
    n_lags = 2 * max_lag + 1
    lags   = np.arange(-max_lag, max_lag + 1)
    rng    = np.random.default_rng((seed, hash(str(session)) & 0xFFFFFFFF))
    avail  = [a for a in areas if a in sess_errs_by_area]
    out    = {}     # curves:  out[angle][period][(a,b)] = mean curve
    ffr    = {}     # FFR:     ffr[angle][period][(a,b)] = (ffr_obs, null_ffrs)
    angle_names = set()
    for a in avail:
        angle_names.update(sess_errs_by_area[a].keys())

    for angle_name in angle_names:
        with_angle = [a for a in avail if angle_name in sess_errs_by_area[a]]
        if len(with_angle) < 1: continue
        prepped = {p: {} for p in period_masks}
        for area in with_angle:
            err = sess_errs_by_area[area][angle_name]
            for period, mask in period_masks.items():
                prepped[period][area] = _prep_errors_window(err, mask)
        out[angle_name] = {}; ffr[angle_name] = {}
        for period in period_masks:
            out[angle_name][period] = {}; ffr[angle_name][period] = {}
            for i, a in enumerate(with_angle):
                A = prepped[period][a]
                for j in range(i, len(with_angle)):
                    b  = with_angle[j]
                    B  = prepped[period][b]
                    curve = np.nanmean(_xcorr_lagged(A, B, max_lag), axis=0)
                    out[angle_name][period][(a, b)] = curve
                    if i == j:   # diagonal = autocorrelation, no direction
                        continue
                    ffr_obs   = _ffr_from_curve(curve, lags)
                    null_ffrs = _null_ffrs_via_gram(A, B, max_lag, lags, n_shuffle, rng)
                    ffr[angle_name][period][(a, b)] = (ffr_obs, null_ffrs)
    return session, out, ffr, n_lags


def compute_interarea_xcorr(errors_angular, time_abs, areas, periods, max_lag_s,
                            n_shuffle=0, seed=0):
    step_s       = float(time_abs[1] - time_abs[0]) if len(time_abs) > 1 else 0.025
    period_masks = {p: (time_abs >= lo) & (time_abs < hi) for p, (lo, hi) in periods.items()}
    max_lag      = int(round(max_lag_s / step_s))
    lags         = np.arange(-max_lag, max_lag + 1)

    print("  Inter-area xcorr window bins:")
    for p, m in period_masks.items():
        print(f"    {p:10s}: {int(m.sum())} bins  [{periods[p][0]:.2f}, {periods[p][1]:.2f})s")
    print(f"  max_lag = {max_lag} bins  (±{max_lag * step_s:.3f}s);  FFR shuffles = {n_shuffle}")

    sess_ids = set()
    for area in errors_angular:
        sess_ids.update(errors_angular[area].keys())
    by_sess = {s: {} for s in sess_ids}
    for area, sess_dict in errors_angular.items():
        for s, ang_dict in sess_dict.items():
            by_sess[s][area] = ang_dict

    results = Parallel(n_jobs=-1, prefer='threads')(
        delayed(_interarea_xcorr_session)(
            s, by_sess[s], areas, period_masks, max_lag, n_shuffle, seed)
        for s in sorted(by_sess.keys())
    )

    per_session, ffr_per_session = {}, {}
    for s, sess_out, sess_ffr, _ in results:
        for angle_name, periods_dict in sess_out.items():
            per_session.setdefault(angle_name, {})
            for period, pair_dict in periods_dict.items():
                per_session[angle_name].setdefault(period, {})
                for pair, curve in pair_dict.items():
                    per_session[angle_name][period].setdefault(pair, {})
                    per_session[angle_name][period][pair][s] = curve
        for angle_name, periods_dict in sess_ffr.items():
            ffr_per_session.setdefault(angle_name, {})
            for period, pair_dict in periods_dict.items():
                ffr_per_session[angle_name].setdefault(period, {})
                for pair, val in pair_dict.items():
                    ffr_per_session[angle_name][period].setdefault(pair, {})
                    ffr_per_session[angle_name][period][pair][s] = val
    return per_session, lags, ffr_per_session


def summarize_ffr(ffr_per_session):
    """Per pair/window: mean FFR across sessions + shuffle-test p-value.
    The null is the distribution of the session-averaged FFR under broken
    trial correspondence (two-sided, add-one corrected)."""
    summary = {}
    for angle, pdict in ffr_per_session.items():
        summary[angle] = {}
        for period, pairdict in pdict.items():
            summary[angle][period] = {}
            for pair, sessdict in pairdict.items():
                obs_vals, null_rows = [], []
                for _s, (ffr_obs, null_arr) in sessdict.items():
                    if not np.isfinite(ffr_obs): continue
                    obs_vals.append(ffr_obs)
                    null_rows.append(np.asarray(null_arr, dtype=float))
                if not obs_vals: continue
                obs_vals = np.asarray(obs_vals)
                mean_obs = float(np.nanmean(obs_vals))
                sem_obs  = float(np.nanstd(obs_vals) / np.sqrt(len(obs_vals)))
                p_val = np.nan
                if null_rows:
                    min_k    = min(len(r) for r in null_rows)
                    null_mat = np.stack([r[:min_k] for r in null_rows], axis=0)  # (n_sess, n_shuf)
                    null_mean = np.nanmean(null_mat, axis=0)                     # (n_shuf,)
                    null_mean = null_mean[np.isfinite(null_mean)]
                    if len(null_mean):
                        p_val = (np.sum(np.abs(null_mean) >= abs(mean_obs)) + 1) / (len(null_mean) + 1)
                summary[angle][period][pair] = {
                    'mean': mean_obs, 'sem': sem_obs, 'p': float(p_val),
                    'n': len(obs_vals), 'values': obs_vals,
                }
    return summary


def aggregate_group_flow(ffr_per_session, sensory, frontal):
    """Collapse all sensory↔frontal area pairs into a single directional flow
    index per window. Orientation: + = sensory leads frontal (feedforward),
    − = frontal leads sensory (feedback). Within each session we average FFR
    over all available cross-group pairs (session = unit of independence), then
    average across sessions; the shuffle p-value compares the session-averaged
    flow to the session-averaged null. Returns summary[angle][period] with
    mean/sem/p/n_sessions and per-session values (for the epoch paired test)."""
    sset, fset = set(sensory), set(frontal)
    summary = {}
    for angle, pdict in ffr_per_session.items():
        summary[angle] = {}
        for period, pairdict in pdict.items():
            sess_obs, sess_null = {}, {}     # session -> list over cross-pairs
            for (a, b), sessd in pairdict.items():
                if   a in sset and b in fset: sign = +1.0    # a(sensory) leads b(frontal) = feedforward
                elif a in fset and b in sset: sign = -1.0    # reorient so + stays "sensory leads"
                else: continue
                for s, (ffr_obs, null_arr) in sessd.items():
                    if not np.isfinite(ffr_obs): continue
                    sess_obs.setdefault(s, []).append(sign * ffr_obs)
                    sess_null.setdefault(s, []).append(sign * np.asarray(null_arr, float))
            if not sess_obs: continue
            s_keys   = sorted(sess_obs.keys())
            sess_mean = {s: float(np.nanmean(sess_obs[s])) for s in s_keys}
            sess_null_mean = {}
            for s in s_keys:
                mk = min(len(r) for r in sess_null[s])
                sess_null_mean[s] = np.nanmean(np.stack([r[:mk] for r in sess_null[s]], 0), 0)

            obs_vals = np.array([sess_mean[s] for s in s_keys])
            mean_obs = float(np.nanmean(obs_vals))
            sem_obs  = float(np.nanstd(obs_vals) / np.sqrt(len(obs_vals))) if len(obs_vals) else np.nan
            mk = min(len(sess_null_mean[s]) for s in s_keys)
            null_agg = np.nanmean(np.stack([sess_null_mean[s][:mk] for s in s_keys], 0), 0)
            null_agg = null_agg[np.isfinite(null_agg)]
            p = ((np.sum(np.abs(null_agg) >= abs(mean_obs)) + 1) / (len(null_agg) + 1)
                 if len(null_agg) else np.nan)
            summary[angle][period] = {
                'mean': mean_obs, 'sem': sem_obs, 'p': float(p),
                'n_sessions': len(s_keys), 'session_vals': sess_mean,
            }
    return summary


def _paired_epoch_flow_test(summary_angle, e1, e2):
    """Wilcoxon signed-rank on per-session flow values shared between two
    windows. Returns (p, n_pairs) or (nan, 0)."""
    if e1 not in summary_angle or e2 not in summary_angle:
        return np.nan, 0
    v1, v2 = summary_angle[e1]['session_vals'], summary_angle[e2]['session_vals']
    shared = sorted(set(v1) & set(v2))
    x = np.array([v1[s] for s in shared]); y = np.array([v2[s] for s in shared])
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 5 or np.allclose(x, y):
        return np.nan, len(x)
    try:
        _, p = wilcoxon(x, y)
    except ValueError:
        return np.nan, len(x)
    return float(p), len(x)


def average_interarea_xcorr(per_session):
    mean_x, sem_x = {}, {}
    for angle_name, periods_dict in per_session.items():
        mean_x[angle_name] = {}; sem_x[angle_name] = {}
        for period, pair_dict in periods_dict.items():
            mean_x[angle_name][period] = {}; sem_x[angle_name][period] = {}
            for pair, sess_dict in pair_dict.items():
                curves = [c for c in sess_dict.values() if c is not None]
                if not curves: continue
                stack = np.stack(curves, axis=0)
                n_eff = np.sum(~np.isnan(stack), axis=0)
                mean_x[angle_name][period][pair] = np.nanmean(stack, axis=0)
                sem_x [angle_name][period][pair] = (
                    np.nanstd(stack, axis=0) / np.where(n_eff > 0, np.sqrt(n_eff), np.nan))
    return mean_x, sem_x


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_interarea_xcorr(mean_x, sem_x, lags, areas, angle_pairs, step_s):
    """One figure per time window. In each NxN grid the upper triangle shows the
    pairwise lead/lag cross-correlation; with AREAS ordered ascending the
    hierarchy, a positive-lag peak means the ROW area leads the COLUMN area
    (feedforward), a negative-lag peak means the column area leads (feedback)."""
    lag_s = lags * step_s
    period_styles = {
        'encoding': {'color': '#2ca02c', 'label': 'Encoding (stimulus)'},
        'delay':    {'color': '#1f77b4', 'label': 'Delay (maintenance)'},
        'response': {'color': '#d62728', 'label': 'Response (saccade)'},
    }

    def _vlim(vals, default=0.1, floor=0.05, pct=99.5):
        if not vals: return default
        flat = np.concatenate(vals); finite = flat[np.isfinite(flat)]
        return max(floor, float(np.percentile(np.abs(finite), pct))) if len(finite) else default

    for angle_name, periods_dict in mean_x.items():
        present = {a for pd in periods_dict.values() for pair in pd for a in pair}
        areas_p = [a for a in areas if a in present]
        n = len(areas_p)
        if n < 2: continue

        # one figure per time window, in encoding → delay → response order
        for period in (p for p in period_styles if p in periods_dict):
            pair_curves = periods_dict[period]
            style       = period_styles[period]

            diag_vals = [c for (pa, pb), c in pair_curves.items()
                         if c is not None and pa == pb]
            off_vals  = [c for (pa, pb), c in pair_curves.items()
                         if c is not None and pa != pb]
            vlim_d = _vlim(diag_vals, default=1.0, floor=0.5, pct=100)
            vlim_o = _vlim(off_vals)

            fig, axes = plt.subplots(n, n, figsize=(1.8 * n, 1.6 * n),
                                      sharex=True, sharey=False, squeeze=False)
            for i, a in enumerate(areas_p):
                for j, b in enumerate(areas_p):
                    ax = axes[i][j]
                    if j < i: ax.set_visible(False); continue
                    pair_key = (a, b)
                    curve = pair_curves.get(pair_key)
                    s_val = sem_x.get(angle_name, {}).get(period, {}).get(pair_key)
                    if curve is not None:
                        ax.plot(lag_s, curve, color=style['color'], lw=1.5)
                        if s_val is not None:
                            ax.fill_between(lag_s, curve - s_val, curve + s_val,
                                            color=style['color'], alpha=0.20, lw=0)
                    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
                    ax.axvline(0, color='k', lw=0.5, alpha=0.3)
                    ax.set_ylim((-vlim_d, vlim_d) if j == i else (-vlim_o, vlim_o))
                    ax.set_xlim(lag_s[0], lag_s[-1])
                    if i == 0: ax.set_title(b, fontsize=10, fontweight='bold',
                                             color=AREA_COLORS.get(b, 'black'))
                    if j == i: ax.set_ylabel(a, fontsize=10, fontweight='bold',
                                              color=AREA_COLORS.get(a, 'black'))
                    elif j > i + 1: ax.tick_params(labelleft=False)
                    if i == n - 1: ax.set_xlabel('Lag (s)', fontsize=8)
                    ax.tick_params(labelsize=7)

            fig.text(0.5, 0.005,
                     'lag > 0: row leads column (feedforward, ↑ hierarchy)   |   '
                     'lag < 0: column leads row (feedback, ↓ hierarchy)',
                     ha='center', fontsize=9, style='italic')
            fig.suptitle(
                f'Inter-area cross-correlation of decoding errors\n'
                f'{style["label"]} — {angle_name}   '
                f'(diagonal ±{vlim_d:.2f}, off-diagonal ±{vlim_o:.2f})',
                fontsize=12, fontweight='bold')
            plt.tight_layout(rect=[0, 0.05, 1, 0.94])
            p = os.path.join(FIG_DIR, f'03_interarea_xcorr_{period}_{angle_name}.svg')
            fig.savefig(p, format='svg', bbox_inches='tight')
            print(f"Saved → {p}"); plt.close(fig)


def plot_ffr_heatmap(ffr_summary, areas, angle_pairs, fname_prefix='13',
                     source='decoding errors'):
    """One heatmap per window: mean feedforward ratio per area pair (upper
    triangle), red = feedforward (row leads col), blue = feedback (col leads
    row); cell text shows FFR ± significance stars from the shuffle null.
    `fname_prefix` / `source` let the CCA analysis reuse this with its own
    figure number and provenance label."""
    period_labels = {'encoding': 'Encoding (stimulus)',
                     'delay':    'Delay (maintenance)',
                     'response': 'Response (saccade)'}
    for angle, pdict in ffr_summary.items():
        present = {a for pairdict in pdict.values() for pair in pairdict for a in pair}
        areas_p = [a for a in areas if a in present]
        n = len(areas_p)
        if n < 2: continue
        idx = {a: i for i, a in enumerate(areas_p)}

        for period in (p for p in period_labels if p in pdict):
            M = np.full((n, n), np.nan)   # mean FFR
            P = np.full((n, n), np.nan)   # p-value
            for (a, b), st in pdict[period].items():
                if a == b or a not in idx or b not in idx: continue
                i, j = idx[a], idx[b]
                if i > j: i, j = j, i        # keep upper triangle
                M[i, j] = st['mean']; P[i, j] = st['p']

            finite = np.abs(M[np.isfinite(M)])
            vmax = max(float(finite.max()), 0.05) if finite.size else 0.05

            fig, ax = plt.subplots(figsize=(1.15 * n + 2.2, 1.15 * n + 1.2))
            im = ax.imshow(M, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            for i in range(n):
                for j in range(n):
                    if not np.isfinite(M[i, j]): continue
                    star = stars(P[i, j])
                    ax.text(j, i, f'{M[i, j]:+.2f}\n{star}', ha='center', va='center',
                            fontsize=8.5,
                            color='white' if abs(M[i, j]) > 0.6 * vmax else 'black')
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(areas_p, fontsize=10)
            ax.set_yticklabels(areas_p, fontsize=10)
            for k, a in enumerate(areas_p):
                ax.get_xticklabels()[k].set_color(AREA_COLORS.get(a, 'black'))
                ax.get_yticklabels()[k].set_color(AREA_COLORS.get(a, 'black'))
            ax.set_xlabel('Column area  (higher in hierarchy →)', fontsize=10)
            ax.set_ylabel('← Row area  (lower in hierarchy)', fontsize=10)
            ax.set_xticks(np.arange(-.5, n, 1), minor=True)
            ax.set_yticks(np.arange(-.5, n, 1), minor=True)
            ax.grid(which='minor', color='white', lw=1.5)
            ax.tick_params(which='minor', length=0)
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Feedforward ratio', fontsize=10)
            ax.set_title(f'Feedforward ratio ({source}) — '
                         f'{period_labels.get(period, period)} — {angle}\n'
                         f'+ row leads (feedforward)   /   − column leads (feedback)',
                         fontsize=11, fontweight='bold')
            plt.tight_layout()
            p = os.path.join(FIG_DIR, f'{fname_prefix}_ffr_heatmap_{period}_{angle}.svg')
            fig.savefig(p, format='svg', bbox_inches='tight')
            print(f"Saved → {p}"); plt.close(fig)


def plot_group_flow(group_flow, sensory, frontal, angle_pairs, fname_prefix='14',
                    source='decoding-error xcorr'):
    """Aggregated sensory→frontal directional flow per window: one bar per
    window (+ feedforward / − feedback), SEM error bars, shuffle-test stars,
    and an encoding-vs-response paired-test bracket. `fname_prefix` / `source`
    let the CCA analysis reuse this with its own figure number and label."""
    order   = ['encoding', 'delay', 'response']
    labels  = {'encoding': 'Encoding\n(stimulus)', 'delay': 'Delay\n(maintenance)',
               'response': 'Response\n(saccade)'}
    for angle, sdict in group_flow.items():
        periods = [p for p in order if p in sdict]
        if not periods: continue
        means = np.array([sdict[p]['mean'] for p in periods])
        sems  = np.array([sdict[p]['sem']  for p in periods])
        pvals = [sdict[p]['p'] for p in periods]
        nsess = [sdict[p]['n_sessions'] for p in periods]
        x     = np.arange(len(periods))

        fig, ax = plt.subplots(figsize=(1.7 * len(periods) + 2.5, 5.5))
        colors = ['#c0392b' if m >= 0 else '#2c6fbb' for m in means]   # red=FF, blue=FB
        ax.bar(x, means, yerr=sems, color=colors, width=0.62,
               edgecolor='black', linewidth=1.2, capsize=5, alpha=0.9)
        ax.axhline(0, color='k', lw=1.2)
        ymax = float(np.nanmax(np.abs(means) + sems)) if len(means) else 0.1
        for xi, (m, s, pv, nn) in enumerate(zip(means, sems, pvals, nsess)):
            yt = (m + s + 0.04 * ymax) if m >= 0 else (m - s - 0.10 * ymax)
            ax.text(xi, yt, stars(pv), ha='center', va='bottom' if m >= 0 else 'top',
                    fontsize=14, fontweight='bold')
            ax.text(xi, -1.18 * ymax, f'n={nn}', ha='center', va='top', fontsize=9, color='0.4')

        # encoding vs response paired bracket
        if 'encoding' in periods and 'response' in periods:
            p_er, n_er = _paired_epoch_flow_test(sdict, 'encoding', 'response')
            if np.isfinite(p_er):
                i1, i2 = periods.index('encoding'), periods.index('response')
                yb = 1.28 * ymax
                ax.plot([i1, i1, i2, i2], [yb, yb + 0.05*ymax, yb + 0.05*ymax, yb],
                        color='k', lw=1.2)
                ax.text((i1 + i2) / 2, yb + 0.07*ymax,
                        f'encoding vs response: p={p_er:.3f} {stars(p_er)} (n={n_er})',
                        ha='center', va='bottom', fontsize=10)

        ax.set_xticks(x); ax.set_xticklabels([labels.get(p, p) for p in periods], fontsize=11)
        ax.set_ylabel('Sensory → frontal flow  (feedforward ratio)', fontsize=12)
        ax.set_ylim(-1.45 * ymax, 1.55 * ymax)
        ax.text(0.015, 0.985, '+ sensory leads (feedforward)', transform=ax.transAxes,
                fontsize=9, va='top', color='#c0392b')
        ax.text(0.015, 0.945, '−  frontal leads (feedback)', transform=ax.transAxes,
                fontsize=9, va='top', color='#2c6fbb')
        ax.set_title(f'Aggregated information flow ({source}): '
                     f'{"/".join(sensory)} → {"/".join(frontal)}\n'
                     f'(stars = shuffle test vs no-flow null) — {angle}',
                     fontsize=12, fontweight='bold')
        for sp in ax.spines.values(): sp.set_linewidth(1.4)
        ax.tick_params(labelsize=11)
        plt.tight_layout()
        p = os.path.join(FIG_DIR, f'{fname_prefix}_group_flow_{angle}.svg')
        fig.savefig(p, format='svg', bbox_inches='tight')
        print(f"Saved → {p}"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# (B) CCA OF RESIDUAL POPULATION ACTIVITY  (Semedo et al. 2022, Fig 3b/3d)
# ══════════════════════════════════════════════════════════════════════════════

def _raw_time_axis(n_bins):
    """Absolute-second centre of each sliding-window bin (matches decoder.py)."""
    W = round(CCA_WINDOW_S / CCA_ORIG_BIN)
    S = round(CCA_STEP_S   / CCA_ORIG_BIN)
    return np.array([CCA_T_START + (i * S + W / 2) * CCA_ORIG_BIN for i in range(n_bins)])


def _target_condition(trial_df):
    """Integer label per trial for the 6 discrete saccade targets. Derived from
    (targetX, targetY) so it works whether or not a 'targAng' column exists."""
    cols = trial_df.columns
    if 'targAng' in cols:
        vals = np.round(np.asarray(trial_df['targAng'].values, float), 3)
    else:
        tx = np.asarray(trial_df['targetX'].values, float)
        ty = np.asarray(trial_df['targetY'].values, float)
        vals = np.round(np.arctan2(ty, tx), 3)
    uniq = {v: i for i, v in enumerate(sorted(set(vals[np.isfinite(vals)])))}
    return np.array([uniq.get(v, -1) for v in vals])


def _residualize(X, cond):
    """Subtract the per-condition mean (PSTH) per neuron per bin → trial-to-trial
    residual activity (the shared-variability substrate Semedo's CCA operates on).
    X: (n_trials, n_neurons, n_bins); cond: (n_trials,) integer labels (−1 = drop)."""
    R = X.astype(float).copy()
    for c in np.unique(cond):
        if c < 0: continue
        m = cond == c
        R[m] -= np.nanmean(X[m], axis=0, keepdims=True)
    return R


def _left_sv(M, n_pcs):
    """Top-k left singular vectors of a centred sample×feature matrix (an
    orthonormal basis for its top-k PCA subspace). Returns None if degenerate."""
    M = np.asarray(M, float)
    M = M - M.mean(0, keepdims=True)
    if M.shape[0] < 3 or M.shape[1] == 0:
        return None
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    if S.size == 0 or S[0] <= 0:
        return None
    k = int(min(n_pcs, np.sum(S > 1e-8 * S[0])))
    return U[:, :k] if k >= 1 else None


def _cca_popcorr_pair(RX, RY, epoch_bins, max_lag, n_pcs, n_shuffle, rng):
    """Population-correlation-vs-lag curve for one area pair within one epoch.

    The population correlation at lag δ is the first canonical correlation
    between residual activity of area X at bin t and area Y at bin t+δ, pooled
    over (trial × bin) samples. For PCA-whitened subspaces the first canonical
    correlation equals the top singular value of Uxᵀ Uy, where Ux, Uy are the
    top-k left singular vectors of the centred sample matrices (cosine of the
    smallest principal angle between the two areas' subspaces).

    δ>0 → X (row, lower hierarchy) leads Y → feedforward, matching the FFR sign
    convention used for the decoding-error xcorr. Shuffle null breaks the trial
    correspondence between the areas; under a trial permutation Uy's rows simply
    permute, so every shuffle reuses the cached Ux, Uy without re-running SVD."""
    n_trials, _, n_bins = RX.shape
    lags = np.arange(-max_lag, max_lag + 1)
    curve = np.full(len(lags), np.nan)
    Ux_l, Uy_l, nb_l = [], [], []
    for li, lag in enumerate(lags):
        tt = [t for t in epoch_bins if 0 <= t + lag < n_bins]
        nb = len(tt)
        if nb < 2:
            Ux_l.append(None); Uy_l.append(None); nb_l.append(0); continue
        ty = [t + lag for t in tt]
        Xs = RX[:, :, tt].transpose(0, 2, 1).reshape(n_trials * nb, -1)
        Ys = RY[:, :, ty].transpose(0, 2, 1).reshape(n_trials * nb, -1)
        Ux = _left_sv(Xs, n_pcs); Uy = _left_sv(Ys, n_pcs)
        Ux_l.append(Ux); Uy_l.append(Uy); nb_l.append(nb)
        if Ux is None or Uy is None:
            continue
        sv = np.linalg.svd(Ux.T @ Uy, compute_uv=False)
        curve[li] = float(sv[0]) if sv.size else np.nan

    ffr_obs   = _ffr_from_curve(curve, lags)
    null_ffrs = np.empty(0)
    if n_shuffle > 0:
        P = np.stack([rng.permutation(n_trials) for _ in range(n_shuffle)])   # (S, T)
        null_curves = np.full((n_shuffle, len(lags)), np.nan)
        for li in range(len(lags)):
            Ux, Uy, nb = Ux_l[li], Uy_l[li], nb_l[li]
            if Ux is None or Uy is None:
                continue
            base = np.arange(nb)
            for si in range(n_shuffle):
                rows = (P[si][:, None] * nb + base[None, :]).ravel()   # permute Y trials, keep bins
                sv = np.linalg.svd(Ux.T @ Uy[rows], compute_uv=False)
                null_curves[si, li] = sv[0] if sv.size else np.nan
        pos = np.nansum(null_curves[:, lags > 0], axis=1)
        neg = np.nansum(null_curves[:, lags < 0], axis=1)
        tot = np.nansum(np.abs(null_curves[:, lags != 0]), axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            null_ffrs = np.where(tot > 0, (pos - neg) / tot, np.nan)
    return curve, ffr_obs, null_ffrs


def _cca_session(session, X, unit_df, trial_df, areas, max_lag, n_pcs, n_shuffle, seed):
    rng  = np.random.default_rng((seed, hash(str(session)) & 0xFFFFFFFF))
    cond = _target_condition(trial_df)
    area_arr = np.asarray(unit_df['area'].values)
    time_raw = _raw_time_axis(X.shape[2])
    epoch_masks = {name: (time_raw >= CCA_EV_TARGET_ON + lo) & (time_raw < CCA_EV_TARGET_ON + hi)
                   for name, (lo, hi) in INTERAREA_WINDOWS.items()}

    res = {}
    for a in areas:
        m = area_arr == a
        if int(m.sum()) >= CCA_MIN_NEURONS:
            res[a] = _residualize(X[:, m, :], cond)
    avail = [a for a in areas if a in res]

    POP = 'pop'
    out = {POP: {p: {} for p in epoch_masks}}
    ffr = {POP: {p: {} for p in epoch_masks}}
    for period, mask in epoch_masks.items():
        ebins = np.where(mask)[0]
        if ebins.size < 2:
            continue
        for i, a in enumerate(avail):
            for j in range(i + 1, len(avail)):
                b = avail[j]
                curve, ffr_obs, nullf = _cca_popcorr_pair(
                    res[a], res[b], ebins, max_lag, n_pcs, n_shuffle, rng)
                out[POP][period][(a, b)] = curve
                ffr[POP][period][(a, b)] = (ffr_obs, nullf)
    return session, out, ffr


def compute_cca_popcorr(raw_sessions, areas, max_lag_s, n_pcs, n_shuffle, seed):
    """Drive the CCA population-correlation analysis across sessions. Returns
    (per_session_curves, lags, ffr_per_session) with the same nesting as
    compute_interarea_xcorr, so summarize_ffr / aggregate_group_flow apply."""
    max_lag = int(round(max_lag_s / CCA_STEP_S))
    lags    = np.arange(-max_lag, max_lag + 1)
    print(f"  CCA: {len(raw_sessions)} sessions, top-{n_pcs} PCs/area, "
          f"max_lag = {max_lag} bins (±{max_lag * CCA_STEP_S:.3f}s), shuffles = {n_shuffle}")

    results = Parallel(n_jobs=-1, prefer='threads')(
        delayed(_cca_session)(sid, X, u, t, areas, max_lag, n_pcs, n_shuffle, seed)
        for sid, X, u, t in raw_sessions
    )

    per_session, ffr_per_session = {}, {}
    for s, sess_out, sess_ffr in results:
        for ang, pdict in sess_out.items():
            per_session.setdefault(ang, {})
            for period, pair_dict in pdict.items():
                per_session[ang].setdefault(period, {})
                for pair, curve in pair_dict.items():
                    per_session[ang][period].setdefault(pair, {})
                    per_session[ang][period][pair][s] = curve
        for ang, pdict in sess_ffr.items():
            ffr_per_session.setdefault(ang, {})
            for period, pair_dict in pdict.items():
                ffr_per_session[ang].setdefault(period, {})
                for pair, val in pair_dict.items():
                    ffr_per_session[ang][period].setdefault(pair, {})
                    ffr_per_session[ang][period][pair][s] = val
    return per_session, lags, ffr_per_session


def _load_raw_sessions(path):
    """Load raw spike counts + unit/trial metadata from the decoder's data pkl."""
    print(f"Loading raw spike counts from:\n  {path}")
    with open(path, 'rb') as fh:
        data = pickle.load(fh)
    n = len(data['spikecounts'])
    return [(s, np.asarray(data['spikecounts'][s], float), data['unit'][s], data['trial'][s])
            for s in range(n)]


def plot_cca_popcorr(mean_x, sem_x, lags, areas, step_s):
    """One figure per window: population-correlation function (first canonical
    correlation of residual activity) vs inter-area lag, for each area pair.
    Positive lag = row area leads column area (feedforward)."""
    lag_s = lags * step_s
    period_styles = {
        'encoding': {'color': '#2ca02c', 'label': 'Encoding (stimulus)'},
        'delay':    {'color': '#1f77b4', 'label': 'Delay (maintenance)'},
        'response': {'color': '#d62728', 'label': 'Response (saccade)'},
    }
    for angle_name, periods_dict in mean_x.items():
        present = {a for pd in periods_dict.values() for pair in pd for a in pair}
        areas_p = [a for a in areas if a in present]
        n = len(areas_p)
        if n < 2:
            continue
        for period in (p for p in period_styles if p in periods_dict):
            pair_curves = periods_dict[period]
            style       = period_styles[period]
            vals = [c for c in pair_curves.values() if c is not None and np.isfinite(c).any()]
            vmax = (max(0.05, float(np.nanpercentile(np.concatenate(vals), 99.5)))
                    if vals else 0.2)

            fig, axes = plt.subplots(n, n, figsize=(1.8 * n, 1.6 * n),
                                      sharex=True, sharey=True, squeeze=False)
            for i, a in enumerate(areas_p):
                for j, b in enumerate(areas_p):
                    ax = axes[i][j]
                    if j <= i:
                        ax.set_visible(False); continue
                    curve = pair_curves.get((a, b))
                    s_val = sem_x.get(angle_name, {}).get(period, {}).get((a, b))
                    if curve is not None:
                        ax.plot(lag_s, curve, color=style['color'], lw=1.5)
                        if s_val is not None:
                            ax.fill_between(lag_s, curve - s_val, curve + s_val,
                                            color=style['color'], alpha=0.20, lw=0)
                    ax.axvline(0, color='k', lw=0.5, alpha=0.3)
                    ax.set_ylim(0, vmax)
                    ax.set_xlim(lag_s[0], lag_s[-1])
                    if i == 0: ax.set_title(b, fontsize=10, fontweight='bold',
                                             color=AREA_COLORS.get(b, 'black'))
                    if j == i + 1: ax.set_ylabel(a, fontsize=10, fontweight='bold',
                                                  color=AREA_COLORS.get(a, 'black'))
                    if i == n - 1: ax.set_xlabel('Lag (s)', fontsize=8)
                    ax.tick_params(labelsize=7)

            fig.text(0.5, 0.005,
                     'lag > 0: row leads column (feedforward, ↑ hierarchy)   |   '
                     'lag < 0: column leads row (feedback, ↓ hierarchy)',
                     ha='center', fontsize=9, style='italic')
            fig.suptitle('CCA population correlation of residual activity\n'
                         f'{style["label"]} — {angle_name}',
                         fontsize=12, fontweight='bold')
            plt.tight_layout(rect=[0, 0.05, 1, 0.94])
            p = os.path.join(FIG_DIR, f'15_cca_popcorr_{period}_{angle_name}.svg')
            fig.savefig(p, format='svg', bbox_inches='tight')
            print(f"Saved → {p}"); plt.close(fig)


def _print_flow_summary(ffr_summary, group_flow, windows):
    for period in windows:
        sig = [f"{a}->{b} FFR={st['mean']:+.2f}{stars(st['p'])}"
               for ang in ffr_summary.values()
               for (a, b), st in ang.get(period, {}).items()
               if np.isfinite(st['p']) and st['p'] < ALPHA]
        print(f"    {period:9s}: {len(sig)} significant pair(s)"
              + ("  | " + ", ".join(sig) if sig else ""))
    for ang, sdict in group_flow.items():
        for period in windows:
            if period not in sdict: continue
            st = sdict[period]
            print(f"    flow {period:9s}: {st['mean']:+.3f}±{st['sem']:.3f} "
                  f"p={st['p']:.3f} {stars(st['p'])} (n={st['n_sessions']})")
        p_er, n_er = _paired_epoch_flow_test(sdict, 'encoding', 'response')
        if np.isfinite(p_er):
            print(f"    flow encoding vs response: p={p_er:.3f} {stars(p_er)} (n={n_er})")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def _run_cca():
    cca_path = os.path.join(PKL_DIR, 'cca_popcorr.pkl')
    if RECOMPUTE or not os.path.exists(cca_path):
        raw_sessions = _load_raw_sessions(CCA_DATA_PATH)
        print("\nComputing CCA population correlation (residual activity) ...")
        per_session, lags, ffr_per_session = compute_cca_popcorr(
            raw_sessions, AREAS, CCA_MAX_LAG_S, CCA_N_PCS,
            n_shuffle=N_CCA_SHUFFLE, seed=CCA_SHUFFLE_SEED)
        mean_x, sem_x = average_interarea_xcorr(per_session)
        ffr_summary = summarize_ffr(ffr_per_session)
        group_flow  = aggregate_group_flow(ffr_per_session, SENSORY_AREAS, FRONTAL_AREAS)
        print("  CCA flow summary:")
        _print_flow_summary(ffr_summary, group_flow, INTERAREA_WINDOWS)
        with open(cca_path, 'wb') as fh:
            pickle.dump({'mean': mean_x, 'sem': sem_x, 'lags': lags,
                         'step_s': CCA_STEP_S, 'ffr_summary': ffr_summary,
                         'ffr_per_session': ffr_per_session, 'group_flow': group_flow}, fh)
        print(f"Saved → {cca_path}")
    else:
        print(f"Loading cached CCA results from:\n  {cca_path}")
        with open(cca_path, 'rb') as fh:
            _c = pickle.load(fh)
        mean_x = _c['mean']; sem_x = _c['sem']; lags = _c['lags']
        ffr_summary     = _c.get('ffr_summary', {})
        ffr_per_session = _c.get('ffr_per_session', {})
        group_flow = (aggregate_group_flow(ffr_per_session, SENSORY_AREAS, FRONTAL_AREAS)
                      if ffr_per_session else _c.get('group_flow', {}))

    print("\n── Generating CCA figures ──")
    plot_cca_popcorr(mean_x, sem_x, lags, AREAS, CCA_STEP_S)                       # 15
    if ffr_summary:
        plot_ffr_heatmap(ffr_summary, AREAS, {}, fname_prefix='16',
                         source='residual activity, CCA')                          # 16
    if group_flow:
        plot_group_flow(group_flow, SENSORY_AREAS, FRONTAL_AREAS, {},
                        fname_prefix='17', source='residual-activity CCA')         # 17


def _run_xcorr():
    iax_path = os.path.join(PKL_DIR, 'interarea_xcorr.pkl')

    if RECOMPUTE or not os.path.exists(iax_path):
        # ── metadata (time axis, step, angle pairs, event onset) ──────────────
        with open(os.path.join(PKL_DIR, 'trial_data.pkl'), 'rb') as fh:
            _td = pickle.load(fh)
        time_abs     = _td['time']
        angle_pairs  = _td['angle_pairs']
        step_s       = _td['step_s']
        ev_target_on = _td.get('ev_target_on', EV_TARGET_ON)

        # ── raw per-trial decoding errors (computed by analysis.py) ───────────
        _bl_path = os.path.join(PKL_DIR, 'neurobeh_baseline.pkl')
        print(f"Loading errors_angular from:\n  {_bl_path}")
        with open(_bl_path, 'rb') as fh:
            _bl = pickle.load(fh)
        errors_angular = _bl['errors_angular']

        # windows are relative to target onset → convert to absolute seconds
        interarea_periods = {name: (ev_target_on + lo, ev_target_on + hi)
                             for name, (lo, hi) in INTERAREA_WINDOWS.items()}

        print("\nComputing inter-area cross-correlation ...")
        per_session, lags, ffr_per_session = compute_interarea_xcorr(
            errors_angular, time_abs, AREAS, interarea_periods,
            INTERAREA_XCORR_MAX_LAG_S,
            n_shuffle=N_INTERAREA_SHUFFLE, seed=INTERAREA_SHUFFLE_SEED)
        mean_x, sem_x = average_interarea_xcorr(per_session)

        print("  Summarising feedforward ratios (shuffle test) ...")
        ffr_summary = summarize_ffr(ffr_per_session)
        for period in INTERAREA_WINDOWS:
            sig = [f"{a}->{b} FFR={st['mean']:+.2f}{stars(st['p'])}"
                   for ang in ffr_summary.values()
                   for (a, b), st in ang.get(period, {}).items()
                   if np.isfinite(st['p']) and st['p'] < ALPHA]
            print(f"    {period:9s}: {len(sig)} significant pair(s)"
                  + ("  | " + ", ".join(sig) if sig else ""))

        print(f"  Aggregating sensory→frontal flow "
              f"({'/'.join(SENSORY_AREAS)} → {'/'.join(FRONTAL_AREAS)}) ...")
        group_flow = aggregate_group_flow(ffr_per_session, SENSORY_AREAS, FRONTAL_AREAS)
        for ang, sdict in group_flow.items():
            for period in INTERAREA_WINDOWS:
                if period not in sdict: continue
                st = sdict[period]
                print(f"    {period:9s}: flow={st['mean']:+.3f}±{st['sem']:.3f} "
                      f"p={st['p']:.3f} {stars(st['p'])} (n={st['n_sessions']})")
            p_er, n_er = _paired_epoch_flow_test(sdict, 'encoding', 'response')
            if np.isfinite(p_er):
                print(f"    encoding vs response: p={p_er:.3f} {stars(p_er)} (n={n_er})")

        with open(iax_path, 'wb') as fh:
            pickle.dump({'per_session': per_session, 'mean': mean_x, 'sem': sem_x,
                         'lags': lags, 'step_s': step_s,
                         'periods': interarea_periods, 'angle_pairs': angle_pairs,
                         'max_lag_s': INTERAREA_XCORR_MAX_LAG_S,
                         'ffr_summary': ffr_summary,
                         'ffr_per_session': ffr_per_session,
                         'group_flow': group_flow}, fh)
        print(f"Saved → {iax_path}")

    else:
        print(f"Loading cached inter-area results from:\n  {iax_path}")
        with open(iax_path, 'rb') as fh:
            _ia = pickle.load(fh)
        mean_x          = _ia['mean']
        sem_x           = _ia['sem']
        lags            = _ia['lags']
        step_s          = _ia['step_s']
        angle_pairs     = _ia.get('angle_pairs', {})
        ffr_summary     = _ia.get('ffr_summary', {})
        ffr_per_session = _ia.get('ffr_per_session', {})
        # Re-aggregate from cached per-session FFRs so SENSORY_AREAS/FRONTAL_AREAS
        # can be changed above without rerunning the (expensive) shuffle null.
        group_flow = (aggregate_group_flow(ffr_per_session, SENSORY_AREAS, FRONTAL_AREAS)
                      if ffr_per_session else _ia.get('group_flow', {}))

    # ── figures ───────────────────────────────────────────────────────────────
    print("\n── Generating inter-area figures ──")

    # 03. Inter-area cross-correlation (one per window)
    plot_interarea_xcorr(mean_x, sem_x, lags, AREAS, angle_pairs, step_s)

    # 13. Feedforward-ratio heatmaps (summary of 03, one per window)
    if ffr_summary:
        plot_ffr_heatmap(ffr_summary, AREAS, angle_pairs)

    # 14. Aggregated sensory→frontal flow (one bar per window)
    if group_flow:
        plot_group_flow(group_flow, SENSORY_AREAS, FRONTAL_AREAS, angle_pairs)

    print("── inter-area xcorr figures done ──")


def main():
    if RUN_XCORR:
        _run_xcorr()
    if RUN_CCA:
        _run_cca()
    print("\n" + "═" * 60)
    print("EXTRA ANALYSIS COMPLETE")
    print(f"  figures: {FIG_DIR}")
    print("═" * 60)


if __name__ == '__main__':
    main()
