"""
neurobeh_corr.py
================
Neural–behavioural error correlation analysis for the distributed WM project.

Compatibility
-------------
Loads files produced by decoder.py:
  neurobeh_baseline.pkl   — time axis, per_session_real (decoding circcorr), errors_angular,
                             trial_data, area/angle metadata
  errors_angular.csv      — per-trial neural decoding errors + behavioural columns
                             (incl. 'err' from BEH_COLS_CSV in decoder.py)

  NOTE: behavior_all.csv is only needed as a fallback if 'err' is missing from the CSV.
        In standard decoder.py runs, 'err' is always present.

Outputs (all SVG, written to RESULTS_DIR)
-----------------------------------------
  neurobeh_timeseries.svg       neural–beh circ-corr over time, per area
  neurobeh_barplot_mean.svg     Method A: mean circ-corr in delay vs response window
  neurobeh_barplot_slope.svg    Method B: linear slope of corr in delay vs response window
  decoding_timeseries.svg       decoding circ-corr over time, per area
  decoding_barplot_mean.svg     Method A: mean decoding corr in delay vs response window
  decoding_barplot_slope.svg    Method B: linear slope of decoding in delay vs response window
  neurobeh_vs_decoding.svg      scatter: neural–beh vs decoding correlation (demeaned per area)

Usage
-----
    python neurobeh_corr.py
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress, ttest_1samp, wilcoxon, mannwhitneyu


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  ← edit paths and parameters here
# ══════════════════════════════════════════════════════════════════════════════

BASELINE_PKL   = '/home/aarghavan/aslan/delsac-neural-decoding/results/neurobeh_baseline.pkl'
ERRORS_CSV     = '/home/aarghavan/aslan/delsac-neural-decoding/results/errors_angular.csv'
BEHAVIOR_CSV   = '/home/aarghavan/aslan/data/behavior_all.csv'   # fallback only
RESULTS_DIR    = '/home/aarghavan/aslan/delsac-neural-decoding/results/'

AREAS      = ['PFC', 'FEF', 'LIP', 'Parietal', 'IT', 'MT', 'V4']
# AREAS      = ['PFC', 'FEF', 'LIP']
# AREAS      = ['Parietal', 'IT', 'MT', 'V4']
ANGLE_NAME = 'targetAngle'

# ── Event times relative to target onset (target on = 0) ─────────────────────
EV_TARGET_ON_RAW = 1.70   # target on in the raw (absolute) time axis stored by decoder.py
EV_TARGET_OFF    = 0.10   # relative to target on
EV_FIXPT_OFF     = 0.85   # approximate fixpoint off / response

# ── Analysis periods (relative to target onset = 0) ──────────────────────────
# These correspond to absolute times EV_TARGET_ON_RAW + DELAY_START, etc.
DELAY_START    =  0.10   # target off
DELAY_END      =  0.85   # fixpoint off
RESPONSE_START =  0.85
RESPONSE_END   =  1.80   # end of post-response window

# ── Minimum trials per time bin for a session to contribute ───────────────────
MIN_TRIALS = 10

# ── 'err' column in errors_angular.csv is in degrees (from behavior data) ─────
BEH_ERR_RADIANS = False   # set True if the 'err' column is already in radians

# ── Statistics ────────────────────────────────────────────────────────────────
ALPHA         = 0.05    # significance threshold
N_PERM        = 1000    # permutations for cluster permutation test
FRONTAL_AREAS = ['PFC', 'FEF', 'LIP']
SENSORY_AREAS = ['IT', 'MT', 'V4']

AREA_COLORS = {
    'PFC':      '#1f77b4',
    'FEF':      '#d62728',
    'LIP':      '#2ca02c',
    'Parietal': '#ff7f0e',
    'IT':       '#17becf',
    'MT':       '#9467bd',
    'V4':       '#8c564b',
}

os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("Loading neurobeh_baseline.pkl ...")
with open(BASELINE_PKL, 'rb') as f:
    baseline = pickle.load(f)

# Time axis (stored as absolute seconds in decoder.py, shift to target-on = 0)
time_raw = baseline['time']                      # absolute seconds
time     = time_raw - EV_TARGET_ON_RAW           # aligned: target on = 0
n_time   = len(time)
print(f"  Time: {n_time} bins  [{time[0]:.2f} s → {time[-1]:.2f} s]  (target on = 0)")

# Per-session decoding circcorr is in the baseline pkl
# Structure: per_session_real[area][session_idx_int][angle_name] → (n_time,)
per_session_real = baseline['per_session_real']

# Per-session z-scored decoding accuracy ("distance from shuffle")
# Structure: errors_angular_z[area][session_idx_int][angle_name] → (n_time,)
errors_angular_z = baseline.get('errors_angular_z', {})

print("\nLoading errors_angular.csv ...")
df_neural = pd.read_csv(ERRORS_CSV)
df_neural = df_neural[df_neural['angle_name'] == ANGLE_NAME].copy()

# ── Ensure we have a behavioural error column ──────────────────────────────────
if 'err' in df_neural.columns:
    print("  Using 'err' column from errors_angular.csv (already merged by decoder.py)")
    beh_col = 'err'
else:
    warnings.warn(
        "'err' not found in errors_angular.csv — falling back to behavior_all.csv merge.\n"
        "Re-run decoder.py with BEH_COLS_CSV containing 'err' to avoid this."
    )
    print(f"  Loading behavior fallback: {BEHAVIOR_CSV}")
    df_beh = pd.read_csv(BEHAVIOR_CSV)

    def _clean_id(s):
        return (s.astype(str).str.strip("[]' ")
                 .replace('', np.nan).astype(float).astype('Int64'))

    df_neural['_s'] = _clean_id(df_neural['original_session'])
    df_neural['_t'] = pd.to_numeric(df_neural['original_trial'], errors='coerce').astype('Int64')
    df_beh['_s']    = _clean_id(df_beh['session_id'])
    df_beh['_t']    = pd.to_numeric(df_beh['trial_id'], errors='coerce').astype('Int64')

    df_neural = df_neural.merge(
        df_beh[['_s', '_t', 'err']], on=['_s', '_t'], how='left')
    df_neural.drop(columns=['_s', '_t'], inplace=True)
    beh_col = 'err'

# Convert behavioural error to radians if necessary
df_neural['beh_err_rad'] = (
    np.radians(df_neural[beh_col].values.astype(float))
    if not BEH_ERR_RADIANS
    else df_neural[beh_col].values.astype(float)
)

# Cast session_idx to int for alignment with per_session_real keys
df_neural['session_idx'] = pd.to_numeric(
    df_neural['session_idx'], errors='coerce').astype('Int64')

# Round time column once for fast groupby lookup
df_neural['time_r'] = df_neural['time'].round(3)
time_raw_r = np.round(time_raw, 3)     # matching rounded reference

n_total = len(df_neural)
n_with_beh = df_neural['beh_err_rad'].notna().sum()
print(f"  {n_total:,} rows  |  {n_with_beh:,} with behavioural error")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _draw_events(ax):
    ax.axvline(0,             color='steelblue', lw=1.5, ls='--', alpha=0.75, label='Target on')
    ax.axvline(EV_TARGET_OFF, color='steelblue', lw=1.0, ls=':',  alpha=0.60, label='Target off')
    ax.axvline(EV_FIXPT_OFF,  color='seagreen',  lw=1.5, ls='--', alpha=0.75, label='Fixpoint off')


def _shade_periods(ax):
    ax.axvspan(DELAY_START,    DELAY_END,
               alpha=0.06, color='gray',      label='Delay period')
    ax.axvspan(RESPONSE_START, RESPONSE_END,
               alpha=0.06, color='steelblue', label='Response period')


def _period_masks():
    delay_mask    = (time >= DELAY_START)    & (time <= DELAY_END)
    response_mask = (time >= RESPONSE_START) & (time <= RESPONSE_END)
    return delay_mask, response_mask


def _mean_sem(arr, axis=0):
    """NaN-safe mean and SEM."""
    m   = np.nanmean(arr, axis=axis)
    n   = np.sum(~np.isnan(arr), axis=axis)
    sem = np.nanstd(arr, axis=axis) / np.where(n > 0, np.sqrt(n), np.nan)
    return m, sem


def _circcorr_vec(pred_v, true_v):
    """
    Vectorised circular correlation for all time bins at once.
    pred_v : (n_valid, n_time)  — may contain NaN (handled via nanmean/nansum)
    true_v : (n_valid,)
    Returns (n_time,)
    """
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


def _circ_corr_series(area_df):
    """
    Compute neural–behavioural circular correlation at each time bin, per session.
    Vectorised: pivots each session to (n_trials, n_time) and applies _circcorr_vec
    once per session instead of calling circcorrcoef in a per-bin Python loop.

    Returns
    -------
    sessions   : sorted list of integer session indices
    r_sessions : (n_sessions, n_time) — NaN where trial count < MIN_TRIALS
    """
    sessions = sorted(
        area_df['session_idx'].dropna().unique().astype(int).tolist()
    )
    r_sessions = np.full((len(sessions), n_time), np.nan)

    for si, sess_idx in enumerate(sessions):
        sess_df = area_df[area_df['session_idx'] == sess_idx]

        # Pivot: rows = trial, cols = time bin (rounded)
        piv = sess_df.pivot_table(index='trial_idx', columns='time_r',
                                   values='error', aggfunc='first')
        piv = piv.reindex(columns=time_raw_r)   # align columns to reference axis

        # Behavioural error per trial (constant across time bins)
        beh = sess_df.groupby('trial_idx')['beh_err_rad'].first().reindex(piv.index)

        valid_trials = np.isfinite(beh.values)
        if valid_trials.sum() < MIN_TRIALS:
            continue

        neural_mat = piv.values[valid_trials]     # (n_valid, n_time)
        behav_vec  = beh.values[valid_trials]     # (n_valid,)

        r_series = _circcorr_vec(neural_mat, behav_vec)

        # Suppress bins where too few trials have valid neural predictions
        n_valid_per_bin = np.sum(np.isfinite(neural_mat), axis=0)
        r_series[n_valid_per_bin < MIN_TRIALS] = np.nan

        r_sessions[si] = r_series

    return sessions, r_sessions


def _stars(p):
    """Return significance stars for a p-value."""
    if p is None or np.isnan(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < ALPHA:
        return '*'
    return 'n.s.'


def _test_vs_zero(values):
    """
    One-sample test against zero.
    Uses Wilcoxon signed-rank (n ≥ 5, non-parametric) or t-test (n < 5).
    Returns p-value, or NaN if insufficient data.
    """
    clean = values[np.isfinite(values)]
    if len(clean) < 3:
        return np.nan
    if len(clean) < 5:
        _, p = ttest_1samp(clean, 0)
        return float(p)
    try:
        _, p = wilcoxon(clean, alternative='two-sided')
        return float(p)
    except ValueError:
        return np.nan


def _cluster_perm_1samp(r_mat, n_perm=N_PERM, thresh_p=0.05, seed=42):
    """
    Sign-flip cluster permutation test: H0 = mean across sessions is zero.
    Vectorised: t-stats computed with numpy for all bins at once; all n_perm
    sign-flip matrices are stacked and processed in a single batched pass.

    Returns
    -------
    sig_mask : (n_time,) bool  — True where cluster-corrected p < ALPHA
    """
    from scipy.stats import t as t_dist

    valid_rows = ~np.all(np.isnan(r_mat), axis=1)
    r = r_mat[valid_rows, :]
    if r.shape[0] < 3:
        return np.zeros(r_mat.shape[1], dtype=bool)

    n_sess, n_t = r.shape
    t_thresh = t_dist.ppf(1 - thresh_p / 2, df=n_sess - 1)

    def _t_stats_vec(mat):
        """Vectorised one-sample t-stats. mat: (n_sess, n_t) or (n_perm, n_sess, n_t)."""
        n = np.sum(np.isfinite(mat), axis=-2)
        m = np.nanmean(mat, axis=-2)
        s = np.nanstd(mat, axis=-2, ddof=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(n >= 3, m / (s / np.sqrt(n)), np.nan)

    def _max_cluster_mass(t_vals):
        mass = cur = 0.0
        for v in t_vals:
            if np.isfinite(v) and abs(v) > t_thresh:
                cur += abs(v)
            else:
                mass = max(mass, cur); cur = 0.0
        return max(mass, cur)

    t_obs    = _t_stats_vec(r)
    obs_mass = _max_cluster_mass(t_obs)
    if obs_mass == 0:
        return np.zeros(n_t, dtype=bool)

    # Batch all permutations: (n_perm, n_sess, n_t) — compute t-stats in one pass
    rng   = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(n_perm, n_sess, 1)).astype(float)
    t_perm = _t_stats_vec(r[None] * signs)          # (n_perm, n_t)
    null_max = np.array([_max_cluster_mass(t_perm[pi]) for pi in range(n_perm)])

    # Label bins belonging to significant clusters
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


def _annotate_bar(ax, x, y_val, y_err, label, offset=0.01):
    """Place significance stars above (or below) a bar."""
    if not label or label == 'n.s.':
        return
    y_err  = y_err if (y_err is not None and np.isfinite(y_err)) else 0.0
    y_top  = (max(y_val, 0) if np.isfinite(y_val) else 0) + y_err + offset
    ax.text(x, y_top, label, ha='center', va='bottom', fontsize=10, fontweight='bold')


def _draw_sig_ribbon(ax, time, sig_mask, color, y_pos, height=0.012):
    """Draw a thin colored ribbon at y_pos for significant time bins."""
    ax.fill_between(time, y_pos, y_pos + height,
                    where=sig_mask, color=color, alpha=0.7,
                    transform=ax.get_xaxis_transform(), clip_on=False)


def _slope_per_session(r_mat, mask, times_subset):
    """
    Fit a linear slope to each session's correlation time-series within a period.
    Vectorised: uses numpy OLS formula for all sessions at once, falling back to
    linregress only for sessions with missing bins.

    Returns
    -------
    slopes : (n_sessions,) — NaN for sessions with < 3 valid bins
    pvals  : (n_sessions,)
    """
    Y      = r_mat[:, mask]          # (n_sess, n_bins)
    x      = times_subset            # (n_bins,)
    n_sess = Y.shape[0]
    slopes = np.full(n_sess, np.nan)
    pvals  = np.full(n_sess, np.nan)

    # Sessions where all selected bins are finite → batched OLS
    all_valid = np.all(np.isfinite(Y), axis=1)
    if all_valid.any():
        Yv = Y[all_valid]
        xm = x - x.mean()
        slopes[all_valid] = (Yv * xm).sum(axis=1) / (xm ** 2).sum()
        # p-value via t-distribution (2-sided)
        n   = len(x)
        yhat = slopes[all_valid, None] * xm + np.nanmean(Yv, axis=1, keepdims=True)
        sse = ((Yv - yhat) ** 2).sum(axis=1)
        se  = np.sqrt(sse / (n - 2) / (xm ** 2).sum())
        from scipy.stats import t as t_dist
        with np.errstate(invalid='ignore', divide='ignore'):
            pvals[all_valid] = 2 * t_dist.sf(np.abs(slopes[all_valid] / se), df=n - 2)

    # Sessions with some NaN bins → fall back to linregress
    for si in np.where(~all_valid)[0]:
        r     = Y[si]
        valid = np.isfinite(r)
        if valid.sum() > 2:
            sl, _, _, pv, _ = linregress(x[valid], r[valid])
            slopes[si] = sl
            pvals[si]  = pv

    return slopes, pvals


def _bar_plot(means_delay, sems_delay, means_response, sems_response,
              ylabel, title, save_path,
              pvals_delay=None, pvals_response=None,
              group_label=None):
    """
    Paired bar plot (delay vs response) with area-coloured edges and optional stats.

    pvals_delay / pvals_response : list of p-values per area (same order as AREAS).
                                   Stars (*/**/***/n.s.) are drawn above each bar.
    group_label : string appended as italic annotation, e.g. 'Frontal > Sensory: p=0.012'.
    """
    x         = np.arange(len(AREAS))
    bar_width  = 0.35
    fig, ax    = plt.subplots(figsize=(11, 5))

    for i, area in enumerate(AREAS):
        ec = AREA_COLORS.get(area, 'k')
        ax.bar(x[i] - bar_width / 2, means_delay[i], bar_width,
               yerr=sems_delay[i], capsize=5,
               color='lightgray', edgecolor=ec, linewidth=2, label=('Delay' if i == 0 else '_'))
        ax.bar(x[i] + bar_width / 2, means_response[i], bar_width,
               yerr=sems_response[i], capsize=5,
               color='lightblue', edgecolor=ec, linewidth=2,
               label=('Response' if i == 0 else '_'))

        if pvals_delay is not None:
            _annotate_bar(ax, x[i] - bar_width / 2,
                          means_delay[i], sems_delay[i], _stars(pvals_delay[i]))
        if pvals_response is not None:
            _annotate_bar(ax, x[i] + bar_width / 2,
                          means_response[i], sems_response[i], _stars(pvals_response[i]))

    ax.axhline(0, color='k', lw=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(AREAS, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, frameon=False)

    if group_label:
        ax.text(0.98, 0.97, group_label, transform=ax.transAxes,
                ha='right', va='top', fontsize=10, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.7))

    plt.tight_layout()
    fig.savefig(save_path, format='svg', bbox_inches='tight')
    print(f"Saved → {save_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Neural–behavioural correlation: time series
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Section 1: neural–behavioural correlation over time ──")

nb_corr: dict = {}   # area → (sessions_list, r_sessions_matrix)

for area in AREAS:
    area_df = df_neural[df_neural['area'] == area]
    if area_df.empty:
        print(f"  {area}: no data — skipped")
        continue
    sessions, r_mat = _circ_corr_series(area_df)
    nb_corr[area] = (sessions, r_mat)
    m, _ = _mean_sem(r_mat, axis=0)
    n_ok = np.sum(~np.isnan(m))
    print(f"  {area}: {len(sessions)} sessions  |  {n_ok}/{n_time} time bins with data")

# ── Cluster permutation test (H0: mean correlation = 0 at each time bin) ──────
print("\n  Running cluster permutation tests ...")
nb_sig: dict = {}   # area → (n_time,) bool significant mask
for area in AREAS:
    if area not in nb_corr:
        continue
    _, r_mat = nb_corr[area]
    nb_sig[area] = _cluster_perm_1samp(r_mat, n_perm=N_PERM)
    n_sig = nb_sig[area].sum()
    print(f"    {area}: {n_sig} significant bins")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
for area in AREAS:
    if area not in nb_corr:
        continue
    sessions, r_mat = nb_corr[area]
    m, sem = _mean_sem(r_mat, axis=0)
    c = AREA_COLORS[area]
    ax.plot(time, m, color=c, linewidth=2.5, label=f'{area} (n={len(sessions)})')
    ax.fill_between(time, m - sem, m + sem, alpha=0.15, color=c)

# Significance ribbons: one row per area at the bottom of the axes
_ribbon_areas = [a for a in AREAS if a in nb_sig and nb_sig[a].any()]
for ri, area in enumerate(_ribbon_areas):
    y_pos = -(ri + 1) * 0.025   # stacked below y=0 in axes-fraction coords
    _draw_sig_ribbon(ax, time, nb_sig[area], AREA_COLORS[area], y_pos)

_shade_periods(ax)
_draw_events(ax)
ax.axhline(0, color='k', lw=0.8, alpha=0.4)
ax.set_xlabel('Time from target onset (s)', fontsize=13)
ax.set_ylabel('Neural–behavioural circ. correlation', fontsize=13)
ax.set_title('Neural decoding error vs behavioural error\n'
             '(ribbons below = cluster-corrected p < 0.05)', fontsize=14, fontweight='bold')
ax.set_xlim(time[0], time[-1])
ax.legend(fontsize=10, frameon=False, loc='upper left', ncol=2)
plt.tight_layout()
_p = os.path.join(RESULTS_DIR, 'neurobeh_timeseries.svg')
fig.savefig(_p, format='svg', bbox_inches='tight')
print(f"\nSaved → {_p}")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Neural–behavioural correlation: bar plot (Method A — mean)
# Mirrors Cell 8 of the original neurobehCorr.ipynb.
# Average circ-corr within the delay / response windows, then mean ± SEM across
# sessions.  Each bar is filled with a neutral colour; edge = area colour.
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Section 2: bar plot — mean correlation per window (Method A) ──")

delay_mask, response_mask = _period_masks()

nb_delay_per_sess:    dict = {}   # area → (n_sessions,)  per-session period mean
nb_response_per_sess: dict = {}

d_means, d_sems = [], []
r_means, r_sems = [], []

d_pvals, r_pvals = [], []

for area in AREAS:
    if area not in nb_corr:
        d_means.append(np.nan); d_sems.append(np.nan); d_pvals.append(np.nan)
        r_means.append(np.nan); r_sems.append(np.nan); r_pvals.append(np.nan)
        continue

    sessions, r_mat = nb_corr[area]
    d_sess = np.nanmean(r_mat[:, delay_mask],    axis=1)
    r_sess = np.nanmean(r_mat[:, response_mask], axis=1)

    nb_delay_per_sess[area]    = (sessions, d_sess)
    nb_response_per_sess[area] = (sessions, r_sess)

    dm, ds = _mean_sem(d_sess)
    rm, rs = _mean_sem(r_sess)
    dp = _test_vs_zero(d_sess)
    rp = _test_vs_zero(r_sess)

    d_means.append(dm); d_sems.append(ds); d_pvals.append(dp)
    r_means.append(rm); r_sems.append(rs); r_pvals.append(rp)
    print(f"  {area} (n={len(sessions)}):  "
          f"delay={dm:.4f}±{ds:.4f} {_stars(dp)}  "
          f"response={rm:.4f}±{rs:.4f} {_stars(rp)}")

# ── Frontal vs sensory: Mann-Whitney on delay-period values ───────────────────
frontal_d = np.concatenate([nb_delay_per_sess[a][1]
                             for a in FRONTAL_AREAS if a in nb_delay_per_sess])
sensory_d  = np.concatenate([nb_delay_per_sess[a][1]
                              for a in SENSORY_AREAS if a in nb_delay_per_sess])
frontal_d  = frontal_d[np.isfinite(frontal_d)]
sensory_d  = sensory_d[np.isfinite(sensory_d)]

if len(frontal_d) >= 3 and len(sensory_d) >= 3:
    _, p_grp = mannwhitneyu(frontal_d, sensory_d, alternative='greater')
    group_label = f'Frontal > Sensory (delay): p={p_grp:.3f} {_stars(p_grp)}'
    print(f"\n  Frontal vs Sensory (delay, Mann-Whitney): p={p_grp:.4f} {_stars(p_grp)}")
else:
    group_label = None

_bar_plot(d_means, d_sems, r_means, r_sems,
          ylabel='Mean circ. correlation',
          title='Neural–behavioural correlation: delay vs response (Method A)',
          save_path=os.path.join(RESULTS_DIR, 'neurobeh_barplot_mean.svg'),
          pvals_delay=d_pvals, pvals_response=r_pvals,
          group_label=group_label)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Neural–behavioural correlation: bar plot (Method B — slope)
# Mirrors Cells 12–13 of the original neurobehCorr.ipynb.
# Linear regression slope of corr over time within each window, per session.
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Section 3: bar plot — correlation slope per window (Method B) ──")

sd_means, sd_sems, sd_pvals = [], [], []
sr_means, sr_sems, sr_pvals = [], [], []

nb_slopes: dict = {}

for area in AREAS:
    if area not in nb_corr:
        sd_means.append(np.nan); sd_sems.append(np.nan); sd_pvals.append(np.nan)
        sr_means.append(np.nan); sr_sems.append(np.nan); sr_pvals.append(np.nan)
        continue

    _, r_mat = nb_corr[area]
    sl_d, _ = _slope_per_session(r_mat, delay_mask,    time[delay_mask])
    sl_r, _ = _slope_per_session(r_mat, response_mask, time[response_mask])

    nb_slopes[area] = {'delay': sl_d, 'response': sl_r}

    dm, ds = _mean_sem(sl_d)
    rm, rs = _mean_sem(sl_r)
    dp = _test_vs_zero(sl_d)
    rp = _test_vs_zero(sl_r)

    sd_means.append(dm); sd_sems.append(ds); sd_pvals.append(dp)
    sr_means.append(rm); sr_sems.append(rs); sr_pvals.append(rp)
    print(f"  {area}: delay slope={dm:.4f}±{ds:.4f} {_stars(dp)}  "
          f"response slope={rm:.4f}±{rs:.4f} {_stars(rp)}")

_bar_plot(sd_means, sd_sems, sr_means, sr_sems,
          ylabel='Slope (corr / s)',
          title='Neural–behavioural correlation slope: delay vs response (Method B)',
          save_path=os.path.join(RESULTS_DIR, 'neurobeh_barplot_slope.svg'),
          pvals_delay=sd_pvals, pvals_response=sr_pvals)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Decoding circcorr: time series
# Source: per_session_real from neurobeh_baseline.pkl.
# Structure: per_session_real[area][session_idx_int][angle_name] → (n_time,)
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Section 4: decoding circcorr over time ──")

dec_sessions: dict = {}   # area → sorted list of session indices (int)
dec_r:        dict = {}   # area → (n_sessions, n_time)

for area in AREAS:
    if area not in per_session_real:
        continue

    sess_dict = per_session_real[area]
    curves, keys = [], []
    for s_key, angle_dict in sess_dict.items():
        c = angle_dict.get(ANGLE_NAME)
        if c is None:
            continue
        arr = np.asarray(c, dtype=float)
        if len(arr) != n_time:
            warnings.warn(
                f"Skipping {area} session {s_key}: length {len(arr)} ≠ {n_time}. "
                "neurobeh_baseline.pkl may have been produced with different settings."
            )
            continue
        curves.append(arr)
        keys.append(int(s_key))

    if not curves:
        continue
    order = np.argsort(keys)
    dec_sessions[area] = [keys[i] for i in order]
    dec_r[area]        = np.stack([curves[i] for i in order], axis=0)
    print(f"  {area}: {len(keys)} sessions")

# ── Cluster permutation test on decoding curves ───────────────────────────────
print("\n  Running cluster permutation tests on decoding ...")
dec_sig: dict = {}
for area in AREAS:
    if area not in dec_r:
        continue
    dec_sig[area] = _cluster_perm_1samp(dec_r[area], n_perm=N_PERM)
    print(f"    {area}: {dec_sig[area].sum()} significant bins")

# ── Plot: decoding time series ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
for area in AREAS:
    if area not in dec_r:
        continue
    m, sem = _mean_sem(dec_r[area], axis=0)
    c = AREA_COLORS[area]
    n = dec_r[area].shape[0]
    ax.plot(time, m, color=c, linewidth=2.5, label=f'{area} (n={n})')
    ax.fill_between(time, m - sem, m + sem, alpha=0.15, color=c)

_ribbon_areas_dec = [a for a in AREAS if a in dec_sig and dec_sig[a].any()]
for ri, area in enumerate(_ribbon_areas_dec):
    y_pos = -(ri + 1) * 0.025
    _draw_sig_ribbon(ax, time, dec_sig[area], AREA_COLORS[area], y_pos)

_shade_periods(ax)
_draw_events(ax)
ax.axhline(0, color='k', lw=0.8, alpha=0.4)
ax.set_xlabel('Time from target onset (s)', fontsize=13)
ax.set_ylabel('Decoding circ. correlation', fontsize=13)
ax.set_title('Target angle decoding over time\n'
             '(ribbons below = cluster-corrected p < 0.05)', fontsize=14, fontweight='bold')
ax.set_xlim(time[0], time[-1])
ax.legend(fontsize=10, frameon=False, loc='upper left', ncol=2)
plt.tight_layout()
_p = os.path.join(RESULTS_DIR, 'decoding_timeseries.svg')
fig.savefig(_p, format='svg', bbox_inches='tight')
print(f"\nSaved → {_p}")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Decoding: bar plot (Method A — mean)
# Mirrors Cell 18 of the original neurobehCorr.ipynb.
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Section 5: decoding bar plot — mean (Method A) ──")

dec_delay_per_sess:    dict = {}   # area → (sessions, values)
dec_response_per_sess: dict = {}

dd_means, dd_sems, dd_pvals = [], [], []
dr_means, dr_sems, dr_pvals = [], [], []

for area in AREAS:
    if area not in dec_r:
        dd_means.append(np.nan); dd_sems.append(np.nan); dd_pvals.append(np.nan)
        dr_means.append(np.nan); dr_sems.append(np.nan); dr_pvals.append(np.nan)
        continue

    sessions = dec_sessions[area]
    r_mat    = dec_r[area]
    d_sess   = np.nanmean(r_mat[:, delay_mask],    axis=1)
    resp_sess = np.nanmean(r_mat[:, response_mask], axis=1)

    dec_delay_per_sess[area]    = (sessions, d_sess)
    dec_response_per_sess[area] = (sessions, resp_sess)

    dm, ds = _mean_sem(d_sess)
    rm, rs = _mean_sem(resp_sess)
    dp = _test_vs_zero(d_sess)
    rp = _test_vs_zero(resp_sess)

    dd_means.append(dm); dd_sems.append(ds); dd_pvals.append(dp)
    dr_means.append(rm); dr_sems.append(rs); dr_pvals.append(rp)
    print(f"  {area} (n={r_mat.shape[0]}):  "
          f"delay={dm:.4f}±{ds:.4f} {_stars(dp)}  "
          f"response={rm:.4f}±{rs:.4f} {_stars(rp)}")

_bar_plot(dd_means, dd_sems, dr_means, dr_sems,
          ylabel='Mean decoding circ. correlation',
          title='Target angle decoding: delay vs response (Method A)',
          save_path=os.path.join(RESULTS_DIR, 'decoding_barplot_mean.svg'),
          pvals_delay=dd_pvals, pvals_response=dr_pvals)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5b — Decoding: bar plot — mean DISTANCE FROM SHUFFLE (z-score)
# Same structure as Section 5 (Method A) but using the z-score
# from errors_angular_z (positive z = better than chance).
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Section 5b: decoding bar plot — mean distance from shuffle (z) ──")

# Build dec_z[area] = (n_sessions, n_time) matrix of per-session z-curves
dec_z_sessions: dict = {}
dec_z:          dict = {}
for area in AREAS:
    if area not in errors_angular_z:
        continue
    sess_dict = errors_angular_z[area]
    curves, keys = [], []
    for s_key, angle_dict in sess_dict.items():
        c = angle_dict.get(ANGLE_NAME)
        if c is None:
            continue
        arr = np.asarray(c, dtype=float)
        if len(arr) != n_time:
            warnings.warn(
                f"Skipping {area} session {s_key}: z-curve length {len(arr)} ≠ {n_time}."
            )
            continue
        curves.append(arr)
        keys.append(int(s_key))
    if not curves:
        continue
    order = np.argsort(keys)
    dec_z_sessions[area] = [keys[i] for i in order]
    dec_z[area]          = np.stack([curves[i] for i in order], axis=0)
    print(f"  {area}: {len(keys)} sessions")

if not dec_z:
    print("  No errors_angular_z data in baseline pkl — re-run decoder.py with "
          "COMPUTE_NULL=True to generate it.")
else:
    zd_means, zd_sems, zd_pvals = [], [], []
    zr_means, zr_sems, zr_pvals = [], [], []
    dec_z_delay_per_sess:    dict = {}
    dec_z_response_per_sess: dict = {}

    for area in AREAS:
        if area not in dec_z:
            zd_means.append(np.nan); zd_sems.append(np.nan); zd_pvals.append(np.nan)
            zr_means.append(np.nan); zr_sems.append(np.nan); zr_pvals.append(np.nan)
            continue

        sessions = dec_z_sessions[area]
        z_mat    = dec_z[area]
        d_sess   = np.nanmean(z_mat[:, delay_mask],    axis=1)
        r_sess   = np.nanmean(z_mat[:, response_mask], axis=1)

        dec_z_delay_per_sess[area]    = (sessions, d_sess)
        dec_z_response_per_sess[area] = (sessions, r_sess)

        dm, ds = _mean_sem(d_sess)
        rm, rs = _mean_sem(r_sess)
        dp = _test_vs_zero(d_sess)
        rp = _test_vs_zero(r_sess)

        zd_means.append(dm); zd_sems.append(ds); zd_pvals.append(dp)
        zr_means.append(rm); zr_sems.append(rs); zr_pvals.append(rp)
        print(f"  {area} (n={z_mat.shape[0]}):  "
              f"delay z={dm:.3f}±{ds:.3f} {_stars(dp)}  "
              f"response z={rm:.3f}±{rs:.3f} {_stars(rp)}")

    _bar_plot(zd_means, zd_sems, zr_means, zr_sems,
              ylabel='Mean distance from shuffle (σ)',
              title='Target angle decoding accuracy: delay vs response\n'
                    '(distance from shuffle)',
              save_path=os.path.join(RESULTS_DIR, 'decoding_barplot_zscore.svg'),
              pvals_delay=zd_pvals, pvals_response=zr_pvals)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Decoding: bar plot (Method B — slope)
# Mirrors Cells 21–22 of the original neurobehCorr.ipynb.
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Section 6: decoding bar plot — slope (Method B) ──")

dsd_means, dsd_sems, dsd_pvals = [], [], []
dsr_means, dsr_sems, dsr_pvals = [], [], []

dec_slopes: dict = {}

for area in AREAS:
    if area not in dec_r:
        dsd_means.append(np.nan); dsd_sems.append(np.nan); dsd_pvals.append(np.nan)
        dsr_means.append(np.nan); dsr_sems.append(np.nan); dsr_pvals.append(np.nan)
        continue

    r_mat = dec_r[area]
    sl_d, _ = _slope_per_session(r_mat, delay_mask,    time[delay_mask])
    sl_r, _ = _slope_per_session(r_mat, response_mask, time[response_mask])

    dec_slopes[area] = {'delay': sl_d, 'response': sl_r}

    dm, ds = _mean_sem(sl_d)
    rm, rs = _mean_sem(sl_r)
    dp = _test_vs_zero(sl_d)
    rp = _test_vs_zero(sl_r)

    dsd_means.append(dm); dsd_sems.append(ds); dsd_pvals.append(dp)
    dsr_means.append(rm); dsr_sems.append(rs); dsr_pvals.append(rp)
    print(f"  {area}: delay slope={dm:.4f}±{ds:.4f} {_stars(dp)}  "
          f"response slope={rm:.4f}±{rs:.4f} {_stars(rp)}")

_bar_plot(dsd_means, dsd_sems, dsr_means, dsr_sems,
          ylabel='Slope (corr / s)',
          title='Target angle decoding slope: delay vs response (Method B)',
          save_path=os.path.join(RESULTS_DIR, 'decoding_barplot_slope.svg'),
          pvals_delay=dsd_pvals, pvals_response=dsr_pvals)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Scatter: neural–beh correlation vs decoding (demeaned per area)
# Mirrors Cells 24–27 of the original neurobehCorr.ipynb.
# Sessions are aligned by integer session_idx (same key in both analyses).
# x-axis = demeaned decoding (delay period mean)
# y-axis = demeaned neural–beh correlation (delay period mean)
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Section 7: scatter — neural–beh vs decoding ──")

fig, ax = plt.subplots(figsize=(7, 6))
all_x, all_y = [], []

for area in AREAS:
    if area not in nb_delay_per_sess or area not in dec_delay_per_sess:
        continue

    nb_sess_ids,  nb_vals  = nb_delay_per_sess[area]   # aligned by session_idx
    dec_sess_ids, dec_vals = dec_delay_per_sess[area]

    # Find common session indices (both analyses may not cover all sessions)
    nb_map  = dict(zip(nb_sess_ids,  nb_vals))
    dec_map = dict(zip(dec_sess_ids, dec_vals))
    common  = sorted(set(nb_map.keys()) & set(dec_map.keys()))

    if len(common) < 2:
        print(f"  {area}: fewer than 2 aligned sessions — scatter skipped")
        continue

    nb_common  = np.array([nb_map[s]  for s in common])
    dec_common = np.array([dec_map[s] for s in common])

    # Demean per area to remove between-area mean offsets
    nb_dm  = nb_common  - np.nanmean(nb_common)
    dec_dm = dec_common - np.nanmean(dec_common)

    mask = np.isfinite(nb_dm) & np.isfinite(dec_dm)
    c    = AREA_COLORS[area]
    ax.scatter(dec_dm[mask], nb_dm[mask],
               color=c, alpha=0.65, s=45, label=f'{area} (n={mask.sum()})',
               edgecolors=c, linewidths=0.5)
    all_x.extend(dec_dm[mask].tolist())
    all_y.extend(nb_dm[mask].tolist())
    print(f"  {area}: {mask.sum()} aligned sessions used")

all_x = np.array(all_x)
all_y = np.array(all_y)
if len(all_x) > 2:
    sl, intercept, r, p, _ = linregress(all_x, all_y)
    x_line = np.linspace(all_x.min(), all_x.max(), 200)
    ax.plot(x_line, sl * x_line + intercept, 'k--', lw=1.5,
            label=f'all areas: r={r:.2f}, p={p:.3f}')
    print(f"\n  Overall (all areas): r={r:.3f}  p={p:.4f}  n={len(all_x)} sessions")

ax.axhline(0, color='k', lw=0.6, alpha=0.4)
ax.axvline(0, color='k', lw=0.6, alpha=0.4)
ax.set_xlabel('Decoding circ. correlation (demeaned, delay period)', fontsize=12)
ax.set_ylabel('Neural–beh circ. correlation (demeaned, delay period)', fontsize=12)
ax.set_title('Decoding vs neural–behavioural correlation\n'
             '(delay period, demeaned per area)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, frameon=False, ncol=2)
plt.tight_layout()
_p = os.path.join(RESULTS_DIR, 'neurobeh_vs_decoding.svg')
fig.savefig(_p, format='svg', bbox_inches='tight')
print(f"Saved → {_p}")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS  (loaded by plot.py for plotting without recomputation)
# ══════════════════════════════════════════════════════════════════════════════

print("\nSaving neurobeh_results.pkl ...")

_results = {

    # ── Metadata ──────────────────────────────────────────────────────────────
    'meta': {
        'areas':         AREAS,
        'time':          time,          # (n_time,) aligned to target on = 0
        'angle_name':    ANGLE_NAME,
        'n_perm':        N_PERM,
        'alpha':         ALPHA,
        'frontal_areas': FRONTAL_AREAS,
        'sensory_areas': SENSORY_AREAS,
        'delay_start':    DELAY_START,
        'delay_end':      DELAY_END,
        'response_start': RESPONSE_START,
        'response_end':   RESPONSE_END,
        'ev_target_off':  EV_TARGET_OFF,
        'ev_fixpt_off':   EV_FIXPT_OFF,
    },

    # ── Raw session-level correlation matrices (for timeseries plots) ──────────
    # r_mat shape: (n_sessions, n_time)
    'nb_r':        {a: r for a, (_, r) in nb_corr.items()},
    'nb_sessions': {a: s for a, (s, _) in nb_corr.items()},
    'dec_r':        {a: r for a, r in dec_r.items()},
    'dec_sessions': {a: s for a, s in dec_sessions.items()},

    # ── Cluster-permutation significance masks (n_time,) bool per area ─────────
    'nb_sig':  nb_sig,
    'dec_sig': dec_sig,

    # ── Method A: mean in delay / response window ──────────────────────────────
    'nb_mean': {
        'delay_means':    d_means,    'delay_sems':     d_sems,    'delay_pvals':    d_pvals,
        'response_means': r_means,    'response_sems':  r_sems,    'response_pvals': r_pvals,
        'group_label':    group_label,
    },
    'dec_mean': {
        'delay_means':    dd_means,   'delay_sems':     dd_sems,   'delay_pvals':    dd_pvals,
        'response_means': dr_means,   'response_sems':  dr_sems,   'response_pvals': dr_pvals,
    },

    # ── Method B: linear slope within window ──────────────────────────────────
    'nb_slope': {
        'delay_means':    sd_means,   'delay_sems':     sd_sems,   'delay_pvals':    sd_pvals,
        'response_means': sr_means,   'response_sems':  sr_sems,   'response_pvals': sr_pvals,
    },
    'dec_slope': {
        'delay_means':    dsd_means,  'delay_sems':     dsd_sems,  'delay_pvals':    dsd_pvals,
        'response_means': dsr_means,  'response_sems':  dsr_sems,  'response_pvals': dsr_pvals,
    },

    # ── Per-session period values (for scatter plot alignment) ─────────────────
    'nb_delay_per_sess':     {a: {'sessions': s, 'values': v}
                              for a, (s, v) in nb_delay_per_sess.items()},
    'nb_response_per_sess':  {a: {'sessions': s, 'values': v}
                              for a, (s, v) in nb_response_per_sess.items()},
    'dec_delay_per_sess':    {a: {'sessions': s, 'values': v}
                              for a, (s, v) in dec_delay_per_sess.items()},
    'dec_response_per_sess': {a: {'sessions': s, 'values': v}
                              for a, (s, v) in dec_response_per_sess.items()},
}

_rpath = os.path.join(RESULTS_DIR, 'neurobeh_results.pkl')
with open(_rpath, 'wb') as f:
    pickle.dump(_results, f)
print(f"  Saved → {_rpath}")


# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("DONE")
print(f"  Results: {RESULTS_DIR}")
print("  Files written:")
for f in [
    'neurobeh_timeseries.svg',
    'neurobeh_barplot_mean.svg',    # Method A: mean circ-corr in window
    'neurobeh_barplot_slope.svg',   # Method B: linear slope in window
    'decoding_timeseries.svg',
    'decoding_barplot_mean.svg',    # Method A
    'decoding_barplot_slope.svg',   # Method B
    'neurobeh_vs_decoding.svg',
]:
    print(f"    {f}")
print("═" * 60)
