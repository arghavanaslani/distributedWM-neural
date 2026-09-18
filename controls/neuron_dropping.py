"""
neuron_dropping.py
==================
Matched-N (neuron-dropping) decoding, to make cross-area decoding COMPARABLE.

Why this exists
---------------
The main decoding figures show the target is decodable in every area (existence),
but the delay-averaged bars can't be *ranked*, because decoding strength is
confounded by how many neurons were recorded per area (PFC ~61/session vs
IT ~12/session). This script removes that confound: it subsamples every area to a
common neuron count N, decodes, and compares areas at equal N.

  - Neuron-dropping CURVE:   accuracy vs N, one line per area  → shows existence
                             (all above chance) AND whether areas differ at
                             matched N or just converge.
  - Matched-N BAR:           accuracy at N = MATCHED_N (headline 15, robustness 10)
                             → the *defensible* ranking.

Metric: mean absolute angular decoding error in DEGREES (chance ~90 deg), plus
circular correlation dec_r as a companion. No shuffle null needed — both are
sample-size-robust effect sizes.

Inputs (already produced by decoder.py; NO heavy recompute)
-----------------------------------------------------------
  <PKL_DIR>/centered.pkl     {'spikecounts': [n_sess], 'unit': [n_sess]}
                             smoothed + z-scored spikecounts, filtered units
                             (exactly what decoder.py decodes from)
  <PKL_DIR>/trial_data.pkl   {'trial_data': {s: df}, 'time', 'delay_idx', ...}
                             trial DataFrames with normalized targetX/targetY

Outputs
-------
  <PKL_DIR>/neuron_dropping.pkl        per-(area,session,N) results + curves
  <FIG_DIR>/neuron_dropping_curve.svg  accuracy vs N, one line per area
  <FIG_DIR>/neuron_dropping_bar.svg    matched-N ranking bar (headline N)

Run ON THE SERVER:
  python neuron_dropping.py
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from scipy.stats import wilcoxon, ttest_1samp
from joblib import Parallel, delayed

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  ← edit here
# ══════════════════════════════════════════════════════════════════════════════

from config import (
    AREAS,
    AREA_COLORS,
    DELAY_END,
    DELAY_START,
    FIG_DIR,
    ORIG_BIN,
    PKL_DIR,
    RESULTS_DIR,
    STEP_S,
    T_START,
    WINDOW_S,
)
from core.metrics import circ_corr
from core.stats import mean_sem

ANGLE_XY    = ('targetX', 'targetY')          # decode these, recombine via arctan2

# Time axis (must match decoder.py). Used to rebuild the delay-window mask over
# the FULL-length spikecounts in centered.pkl. NOTE: trial_data.pkl's delay_idx is
# on the decode-window-CLIPPED axis (fewer bins) and does NOT match centered.pkl.

# Late-delay robustness toggle. When True, starts the delay window later (~1
# smoothing time-constant past target-off) to test whether the ranking reflects
# SUSTAINED delay activity rather than a smeared stimulus transient. NB:
# centered.pkl is ALREADY exponentially smoothed (tau~0.375 s), so a late window
# REDUCES but does not fully remove forward smear — for an airtight control,
# re-smooth from raw (test.pkl) with a shorter kernel. Outputs get a '_latedelay'
# suffix so both runs coexist.
LATE_DELAY       = True
LATE_DELAY_START = 2.15        # seconds (~1 tau past target-off at 1.80)
_DELAY_START_EFF = LATE_DELAY_START if LATE_DELAY else DELAY_START

# Trial-matching control (removes the training-set-size confound). When set to an
# int, subsample each session's trials to this count before decoding (per draw)
# and include only sessions with >= this many valid trials. Areas differ in
# trials (frontoparietal ~126-134, sensory ~86-91), so ~85 handicaps all areas to
# the sensory trial level. None = off. Outputs get a '_t<N>' suffix.
MATCH_TRIALS     = 80

_SUFFIX = ('_latedelay' if LATE_DELAY else '') + (f'_t{MATCH_TRIALS}' if MATCH_TRIALS else '')

# Neuron-dropping ladder (# neurons). Each area/session is only used at N <= its
# neuron count; a curve point is kept only if >= MIN_SESSIONS_PER_POINT sessions
# qualify. IT caps the fully-matched comparison at N=15 (see diagnostic).
N_LADDER    = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50]
MATCHED_N   = [15, 10]        # first = headline ranking, rest = robustness
MIN_SESSIONS_PER_POINT = 10

R_DRAWS     = 30              # random neuron subsets per (area, session, N)
CV_FOLDS    = 5
ALPHA_CV    = 3
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
MIN_TRIALS  = 10
CHANCE_DEG  = 90.0           # reference: mean |error| for uniform angles
SEED        = 42
N_JOBS      = -1


_pkl_out = os.path.join(PKL_DIR, f'neuron_dropping{_SUFFIX}.pkl')
os.makedirs(FIG_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _delay_mask(n_bins):
    """Boolean mask of delay-window bins over an n_bins time axis (matches decoder.py)."""
    W = round(WINDOW_S / ORIG_BIN); S = round(STEP_S / ORIG_BIN)
    t = np.array([T_START + (i * S + W / 2) * ORIG_BIN for i in range(n_bins)])
    return (t >= _DELAY_START_EFF) & (t <= DELAY_END)




def _decode_once(Xd, tX, tY):
    """Cross-validated decode of one neuron subset from delay-averaged activity.
    Xd: (n_trials, N)   tX, tY: (n_trials,)   →  (mean_abs_error_deg, dec_r)."""
    n = Xd.shape[0]
    predX = np.full(n, np.nan); predY = np.full(n, np.nan)
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in cv.split(Xd):
        mu = Xd[tr].mean(0); sd = Xd[tr].std(0); sd[sd == 0] = 1.0
        Xtr = (Xd[tr] - mu) / sd
        Xte = (Xd[te] - mu) / sd
        predX[te] = RidgeCV(alphas=RIDGE_ALPHAS, cv=ALPHA_CV).fit(Xtr, tX[tr]).predict(Xte)
        predY[te] = RidgeCV(alphas=RIDGE_ALPHAS, cv=ALPHA_CV).fit(Xtr, tY[tr]).predict(Xte)
    pred_ang = np.arctan2(predY, predX)
    true_ang = np.arctan2(tY, tX)
    err = np.arctan2(np.sin(true_ang - pred_ang), np.cos(true_ang - pred_ang))
    return np.degrees(np.nanmean(np.abs(err))), circ_corr(pred_ang, true_ang)


def _process_session(area, s, X_area, tX, tY, seed):
    """All N in the ladder for one (area, session): average over R random subsets.
    Returns {N: (deg, dec_r)} for feasible N only."""
    rng = np.random.default_rng(seed)
    n_neurons = X_area.shape[1]
    n_trials  = X_area.shape[0]
    out = {}
    for N in N_LADDER:
        if N > n_neurons:
            continue
        degs, rs = [], []
        for _ in range(R_DRAWS):
            sub = rng.choice(n_neurons, size=N, replace=False)
            if MATCH_TRIALS and n_trials > MATCH_TRIALS:
                tr_sub = rng.choice(n_trials, size=MATCH_TRIALS, replace=False)
                d, r = _decode_once(X_area[np.ix_(tr_sub, sub)], tX[tr_sub], tY[tr_sub])
            else:
                d, r = _decode_once(X_area[:, sub], tX, tY)
            degs.append(d); rs.append(r)
        out[N] = (float(np.nanmean(degs)), float(np.nanmean(rs)))
    return area, s, out


# ══════════════════════════════════════════════════════════════════════════════
# LOAD CACHED PIPELINE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

print("Loading cached pipeline outputs ...")
with open(os.path.join(PKL_DIR, 'centered.pkl'), 'rb') as fh:
    centered = pickle.load(fh)
with open(os.path.join(PKL_DIR, 'trial_data.pkl'), 'rb') as fh:
    tdata = pickle.load(fh)

spikecounts = centered['spikecounts']          # list[s] (trials, neurons, bins)
units       = centered['unit']                 # list[s] DataFrame with 'area'
trial_data  = tdata['trial_data']              # dict s -> DataFrame
varX, varY  = ANGLE_XY
n_sessions  = len(spikecounts)
_n_bins0    = np.asarray(spikecounts[0]).shape[-1]
_dmask0     = _delay_mask(_n_bins0)
print(f"  {n_sessions} sessions | spikecounts bins={_n_bins0} | "
      f"{int(_dmask0.sum())} delay bins [{_DELAY_START_EFF}–{DELAY_END} s]"
      f"{'  (LATE-DELAY)' if LATE_DELAY else ''}"
      f"{f'  (MATCH_TRIALS={MATCH_TRIALS})' if MATCH_TRIALS else ''}\n")


# ══════════════════════════════════════════════════════════════════════════════
# BUILD (area, session) JOBS  — delay-averaged activity + targets
# ══════════════════════════════════════════════════════════════════════════════

jobs = []
for s in range(n_sessions):
    sc = np.asarray(spikecounts[s])                     # (trials, neurons, bins)
    if s not in trial_data:
        continue
    tdf = trial_data[s]
    if varX not in tdf.columns or varY not in tdf.columns:
        continue
    n_tr = min(sc.shape[0], len(tdf))
    tX = tdf[varX].values.astype(float)[:n_tr]
    tY = tdf[varY].values.astype(float)[:n_tr]
    valid = np.isfinite(tX) & np.isfinite(tY)
    if valid.sum() < max(MIN_TRIALS, MATCH_TRIALS or 0):
        continue
    # delay-window-averaged activity per neuron: (trials, neurons)
    dmask = _delay_mask(sc.shape[-1])
    X_delay = np.nanmean(sc[:n_tr][:, :, dmask], axis=2)[valid]
    tX, tY  = tX[valid], tY[valid]
    area_arr = np.asarray(units[s]['area'].values)
    for a in AREAS:
        mask = area_arr == a
        if mask.sum() < min(N_LADDER):
            continue
        jobs.append((a, s, X_delay[:, mask], tX, tY, SEED + 1000 * s + AREAS.index(a)))

print(f"Running {len(jobs)} (area, session) decodes "
      f"× {R_DRAWS} draws × {len(N_LADDER)} N-levels ...")

results = Parallel(n_jobs=N_JOBS, prefer='processes')(
    delayed(_process_session)(a, s, X, tX, tY, seed) for (a, s, X, tX, tY, seed) in jobs
)


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE  → per-(area, N) mean ± SEM across sessions
# ══════════════════════════════════════════════════════════════════════════════

# per_area[a][N] = list of (session, deg, dec_r)
per_area = {a: {N: [] for N in N_LADDER} for a in AREAS}
for area, s, out in results:
    for N, (deg, r) in out.items():
        per_area[area][N].append((s, deg, r))


curve = {a: {'N': [], 'deg_mean': [], 'deg_sem': [], 'r_mean': [], 'r_sem': [], 'n_sess': []}
         for a in AREAS}
for a in AREAS:
    for N in N_LADDER:
        rows = per_area[a][N]
        if len(rows) < MIN_SESSIONS_PER_POINT:
            continue
        dmean, dsem, nd = mean_sem([d for _, d, _ in rows], ddof=1)
        rmean, rsem, _  = mean_sem([r for _, _, r in rows], ddof=1)
        curve[a]['N'].append(N)
        curve[a]['deg_mean'].append(dmean); curve[a]['deg_sem'].append(dsem)
        curve[a]['r_mean'].append(rmean);   curve[a]['r_sem'].append(rsem)
        curve[a]['n_sess'].append(nd)


# ══════════════════════════════════════════════════════════════════════════════
# MATCHED-N RANKING (+ significance vs chance)
# ══════════════════════════════════════════════════════════════════════════════

def _p_vs_chance(degs):
    """One-sample test that mean error < CHANCE_DEG (below-chance error = decoding)."""
    v = np.asarray([x for x in degs if np.isfinite(x)], float)
    if len(v) < 3:
        return np.nan
    if len(v) >= 5:
        try:
            return wilcoxon(v - CHANCE_DEG, alternative='less').pvalue
        except ValueError:
            return np.nan
    return ttest_1samp(v, CHANCE_DEG, alternative='less').pvalue

matched = {}
for N in MATCHED_N:
    tbl = []
    for a in AREAS:
        rows = per_area[a][N]
        degs = [d for _, d, _ in rows]
        rs   = [r for _, _, r in rows]
        dmean, dsem, nd = mean_sem(degs, ddof=1)
        rmean, rsem, _  = mean_sem(rs, ddof=1)
        tbl.append({'area': a, 'deg_mean': dmean, 'deg_sem': dsem,
                    'r_mean': rmean, 'r_sem': rsem, 'n_sess': nd,
                    'p_vs_chance': _p_vs_chance(degs)})
    df = pd.DataFrame(tbl).sort_values('deg_mean')   # ascending error = best first
    matched[N] = df
    print(f"\n=== Matched-N = {N}: ranking (lowest error = best decoding) ===")
    print(df.to_string(index=False,
          formatters={'deg_mean': '{:.1f}'.format, 'deg_sem': '{:.1f}'.format,
                      'r_mean': '{:.3f}'.format, 'r_sem': '{:.3f}'.format,
                      'p_vs_chance': '{:.2g}'.format}))


# ══════════════════════════════════════════════════════════════════════════════
# SAVE + PLOT
# ══════════════════════════════════════════════════════════════════════════════

with open(_pkl_out, 'wb') as fh:
    pickle.dump({'per_area': per_area, 'curve': curve,
                 'matched': {N: df.to_dict('records') for N, df in matched.items()},
                 'config': {'N_LADDER': N_LADDER, 'MATCHED_N': MATCHED_N,
                            'R_DRAWS': R_DRAWS, 'chance_deg': CHANCE_DEG,
                            'late_delay': LATE_DELAY,
                            'delay_start_eff': _DELAY_START_EFF,
                            'match_trials': MATCH_TRIALS}}, fh)
print(f"\nSaved → {_pkl_out}")

# ── Curve: mean |error| (deg) vs N, one line per area ─────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for a in AREAS:
    if not curve[a]['N']:
        continue
    ax.errorbar(curve[a]['N'], curve[a]['deg_mean'], yerr=curve[a]['deg_sem'],
                color=AREA_COLORS[a], marker='o', ms=4, lw=2, capsize=2, label=a)
ax.axhline(CHANCE_DEG, ls='--', color='gray', lw=1, label='chance (~90°)')
for N in MATCHED_N:
    ax.axvline(N, ls=':', color='k', alpha=0.35)
ax.set_xlabel('# neurons (matched)'); ax.set_ylabel('mean abs decoding error (°)')
ax.set_title(f'Neuron-dropping: decoding vs population size '
             f'({"late " if LATE_DELAY else ""}delay {_DELAY_START_EFF}–{DELAY_END}s'
             f'{f", {MATCH_TRIALS}tr" if MATCH_TRIALS else ""})')
ax.invert_yaxis()                      # lower error = better, so up = better
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
_curve_path = os.path.join(FIG_DIR, f'neuron_dropping_curve{_SUFFIX}.svg')
fig.savefig(_curve_path)
print(f"Saved → {_curve_path}")

# ── Bar: matched-N ranking (headline = MATCHED_N[0]) ──────────────────────────
Nh = MATCHED_N[0]
df = matched[Nh]
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(df['area'], df['deg_mean'], yerr=df['deg_sem'],
       color=[AREA_COLORS[a] for a in df['area']], capsize=3)
ax.axhline(CHANCE_DEG, ls='--', color='gray', lw=1)
ax.set_ylabel('mean abs decoding error (°)')
ax.set_title(f'Matched-N = {Nh}: ranking ({"late " if LATE_DELAY else ""}delay '
             f'{_DELAY_START_EFF}–{DELAY_END}s{f", {MATCH_TRIALS}tr" if MATCH_TRIALS else ""})')
ax.invert_yaxis()
fig.tight_layout()
_bar_path = os.path.join(FIG_DIR, f'neuron_dropping_bar{_SUFFIX}.svg')
fig.savefig(_bar_path)
print(f"Saved → {_bar_path}")
print("\nDone.")
