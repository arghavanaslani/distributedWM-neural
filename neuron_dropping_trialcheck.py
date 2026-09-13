"""
neuron_dropping_trialcheck.py
=============================
Diagnostic for the residual training-set-size effect: does per-session decoding
get BETTER (lower error) with more trials, and — the part that matters for the
ranking — do areas differ systematically in trial count?

Reads neuron_dropping.pkl (per-area/session/N decoding) + trial_data.pkl (trial
counts). NO re-decoding.

Two questions:
  Q1  Within area, does error fall as trials rise?   Spearman rho(error, trials).
      Also pooled across areas after removing each area's mean (area-centered),
      so the area ranking itself doesn't drive the correlation.
      rho < 0  = more trials -> lower error = training-size effect present.
  Q2  Do areas differ in trial count?  Kruskal-Wallis across areas' per-session
      trial counts. If NOT, trial count cannot bias the cross-area ranking even
      if Q1 is nonzero (trial count is a session property shared across areas).

Run:  python neuron_dropping_trialcheck.py
"""

import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kruskal

RESULTS_DIR = '/home/aarghavan/aslan/distributedWM-neural/results/'
PKL_DIR     = os.path.join(RESULTS_DIR, 'pkl')
FIG_DIR     = os.path.join(RESULTS_DIR, 'figures')
AREAS       = ['PFC', 'FEF', 'LIP', 'Parietal', 'IT', 'MT', 'V4']
ANGLE_XY    = ('targetX', 'targetY')

# Match neuron_dropping.py to read the corresponding results file.
LATE_DELAY  = False
_SUFFIX     = '_latedelay' if LATE_DELAY else ''

AREA_COLORS = {
    'PFC': '#1f77b4', 'FEF': '#d62728', 'LIP': '#2ca02c', 'Parietal': '#ff7f0e',
    'IT': '#17becf', 'MT': '#9467bd', 'V4': '#8c564b',
}

# ── Load ──────────────────────────────────────────────────────────────────────
with open(os.path.join(PKL_DIR, f'neuron_dropping{_SUFFIX}.pkl'), 'rb') as fh:
    res = pickle.load(fh)
per_area  = res['per_area']
MATCHED_N = res['config']['MATCHED_N']

with open(os.path.join(PKL_DIR, 'trial_data.pkl'), 'rb') as fh:
    tdata = pickle.load(fh)
trial_data = tdata['trial_data']
varX, varY = ANGLE_XY

# trial count per session = # valid (finite target) trials, as used in decoding
trials_per_session = {}
for s, tdf in trial_data.items():
    if varX in tdf.columns and varY in tdf.columns:
        tX = tdf[varX].values.astype(float)
        tY = tdf[varY].values.astype(float)
        trials_per_session[s] = int(np.sum(np.isfinite(tX) & np.isfinite(tY)))

# ── Per matched-N ─────────────────────────────────────────────────────────────
for N in MATCHED_N:
    print(f"\n══════════════  Matched-N = {N}  ══════════════")

    # Q1: within-area rho(error, trials) + area-centered pooled
    print("\nQ1  decoding error vs trial count")
    print(f"  {'area':9s} {'n':>3s} {'trials med':>10s} {'rho':>6s} {'p':>8s}")
    pooled_resid, pooled_trials, per_area_trials = [], [], {}
    for a in AREAS:
        rows = per_area[a][N]
        deg  = np.array([d for _, d, _ in rows], float)
        sess = [s for s, _, _ in rows]
        tr   = np.array([trials_per_session.get(s, np.nan) for s in sess], float)
        m = np.isfinite(deg) & np.isfinite(tr)
        deg, tr = deg[m], tr[m]
        per_area_trials[a] = tr
        if len(deg) >= 5:
            rho, p = spearmanr(tr, deg)
        else:
            rho, p = np.nan, np.nan
        print(f"  {a:9s} {len(deg):3d} {np.median(tr):10.0f} {rho:6.2f} {p:8.2g}")
        pooled_resid.append(deg - deg.mean())     # remove area mean
        pooled_trials.append(tr)

    rr = np.concatenate(pooled_resid); tt = np.concatenate(pooled_trials)
    m = np.isfinite(rr) & np.isfinite(tt)
    rho_p, p_p = spearmanr(tt[m], rr[m])
    print(f"\n  POOLED (area-centered): rho = {rho_p:+.2f}, p = {p_p:.2g}, n = {int(m.sum())}")
    print("  (rho < 0  = more trials -> lower error = training-size effect)")

    # Q2: do areas differ in trial count?
    groups = [per_area_trials[a] for a in AREAS if len(per_area_trials[a]) >= 5]
    if len(groups) >= 2:
        H, p_kw = kruskal(*groups)
        print(f"\nQ2  areas differ in trial count?  Kruskal-Wallis H = {H:.1f}, "
              f"p = {p_kw:.2g}  ({'YES' if p_kw < 0.05 else 'no'})")

    # verdict
    print("\n  Verdict:")
    if (np.isnan(rho_p) or p_p >= 0.05):
        print("    Q1 flat → no training-size effect in range → trial count is NOT an issue.")
    elif len(groups) >= 2 and p_kw >= 0.05:
        print("    Q1 present but Q2 flat → areas share trial counts → does NOT bias ranking.")
    else:
        print("    Q1 present AND areas differ in trials → possible ranking bias → consider matched-trials.")

# ── Scatter (headline N) ──────────────────────────────────────────────────────
Nh = MATCHED_N[0]
fig, ax = plt.subplots(figsize=(7, 5))
for a in AREAS:
    rows = per_area[a][Nh]
    deg  = np.array([d for _, d, _ in rows], float)
    tr   = np.array([trials_per_session.get(s, np.nan) for s, _, _ in rows], float)
    ax.scatter(tr, deg, s=18, color=AREA_COLORS[a], label=a, alpha=0.7)
ax.set_xlabel('# trials (session)'); ax.set_ylabel('decoding error (°)')
ax.invert_yaxis()
ax.set_title(f'Decoding vs trial count (matched-N = {Nh}{", late" if LATE_DELAY else ""})')
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
_p = os.path.join(FIG_DIR, f'neuron_dropping_trialcheck{_SUFFIX}.svg')
fig.savefig(_p)
print(f"\nSaved → {_p}\nDone.")
