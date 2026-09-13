"""
coupling_stats.py
=================
Tier-1 + Tier-2 post-hoc for the neural-behavioural coupling caveats.
NO re-decoding — reads two existing pkls.

Tier 1  (caveats #4, #6)
  - Effective n per area  (finite per-session values; the legend n over-counts
    because sessions failing the trial/behaviour gate are kept as NaN rows).
  - vs-zero significance per area, matching analysis.py's _test_vs_zero
    (two-sided Wilcoxon for n>=5, t-test for 3-4), then Holm- AND FDR-corrected
    across the 7 areas — the coupling effects sit near threshold, so correction
    matters here.

Tier 2  (caveat #2)
  - Scatter of coupling (y) vs matched-N decoding (x, degrees), one point/area,
    showing coupling does NOT track decoding strength. Prints the paired table
    and a descriptive across-area rank correlation (n=7 → descriptive only).

Inputs
  <PKL_DIR>/neurobeh_results.pkl   nb_delay_per_sess[area] = {'sessions','values'}
  <PKL_DIR>/neuron_dropping.pkl    matched[N] = [{area, deg_mean, r_mean, ...}]

Run:  python coupling_stats.py
"""

import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, ttest_1samp, spearmanr

RESULTS_DIR = '/home/aarghavan/aslan/distributedWM-neural/results/'
PKL_DIR     = os.path.join(RESULTS_DIR, 'pkl')
FIG_DIR     = os.path.join(RESULTS_DIR, 'figures')
AREAS       = ['PFC', 'FEF', 'LIP', 'Parietal', 'IT', 'MT', 'V4']
ALPHA       = 0.05
DEC_N       = 15          # matched-N decoding level for the scatter (neuron-only)

AREA_COLORS = {
    'PFC': '#1f77b4', 'FEF': '#d62728', 'LIP': '#2ca02c', 'Parietal': '#ff7f0e',
    'IT': '#17becf', 'MT': '#9467bd', 'V4': '#8c564b',
}


# ── helpers ───────────────────────────────────────────────────────────────────
def _test_vs_zero(v):
    """Match analysis.py: two-sided Wilcoxon (n>=5), t-test (3-4), NaN (<3)."""
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.nan
    if len(v) < 5:
        return float(ttest_1samp(v, 0).pvalue)
    try:
        return float(wilcoxon(v, alternative='two-sided').pvalue)
    except ValueError:
        return np.nan


def _holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx]); adj[idx] = min(run, 1.0)
    return adj


def _bh(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        prev = min(prev, p[idx] * m / (rank + 1)); adj[idx] = min(prev, 1.0)
    return adj


def _stars(p):
    if p is None or np.isnan(p): return ''
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < ALPHA else 'n.s.'


# ── load coupling ─────────────────────────────────────────────────────────────
with open(os.path.join(PKL_DIR, 'neurobeh_results.pkl'), 'rb') as fh:
    res = pickle.load(fh)
nb = res.get('nb_delay_per_sess')
if nb is None:
    raise KeyError("neurobeh_results.pkl has no 'nb_delay_per_sess' — re-run analysis.py.")

present = [a for a in AREAS if a in nb and np.any(np.isfinite(np.asarray(nb[a]['values'], float)))]
if len(present) < len(AREAS):
    missing = [a for a in AREAS if a not in present]
    print(f"⚠  Coupling results present for {len(present)}/{len(AREAS)} areas. "
          f"Missing: {missing}")
    print("   Re-run analysis.py with AREAS = all 7 and RECOMPUTE=True for the full set.\n")


# ── Tier 1: effective n + corrected vs-zero significance ──────────────────────
rows = []
for a in present:
    v   = np.asarray(nb[a]['values'], float)
    fin = v[np.isfinite(v)]
    n_total, n_eff = len(v), len(fin)
    mean = fin.mean()
    sem  = fin.std(ddof=1) / np.sqrt(n_eff) if n_eff > 1 else np.nan
    rows.append({'area': a, 'n_total': n_total, 'n_eff': n_eff,
                 'mean': mean, 'sem': sem, 'p_raw': _test_vs_zero(fin)})

praw   = np.array([r['p_raw'] for r in rows])
p_holm = _holm(praw); p_fdr = _bh(praw)

print("═════════  Tier 1: coupling significance (delay window)  ═════════")
print(f"  {'area':9s} {'n_leg':>5s} {'n_eff':>5s} {'coupling':>9s} {'sem':>6s} "
      f"{'p_raw':>8s} {'p_holm':>8s} {'p_fdr':>8s}")
for r, ph, pf in zip(rows, p_holm, p_fdr):
    print(f"  {r['area']:9s} {r['n_total']:5d} {r['n_eff']:5d} "
          f"{r['mean']:+9.3f} {r['sem']:6.3f} "
          f"{r['p_raw']:8.2g} {ph:8.2g}{_stars(ph):>4s} {pf:8.2g}")
print("  (n_leg = legend/row count, n_eff = sessions actually contributing)")


# ── Tier 2: coupling vs matched-N decoding ────────────────────────────────────
coupling = {r['area']: r['mean'] for r in rows}
coup_holm = {r['area']: ph for r, ph in zip(rows, p_holm)}

dec_deg = None
try:
    with open(os.path.join(PKL_DIR, 'neuron_dropping.pkl'), 'rb') as fh:
        dec = pickle.load(fh)
    matched = dec['matched'].get(DEC_N) or dec['matched'].get(str(DEC_N))
    dec_deg = {d['area']: d['deg_mean'] for d in matched}
except (FileNotFoundError, KeyError, TypeError) as e:
    print(f"\n⚠  Could not load matched-N decoding for the scatter ({e}). "
          f"Run neuron_dropping.py first.")

if dec_deg:
    areas_sc = [a for a in present if a in dec_deg]
    xs = np.array([dec_deg[a] for a in areas_sc])      # decoding error (deg); lower=better
    ys = np.array([coupling[a] for a in areas_sc])     # coupling
    rho, p_rho = spearmanr(xs, ys)                      # descriptive (n=7)

    print(f"\n═════════  Tier 2: coupling vs decoding (matched-N={DEC_N})  ═════════")
    print(f"  {'area':9s} {'dec_err(°)':>10s} {'coupling':>9s} {'p_holm':>8s}")
    for a in sorted(areas_sc, key=lambda a: dec_deg[a]):   # best decoder first
        print(f"  {a:9s} {dec_deg[a]:10.1f} {coupling[a]:+9.3f} "
              f"{coup_holm[a]:8.2g}{_stars(coup_holm[a]):>4s}")
    print(f"\n  across-area Spearman(decoding, coupling) = {rho:+.2f}, p = {p_rho:.2g}  "
          f"(n={len(areas_sc)} → descriptive only)")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for a in areas_sc:
        ax.scatter(dec_deg[a], coupling[a], s=80, color=AREA_COLORS[a], zorder=3)
        ax.annotate(a, (dec_deg[a], coupling[a]), textcoords='offset points',
                    xytext=(6, 4), fontsize=9)
    ax.axhline(0, color='k', lw=0.8, alpha=0.4)
    ax.invert_xaxis()                      # better decoding → right
    ax.set_xlabel(f'decoding error (°, matched-N={DEC_N})  ← better')
    ax.set_ylabel('neural–behavioural coupling (delay)')
    ax.set_title('Coupling does not track decoding strength')
    fig.tight_layout()
    _p = os.path.join(FIG_DIR, 'coupling_vs_decoding.svg')
    fig.savefig(_p)
    print(f"\nSaved → {_p}")

print("\nDone.")
