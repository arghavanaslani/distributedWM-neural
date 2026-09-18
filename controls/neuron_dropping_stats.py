"""
neuron_dropping_stats.py
========================
Post-hoc pairwise statistics on the matched-N decoding ranking.
Reads neuron_dropping.pkl (already produced by neuron_dropping.py) — NO re-decoding.

For each matched-N, compares every pair of areas on their per-session decoding
error (degrees) with a two-sided Mann-Whitney U test (areas = different, mostly
non-overlapping session sets → unpaired), Holm-corrected across all pairs.
Prints the corrected significance for each pair, most-significant first, and
flags the two claims of interest: LIP > all, and FEF vs PFC (matched pair).

Run:
  python neuron_dropping_stats.py
"""

import os
import pickle
from itertools import combinations

import numpy as np
from scipy.stats import mannwhitneyu

from config import AREAS, PKL_DIR, RESULTS_DIR

# Match these to neuron_dropping.py to read the corresponding results file.
from core.stats import stars
LATE_DELAY   = True
MATCH_TRIALS = 80
_SUFFIX      = ('_latedelay' if LATE_DELAY else '') + (f'_t{MATCH_TRIALS}' if MATCH_TRIALS else '')

with open(os.path.join(PKL_DIR, f'neuron_dropping{_SUFFIX}.pkl'), 'rb') as fh:
    res = pickle.load(fh)
per_area  = res['per_area']
MATCHED_N = res['config']['MATCHED_N']


def _holm(pvals):
    """Holm–Bonferroni step-down adjusted p-values (order preserved)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj




for N in MATCHED_N:
    # per-session degrees per area
    vals = {}
    for a in AREAS:
        v = np.array([d for _, d, _ in per_area[a][N] if np.isfinite(d)], float)
        if len(v) >= 3:
            vals[a] = v
    present = [a for a in AREAS if a in vals]
    means = {a: vals[a].mean() for a in present}
    rank  = sorted(present, key=lambda a: means[a])   # best (lowest error) first

    pairs = list(combinations(present, 2))
    praw = []
    for a, b in pairs:
        try:
            praw.append(mannwhitneyu(vals[a], vals[b], alternative='two-sided').pvalue)
        except ValueError:
            praw.append(np.nan)
    padj = _holm(praw)

    print(f"\n================  Matched-N = {N}  ================")
    print("Ranking (best → worst, mean error °):",
          "  ".join(f"{a}({means[a]:.1f})" for a in rank))

    # All pairs, most significant first
    print("\nPairwise (Holm-corrected):")
    order = np.argsort(padj)
    for i in order:
        a, b = pairs[i]
        better, worse = (a, b) if means[a] < means[b] else (b, a)
        print(f"  {better:8s} < {worse:8s}  Δ={abs(means[a]-means[b]):4.1f}°  "
              f"p_holm={padj[i]:.2g} {stars(padj[i])}")

    # Claims of interest
    def _get(a, b):
        for i, (x, y) in enumerate(pairs):
            if {x, y} == {a, b}:
                return padj[i]
        return np.nan
    print("\nClaims of interest:")
    if 'LIP' in present:
        lip_all = [( _get('LIP', a)) for a in present if a != 'LIP']
        print(f"  LIP vs all others:  max p_holm = {np.nanmax(lip_all):.2g} "
              f"({'all significant' if np.nanmax(lip_all) < 0.05 else 'NOT all significant'})")
    if 'FEF' in present and 'PFC' in present:
        p = _get('FEF', 'PFC')
        print(f"  FEF vs PFC:         p_holm = {p:.2g} {stars(p)}  "
              f"({'matched (n.s.) — clean dissociation' if p >= 0.05 else 'differ'})")
    if 'MT' in present:
        for other in ('FEF', 'PFC'):
            if other in present:
                p = _get('MT', other)
                print(f"  MT vs {other}:          p_holm = {p:.2g} {stars(p)}")

print("\nDone.")
