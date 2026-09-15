"""
core/stats.py
=============
Significance testing and summary statistics shared across the pipeline.

Note on `ddof`: the pipeline currently uses ddof=0 for the main decoding and
coupling figures and ddof=1 for the matched-N and eye-control analyses. The
parameter is explicit at every call site so that inconsistency is visible rather
than buried in three separate implementations. ddof=1 is the statistically
correct choice for a standard error estimated from n sessions; switching the
ddof=0 sites is a deliberate change, not a refactor, and would widen the error
bars on figures 01, 02, 06 and 12 by sqrt((n-1)/n).
"""

import numpy as np
from scipy.stats import ttest_1samp, wilcoxon

ALPHA = 0.05


def mean_sem(values, axis=None, ddof=0):
    """Mean and standard error over finite values, ignoring NaN and +/-inf.

    Returns (mean, sem, n). `sem` is NaN where n <= ddof.
    """
    a = np.asarray(values, dtype=float)
    a = np.where(np.isfinite(a), a, np.nan)
    n = np.sum(np.isfinite(a), axis=axis)
    with np.errstate(invalid='ignore', divide='ignore'):
        m   = np.nanmean(a, axis=axis)
        sd  = np.nanstd(a, axis=axis, ddof=ddof)
        sem = np.where(n > ddof, sd / np.sqrt(n), np.nan)
    return m, sem, n


def test_vs_zero(values):
    """Two-sided test that values differ from zero.

    n >= 5 : Wilcoxon signed-rank
    n 3-4  : one-sample t-test
    n < 3  : NaN (not testable)
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.nan
    if len(v) < 5:
        return float(ttest_1samp(v, 0).pvalue)
    try:
        return float(wilcoxon(v, alternative='two-sided').pvalue)
    except ValueError:
        return np.nan


def stars(p):
    """Significance marker. Empty string for an untested (NaN) p-value -
    'not tested' is a different claim from 'not significant'."""
    if p is None or not np.isfinite(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < ALPHA:
        return '*'
    return 'n.s.'
