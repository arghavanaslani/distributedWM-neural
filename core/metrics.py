"""
core/metrics.py
===============
Circular statistics used throughout the pipeline.

All angles are in RADIANS. The correlation is the Jammalamadaka circular
correlation coefficient; `circ_corr` and `circ_corr_vec` are the same estimator,
the latter vectorised so every time bin is computed in one pass.
"""

import numpy as np


def circ_mean(x):
    """Circular mean of angles (radians), ignoring NaN."""
    return np.arctan2(np.nanmean(np.sin(x)), np.nanmean(np.cos(x)))


def circ_corr(a, b):
    """Circular correlation between two angle vectors (radians). Scalar."""
    sa = np.sin(a - circ_mean(a))
    sb = np.sin(b - circ_mean(b))
    num = np.nansum(sa * sb)
    den = np.sqrt(np.nansum(sa ** 2) * np.nansum(sb ** 2))
    return num / den if den > 0 else np.nan


def circ_corr_vec(pred_v, true_v):
    """Circular correlation for every time bin at once.

    pred_v : (n_valid, n_time) predicted angles
    true_v : (n_valid,)        true angles
    returns: (n_time,)
    """
    alpha_bar = np.arctan2(np.nanmean(np.sin(pred_v), axis=0),
                           np.nanmean(np.cos(pred_v), axis=0))
    sin_a = np.sin(pred_v - alpha_bar)
    sin_b = np.sin(true_v - circ_mean(true_v))[:, None]
    num   = np.nansum(sin_a * sin_b, axis=0)
    denom = np.sqrt(np.nansum(sin_a ** 2, axis=0) * np.nansum(sin_b ** 2))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(denom > 0, num / denom, np.nan)
