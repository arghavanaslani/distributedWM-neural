"""
coupling_eyecontrol.py
======================
Tier-3 eye-position control for the neural-behavioural coupling (caveats #3, #8).

Question: are the couplings (esp. MT's NEGATIVE one) driven by fixational gaze
position? MT is retinotopic; if the eye drifts during the delay, its decoded
target and the saccade endpoint move together, faking coupling with the wrong
sign. We partial out fixation gaze and see which couplings survive.

Method (per session, on identical trials):
  a = sin(neural_err  − circ_mean)      # centred, as in core.metrics.circ_corr_vec
  b = sin(beh_err      − circ_mean)
  r_full    = corr(a, b)                # == the circular coupling (validation)
  r_partial = corr(resid_a, resid_b)    # after regressing a,b on [fixX, fixY]
Aggregate per area across sessions; compare r_full vs r_partial; test vs zero.

Inputs
  errors_angular.csv   per-trial-per-bin neural error + beh_err_rad + keys
  behavior_all.csv     fixation gaze: FIX_X, FIX_Y, keyed by session_id/trial_id
NO re-decoding.

Run:  python coupling_eyecontrol.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_1samp

from config import AREAS, BEHAVIOR_CSV, DELAY_START, DELAY_END, PKL_DIR

ERRORS_CSV  = os.path.join(PKL_DIR, 'errors_angular.csv')
from core.metrics import circ_mean
from core.stats import mean_sem
ANGLE_NAME  = 'targetAngle'
MIN_TRIALS  = 10
BEH_ERR_RADIANS = False                 # 'err' is in degrees

FIX_X, FIX_Y = 'fixBaseX_raw', 'fixBaseY_raw'   # fixation gaze columns


def _clean_id(s):
    return (s.astype(str).str.strip("[]' ").replace('', np.nan)
             .astype(float).astype('Int64'))




def _full_corr(a, b):
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return np.sum(a * b) / d if d > 0 else np.nan


def _partial_corr(a, b, C):
    C1 = np.column_stack([np.ones(len(a)), C])
    ra = a - C1 @ np.linalg.lstsq(C1, a, rcond=None)[0]
    rb = b - C1 @ np.linalg.lstsq(C1, b, rcond=None)[0]
    return _full_corr(ra, rb)


def _vs_zero(v):
    v = v[np.isfinite(v)]
    if len(v) < 3: return np.nan
    if len(v) < 5: return float(ttest_1samp(v, 0).pvalue)
    try:    return float(wilcoxon(v, alternative='two-sided').pvalue)
    except ValueError: return np.nan




# ── Load + merge gaze ─────────────────────────────────────────────────────────
print("Loading errors_angular.csv ...")
dn = pd.read_csv(ERRORS_CSV)
dn = dn[dn['angle_name'] == ANGLE_NAME].copy()

# behavioural error in radians
if 'beh_err_rad' in dn.columns:
    dn['b_rad'] = dn['beh_err_rad'].astype(float)
elif 'err' in dn.columns:
    dn['b_rad'] = dn['err'].astype(float) * (1.0 if BEH_ERR_RADIANS else np.pi / 180)
else:
    raise KeyError("No 'beh_err_rad' or 'err' in errors_angular.csv")

print("Loading behavior_all.csv (gaze) ...")
db = pd.read_csv(BEHAVIOR_CSV)
for c in (FIX_X, FIX_Y):
    if c not in db.columns:
        raise KeyError(f"'{c}' not in behavior_all.csv — check FIX_X/FIX_Y names.")

dn['_s'] = _clean_id(dn['original_session'])
dn['_t'] = pd.to_numeric(dn['original_trial'], errors='coerce').astype('Int64')
db['_s'] = _clean_id(db['session_id'])
db['_t'] = pd.to_numeric(db['trial_id'], errors='coerce').astype('Int64')

dn = dn.merge(db[['_s', '_t', FIX_X, FIX_Y]].drop_duplicates(['_s', '_t']),
              on=['_s', '_t'], how='left')

# restrict to delay window
dn = dn[(dn['time'] >= DELAY_START) & (dn['time'] <= DELAY_END)].copy()
matched = np.isfinite(dn[FIX_X].values) & np.isfinite(dn[FIX_Y].values)
print(f"  delay-window rows: {len(dn)} | with gaze: {int(matched.sum())} "
      f"({100*matched.mean():.0f}%)\n")


# ── Per area / session: r_full vs r_partial ───────────────────────────────────
print(f"  {'area':9s} {'n_sess':>6s} {'r_full':>8s} {'r_partial':>9s} "
      f"{'Δ':>7s} {'p_partial':>9s}")
summary = {}
for area in AREAS:
    da = dn[dn['area'] == area]
    if da.empty:
        continue
    r_full_l, r_part_l = [], []
    for sess, g in da.groupby('session_idx'):
        # per-trial delay-averaged neural error (circular mean over bins)
        piv = g.groupby('trial_idx').agg(
            nerr=('error', lambda x: circ_mean(x.values.astype(float))),
            brad=('b_rad', 'first'),
            fx=(FIX_X, 'first'),
            fy=(FIX_Y, 'first'))
        piv = piv.dropna(subset=['nerr', 'brad', 'fx', 'fy'])
        if len(piv) < MIN_TRIALS:
            continue
        nerr = piv['nerr'].values; brad = piv['brad'].values
        a = np.sin(nerr - circ_mean(nerr))
        b = np.sin(brad - circ_mean(brad))
        C = np.column_stack([piv['fx'].values - piv['fx'].mean(),
                             piv['fy'].values - piv['fy'].mean()])
        r_full_l.append(_full_corr(a, b))
        r_part_l.append(_partial_corr(a, b, C))
    rf = np.array(r_full_l); rp = np.array(r_part_l)
    mf, sf, n = mean_sem(rf, ddof=1)
    mp, sp, _ = mean_sem(rp, ddof=1)
    pp = _vs_zero(rp)
    summary[area] = (mf, mp, n)
    print(f"  {area:9s} {n:6d} {mf:+8.3f} {mp:+9.3f} {mp-mf:+7.3f} {pp:9.2g}")

print("\n  r_full should reproduce the circular coupling (LIP +0.074, PFC +0.043,"
      " MT -0.061).\n  If MT's r_partial → ~0, gaze explains it; if it stays "
      "negative, it's genuine read-out.")
print("\nDone.")
