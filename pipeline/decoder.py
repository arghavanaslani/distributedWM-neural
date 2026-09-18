"""
decoder.py
==========
Neural decoding pipeline — HEAVY computation step.
Run once per dataset / per change in decoding hyperparameters.

Input
-----
  A filtered data dict produced by load_and_filter.load_pipeline(), stored as a PKL.
  Expected keys per session index s:
    data['spikecounts'][s]   ndarray (n_trials, n_neurons, n_time)
    data['trial'][s]         DataFrame  —  must contain targetX/Y, responseX/Y,
                                           plus behavioral columns from behavior_all.csv
    data['unit'][s]          DataFrame  —  must contain 'area' column

  The spikecounts are assumed to be sliding-window bins (window=0.1 s, step=0.025 s).

Outputs  (all written to RESULTS_DIR)
--------------------------------------
  smoothed.pkl                 cached smoothed spikecounts
  centered.pkl                 cached z-scored spikecounts + filtered unit metadata
  predicted_data.pkl           raw predictions + config fingerprint
  shuffled_data.pkl            null (label-permuted) predictions
  cross_temp_predictions.pkl   cross-temporal predictions  (if COMPUTE_CROSS_TEMPORAL)
  cross_temp_shuf_stats.pkl    per-session shuffle mean/std for CT z-scoring
  trial_data.pkl               normalised trial DataFrames + time / event metadata

Caching
-------
  If predicted_data.pkl exists and its embedded config fingerprint matches the current
  DECODE_CONFIG dict, decoding is skipped entirely.  Set FORCE_RECOMPUTE=True or delete
  predicted_data.pkl to force a fresh run.  A mismatch prints which keys changed.
"""

# ─── stdlib / third-party ─────────────────────────────────────────────────────
import os
import sys
import pickle
import itertools
import time as _time
from math import ceil

# Limit BLAS/LAPACK internal threading so joblib's session-level parallelism
# doesn't compete with numpy's own thread pool (would cause n_cpu² threads).
# Must be set before numpy is imported.
os.environ.setdefault('OMP_NUM_THREADS',       '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS',  '1')
os.environ.setdefault('MKL_NUM_THREADS',       '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',   '1')

import numpy as np
import pandas as pd
import scipy.signal

from sklearn.model_selection import LeaveOneOut, KFold
from joblib import Parallel, delayed
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  ← edit paths and decoding parameters here
# ══════════════════════════════════════════════════════════════════════════════

from config import (
    AREAS,
    DATA_PATH,
    DELAY_END,
    DELAY_START,
    EV_RESPONSE,
    EV_TARGET_OFF,
    EV_TARGET_ON,
    ORIG_BIN,
    PKL_DIR,
    RESULTS_DIR,
    STEP_S,
    T_START,
    WINDOW_S,
    env_bool,
    env_int,
)

FORCE_RECOMPUTE = env_bool('DISTWM_FORCE', False)   # run.py --force

# ── Time-axis parameters (must match load_and_filter.py settings) ─────────────
N_BINS_RAW = 240     # number of raw bins in the epoch

# ── Smoothing (causal exponential kernel applied after sliding window) ─────────
SMOOTH_WIDTH = 1.5   # kernel temporal extent (seconds)
SMOOTH_K     = 2.0   # shape parameter

# ── Decoding ──────────────────────────────────────────────────────────────────
MIN_NEURONS  = 10
COMPUTE_NULL = True
N_SHUFFLES   = 200    # shuffles for standard decoding null
N_SHUFFLES_CT = 50    # shuffles for cross-temporal null (more expensive)
CV_FOLDS     = 5
USE_LOO      = False
RIDGE_ALPHAS   = [0.01, 0.1, 1.0, 10.0, 100.0]
ALPHA_CV_FOLDS = 3

# ── PCA pre-processing ────────────────────────────────────────────────────────
USE_PCA = False
N_PCS   = 15

MAX_SESSIONS = env_int('DISTWM_MAX_SESSIONS', None)   # run.py --max-sessions

# ── Cross-temporal decoding ────────────────────────────────────────────────────
COMPUTE_CROSS_TEMPORAL = True

# ── Decode time window ────────────────────────────────────────────────────────
DECODE_START = 1      # only decode from this time (seconds); None = full epoch
DECODE_END   = None   # clip end (seconds); None = keep all

# ── Variables to decode ───────────────────────────────────────────────────────
VARIABLES_TO_DECODE = ['targetX', 'targetY']
ANGLE_PAIRS = {'targetAngle': ('targetX', 'targetY')}

# ── Event times (seconds, relative to epoch start) ────────────────────────────

# ── Behavioral columns written to trial_data.pkl ──────────────────────────────
BEH_COLS = ['targAng', 'respAng_dva', 'err', 'folded_err', 'abs_err',
            'memoryDelay', 'ITI']

os.makedirs(PKL_DIR, exist_ok=True)
SMOOTHED_PATH = os.path.join(PKL_DIR, 'smoothed.pkl')
CENTERED_PATH = os.path.join(PKL_DIR, 'centered.pkl')

_pred_path    = os.path.join(PKL_DIR, 'predicted_data.pkl')
_shuf_path    = os.path.join(PKL_DIR, 'shuffled_data.pkl')
_ct_pred_path = os.path.join(PKL_DIR, 'cross_temp_predictions.pkl')
_ct_shuf_path = os.path.join(PKL_DIR, 'cross_temp_shuf_stats.pkl')
_trial_path   = os.path.join(PKL_DIR, 'trial_data.pkl')


# ── Config fingerprint — used to detect stale cached predictions ──────────────
DECODE_CONFIG = {
    'smooth_width':        SMOOTH_WIDTH,
    'smooth_k':            SMOOTH_K,
    'window_s':            WINDOW_S,
    'step_s':              STEP_S,
    'use_loo':             USE_LOO,
    'cv_folds':            CV_FOLDS,
    'ridge_alphas':        sorted(RIDGE_ALPHAS),
    'alpha_cv_folds':      ALPHA_CV_FOLDS,
    'n_shuffles':          N_SHUFFLES,
    'n_shuffles_ct':       N_SHUFFLES_CT,
    'use_pca':             USE_PCA,
    'n_pcs':               N_PCS,
    'compute_null':        COMPUTE_NULL,
    'min_neurons':         MIN_NEURONS,
    'decode_start':        DECODE_START,
    'decode_end':          DECODE_END,
    'variables_to_decode': sorted(VARIABLES_TO_DECODE),
    'angle_pairs':         {k: tuple(sorted(v)) for k, v in ANGLE_PAIRS.items()},
    'areas':               list(AREAS),
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nLoading data from:\n  {DATA_PATH}")
with open(DATA_PATH, 'rb') as f:
    data = pickle.load(f)

num_sessions = len(data['spikecounts'])
print(f"  {num_sessions} sessions loaded.")
if num_sessions == 0:
    raise ValueError("No sessions found in data!")

print("\nVerifying spikecounts ↔ unit alignment:")
for s in range(num_sessions):
    sc  = np.asarray(data['spikecounts'][s])
    n_sc, n_u = sc.shape[1], len(data['unit'][s])
    if n_sc != n_u:
        raise ValueError(
            f"  Session {s}: spikecounts has {n_sc} neurons "
            f"but unit DataFrame has {n_u}."
        )
print("  OK\n")

# Build time axis from actual data shape (avoids N_BINS_RAW mismatch)
_n_time_actual = np.asarray(data['spikecounts'][0]).shape[-1]
W = round(WINDOW_S / ORIG_BIN)
S = round(STEP_S   / ORIG_BIN)
time      = np.array([T_START + (i * S + W / 2) * ORIG_BIN for i in range(_n_time_actual)])
n_time    = _n_time_actual
delay_idx = (time >= DELAY_START) & (time <= DELAY_END)
print(f"Time axis: {n_time} bins  [{time[0]:.3f} s → {time[-1]:.3f} s]  step={STEP_S} s\n")


# ══════════════════════════════════════════════════════════════════════════════
# TARGET VARIABLE PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def _minmax_norm(x):
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi == lo:
        return np.zeros_like(x, dtype=float)
    return 2.0 * (x - lo) / (hi - lo) - 1.0


_REQUIRED_COLS = {'targetX', 'targetY', 'responseX', 'responseY'}

print("Normalising trial variables ...")
for i, df in enumerate(data['trial']):
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Session {i} is missing columns: {sorted(missing)}")
    df = df.copy()
    for col in ['targetX', 'targetY', 'responseX', 'responseY']:
        df[col] = _minmax_norm(df[col].values.astype(float))
    df['prevtargetX']   = df['targetX'].shift(1).fillna(0)
    df['prevtargetY']   = df['targetY'].shift(1).fillna(0)
    df['prevresponseX'] = df['responseX'].shift(1).fillna(0)
    df['prevresponseY'] = df['responseY'].shift(1).fillna(0)
    data['trial'][i] = df
print("  Done.\n")


# ══════════════════════════════════════════════════════════════════════════════
# TRIAL DATA SAVER  (called on cache hit AND at end of full run)
# ══════════════════════════════════════════════════════════════════════════════

def _save_trial_data():
    pkg = {
        'trial_data':     {s: data['trial'][s] for s in range(num_sessions)},
        'time':           time,
        'n_time':         n_time,
        'delay_idx':      delay_idx,
        'areas':          AREAS,
        'angle_pairs':    ANGLE_PAIRS,
        'ev_target_on':   EV_TARGET_ON,
        'ev_target_off':  EV_TARGET_OFF,
        'ev_response':    EV_RESPONSE,
        'delay_start':    DELAY_START,
        'delay_end':      DELAY_END,
        'step_s':         STEP_S,
        'beh_cols':       BEH_COLS,
    }
    with open(_trial_path, 'wb') as fh:
        pickle.dump(pkg, fh)
    print(f"Saved → {_trial_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CACHE CHECK
# ══════════════════════════════════════════════════════════════════════════════

if not FORCE_RECOMPUTE and os.path.exists(_pred_path):
    print(f"Found cached predictions: {_pred_path}")
    with open(_pred_path, 'rb') as fh:
        _cached = pickle.load(fh)
    cached_cfg = _cached.get('config', {})

    if cached_cfg == DECODE_CONFIG:
        print("Config fingerprint matches — skipping decoding.")
        _save_trial_data()
        print("\nTo rerun, delete predicted_data.pkl or set FORCE_RECOMPUTE=True.")
        sys.exit(0)
    else:
        changed = sorted(k for k in set(DECODE_CONFIG) | set(cached_cfg)
                         if DECODE_CONFIG.get(k) != cached_cfg.get(k))
        print(f"⚠  Config mismatch in: {', '.join(changed)}")
        print("   Rerunning decoding with updated config.\n")


# ══════════════════════════════════════════════════════════════════════════════
# SMOOTHING & CENTERING
# ══════════════════════════════════════════════════════════════════════════════

def _make_exp_kernel(bin_size=STEP_S, width=SMOOTH_WIDTH, K=SMOOTH_K):
    """Causal exponential smoothing kernel."""
    bin_w = int(ceil(width / bin_size))
    win   = scipy.signal.windows.exponential(2 * bin_w + 1, tau=bin_w / (2 * K))
    win[:bin_w] = 0
    win /= win.sum() * bin_size
    return win


def _smooth_session(X, kernel):
    out = np.empty_like(X)
    for c, n in itertools.product(range(X.shape[0]), range(X.shape[1])):
        out[c, n, :] = np.convolve(X[c, n, :], kernel, mode='same')
    return out


def load_or_compute_smoothed_centered(data):
    kernel = _make_exp_kernel()

    if os.path.exists(CENTERED_PATH):
        print(f"Loading centered data from:\n  {CENTERED_PATH}")
        with open(CENTERED_PATH, 'rb') as fh:
            pkg = pickle.load(fh)
        if not isinstance(pkg, dict) or 'spikecounts' not in pkg:
            raise ValueError(f"Invalid format in {CENTERED_PATH}")
        data['spikecounts'] = pkg['spikecounts']
        data['unit']        = pkg['unit']
        print("  Done.\n")
        return data

    if os.path.exists(SMOOTHED_PATH):
        print(f"Loading smoothed data from:\n  {SMOOTHED_PATH}")
        with open(SMOOTHED_PATH, 'rb') as fh:
            data['spikecounts'] = pickle.load(fh)
        print("  Done.")
    else:
        print("Computing exponential smoothing ...")
        data['spikecounts'] = [
            _smooth_session(np.asarray(sc), kernel)
            for sc in tqdm(data['spikecounts'], desc='Smoothing')
        ]
        with open(SMOOTHED_PATH, 'wb') as fh:
            pickle.dump(data['spikecounts'], fh)
        print(f"  Saved → {SMOOTHED_PATH}")

    print("\nCentering (z-scoring neurons across trials × time) ...")
    for s, sc in enumerate(tqdm(data['spikecounts'], desc='Centering')):
        sc   = np.asarray(sc)
        flat = sc.reshape(sc.shape[1], -1)
        std  = np.std(flat, axis=1)
        keep = std > 0
        mu   = np.mean(flat[keep], axis=1)
        data['unit'][s]        = data['unit'][s].loc[keep].reset_index(drop=True)
        data['spikecounts'][s] = (
            (sc[:, keep, :] - mu[None, :, None]) / std[keep][None, :, None]
        )
        assert data['spikecounts'][s].shape[1] == len(data['unit'][s])

    with open(CENTERED_PATH, 'wb') as fh:
        pickle.dump({'spikecounts': data['spikecounts'], 'unit': data['unit']}, fh)
    print(f"  Saved → {CENTERED_PATH}\n")
    return data


data = load_or_compute_smoothed_centered(data)

# Restrict to decode window
_dm = np.ones(n_time, dtype=bool)
if DECODE_START is not None:
    _dm &= (time >= DECODE_START)
if DECODE_END is not None:
    _dm &= (time <= DECODE_END)
if not _dm.all():
    for s in range(num_sessions):
        data['spikecounts'][s] = np.asarray(data['spikecounts'][s])[:, :, _dm]
    time      = time[_dm]
    n_time    = len(time)
    delay_idx = (time >= DELAY_START) & (time <= DELAY_END)
    print(f"Decode window: {time[0]:.3f} s → {time[-1]:.3f} s  ({n_time} bins)\n")


# ══════════════════════════════════════════════════════════════════════════════
# DECODING  (Ridge + KFold/LOO, parallelised across sessions)
# ══════════════════════════════════════════════════════════════════════════════

def _make_cv():
    if USE_LOO:
        return LeaveOneOut()
    return KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)


def _select_alpha_batched(Xtr_b, y_tr):
    """Pick best ridge alpha from RIDGE_ALPHAS via inner KFold CV."""
    n_bins, n_train, n_feat = Xtr_b.shape
    inner_cv      = KFold(n_splits=ALPHA_CV_FOLDS, shuffle=False)
    eye_i         = np.eye(n_feat)
    mse_per_alpha = np.zeros(len(RIDGE_ALPHAS))

    for ai, alpha in enumerate(RIDGE_ALPHAS):
        fold_mse = []
        for itr_idx, ival_idx in inner_cv.split(np.arange(n_train)):
            Xin  = Xtr_b[:, itr_idx, :]
            Xval = Xtr_b[:, ival_idx, :]
            XtX  = np.einsum('bni,bnj->bij', Xin, Xin)
            Xty  = np.einsum('bni,n->bi',    Xin, y_tr[itr_idx])
            W    = np.linalg.solve(XtX + alpha * eye_i, Xty)
            pred = np.einsum('bni,bi->bn', Xval, W)
            fold_mse.append(np.mean((pred - y_tr[ival_idx][None, :]) ** 2))
        mse_per_alpha[ai] = np.mean(fold_mse)

    return float(RIDGE_ALPHAS[int(np.argmin(mse_per_alpha))])


def _decode_session(session, data, area, variables,
                    n_shuffles=N_SHUFFLES, compute_null=COMPUTE_NULL):
    """Decode all variables for one session/area via batched ridge regression."""
    area_mask = data['unit'][session]['area'].values == area
    if area_mask.sum() < MIN_NEURONS:
        return session, None, None, None

    X        = np.asarray(data['spikecounts'][session])[:, area_mask, :]
    trial_df = data['trial'][session]
    n_trials, n_neurons, n_bins = X.shape

    import time as _t, sys as _sys
    def _log(msg): _sys.stderr.write(msg + '\n'); _sys.stderr.flush()

    sess_start = _t.time()
    _log(f"  [sess {session}]  area={area}  neurons={n_neurons}  trials={n_trials}")

    cv        = _make_cv()
    eye       = np.eye(n_neurons)
    zero_bins = np.array([np.all(X[:, :, t] == 0) for t in range(n_bins)])
    preds_dict = {}
    shuf_dict  = {} if compute_null else None
    n_vars     = len(variables)

    if compute_null:
        sess_seed = int(session) if np.isscalar(session) else hash(str(session)) % (2**31)
        rng       = np.random.RandomState(sess_seed)
        perm_mat  = np.stack([rng.permutation(n_trials) for _ in range(n_shuffles)], axis=0)
    else:
        perm_mat = None

    for v_idx, v in enumerate(variables, 1):
        y = trial_df[v].values.astype(float)
        if np.unique(y[np.isfinite(y)]).size < 2:
            _log(f"  [sess {session}/{area}]  ({v_idx}/{n_vars}) {v}  — skipped (constant)")
            continue

        var_start = _t.time()
        _log(f"  [sess {session}/{area}]  ({v_idx}/{n_vars}) decoding {v} ...")

        preds = np.full((n_trials, n_bins), np.nan)
        shuf  = np.full((n_shuffles, n_trials, n_bins), np.nan) if compute_null else None

        for tr_idx, te_idx in cv.split(X[:, :, 0]):
            Xtr = X[tr_idx]; Xte = X[te_idx]; y_tr = y[tr_idx]

            mu  = Xtr.mean(axis=0); std = Xtr.std(axis=0); std[std == 0] = 1.0
            Xtr_b = ((Xtr - mu) / std).transpose(2, 0, 1)
            Xte_b = ((Xte - mu) / std).transpose(2, 0, 1)

            if USE_PCA:
                _, _, Vt  = np.linalg.svd(Xtr_b, full_matrices=False)
                n_keep    = min(N_PCS, Vt.shape[1])
                Vt_k      = Vt[:, :n_keep, :]
                Xtr_b     = np.einsum('bni,bpi->bnp', Xtr_b, Vt_k)
                Xte_b     = np.einsum('bni,bpi->bnp', Xte_b, Vt_k)
                eye_r     = np.eye(n_keep)
            else:
                eye_r = eye

            alpha = _select_alpha_batched(Xtr_b, y_tr)
            XtX   = np.einsum('bni,bnj->bij', Xtr_b, Xtr_b)
            A     = XtX + alpha * eye_r
            W     = np.linalg.solve(A, np.einsum('bni,n->bi', Xtr_b, y_tr))
            preds[te_idx] = np.einsum('bni,bi->nb', Xte_b, W)

            if compute_null:
                y_sh_tr  = y[perm_mat][:, tr_idx]
                Xty_sh_T = np.einsum('bni,sn->sbi', Xtr_b, y_sh_tr).transpose(1, 2, 0)
                W_sh_T   = np.linalg.solve(A, Xty_sh_T)
                shuf[:, te_idx, :] = np.einsum('bni,bif->fnb', Xte_b, W_sh_T)

        preds[:, zero_bins] = np.nan
        preds_dict[v] = {'predictions': preds}
        if compute_null:
            shuf[:, :, zero_bins] = np.nan
            shuf_dict[v] = shuf

        _log(f"  [sess {session}/{area}]  ({v_idx}/{n_vars}) {v}  done in {_t.time()-var_start:.1f}s")

    _log(f"  [sess {session}/{area}]  ✓ all variables done in {_t.time()-sess_start:.1f}s")
    return session, preds_dict, trial_df, shuf_dict


def decode_area(data, sessions, area):
    """Run decoding for all qualifying sessions in area; parallelised."""
    valid = [s for s in sessions
             if (data['unit'][s]['area'].values == area).sum() >= MIN_NEURONS]
    if not valid:
        print(f"  {area}: no qualifying sessions — skipped.")
        return {}, {}

    print(f"\n  {area}: decoding {len(valid)} / {len(sessions)} sessions ...")
    t0 = _time.time()

    results = Parallel(n_jobs=-1, prefer='threads')(
        delayed(_decode_session)(s, data, area, VARIABLES_TO_DECODE)
        for s in valid
    )

    preds_all, shuf_all = {}, {}
    for s, pd_, td, sd in results:
        if pd_ is not None:
            preds_all[s] = pd_
            data['trial'][s] = td
        if sd is not None:
            shuf_all[s] = sd

    print(f"  {area}: done in {_time.time() - t0:.1f} s")
    return preds_all, shuf_all


# ── Run standard decoding ─────────────────────────────────────────────────────
sessions = list(range(min(MAX_SESSIONS, num_sessions) if MAX_SESSIONS else num_sessions))

print("\n" + "═" * 60)
print("STANDARD DECODING")
print("═" * 60)

all_predictions: dict = {}
all_shuffles:    dict = {}

for area in AREAS:
    p, sh = decode_area(data, sessions, area)
    if p:
        all_predictions[area] = p
        all_shuffles[area]    = sh

# Save with config fingerprint embedded so cache check can detect stale files
with open(_pred_path, 'wb') as fh:
    pickle.dump({'predictions': all_predictions, 'config': DECODE_CONFIG}, fh)
with open(_shuf_path, 'wb') as fh:
    pickle.dump(all_shuffles, fh)
print(f"\nSaved → {_pred_path}")
print(f"Saved → {_shuf_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-TEMPORAL DECODING
# Train on bin t_train, test on all bins t_test → (n_trials, n_train, n_test).
# ══════════════════════════════════════════════════════════════════════════════

def _decode_session_cross_temporal(session, data, area, variables,
                                    angle_pairs=None, n_shuffles_ct=N_SHUFFLES_CT,
                                    compute_null=COMPUTE_NULL):
    """Cross-temporal decoding for one session/area."""
    import time as _t, sys as _sys
    def _log(msg): _sys.stderr.write(msg + '\n'); _sys.stderr.flush()

    area_mask = data['unit'][session]['area'].values == area
    if area_mask.sum() < MIN_NEURONS:
        return session, None, None, None

    X        = np.asarray(data['spikecounts'][session])[:, area_mask, :]
    trial_df = data['trial'][session]
    n_trials, n_neurons, n_bins = X.shape

    _log(f"  [CT sess {session}]  area={area}  neurons={n_neurons}  trials={n_trials}  bins={n_bins}")

    cv   = _make_cv()
    eye  = np.eye(n_neurons)
    cross_preds = {}
    cross_shuf  = {} if (compute_null and angle_pairs) else None
    n_vars      = len(variables)

    if compute_null and angle_pairs:
        sess_seed = int(session) if np.isscalar(session) else hash(str(session)) % (2**31)
        rng       = np.random.RandomState(sess_seed)
        perm_mat  = np.stack([rng.permutation(n_trials) for _ in range(n_shuffles_ct)], axis=0)
    else:
        perm_mat = None

    for v_idx, v in enumerate(variables, 1):
        y = trial_df[v].values.astype(float)
        if np.unique(y[np.isfinite(y)]).size < 2:
            _log(f"  [CT sess {session}/{area}]  ({v_idx}/{n_vars}) {v}  — skipped (constant)")
            continue

        var_start = _t.time()
        _log(f"  [CT sess {session}/{area}]  ({v_idx}/{n_vars}) decoding {v} ...")

        preds = np.full((n_trials, n_bins, n_bins), np.nan)
        shuf  = (np.full((n_shuffles_ct, n_trials, n_bins, n_bins), np.nan, dtype=np.float32)
                 if cross_shuf is not None else None)

        for fold_i, (tr_idx, te_idx) in enumerate(cv.split(X[:, :, 0])):
            Xtr = X[tr_idx]; Xte = X[te_idx]; y_tr = y[tr_idx]

            mu  = Xtr.mean(axis=0); std = Xtr.std(axis=0); std[std == 0] = 1.0
            Xtr_b = ((Xtr - mu) / std).transpose(2, 0, 1)

            if USE_PCA:
                _, _, Vt = np.linalg.svd(Xtr_b, full_matrices=False)
                n_keep   = min(N_PCS, Vt.shape[2])
                Vt_k     = Vt[:, :n_keep, :]
                Xtr_b    = np.einsum('bni,bpi->bnp', Xtr_b, Vt_k)
                eye_ct   = np.eye(n_keep)
            else:
                eye_ct = eye; Vt_k = None

            alpha = _select_alpha_batched(Xtr_b, y_tr)
            XtX   = np.einsum('bni,bnj->bij', Xtr_b, Xtr_b)
            A     = XtX + alpha * eye_ct
            W     = np.linalg.solve(A, np.einsum('bni,n->bi', Xtr_b, y_tr))

            W_shuf = None
            if cross_shuf is not None:
                y_sh_tr  = y[perm_mat][:, tr_idx]
                Xty_sh_T = np.einsum('bni,sn->sbi', Xtr_b, y_sh_tr).transpose(1, 2, 0)
                W_shuf   = np.linalg.solve(A, Xty_sh_T)

            for t_train in range(n_bins):
                if np.all(Xtr[:, :, t_train] == 0):
                    continue
                mu_a  = mu[:, t_train]; std_a = std[:, t_train]; w_a = W[t_train]
                Xte_sc = (Xte - mu_a[None, :, None]) / std_a[None, :, None]

                if USE_PCA:
                    Xte_proj = Xte_sc.transpose(2, 0, 1) @ Vt_k[t_train].T
                else:
                    Xte_proj = Xte_sc.transpose(2, 0, 1)

                preds[te_idx, t_train, :] = (Xte_proj @ w_a).T

                if W_shuf is not None:
                    pred_shuf = Xte_proj @ W_shuf[t_train]
                    shuf[:, te_idx, t_train, :] = pred_shuf.transpose(2, 1, 0).astype(np.float32)

            _log(f"  [CT sess {session}/{area}]  {v}  fold {fold_i+1}  ({_t.time()-var_start:.0f}s)")

        cross_preds[v] = preds
        if cross_shuf is not None:
            cross_shuf[v] = shuf
        _log(f"  [CT sess {session}/{area}]  ({v_idx}/{n_vars}) {v}  done in {_t.time()-var_start:.1f}s")

    # Collapse shuffle predictions → per-angle (mu, sd) stats, then free raw arrays
    ct_shuf_stats = None
    if cross_shuf is not None:
        ct_shuf_stats = {}
        for angle_name, (varX, varY) in angle_pairs.items():
            if varX not in cross_shuf or varY not in cross_shuf:
                continue
            pX = cross_shuf[varX]; pY = cross_shuf[varY]
            pred_ang = np.arctan2(pY, pX)
            tX = trial_df[varX].values.astype(float)[None, :, None, None]
            tY = trial_df[varY].values.astype(float)[None, :, None, None]
            true_ang = np.arctan2(tY, tX)
            err = np.arctan2(np.sin(true_ang - pred_ang), np.cos(true_ang - pred_ang))
            abs_err_per_shuf = np.nanmean(np.abs(err), axis=1)   # (n_shuf, n_tr, n_te)
            ct_shuf_stats[angle_name] = (
                np.nanmean(abs_err_per_shuf, axis=0),             # (n_tr, n_te) mu
                np.nanstd (abs_err_per_shuf, axis=0, ddof=1),     # (n_tr, n_te) sd
            )
        cross_shuf = None   # free memory

    return session, cross_preds, trial_df, ct_shuf_stats


def decode_area_cross_temporal(data, sessions, area):
    """Parallelised cross-temporal decoding for all qualifying sessions in area."""
    valid = [s for s in sessions
             if (data['unit'][s]['area'].values == area).sum() >= MIN_NEURONS]
    if not valid:
        print(f"  {area}: no qualifying sessions — skipped.")
        return {}, {}

    print(f"\n  {area}: CT decoding {len(valid)} / {len(sessions)} sessions ...")
    t0 = _time.time()

    results = Parallel(n_jobs=-1, prefer='threads')(
        delayed(_decode_session_cross_temporal)(
            s, data, area, VARIABLES_TO_DECODE, angle_pairs=ANGLE_PAIRS)
        for s in valid
    )

    cross_all, ct_shuf_all = {}, {}
    for s, cross_preds, td, ct_shuf_stats in results:
        if cross_preds is not None:
            cross_all[s] = cross_preds
            data['trial'][s] = td
            if ct_shuf_stats is not None:
                ct_shuf_all[s] = ct_shuf_stats

    print(f"  {area}: CT done in {_time.time() - t0:.1f} s")
    return cross_all, ct_shuf_all


if COMPUTE_CROSS_TEMPORAL:
    print("\n" + "═" * 60)
    cv_label = "LOO" if USE_LOO else f"KFold({CV_FOLDS})"
    print(f"CROSS-TEMPORAL DECODING  [{cv_label}]")
    print("═" * 60)

    all_cross_preds: dict = {}
    all_ct_shuf:     dict = {}
    for area in AREAS:
        cp, ct_sh = decode_area_cross_temporal(data, sessions, area)
        if cp:
            all_cross_preds[area] = cp
        if ct_sh:
            all_ct_shuf[area] = ct_sh

    with open(_ct_pred_path, 'wb') as fh:
        pickle.dump(all_cross_preds, fh)
    with open(_ct_shuf_path, 'wb') as fh:
        pickle.dump(all_ct_shuf, fh)
    print(f"\nSaved → {_ct_pred_path}")
    print(f"Saved → {_ct_shuf_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE TRIAL DATA
# ══════════════════════════════════════════════════════════════════════════════

_save_trial_data()

print("\n" + "═" * 60)
print("DECODING COMPLETE")
print(f"  Results: {PKL_DIR}")
print("  Files written:")
for fname in ['smoothed.pkl', 'centered.pkl',
              'predicted_data.pkl', 'shuffled_data.pkl', 'trial_data.pkl']:
    print(f"    {fname}")
if COMPUTE_CROSS_TEMPORAL:
    print("    cross_temp_predictions.pkl")
    print("    cross_temp_shuf_stats.pkl")
print("═" * 60)
