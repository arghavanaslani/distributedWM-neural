"""
load_and_filter.py
==================
Clean data loading and filtering pipeline for the distributed working memory project.

Raw data structure (both PKL files)
-------------------------------------
  data['trial'][s]       — dict: {'allTasks': DataFrame, 'delsac': DataFrame, ...}
  data['unit'][s]        — DataFrame of neuron metadata
  data['session'][s]     — dict: {'delsac': session-metadata-dict, ...}
  data['spiketimes'][s]  — ndarray (trials_all, units, timebins) @ 0.025 s  ← PKL 1 key
  data['spikecounts'][s] — ndarray (trials_all, units, timebins) @ 0.025 s  ← PKL 2 key
  data['trialsNeuron'][s]— list/array, per-neuron trial count

Key inconsistencies resolved here
-----------------------------------
  1. PKL1 spike key  : 'spiketimes'   → normalised to 'spikecounts'
     PKL2 spike key  : 'spikecounts'  → already correct
  2. Delsac trial extraction:
       - spikecounts are aligned to allTasks (all task trials)
       - delsac_idx is derived from allTasks['task'] == 'delsac'
       - spikecounts sliced by delsac_idx before any other filtering
       - trial DataFrame taken from trial[s]['delsac']
  3. task column in allTasks may be scalar strings OR 1-element arrays
     → both handled in _get_delsac_mask()
  4. Session IDs with underscores (e.g. '110107_01') are preserved as-is
     → int-conversion is only attempted if the string has no underscore

Pipeline  (sliding-window mode, default)
------------------------------------------
  load_pipeline(pkl1, pkl2, behavior_csv, ...)
    └── load_and_concat_raw_pkls(pkl1, pkl2)      # load + normalise + concat
    └── extract_delsac_sessions(raw)              # select delsac, slice spikes
    └── filter_neurons(data)                      # area + min-trials threshold
    └── align_to_target_on(data, bin=0.025 s)     # shift so targetOn = 1.7 s
    └── sliding_window_spikecounts(data)          # 0.025 s → 237 bins (0.1/0.025)
    └── merge_behavior(data, beh_df)              # add behavioral columns from CSV
    └── filter_by_subject / nMapStim / iti        # optional session/trial filters
    └── remove_nan_trials(data)                   # drop trials with NaN in key cols
    └── drop_empty_sessions(data)                 # remove sessions with 0 trials

  Alignment happens BEFORE sliding window so it operates at full 0.025 s precision.
  Pass use_sliding_window=False for legacy non-overlapping rebin (→ 60 bins at 0.1 s).

Usage
------
    from load_and_filter import load_pipeline

    data = load_pipeline(
        pkl1               = "/path/to/all_data_siegel_0.025.pkl",
        pkl2               = "/path/to/all_new_13_tasks.pkl",
        behavior_csv       = "/path/to/behavior_all.csv",
        subject            = "Paula",             # None = both
        nMapStim           = 6,                   # None = all;  or [6, 8]
        max_iti            = 5.0,                 # None = no ITI filter
        original_bin       = 0.025,
        use_sliding_window = True,                # default: sliding window, 50% overlap
        window_s           = 0.2,                # 0.2 s window, 0.1 s step → 59 bins
        step_s             = 0.1,
        output_path        = "/path/to/delsac_filtered.pkl",  # None = don't save
    )
"""

from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
from typing import Optional
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AREAS = ['PFC', 'FEF', 'LIP', 'Parietal', 'IT', 'MT', 'V4']

# Minimum number of recorded trials for a neuron to be included
MIN_TRIALS_PER_NEURON = 100

# Minimum number of neurons in an area for a session to qualify
MIN_NEURONS_PER_AREA = 10

# Target time (seconds) to which targetOn is aligned across sessions
TARGET_ON_ALIGNED = 1.7

# Behavioral columns to pull from behavior_all.csv into each trial DataFrame.
# Columns already present in the neural pkl are overwritten by these values.
BEHAVIORAL_COLS = [
    'targEcc', 'targAng', 'targX', 'targY',
    'respEcc_raw', 'respAng_raw',
    'respX_dva', 'respY_dva', 'respEcc_dva', 'respAng_dva',
    'memoryDelay',
    'curr', 'resp', 'prev',
    'diff_signed', 'diff_abs', 'diff',
    'err', 'folded_err', 'abs_err',
    'trial_ends', 'trial_starts', 'ITI',
]

# Column renames applied after merging behavioral columns.
# Keys are the names as they appear in behavior_all.csv;
# values are the final names used throughout the rest of the pipeline.
COLUMN_RENAMES: dict[str, str] = {
    'targX':    'targetX',
    'targY':    'targetY',
    'respX_dva': 'responseX',
    'respY_dva': 'responseY',
}

# ---------------------------------------------------------------------------
# Subject → session mapping
# ---------------------------------------------------------------------------
PAULA_SESSIONS: list[str] = [
    '100706', '100724', '100725', '100726', '100730', '100731', '100802',
    '100803', '100804', '100817', '100818', '100819', '100820', '100823',
    '100824', '100826', '100827', '100828', '101122', '101123', '101124',
    '101127', '101128', '101202', '101206', '101207', '101209', '101210',
    '101216', '110106',
    '110107_01', '110107_02',
    '110110_01', '110110_02',
    '110111_01', '110111_02',
    '110112_01', '110112_02',
    '110115_01', '110115_02',
    '110116_01', '110116_02',
    '110117_01', '110117_02',
]

REX_SESSIONS: list[str] = [
    '100907', '100910', '100913', '100915', '100917', '100920', '100921',
    '101008', '101009', '101023', '101024', '101027', '101028', '101030',
    '110120', '110121',
    '110214', '110215', '110216', '110218', '110219', '110220',
]

# ---------------------------------------------------------------------------
# nMapStim session lists
# ---------------------------------------------------------------------------

# nMapStim == 6  (45 sessions)
NMAP6_SESSIONS: list[str] = [
    '100706', '100819', '100917', '101123',
    '100724', '100820', '100920', '101124',
    '100725', '100823', '100921', '101127', '110120',
    '100726', '100824', '101008', '101128', '110121',
    '100730', '100826', '101009', '101202',
    '100731', '100827', '101023', '101206',
    '100802', '100828', '101024', '101207',
    '100803', '100907', '101027', '101209',
    '100804', '100910', '101028', '101210',
    '100817', '100913', '101030', '101216',
    '100818', '100915', '101122',
]

# nMapStim == 8  (18 sessions; excludes 110106, 110214, 110215)
NMAP8_SESSIONS: list[str] = [
    '110107_01', '110107_02',
    '110110_01', '110110_02',
    '110111_01', '110111_02',
    '110112_01', '110112_02',
    '110115_01', '110115_02',
    '110116_01', '110116_02',
    '110117_01', '110117_02',
    '110216', '110218',
    '110219', '110220',
]

NMAP6_OR_8_SESSIONS: list[str] = NMAP6_SESSIONS + NMAP8_SESSIONS


# ---------------------------------------------------------------------------
# 1. Load both PKL files and concatenate
# ---------------------------------------------------------------------------

def load_and_concat_raw_pkls(pkl1_path: str, pkl2_path: str) -> dict:
    """
    Load the two raw PKL files and return a single concatenated raw dict.

    Inconsistency resolved:
        PKL1 stores spikes under 'spiketimes'
        PKL2 stores spikes under 'spikecounts'
        → both are unified under 'spikecounts' in the output

    The trial and session values are still nested dicts at this point
    (e.g. data['trial'][s] = {'allTasks': df, 'delsac': df, ...}).
    Call extract_delsac_sessions() next.
    """
    print(f"Loading PKL1: {pkl1_path}")
    with open(pkl1_path, 'rb') as f:
        d1 = pickle.load(f)

    print(f"Loading PKL2: {pkl2_path}")
    with open(pkl2_path, 'rb') as f:
        d2 = pickle.load(f)

    # --- resolve spike key inconsistency ---
    spikes1 = _get_spike_list(d1, label='PKL1')
    spikes2 = _get_spike_list(d2, label='PKL2')

    # --- verify required keys exist in both ---
    for label, d in [('PKL1', d1), ('PKL2', d2)]:
        for key in ('trial', 'unit', 'session', 'trialsNeuron'):
            if key not in d:
                raise KeyError(f"{label} is missing required key '{key}'")

    merged = {
        'trial':        d1['trial']        + d2['trial'],
        'unit':         d1['unit']         + d2['unit'],
        'session':      d1['session']      + d2['session'],
        'spikecounts':  spikes1            + spikes2,
        'trialsNeuron': d1['trialsNeuron'] + d2['trialsNeuron'],
    }

    print(f"  PKL1: {len(d1['trial'])} sessions")
    print(f"  PKL2: {len(d2['trial'])} sessions")
    print(f"  Combined: {len(merged['trial'])} sessions total")
    return merged


def _get_spike_list(d: dict, label: str) -> list:
    """Return the spike list from a raw pkl, resolving the spiketimes/spikecounts ambiguity."""
    if 'spikecounts' in d:
        return list(d['spikecounts'])
    if 'spiketimes' in d:
        return list(d['spiketimes'])
    raise KeyError(
        f"{label} has neither 'spikecounts' nor 'spiketimes'. "
        f"Found keys: {list(d.keys())}"
    )


# ---------------------------------------------------------------------------
# 2. Extract delsac sessions from nested raw structure
# ---------------------------------------------------------------------------

def extract_delsac_sessions(raw: dict) -> dict:
    """
    From the raw concatenated dict (nested trial/session dicts), produce a
    flat dict where each entry corresponds to one delsac session:

        data['trial'][s]        → delsac trial DataFrame
        data['unit'][s]         → unit DataFrame (all neurons, unfiltered)
        data['session'][s]      → delsac session metadata dict
        data['spikecounts'][s]  → ndarray (delsac_trials, units, timebins)
        data['trialsNeuron'][s] → per-neuron trial count array

    Two things happen here:
        1. Sessions without a 'delsac' key are skipped.
        2. spikecounts are sliced along the trial axis so they contain only
           delsac trials (using allTasks['task'] == 'delsac' as the mask).
    """
    out = {'trial': [], 'unit': [], 'session': [], 'spikecounts': [], 'trialsNeuron': []}
    n_skipped = 0

    for s in tqdm(range(len(raw['trial'])), desc='Extracting delsac sessions'):
        sess_meta = raw['session'][s]

        # skip sessions without delsac
        if not isinstance(sess_meta, dict) or 'delsac' not in sess_meta:
            n_skipped += 1
            continue

        all_tasks_df = raw['trial'][s].get('allTasks', None)
        delsac_df    = raw['trial'][s].get('delsac',   None)

        if all_tasks_df is None or delsac_df is None:
            n_skipped += 1
            continue

        # build boolean mask over allTasks rows for delsac trials
        delsac_mask = _get_delsac_mask(all_tasks_df)
        if delsac_mask.sum() == 0:
            n_skipped += 1
            continue

        # slice spikecounts to delsac trials only  (trials, units, timebins)
        sc = np.asarray(raw['spikecounts'][s])
        if sc.shape[0] != len(all_tasks_df):
            # fall back: assume spikecounts already aligned to delsac
            pass
        else:
            sc = sc[delsac_mask]

        out['trial'].append(delsac_df.reset_index(drop=True))
        out['unit'].append(raw['unit'][s])
        out['session'].append(sess_meta['delsac'])
        out['spikecounts'].append(sc)
        out['trialsNeuron'].append(np.asarray(raw['trialsNeuron'][s]))

    print(f"  {len(out['trial'])} delsac sessions extracted, {n_skipped} skipped (no delsac).")
    return out


def _get_delsac_mask(all_tasks_df: pd.DataFrame) -> np.ndarray:
    """
    Return a boolean mask over allTasks rows where task == 'delsac'.

    Handles two formats:
        - task column contains scalar strings: 'delsac'
        - task column contains 1-element arrays: ['delsac']
    """
    raw = all_tasks_df['task'].values
    # scalar strings
    if all(np.ndim(t) == 0 for t in raw):
        labels = np.array([str(t) for t in raw])
    else:
        # arrays → concatenate to flat
        labels = np.concatenate([np.atleast_1d(t) for t in raw])
    return labels == 'delsac'


# ---------------------------------------------------------------------------
# 3. Neuron filtering
# ---------------------------------------------------------------------------

def filter_neurons(data: dict,
                   areas: list[str] = AREAS,
                   min_trials: int = MIN_TRIALS_PER_NEURON) -> dict:
    """
    Keep only neurons that:
      - belong to a known area in `areas`
      - have trialsNeuron > min_trials

    Updates data['unit'], data['spikecounts'], data['trialsNeuron'].
    spikecounts shape: (trials, units, timebins)
    """
    for s in range(len(data['unit'])):
        unit_df       = data['unit'][s]
        trials_neuron = np.asarray(data['trialsNeuron'][s])

        idx_area = np.isin(unit_df['area'].values, areas)
        idx_min  = trials_neuron > min_trials
        keep     = idx_area & idx_min

        data['unit'][s]         = unit_df[keep].reset_index(drop=True)
        data['spikecounts'][s]  = data['spikecounts'][s][:, keep, :]
        data['trialsNeuron'][s] = trials_neuron[keep]

    return data


# ---------------------------------------------------------------------------
# 4. Rebinning  (non-overlapping  OR  sliding window)
# ---------------------------------------------------------------------------

def rebin_spikecounts(data: dict,
                      original_bin: float = 0.025,
                      target_bin:   float = 0.1) -> dict:
    """
    Sum spike counts into coarser, NON-OVERLAPPING time bins.

    target_bin must be an integer multiple of original_bin.
    spikecounts shape: (trials, units, timebins)
    """
    factor = round(target_bin / original_bin)
    if factor == 1:
        return data

    if not np.isclose(factor * original_bin, target_bin):
        raise ValueError(
            f"target_bin ({target_bin}) must be an integer multiple of "
            f"original_bin ({original_bin})."
        )

    for s in range(len(data['spikecounts'])):
        sc = np.asarray(data['spikecounts'][s])   # (T, U, B)
        T, U, B = sc.shape
        trim = (B // factor) * factor
        data['spikecounts'][s] = (
            sc[..., :trim]
            .reshape(T, U, trim // factor, factor)
            .sum(axis=-1)
        )

    return data


def sliding_window_spikecounts(data: dict,
                                original_bin: float = 0.025,
                                window:       float = 0.1,
                                step:         float = 0.025) -> dict:
    """
    Apply a SLIDING WINDOW sum over spike counts along the time axis.

    Unlike rebin_spikecounts (non-overlapping), consecutive windows here
    share most of their bins, so adjacent decoded time points are correlated
    and the decoding curve is naturally smoother.

    Parameters
    ----------
    original_bin : bin size of raw spikecounts (seconds)
    window       : integration window width (seconds); default 0.1 s = 4 bins
    step         : step between consecutive windows (seconds); default 0.025 s
                   step < window → overlapping windows

    Output
    ------
    spikecounts shape: (trials, units, n_windows)
        n_windows = (B - W) // S + 1
        where B = original timebins, W = window in bins, S = step in bins

    Use sliding_window_time_axis() to get the matching time axis.
    """
    W = round(window / original_bin)
    S = round(step   / original_bin)

    if W < 1 or S < 1:
        raise ValueError("window and step must each be ≥ one original bin.")

    for s in range(len(data['spikecounts'])):
        sc = np.asarray(data['spikecounts'][s])   # (T, U, B)
        T, U, B = sc.shape
        n_windows = (B - W) // S + 1

        # Use stride tricks for memory-efficient windowed sum
        from numpy.lib.stride_tricks import as_strided
        shape   = (T, U, n_windows, W)
        strides = (sc.strides[0], sc.strides[1],
                   sc.strides[2] * S, sc.strides[2])
        windowed = as_strided(sc, shape=shape, strides=strides)
        data['spikecounts'][s] = windowed.sum(axis=-1)   # (T, U, n_windows)

    return data


def sliding_window_time_axis(t_start:      float = -2.5,
                              original_bin: float = 0.025,
                              n_bins:       int   = 240,
                              window:       float = 0.1,
                              step:         float = 0.025) -> np.ndarray:
    """
    Return the time axis (window centres) matching sliding_window_spikecounts.

    Each value is the centre time of the corresponding integration window.
    """
    W = round(window / original_bin)
    S = round(step   / original_bin)
    n_windows = (n_bins - W) // S + 1
    centres = np.array([t_start + (i * S + W / 2) * original_bin
                        for i in range(n_windows)])
    return centres


# ---------------------------------------------------------------------------
# 5. Time alignment  (shift spikecounts + event columns so targetOn = 1.7 s)
# ---------------------------------------------------------------------------

def align_to_target_on(data: dict,
                       target_time: float = TARGET_ON_ALIGNED,
                       bin_size:    float = 0.1) -> dict:
    """
    Shift each session's spikecounts and event-time columns so that
    targetOn == target_time (default 1.7 s) for all sessions.

    Sessions without a valid targetOn are left unchanged.
    """
    event_cols = [
        'stimOn', 'stimOff', 'targetOn', 'targetOff', 'fixptOff',
        'responseTime', 'responseDone', 'fixptOn', 'trialRefTimes',
    ]

    for s in range(len(data['trial'])):
        df = data['trial'][s]

        if 'targetOn' not in df.columns:
            continue

        vals = pd.to_numeric(df['targetOn'], errors='coerce').dropna()
        if len(vals) == 0:
            continue

        current_target_on = float(np.median(vals))
        time_shift = target_time - current_target_on
        bin_shift  = round(time_shift / bin_size)

        if time_shift == 0:
            continue

        # shift event columns in trial DataFrame
        df = df.copy()
        for col in event_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') + time_shift
        data['trial'][s] = df

        # shift spikecounts along time axis (axis=-1), zero-pad rolled edge
        if bin_shift != 0:
            sc = data['spikecounts'][s]
            sc_shifted = np.roll(sc, bin_shift, axis=-1)
            if bin_shift > 0:
                sc_shifted[..., :bin_shift] = 0
            else:
                sc_shifted[..., bin_shift:] = 0
            data['spikecounts'][s] = sc_shifted

    return data


# ---------------------------------------------------------------------------
# 6. Merge behavioral columns from behavior_all.csv
# ---------------------------------------------------------------------------

def merge_behavior(data: dict,
                   behavior_df: pd.DataFrame,
                   cols_to_add: list[str] = BEHAVIORAL_COLS,
                   renames: dict[str, str] = COLUMN_RENAMES) -> dict:
    """
    Merge selected columns from behavior_df into each session's trial DataFrame,
    then apply COLUMN_RENAMES.

    Match key: (session_id, trial_id)  →  normalised to plain strings.
    Columns already present in the trial DataFrame are overwritten.

    Rename logic (applied after merge):
        targX    → targetX
        targY    → targetY
        respX_dva → responseX
        respY_dva → responseY
    Any pre-existing column whose name matches a rename destination
    (e.g. an old 'targetX' computed from targetPos) is dropped first
    to avoid duplicates.
    """
    available = [c for c in cols_to_add if c in behavior_df.columns]
    missing   = set(cols_to_add) - set(available)
    if missing:
        print(f"[merge_behavior] WARNING: columns not found in behavior_df "
              f"(skipped): {sorted(missing)}")

    beh_slim = behavior_df[['session_id', 'trial_id'] + available].copy()

    # columns that will be renamed → their destination names
    rename_destinations = set(renames.values())

    for s in range(len(data['trial'])):
        df = data['trial'][s].copy()

        # normalise join keys on neural side
        df['_sid'] = df['session'].apply(_norm_session_key)
        df['_tid'] = df['trial'].apply(_norm_trial_key)

        # drop columns that will be replaced by the merge
        df = df.drop(columns=[c for c in available if c in df.columns])

        # drop any pre-existing columns whose names clash with rename destinations
        # (e.g. old 'targetX' computed from targetPos inside the pkl)
        df = df.drop(columns=[c for c in rename_destinations if c in df.columns])

        merged = df.merge(
            beh_slim.rename(columns={'session_id': '_sid', 'trial_id': '_tid'}),
            on=['_sid', '_tid'],
            how='left',
            validate='m:1',
        ).drop(columns=['_sid', '_tid']).reset_index(drop=True)

        # apply renames
        merged = merged.rename(columns={k: v for k, v in renames.items()
                                         if k in merged.columns})

        data['trial'][s] = merged

    return data


def load_behavior(filepath: str) -> pd.DataFrame:
    """Load behavior_all.csv and normalise its session_id / trial_id keys."""
    df = pd.read_csv(filepath, low_memory=False)

    required = {'session_id', 'trial_id'}
    if not required.issubset(df.columns):
        raise ValueError(f"behavior_all.csv must contain columns: {required}")

    df['session_id'] = df['session_id'].apply(_norm_session_key)
    df['trial_id']   = df['trial_id'].apply(_norm_trial_key)
    return df


# ---------------------------------------------------------------------------
# 7. Trial-level filters
# ---------------------------------------------------------------------------

def remove_nan_trials(data: dict,
                      cols: list[str] | None = None) -> dict:
    """
    Drop trials that have NaN or Inf in any of `cols`.

    Default: ['targetX', 'targetY', 'respX_dva', 'respY_dva']
    """
    if cols is None:
        cols = ['targetX', 'targetY', 'responseX', 'responseY']

    for s in range(len(data['trial'])):
        df = data['trial'][s]
        sc = data['spikecounts'][s]

        present = [c for c in cols if c in df.columns]
        if not present:
            continue

        keep = np.isfinite(df[present].to_numpy(dtype=float)).all(axis=1)
        if (~keep).any():
            data['trial'][s]       = df[keep].reset_index(drop=True)
            data['spikecounts'][s] = sc[keep]

    return data


def filter_by_iti(data: dict, max_iti: float = 5.0) -> dict:
    """
    Keep only trials with ITI ≤ max_iti seconds.
    Requires 'ITI' column (present after merge_behavior).
    The last trial of each session (ITI = NaN) is always removed.
    """
    for s in range(len(data['trial'])):
        df = data['trial'][s]
        sc = data['spikecounts'][s]

        if 'ITI' not in df.columns:
            print(f"[filter_by_iti] Session {s}: 'ITI' column missing — skipping.")
            continue

        iti  = pd.to_numeric(df['ITI'], errors='coerce')
        keep = iti.notna() & (iti <= max_iti)

        data['trial'][s]       = df[keep].reset_index(drop=True)
        data['spikecounts'][s] = sc[keep.values]

    return data


# ---------------------------------------------------------------------------
# 8. Session-level filters
# ---------------------------------------------------------------------------

def filter_by_sessions(data: dict, session_list: list[str]) -> dict:
    """Keep only sessions whose normalised session_id is in session_list."""
    session_set = {str(x) for x in session_list}
    keep = [s for s in range(len(data['trial']))
            if _get_session_id(data, s) in session_set]
    return _subset_sessions(data, keep)


def filter_by_subject(data: dict, subject: str) -> dict:
    """
    Keep only sessions for 'Paula' or 'Rex'.
    Requires PAULA_SESSIONS / REX_SESSIONS to be populated above.
    """
    subject = subject.strip().capitalize()
    mapping = {'Paula': PAULA_SESSIONS, 'Rex': REX_SESSIONS}
    if subject not in mapping:
        raise ValueError(f"Unknown subject '{subject}'. Choose 'Paula' or 'Rex'.")
    session_list = mapping[subject]
    if not session_list:
        raise ValueError(
            f"Session list for {subject} is empty. "
            "Populate PAULA_SESSIONS / REX_SESSIONS at the top of this file."
        )
    return filter_by_sessions(data, session_list)


def filter_by_nMapStim(data: dict, n: int | list[int]) -> dict:
    """
    Keep only sessions with nMapStim in `n`.
    e.g.  filter_by_nMapStim(data, 6)
          filter_by_nMapStim(data, [6, 8])
    """
    allowed = {n} if isinstance(n, int) else set(n)
    valid   = {6: NMAP6_SESSIONS, 8: NMAP8_SESSIONS}
    session_list: list[str] = []
    for v in allowed:
        if v not in valid:
            raise ValueError(f"nMapStim={v} not recognised. Choose from {list(valid)}.")
        session_list.extend(valid[v])
    return filter_by_sessions(data, session_list)


# ---------------------------------------------------------------------------
# 9. Drop empty sessions
# ---------------------------------------------------------------------------

def drop_empty_sessions(data: dict,
                        min_trials:  int = 1,
                        min_neurons: int = MIN_NEURONS_PER_AREA) -> dict:
    """
    Remove sessions with fewer than min_trials trials OR no area
    with at least min_neurons neurons.
    """
    keep = []
    for s in range(len(data['trial'])):
        if len(data['trial'][s]) < min_trials:
            continue
        unit_df = data['unit'][s]
        if max((unit_df['area'] == a).sum() for a in AREAS) < min_neurons:
            continue
        keep.append(s)

    dropped = len(data['trial']) - len(keep)
    if dropped:
        print(f"[drop_empty_sessions] Dropped {dropped} session(s).")
    return _subset_sessions(data, keep)


# ---------------------------------------------------------------------------
# 10. Save / load processed data
# ---------------------------------------------------------------------------

def save_data(data: dict, filepath: str) -> None:
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved → {filepath}  ({len(data['trial'])} sessions)")


# ---------------------------------------------------------------------------
# 11. Top-level pipeline
# ---------------------------------------------------------------------------

def load_pipeline(
    pkl1:               str,
    pkl2:               str,
    behavior_csv:       str,
    subject:            Optional[str]             = None,
    nMapStim:           Optional[int | list[int]] = None,
    max_iti:            Optional[float]            = None,
    remove_nans:        bool                       = True,
    original_bin:       float                      = 0.025,
    # ── rebinning (non-overlapping, kept for backward compatibility) ──────────
    target_bin:         float                      = 0.1,
    # ── sliding window (default; overrides non-overlapping rebin) ────────────
    use_sliding_window: bool                       = True,
    window_s:           float                      = 0.2,
    step_s:             float                      = 0.1,
    # ─────────────────────────────────────────────────────────────────────────
    align:              bool                       = True,
    min_neuron_trials:  int                        = MIN_TRIALS_PER_NEURON,
    output_path:        Optional[str]              = None,
) -> dict:
    """
    Full loading + filtering pipeline.

    Parameters
    ----------
    pkl1, pkl2          : paths to the two raw neural PKL files
    behavior_csv        : path to behavior_all.csv
    subject             : 'Paula', 'Rex', or None (both)
    nMapStim            : 6, 8, [6, 8], or None (all)
    max_iti             : max ITI in seconds, or None (no ITI filter)
    remove_nans         : drop trials with NaN in key behavioral columns
    original_bin        : bin size in the raw PKL files (seconds), default 0.025
    target_bin          : bin size for non-overlapping rebin (ignored when
                          use_sliding_window=True)
    use_sliding_window  : if True (default), apply a sliding-window sum instead
                          of non-overlapping rebinning; alignment is performed
                          BEFORE the sliding window at original_bin precision
    window_s            : sliding-window integration width (seconds), default 0.1
    step_s              : sliding-window step size (seconds), default 0.025
                          → n_time = (N_bins − W) // S + 1  ≈ 237 for 240-bin epoch
    align               : align spikecounts + event times to targetOn = 1.7 s
    min_neuron_trials   : minimum recorded trials for a neuron to be kept
    output_path         : if provided, save the filtered data dict to this path

    Returns
    -------
    data : flat dict ready for decoder.py
           keys: 'trial', 'unit', 'session', 'spikecounts', 'trialsNeuron'

    Notes
    -----
    Pipeline order (sliding window mode, recommended):
        load + concat
        → extract delsac
        → filter neurons
        → align to targetOn (at original_bin = 0.025 s precision)
        → sliding_window_spikecounts  (→ 237 time bins)
        → merge behavioral columns
        → optional session / trial filters
        → remove NaN trials
        → drop empty sessions

    Pipeline order (non-overlapping rebin mode, legacy):
        ... → filter neurons → rebin → align → merge ...
    """
    import time as _time
    pipeline_start = _time.time()

    def _step(msg):
        print(f"\n[{_time.time() - pipeline_start:6.1f}s]  {msg}")

    # 1. Load + concat
    _step("Loading PKL files ...")
    raw = load_and_concat_raw_pkls(pkl1, pkl2)
    print(f"         → {len(raw['trial'])} total sessions loaded")

    # 2. Extract delsac sessions
    _step("Extracting delsac sessions ...")
    data = extract_delsac_sessions(raw)
    print(f"         → {len(data['trial'])} delsac sessions")

    # 3. Filter neurons
    _step("Filtering neurons ...")
    n_neurons_before = sum(len(u) for u in data['unit'])
    data = filter_neurons(data, areas=AREAS, min_trials=min_neuron_trials)
    n_neurons_after  = sum(len(u) for u in data['unit'])
    print(f"         → {n_neurons_after} / {n_neurons_before} neurons kept")

    if use_sliding_window:
        # 4a. Align at fine resolution, then slide
        if align:
            _step(f"Aligning to targetOn = {TARGET_ON_ALIGNED} s "
                  f"(at {original_bin} s precision) ...")
            data = align_to_target_on(data,
                                      target_time=TARGET_ON_ALIGNED,
                                      bin_size=original_bin)

        W          = round(window_s / original_bin)
        S          = round(step_s   / original_bin)
        n_bins_raw = data['spikecounts'][0].shape[-1]
        n_wins     = (n_bins_raw - W) // S + 1
        _step(f"Sliding window: window={window_s} s, step={step_s} s "
              f"→ {n_wins} bins, {int((W-S)/W*100)}% overlap ...")
        data = sliding_window_spikecounts(data,
                                          original_bin=original_bin,
                                          window=window_s,
                                          step=step_s)
        print(f"         → spikecounts shape: "
              f"{np.asarray(data['spikecounts'][0]).shape}  "
              f"(trials × neurons × timebins)")
    else:
        _step(f"Rebinning {original_bin} s → {target_bin} s ...")
        data = rebin_spikecounts(data,
                                 original_bin=original_bin,
                                 target_bin=target_bin)
        if align:
            _step(f"Aligning to targetOn = {TARGET_ON_ALIGNED} s ...")
            data = align_to_target_on(data,
                                      target_time=TARGET_ON_ALIGNED,
                                      bin_size=target_bin)
        print(f"         → spikecounts shape: "
              f"{np.asarray(data['spikecounts'][0]).shape}  "
              f"(trials × neurons × timebins)")

    # 5. Merge behavioral columns
    _step("Merging behavioral data ...")
    beh  = load_behavior(behavior_csv)
    data = merge_behavior(data, beh)

    # 6. Optional session-level filters
    if subject is not None:
        _step(f"Filtering by subject: {subject} ...")
        data = filter_by_subject(data, subject)
        print(f"         → {len(data['trial'])} sessions remaining")

    if nMapStim is not None:
        _step(f"Filtering by nMapStim = {nMapStim} ...")
        data = filter_by_nMapStim(data, nMapStim)
        print(f"         → {len(data['trial'])} sessions remaining")

    if max_iti is not None:
        _step(f"Filtering by ITI ≤ {max_iti} s ...")
        data = filter_by_iti(data, max_iti=max_iti)

    # 7. NaN removal
    if remove_nans:
        _step("Removing NaN trials ...")
        data = remove_nan_trials(data)

    # 8. Drop empty sessions
    data = drop_empty_sessions(data)

    total = _time.time() - pipeline_start
    print(f"\n{'─'*55}")
    print(f"Pipeline done in {total:.1f}s  —  "
          f"{len(data['trial'])} sessions ready for decoding.")
    print(f"{'─'*55}")
    _print_summary(data)

    if output_path is not None:
        _step(f"Saving to {output_path} ...")
        save_data(data, output_path)

    return data


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _norm_session_key(x) -> str:
    """
    Normalise a session identifier to a plain string.

    Examples
        [100731]      → '100731'
        '100731.0'    → '100731'
        '110107_01'   → '110107_01'   (underscore preserved)
        array([...])  → first element, then above rules
    """
    # unwrap list/array wrappers
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(x, dtype=object).ravel()
        x   = arr[0] if len(arr) >= 1 else x

    s = str(x).strip().strip('[]').strip("'\"")

    # remove file extension
    if '.' in s and '_' not in s:
        # only strip extension if there is no underscore (e.g. '110107_01' stays)
        s = s.split('.')[0]

    # try int-conversion to remove trailing .0  (only when no underscore)
    if '_' not in s:
        try:
            s = str(int(float(s)))
        except (ValueError, OverflowError):
            pass

    return s


def _norm_trial_key(x) -> str:
    """Normalise a trial id to a plain integer string."""
    try:
        return str(int(float(x)))
    except (ValueError, TypeError):
        return str(x).strip()


def _get_session_id(data: dict, s: int) -> str:
    """Extract normalised session ID for session index s."""
    df = data['trial'][s]
    if 'session' in df.columns and len(df) > 0:
        return _norm_session_key(df['session'].iloc[0])
    sess_meta = data.get('session', [None] * (s + 1))[s]
    if isinstance(sess_meta, dict):
        name = sess_meta.get('name', sess_meta.get('session', ''))
        return _norm_session_key(name)
    return ''


def _subset_sessions(data: dict, keep_idx: list[int]) -> dict:
    """Return a new data dict with only the sessions at keep_idx."""
    n = len(data['trial'])
    new = {}
    for key, val in data.items():
        if isinstance(val, list) and len(val) == n:
            new[key] = [val[i] for i in keep_idx]
        else:
            new[key] = val
    return new


def _print_summary(data: dict) -> None:
    n_sessions = len(data['trial'])

    # --- classify each session ---
    paula_set  = set(PAULA_SESSIONS)
    rex_set    = set(REX_SESSIONS)
    nmap6_set  = set(NMAP6_SESSIONS)
    nmap8_set  = set(NMAP8_SESSIONS)

    paula_sess  = [s for s in range(n_sessions) if _get_session_id(data, s) in paula_set]
    rex_sess    = [s for s in range(n_sessions) if _get_session_id(data, s) in rex_set]
    nmap6_sess  = [s for s in range(n_sessions) if _get_session_id(data, s) in nmap6_set]
    nmap8_sess  = [s for s in range(n_sessions) if _get_session_id(data, s) in nmap8_set]
    unknown_sess = [s for s in range(n_sessions)
                    if _get_session_id(data, s) not in paula_set | rex_set]

    def _trials(idx): return sum(len(data['trial'][s]) for s in idx)

    # --- per subject ---
    print("\n── Sessions & trials per subject ──────────────────")
    print(f"  {'Subject':<10} {'Sessions':>8} {'Trials':>8}")
    print(f"  {'Paula':<10} {len(paula_sess):>8} {_trials(paula_sess):>8}")
    print(f"  {'Rex':<10} {len(rex_sess):>8} {_trials(rex_sess):>8}")
    if unknown_sess:
        print(f"  {'Unknown':<10} {len(unknown_sess):>8} {_trials(unknown_sess):>8}")
    print(f"  {'TOTAL':<10} {n_sessions:>8} {_trials(range(n_sessions)):>8}")

    # --- per nMapStim condition ---
    print("\n── Sessions & trials per nMapStim condition ────────")
    print(f"  {'nMapStim':<10} {'Sessions':>8} {'Trials':>8}")
    print(f"  {'6':<10} {len(nmap6_sess):>8} {_trials(nmap6_sess):>8}")
    print(f"  {'8':<10} {len(nmap8_sess):>8} {_trials(nmap8_sess):>8}")
    other_sess = [s for s in range(n_sessions)
                  if _get_session_id(data, s) not in nmap6_set | nmap8_set]
    if other_sess:
        print(f"  {'other':<10} {len(other_sess):>8} {_trials(other_sess):>8}")
    print(f"  {'TOTAL':<10} {n_sessions:>8} {_trials(range(n_sessions)):>8}")

    # --- neurons per area ---
    print("\n── Neurons per area (across all sessions) ──────────")
    counts = {a: 0 for a in AREAS}
    for unit_df in data['unit']:
        for a in AREAS:
            counts[a] += int((unit_df['area'] == a).sum())
    for a, n in counts.items():
        print(f"  {a:<12} {n:>5}")
    print(f"\n  Total trials: {_trials(range(n_sessions))}")
