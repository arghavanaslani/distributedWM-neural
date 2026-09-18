"""
config.py
=========
Single source of truth for values shared across the pipeline.

Rule: a constant lives here if two or more scripts need it. Constants used by
exactly one script stay in that script.

Paths are resolved automatically: the cluster layout if it exists, otherwise a
`data/` and `results/` folder beside this file, so the pipeline can run against a
small local sample. Any path can be overridden with an environment variable.

Run-state toggles (which areas, force a recompute) also read the environment, so
changing them never means editing a tracked file. `run.py` sets these from
command-line flags.
"""

import os
from pathlib import Path


# ── Environment helpers ────────────────────────────────────────────────────────

def env_bool(name, default):
    """Read a boolean from the environment. '1', 'true', 'yes', 'on' are True."""
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ('1', 'true', 'yes', 'on')


def env_list(name, default):
    """Read a comma-separated list from the environment."""
    v = os.environ.get(name)
    if not v:
        return default
    return [s.strip() for s in v.split(',') if s.strip()]


def env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v not in (None, '') else default


# ── Paths ────────────────────────────────────────────────────────────────────

_REPO_ROOT    = Path(__file__).resolve().parent
_CLUSTER_ROOT = Path('/home/aarghavan/aslan')

if _CLUSTER_ROOT.is_dir():
    _DATA_DIR    = _CLUSTER_ROOT / 'data'
    _RESULTS_DIR = _CLUSTER_ROOT / 'distributedWM-neural' / 'results'
else:
    _DATA_DIR    = _REPO_ROOT / 'data'
    _RESULTS_DIR = _REPO_ROOT / 'results'

DATA_DIR     = Path(os.environ.get('DISTWM_DATA',    _DATA_DIR))
RESULTS_DIR  = str(Path(os.environ.get('DISTWM_RESULTS', _RESULTS_DIR))) + os.sep

DATA_PATH    = str(DATA_DIR / os.environ.get('DISTWM_DATA_FILE', 'test.pkl'))
BEHAVIOR_CSV = str(DATA_DIR / 'behavior_all.csv')

PKL_DIR      = os.path.join(RESULTS_DIR, 'pkl')
FIG_DIR      = os.path.join(RESULTS_DIR, 'figures')

# ── Areas ─────────────────────────────────────────────────────────────────────
# Canonical order: frontal → sensory. Used for every figure axis except the
# inter-area flow analyses, which read sensory → frontal (information-flow
# direction) and use AREAS_SENSORY_FIRST.
_ALL_AREAS = ['PFC', 'FEF', 'LIP', 'Parietal', 'IT', 'MT', 'V4']
AREAS = env_list('DISTWM_AREAS', _ALL_AREAS)
AREAS_SENSORY_FIRST = [a for a in reversed(_ALL_AREAS) if a in AREAS]

AREA_COLORS = {
    'PFC':      '#1f77b4',
    'FEF':      '#d62728',
    'LIP':      '#2ca02c',
    'Parietal': '#ff7f0e',
    'IT':       '#17becf',
    'MT':       '#9467bd',
    'V4':       '#8c564b',
}

# ── Time axis ─────────────────────────────────────────────────────────────────
# Set at decoding time; everything downstream that rebuilds a time mask over the
# spikecounts must use the same values.
T_START  = -2.5     # epoch start, seconds
ORIG_BIN = 0.025    # raw bin width before sliding window, seconds
WINDOW_S = 0.1      # sliding-window integration width, seconds
STEP_S   = 0.025    # step between sliding-window bins, seconds

# ── Task events, seconds from epoch start ─────────────────────────────────────
EV_TARGET_ON  = 1.70
EV_TARGET_OFF = 1.80
EV_RESPONSE   = 2.55

DELAY_START   = 1.80
DELAY_END     = 2.55
