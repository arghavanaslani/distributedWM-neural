"""
config.py
=========
Single source of truth for values shared across the pipeline.

Rule: a constant lives here if two or more scripts need it. Constants used by
exactly one script stay in that script.

Why this exists: AREAS was previously declared in all ten scripts, and the
time-axis constants were duplicated between decoder.py and neuron_dropping.py,
whose docstring warned "must match decoder.py". Changing a window in one place
and not the other would silently shift the delay mask onto the wrong bins and
quietly corrupt the matched-N ranking, with no error raised.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH    = '/home/aarghavan/aslan/data/test.pkl'
BEHAVIOR_CSV = '/home/aarghavan/aslan/data/behavior_all.csv'

RESULTS_DIR  = '/home/aarghavan/aslan/distributedWM-neural/results/'
PKL_DIR      = os.path.join(RESULTS_DIR, 'pkl')
FIG_DIR      = os.path.join(RESULTS_DIR, 'figures')

# ── Areas ─────────────────────────────────────────────────────────────────────
# Canonical order: frontal → sensory. Used for every figure axis except the
# inter-area flow analyses, which read sensory → frontal (information-flow
# direction) and use AREAS_SENSORY_FIRST.
AREAS = ['PFC', 'FEF', 'LIP', 'Parietal', 'IT', 'MT', 'V4']
AREAS_SENSORY_FIRST = list(reversed(AREAS))

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
