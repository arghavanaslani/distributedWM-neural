"""
neuron_count_diagnostic.py
==========================
Per-area neuron-count distribution, to choose the matched-N for the
neuron-dropping / matched-N decoding analysis.

Counts units per area per session in the *filtered* data that actually
enters decoding (test.pkl = output of load_and_filter, already gated by
area membership + MIN_TRIALS_PER_NEURON). MIN_NEURONS gating (>=10) in
decoder.py is applied on top, so matched-N >= 10 is the natural floor.

Run this ON THE SERVER (where DATA_PATH exists):
    python neuron_count_diagnostic.py
"""

import pickle
import numpy as np
import pandas as pd

# Must match decoder.py
DATA_PATH = '/home/aarghavan/aslan/data/test.pkl'
AREAS     = ['PFC', 'FEF', 'LIP', 'Parietal', 'IT', 'MT', 'V4']
CANDIDATE_N = [5, 10, 15, 20, 25, 30, 40, 50]
SAVE_CSV  = 'neuron_counts_per_area.csv'   # None to skip

with open(DATA_PATH, 'rb') as f:
    data = pickle.load(f)

n_sessions = len(data['unit'])
print(f"Loaded {n_sessions} sessions from {DATA_PATH}\n")

# One row per (area, session) with the neuron count
rows = []
for s in range(n_sessions):
    area_arr = np.asarray(data['unit'][s]['area'].values)
    for a in AREAS:
        n = int((area_arr == a).sum())
        if n > 0:
            rows.append((a, s, n))
df = pd.DataFrame(rows, columns=['area', 'session', 'n_neurons'])

# ── Per-area distribution ────────────────────────────────────────────────────
print("Per-area neuron-count distribution:")
print(f"  {'area':9s} {'sessions':>8s} {'min':>4s} {'25%':>4s} {'median':>6s} {'75%':>4s} {'max':>4s}")
for a in AREAS:
    v = df.loc[df.area == a, 'n_neurons'].values
    if len(v) == 0:
        print(f"  {a:9s} {'--':>8s}")
        continue
    q25, q50, q75 = np.percentile(v, [25, 50, 75]).astype(int)
    print(f"  {a:9s} {len(v):8d} {v.min():4d} {q25:4d} {q50:6d} {q75:4d} {v.max():4d}")

# ── Session survival at each candidate matched-N ─────────────────────────────
# matched-N rule: largest N keeping >= ~10 sessions in EVERY area.
print("\nSessions with >= N neurons, per area (drives matched-N choice):")
header = "  " + "N".rjust(4) + "".join(f"{a:>9s}" for a in AREAS) + f"{'min_area':>10s}"
print(header)
for N in CANDIDATE_N:
    counts = {a: int((df.loc[df.area == a, 'n_neurons'] >= N).sum()) for a in AREAS}
    worst = min(counts.values())
    print("  " + f"{N:4d}" + "".join(f"{counts[a]:9d}" for a in AREAS) + f"{worst:10d}")

print("\nGuide: pick the largest N whose 'min_area' column is still >= ~10.")

if SAVE_CSV:
    df.to_csv(SAVE_CSV, index=False)
    print(f"\nSaved per-(area,session) counts -> {SAVE_CSV}")
