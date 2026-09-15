# distributedWM — neural pipeline

Decoding and analysis of delayed-saccade neural data. Consumes `behavior_all.csv`
produced by [distributedWM-behavior](https://github.com/arghavanaslani/distributedWM-behavior).

## Layout

```
config.py            shared constants: paths, AREAS, time axis, event times
pipeline/            the established analysis path, in run order
  load_and_filter.py   raw pkls + behavior_all.csv -> test.pkl
  decoder.py           test.pkl -> predictions, shuffles, cross-temporal (HEAVY)
  analysis.py          predictions -> figures 01, 02, 04, 05, 06, 12, 12b
controls/            robustness checks on pipeline results
  neuron_dropping.py            matched-N decoding (removes neuron-count confound)
  neuron_dropping_stats.py      pairwise stats on the matched-N ranking
  neuron_dropping_trialcheck.py training-set-size diagnostic
  neuron_count_diagnostic.py    per-area neuron counts, to choose matched-N
  coupling_stats.py             tier 1-2 coupling caveats
  coupling_eyecontrol.py        tier 3: partial out fixation gaze
exploratory/         unstable, expected to churn
  extra_analysis.py    inter-area information flow (xcorr, CCA)
filter_data.ipynb    applies pipeline/load_and_filter.py to the data
```

## Running

Run from the repository root, as modules:

```bash
python -m pipeline.decoder
python -m pipeline.analysis
python -m controls.neuron_dropping
```

`python pipeline/decoder.py` will **not** work — the repo root must be on
`sys.path` for `from config import ...` to resolve, which `-m` handles and a
direct path invocation does not.

Launch Jupyter from the repository root too, so `filter_data.ipynb` can import
`pipeline.load_and_filter`.

## Data flow

```
behavior_all.csv ─┐
raw pkls ─────────┴─> load_and_filter -> test.pkl -> decoder -> predicted_data.pkl
                                                                shuffled_data.pkl
                                                                trial_data.pkl
                                                                centered.pkl
                                                                        |
                          analysis.py <---------------------------------+
                                |                                       |
                    errors_angular.csv                        neuron_dropping.py
                    neurobeh_results.pkl                      neuron_dropping.pkl
                                |                                       |
                                +-> coupling_stats, coupling_eyecontrol <+
```

Outputs land in `results/` (gitignored): `results/pkl/` and `results/figures/`.
