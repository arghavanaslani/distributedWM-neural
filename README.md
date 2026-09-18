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

Everything goes through `run.py` at the repository root:

```bash
python run.py decode                                # full decoding run
python run.py decode --areas PFC --max-sessions 2   # quick smoke test
python run.py decode --force                        # ignore cached predictions
python run.py analysis                              # recompute + plot
python run.py analysis --replot                     # replot from cache only
python run.py matched-n
python run.py --list                                # all commands
python run.py --show-config                         # resolved paths, run nothing
```

Run-state is passed as flags, never by editing a file. A debug run therefore
leaves the working tree clean and cannot collide on merge.

`python -m pipeline.decoder` still works if you prefer it, but must be run from
the repository root. `python pipeline/decoder.py` does **not** work — invoking a
file by path puts `pipeline/` on `sys.path` instead of the root, so `config.py`
becomes invisible.

### Paths

Resolved automatically: the cluster layout if `/home/aarghavan/aslan` exists,
otherwise `data/` and `results/` beside the repository, so you can run against a
small local sample. Override either with `--data` / `--results`, or with the
`DISTWM_DATA` / `DISTWM_RESULTS` environment variables.

Check what a run will use before starting a long job:

```bash
python run.py --show-config
```

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
