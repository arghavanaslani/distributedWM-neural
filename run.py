#!/usr/bin/env python3
"""
run.py
======
Single entry point for the pipeline.

Run-state (which areas, force a recompute, replot only) is passed as flags and
handed to the scripts through environment variables that config.py reads. Nothing
in the repository is edited to change how a run behaves, so a debug run never
dirties the working tree and never collides on merge.

Examples
--------
    python run.py decode                          # full decoding run
    python run.py decode --areas PFC --max-sessions 2   # quick smoke test
    python run.py decode --force                  # ignore cached predictions
    python run.py analysis                        # recompute + plot
    python run.py analysis --replot               # replot from cache only
    python run.py matched-n
    python run.py --list

Because this file sits at the repository root, running it puts the root on
sys.path — so `python run.py decode` works where `python pipeline/decoder.py`
does not.
"""

import argparse
import os
import runpy
import sys

COMMANDS = {
    'decode':          'pipeline.decoder',
    'analysis':        'pipeline.analysis',
    'matched-n':       'controls.neuron_dropping',
    'matched-n-stats': 'controls.neuron_dropping_stats',
    'trialcheck':      'controls.neuron_dropping_trialcheck',
    'neuron-counts':   'controls.neuron_count_diagnostic',
    'coupling-stats':  'controls.coupling_stats',
    'eye-control':     'controls.coupling_eyecontrol',
    'flow':            'exploratory.extra_analysis',
}


def build_parser():
    p = argparse.ArgumentParser(
        prog='run.py',
        description='Entry point for the distributed WM neural pipeline.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='commands:\n' + '\n'.join(f'  {k:17s} {v}' for k, v in COMMANDS.items()))
    p.add_argument('command', nargs='?', choices=sorted(COMMANDS), metavar='COMMAND')
    p.add_argument('--list', action='store_true', help='list commands and exit')
    p.add_argument('--show-config', action='store_true',
                   help='print the resolved paths and areas, then exit')

    g = p.add_argument_group('scope (any command)')
    g.add_argument('--areas', nargs='+', metavar='AREA',
                   help='restrict to these areas (default: all seven)')
    g.add_argument('--data', metavar='DIR', help='override the data directory')
    g.add_argument('--results', metavar='DIR', help='override the results directory')

    g = p.add_argument_group('decode')
    g.add_argument('--force', action='store_true',
                   help='ignore cached predictions and rerun decoding')
    g.add_argument('--max-sessions', type=int, metavar='N',
                   help='decode only the first N sessions (quick test)')

    g = p.add_argument_group('analysis')
    g.add_argument('--replot', action='store_true',
                   help='replot from cached results without recomputing')
    return p


def main():
    args = build_parser().parse_args()

    if args.list or (not args.command and not args.show_config):
        for k, v in COMMANDS.items():
            print(f"  {k:17s} -> {v}")
        return 0

    env = {}
    if args.areas:        env['DISTWM_AREAS']        = ','.join(args.areas)
    if args.data:         env['DISTWM_DATA']         = args.data
    if args.results:      env['DISTWM_RESULTS']      = args.results
    if args.force:        env['DISTWM_FORCE']        = '1'
    if args.replot:       env['DISTWM_REPLOT']       = '1'
    if args.max_sessions: env['DISTWM_MAX_SESSIONS'] = str(args.max_sessions)
    os.environ.update(env)

    if args.show_config:
        import config
        print("── resolved configuration")
        for k in ('DATA_PATH', 'BEHAVIOR_CSV', 'RESULTS_DIR', 'PKL_DIR', 'FIG_DIR'):
            v = getattr(config, k)
            mark = 'ok     ' if os.path.exists(str(v)) else 'MISSING'
            print(f"   {mark} {k:13s} {v}")
        print(f"           {'AREAS':13s} {config.AREAS}")
        return 0

    module = COMMANDS[args.command]
    print(f"── run.py: {args.command} -> {module}")
    for k, v in sorted(env.items()):
        print(f"   {k} = {v}")
    if not env:
        print("   (defaults)")
    print()

    runpy.run_module(module, run_name='__main__')
    return 0


if __name__ == '__main__':
    sys.exit(main())
