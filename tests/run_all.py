"""Run every suite in the project.

    python tests/run_all.py
    python tests/run_all.py --quick     skip the PLY-backed suite

There are three, because they need three different things: the Python
package needs numpy and plyfile, the popping metric needs numpy alone, and
the browser modules need node. Splitting them is what lets each be run
where it makes sense - the metric suite on a machine with no PLY stack, the
browser suite from a terminal with no Python at all. This file exists so
that "did I break anything" is still one command.

A missing dependency is reported as skipped rather than failed. A suite
that cannot run is not a suite that failed, and conflating the two teaches
people to ignore the output.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SUITES = [
    ('python package', [sys.executable, 'tests/test_all.py'],
     'numpy and plyfile'),
    ('popping metric', [sys.executable, 'tests/test_popping.py'],
     'numpy'),
    ('patch search', [sys.executable, 'tests/test_autopick.py'],
     'numpy and plyfile'),
    ('browser modules', ['node', 'tests/test_web.mjs'],
     'node'),
]


def runnable(cmd):
    """Whether the interpreter a suite needs is on the path."""
    return cmd[0] == sys.executable or shutil.which(cmd[0]) is not None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quick', action='store_true',
                    help='skip the suite that needs the PLY stack')
    args = ap.parse_args(argv)

    results = []
    for name, cmd, needs in SUITES:
        if args.quick and name == 'python package':
            results.append((name, 'skipped', 'asked for --quick'))
            continue
        if not runnable(cmd):
            results.append((name, 'skipped', f'needs {needs}'))
            continue

        print(f'\n=== {name} ' + '=' * (60 - len(name)))
        # stderr is captured rather than inherited so a missing import can be
        # told apart from a failing assertion. Both exit 1, and reporting an
        # uninstalled dependency as a test failure would send someone
        # looking for a bug that is not there.
        p = subprocess.run(cmd, cwd=ROOT, stderr=subprocess.PIPE, text=True)
        err = p.stderr or ''
        missing = None
        for marker in ('ModuleNotFoundError: No module named ',
                       "Cannot find module "):
            if marker in err:
                missing = err.split(marker, 1)[1].split('\n')[0].strip().strip("'\"")
                break

        if p.returncode == 0:
            results.append((name, 'passed', ''))
        elif missing:
            results.append((name, 'skipped', f'no {missing} installed'))
        else:
            if err.strip():
                print(err.rstrip())
            results.append((name, 'FAILED', 'see output above'))

    print('\n' + '=' * 68)
    for name, state, note in results:
        print(f'  {name:<18} {state:<8} {note}')

    failed = [r for r in results if r[1] == 'FAILED']
    skipped = [r for r in results if r[1] == 'skipped']
    if failed:
        print(f'\n{len(failed)} suite(s) failed')
        return 1
    if skipped:
        print(f'\nall runnable suites passed, {len(skipped)} skipped')
        return 0
    print('\neverything passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())