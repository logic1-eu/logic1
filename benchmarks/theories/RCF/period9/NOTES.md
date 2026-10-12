# Period-9 QE Benchmark Notes

## Provenance

The benchmark input comes from
`logic1/theories/RCF/test_qe_parallel.txt`. It describes the period-9 case for
the recurrence

```text
x_(i+1) = |x_i| - x_(i-1)
```

and asks whether nine recurrence steps return to the initial pair. The source
constructs the implication, sorts all of its free variables with
`Term.sort_key`, universally quantifies them, and applies `qe`.

## Layout

- `period9_inputs.py` constructs the formula and records its input atom count.
- `test_qe_period9.py` contains only the pytest-benchmark matrix, execution,
  and result checks.

Formula construction therefore happens during module import and is excluded
from every timing.

## Input and expected result

- Free-variable order after sorting: `x11`, `x10`, ..., `x1`.
- Quantifier: universal over all 11 free variables.
- Input atom count: 38.
- Expected quantifier-elimination result: `T`.

## Benchmark matrix

The cases use `workers=0`, `1`, `2`, `4`, and `8`, in that order. They share
the benchmark group `qe-period9`, producing one five-row comparison table. Each
case uses one round and one iteration because a single QE call takes several
seconds.

## Verification completed

On 2026-09-29:

- Both modules compiled successfully.
- Both modules passed mypy with `--disallow-untyped-defs`.
- Pytest collected exactly five cases in the requested worker order.
- All five cases passed with benchmarking disabled.
- A complete `--benchmark-only` run passed all five cases and rendered one
  `qe-period9` table.

The single-run timings observed during that verification were approximately
5.18 s (`workers=0`), 10.16 s (`workers=1`), 7.75 s (`workers=2`), 6.86 s
(`workers=4`), and 6.50 s (`workers=8`). These values are machine- and
load-dependent and are recorded only as a smoke-test reference, not as
performance expectations.

## Useful commands

Run these commands from the Logic1 package directory:

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache-period9 python3 -m py_compile benchmarks/theories/RCF/period9/period9_inputs.py benchmarks/theories/RCF/period9/test_qe_period9.py
DOT_SAGE=/tmp/logic1-sage-full conda run -n logic1_dev mypy --disallow-untyped-defs benchmarks/theories/RCF/period9/period9_inputs.py benchmarks/theories/RCF/period9/test_qe_period9.py
DOT_SAGE=/tmp/logic1-sage-full conda run -n logic1_dev pytest --collect-only -q benchmarks/theories/RCF/period9/test_qe_period9.py
DOT_SAGE=/tmp/logic1-sage-full conda run -n logic1_dev pytest -q benchmarks/theories/RCF/period9/test_qe_period9.py --benchmark-disable
DOT_SAGE=/tmp/logic1-sage-full conda run -n logic1_dev pytest -q benchmarks/theories/RCF/period9/test_qe_period9.py --benchmark-only
```

The parallel cases use multiprocessing manager sockets. In a restricted
sandbox, the last two commands may need to run with local socket creation
enabled.
