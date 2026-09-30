# Motor Series Benchmark Notes

Updated: 2026-09-30

Operational instructions for AI agents are in `AGENTS.md`. This document keeps
the provenance, structural findings, and completed verification in more detail.

## Objective and files

The goal was to provide a compact, readable, structurally factorized pytest
benchmark without modifying the supplied files.

- Original benchmark (unchanged):
  `/Users/sturm/Documents/Dynamic/src/python/Logic1/benchmarks/logic1/gsimplify-motor/test_gsimplify_motor_benchmark.py`
- Style reference (unchanged):
  `/Users/sturm/Documents/Dynamic/src/python/Logic1/logic1/logic1/theories/RCF/test_gsimplify.py`
- Shared input module:
  `/Users/sturm/Documents/Dynamic/src/python/Logic1/logic1/benchmarks/theories/RCF/motor_series/motor_inputs.py`
- `gsimplify` benchmark:
  `/Users/sturm/Documents/Dynamic/src/python/Logic1/logic1/benchmarks/theories/RCF/motor_series/test_gsimplify_motor.py`
- `simplify` benchmark:
  `/Users/sturm/Documents/Dynamic/src/python/Logic1/logic1/benchmarks/theories/RCF/motor_series/test_simplify_motor.py`

The input module exposes `series1` through `series13`, corresponding to the
original `testseries1` through `testseries13`. The original `testseries14` is
exactly AST-identical to `testseries1` and is intentionally omitted.

## Required invariants

- Each constructed formula must retain the exact original formula tree,
  argument order, duplicates, and `repr`; logical equivalence alone is not
  sufficient for this benchmark.
- Factoring may name exact repeated subtrees or contiguous argument sequences,
  but must not distribute, reorder, deduplicate, or Boolean-simplify them.
- `p1_split`, `q_td_split`, and `td_split` reconstruct recurring branch trees
  exactly. Starred tuples preserve flat `And` argument sequences and order.
- Formula construction remains outside the timed call. Both benchmark modules
  use `benchmark.pedantic` with one round and one iteration.
- Keep every polynomial expression on one physical source line.
- Use hanging logical-call formatting: do not break immediately after `And(` or
  `Or(`, and do not place a formula's closing parenthesis alone on a line.

## Factoring structure

Shared definitions cover the principal polynomial boundaries, motor bands,
temperature cases, q/td and p1 splits, operating regions, and repeated z and
n/td/z case blocks. Per-series builders add local `boundary_*`, `region_*`,
`case_*`, and `cases_*` definitions only where useful.

The shared helpers and tuples are construction devices only: evaluation yields
the same Logic1 formulas as the original flat source.

## Intentional duplicate disjuncts

An exhaustive AST scan of every sibling argument in every original `And` and
`Or` found exact duplicates only in the outer `Or` of these series (positions
are one-based):

- `testseries7`: operands 4/5 and 9/10
- `testseries8`: operands 1/3 and 6/10
- `testseries9`: operands 4/5 and 9/10

The factorized file retains those duplicates and documents them beside the
corresponding local `case_1` and `case_2` definitions. There are no other exact
duplicate sibling formulas in `testseries1`--`testseries13`, including nested
logical calls.

## Benchmark cases

`motor_inputs.MOTOR_SERIES` records the input atom counts;
`test_gsimplify_motor.EXPECTED_OUTPUT_ATOMS` records the `gsimplify` output
counts, which are identical for `radical=False` and `radical=True`:

| Series | Input | Output |
|---:|---:|---:|
| 1 | 710 | 158 |
| 2 | 1420 | 396 |
| 3 | 94 | 35 |
| 4 | 292 | 135 |
| 5 | 157 | 87 |
| 6 | 994 | 351 |
| 7 | 248 | 97 |
| 8 | 473 | 129 |
| 9 | 235 | 87 |
| 10 | 478 | 280 |
| 11 | 168 | 78 |
| 12 | 2176 | 482 |
| 13 | 358 | 177 |

`pytest --benchmark-only` selects tests using the benchmark fixture; it does not
disable their assertions. All 26 tests collected from `test_gsimplify_motor.py`
are benchmarks: 13 each for `radical=False` and `radical=True`. Both use
`use_redlog_cnf=True`, and each radical setting has its own benchmark table.

`test_simplify_motor.py` has 78 cases: every motor series is tested with all
six combinations of `substitute` in `{0, 1, 2}` and `implicit_ranges` in
`{False, True}`. The configurations and their six benchmark tables are ordered
first by `implicit_ranges` (`False`, then `True`) and then by `substitute`
(`0`, `1`, `2`). Configuration IDs use the same order, for example
`implicit-ranges-false-substitute-0`.

The measured `simplify` output atom counts are:

| Series | subst=0, ranges=F | subst=1, ranges=F | subst=2, ranges=F | subst=0, ranges=T | subst=1, ranges=T | subst=2, ranges=T |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 674 | 641 | 491 | 674 | 641 | 491 |
| 2 | 966 | 815 | 815 | 922 | 801 | 801 |
| 3 | 88 | 76 | 72 | 74 | 76 | 72 |
| 4 | 259 | 283 | 190 | 250 | 274 | 184 |
| 5 | 139 | 141 | 141 | 130 | 132 | 132 |
| 6 | 908 | 715 | 412 | 667 | 601 | 412 |
| 7 | 199 | 186 | 135 | 199 | 186 | 135 |
| 8 | 410 | 328 | 242 | 361 | 312 | 242 |
| 9 | 188 | 175 | 127 | 188 | 175 | 127 |
| 10 | 461 | 426 | 413 | 461 | 426 | 413 |
| 11 | 156 | 151 | 146 | 140 | 145 | 140 |
| 12 | 2100 | 1977 | 1918 | 2054 | 1977 | 1914 |
| 13 | 342 | 303 | 294 | 342 | 303 | 294 |

## Deferred Redlog equivalence check

A possible future correctness check for every `test_simplify_motor.py` case is

```python
assert redlog.qe(Equivalent(formula, result).all()) is T
```

This would prove that each simplified result is equivalent to its input, but
it is slow and is not itself a benchmark. If implemented, keep the check in
the existing parameterized benchmark function so it consumes the exact
`result` already produced for all 78 option combinations; do not duplicate the
matrix or recompute `simplify` in a separate test module. Run the Redlog call
after `benchmark.pedantic` so it remains outside the measured interval.

The proposed opt-in mechanism is a local
`benchmarks/theories/RCF/motor_series/conftest.py` that registers a
`--check-redlog-equivalence` flag. The benchmark function would inspect that
flag through a typed `pytest.FixtureRequest` and perform the assertion only
when explicitly requested. A correctness-only invocation would be:

```bash
pytest benchmarks/theories/RCF/motor_series/test_simplify_motor.py --benchmark-disable --check-redlog-equivalence
```

Normal pytest and benchmark runs must not perform the Redlog checks. Selection
with `-k`, for example `-k series3`, should remain available. This design has
been discussed but is intentionally not implemented yet.

## Verification completed

- All 13 constructed formulas compare equal to their originals.
- All 13 formula `repr` values exactly match their originals.
- All input atom counts match the table above.
- `series14` is absent from the result.
- Python compilation succeeds.
- Pytest collection finds exactly 26 `gsimplify` cases and 78 `simplify`
  cases.
- All 13 `radical=True` `gsimplify` results were computed and have the same
  output atom counts as `radical=False`.
- Both `gsimplify` radical settings passed for `series3` with benchmarking
  disabled.
- All 78 `simplify` cases passed with benchmarking disabled when the full
  option matrix was introduced. After the final ordering and ID changes, all
  six `series3` cases passed and collection still found 78 cases.
- Both benchmark modules pass mypy with `--disallow-untyped-defs`.
- An AST formatting check found no multiline polynomial expressions.
- No full timed pytest-benchmark run was performed after the latest matrix
  expansions.

Useful checks from the Logic1 package directory:

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache-motor python3 -m py_compile benchmarks/theories/RCF/motor_series/motor_inputs.py benchmarks/theories/RCF/motor_series/test_gsimplify_motor.py benchmarks/theories/RCF/motor_series/test_simplify_motor.py
DOT_SAGE=/tmp/logic1-sage-full conda run -n logic1_dev mypy --disallow-untyped-defs benchmarks/theories/RCF/motor_series/test_gsimplify_motor.py benchmarks/theories/RCF/motor_series/test_simplify_motor.py
DOT_SAGE=/tmp/logic1-sage-full conda run -n logic1_dev pytest --collect-only -q benchmarks/theories/RCF/motor_series/test_gsimplify_motor.py
DOT_SAGE=/tmp/logic1-sage-full conda run -n logic1_dev pytest -q benchmarks/theories/RCF/motor_series/test_gsimplify_motor.py -k series3 --benchmark-disable
DOT_SAGE=/tmp/logic1-sage-full conda run -n logic1_dev pytest --collect-only -q benchmarks/theories/RCF/motor_series/test_simplify_motor.py
DOT_SAGE=/tmp/logic1-sage-full conda run -n logic1_dev pytest -q benchmarks/theories/RCF/motor_series/test_simplify_motor.py -k series3 --benchmark-disable
```

## Current state

The shared input module is the source of truth for the formulas; each benchmark
module owns only its function's options and output expectations. There are no
known structural discrepancies. The original and reference files remain
unchanged.

A temporary generator exists at `/tmp/generate_factorized_motor_benchmark.py`,
but it predates the file split, latest docstrings, and explanatory comments. Do
not regenerate the input module from it without first preserving those edits.

After any structural change, re-run the exact `repr` comparison and pytest
collection described above.
