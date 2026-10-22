# Motor Series Benchmark Notes

Updated: 2026-10-22

This document covers `benchmarks/theories/RCF/motor_series/` and all its
descendants, excluding sibling benchmark directories. It records provenance,
structural findings, measurements, and completed verification for that subtree.
Operational instructions are in this directory's `AGENTS.md`; `NOTES.md` is
supporting documentation, not an automatically loaded agent instruction file.

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
- DNF benchmark:
  `/Users/sturm/Documents/Dynamic/src/python/Logic1/logic1/benchmarks/theories/RCF/motor_series/test_dnf_motor.py`

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
- Formula construction remains outside the timed call. All three benchmark
  modules use `benchmark.pedantic` with one round and one iteration. The DNF
  benchmark also disables warmup and counts result atoms after the timed call.
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

## DNF benchmark

`test_dnf_motor.py` now has 91 cases: Redlog off and on; Cartesian with
`Simplification.NONE`, `FINAL`, `AFTER_STEP`, and `DURING_STEP`; and PyEDA,
each on 13 inputs. The default `NONE` schedule reproduces plain Cartesian
behavior. The other schedules share the atomic rules and a final semantic
budget of 500 pairs. Every configuration has its own 13-row benchmark group.

The schedule comparison on 2026-10-22 excluded all 13 PyEDA cases at the
user's request and passed all 78 remaining cases. The module docstring has
current input/result atom counts and computation times, plus a separate
historical PyEDA table. Its PyEDA values were not recomputed.
The PyEDA table now includes all 13 recorded input atom counts, checked
against the main table. This presentation change
did not recompute PyEDA results or times.

The following table preserves the six-configuration results from 2026-10-21.
Here early corresponds to `DURING_STEP`, and subtree to `AFTER_STEP`:

| Series | Input | Redlog off | Redlog on | Cartesian | Early | Subtree | PyEDA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 710 | 2105 | 2105 | 1528 | 1512 | 1512 | timeout |
| 2 | 1420 | 2368 | 2300 | 2297 | 2263 | 2263 | timeout |
| 3 | 94 | 112 | 35 | 112 | 38 | 38 | 49 |
| 4 | 292 | 389 | 243 | 219 | 87 | 87 | 219 |
| 5 | 157 | 152 | 96 | 143 | 48 | 48 | 143 |
| 6 | 994 | 2205 | 2142 | 802 | 761 | 761 | timeout |
| 7 | 248 | 308 | 308 | 230 | 176 | 176 | 230 |
| 8 | 473 | 1203 | 1203 | 662 | 621 | 621 | timeout |
| 9 | 235 | 282 | 282 | 210 | 162 | 162 | 210 |
| 10 | 478 | 1104 | 1104 | 1005 | 1005 | 1005 | timeout |
| 11 | 168 | 255 | 87 | 240 | 78 | 78 | 130 |
| 12 | 2176 | 5020 | 2366 | 4560 | 2116 | 2116 | timeout |
| 13 | 358 | 792 | 792 | 717 | 717 | 717 | timeout |

Both simplified Cartesian methods use the same atomic relation rules and up
to 500 final semantic cube-pair comparisons. The early method reduces cubes
as they are generated; the subtree method reduces each completed local
product. Both now choose cubes by size and sorted atoms instead of relying on
set iteration order. The insertion routine selects the first applicable cube
in one scan, avoiding a full sort at every insertion. In the repeated full
run, both returned exactly the same output atom counts as before on all 13
series. Their strict DNF sets also compared equal for all 13 after the shared
initial simplification and before final semantic subsumption. Thus the motor
inputs give no evidence that early insertion improves result quality. The
full run gave mixed single-call timings: subtree was faster on four series
and slower on nine. That timing run did not measure peak intermediate memory.

The direct set comparison used the matrix of
`early.simplify(formula.to_pnf())` for each input and checked
`early._strict_dnf(matrix) == subtree._strict_dnf(matrix)`.

Each case starts one spawned worker and waits for it to be ready before
timing. The timed call requests one DNF computation and waits at most 30
seconds for its result; it includes a small pipe round trip. Formula
construction, worker startup, and result atom counting are outside the timed
call. A separate 30-second wait guards worker startup and another guards
result atom counting. The worker runs in its own process group so a timed-out
computation and its child processes can be killed before the next case.

The first full attempt used a signal-based pytest timeout. It did not
interrupt a long-running computation, so that run was stopped. The spawned
worker design completed two full 52-case runs and a 65-case run after the
early simplified method was added. With the subtree method added, the full
78-case run finished in 528.45 seconds: 71 cases passed and seven PyEDA
cases (`series1`, `series2`, `series6`, `series8`, `series10`, `series12`, and
`series13`) raised `TimeoutError` at 30 seconds. These are the same seven
timeouts as in the earlier runs. Pytest exits with status 1 for these
timeouts. A timeout means no result size was obtained, rather than a
zero-atom result.

After deterministic cube selection was added, the full 78-case matrix was
run again in 527.20 seconds. It had the same 71 passes, seven PyEDA
timeouts, and all result atom counts. At that point, the computation-time
table in `test_dnf_motor.py` recorded this run. A separate comparison
confirmed that the two simplified methods still produced equal strict DNF
sets on every series before semantic subsumption. Summing the 13 timed calls
gives 51.030 seconds for early insertion and 51.484 seconds for subtree
reduction, compared with 46.740 and 47.804 seconds in the preceding run.
These are single measurements, so the differences do not isolate sorting
overhead from run-to-run variation.

After `CartesianBooleanNormalForm.simplify` was fixed to
`explode_always=False, substitute=1`, the full 78-case matrix was run again.
It finished in 533.49 seconds: 71 cases passed, and PyEDA timed out on the
same seven series (1, 2, 6, 8, 10, 12, 13). All successful result sizes
remained the same except for early and subtree on series12, which both
changed from 2100 to 2116 atom occurrences. The module docstring records
the result sizes and one-run computation times from this latest run. The
benchmark JSON is `/tmp/dnf-substitute1-motor-full.json`; the full pytest
output is `/tmp/dnf-substitute1-motor-full.log`. A direct comparison with
the new initial simplification again found equal strict DNF sets for early
and subtree on all 13 series.

### Initial implicit-ranges experiment

A separate experiment disabled `implicit_ranges` only in the initial
Cartesian simplification, retaining `explode_always=False, substitute=1`
and the defaults for semantic checks and final simplification. For both early
and subtree, disabling the option increased the final atom counts on series2
(2263 to 2265), series4 (87 to 93), series5 (48 to 54), series6 (761 to 770),
and series11 (78 to 87). The other eight series had unchanged atom counts.
The two methods agreed on every result size under each setting.

Single-call timings were mixed. The totals across all 13 inputs were
50.725 seconds with implicit ranges and 48.910 without for early, and
50.123 with and 48.694 without for subtree. These are separate exploratory
measurements, not replacements for the default-option module tables.
The saved data are in `/tmp/logic1-implicit-ranges-results.jsonl`, which also
contains the corresponding SMT-LIB experiment. Temporary files are local
artifacts and may disappear; the recorded findings do not depend on retaining
them. The observed motor result sizes support keeping `implicit_ranges=True`.

### Four simplification schedules (2026-10-22)

The new `FINAL` schedule constructs the strict DNF without recursive
reductions, then normalizes cubes and applies structural subsumption and cut
once before final semantic subsumption. The after-step and during-step
implementations now inherit their helpers from a shared base, rather than
from each other. Existing recursive algorithms retain their reduction order.

`FINAL`, `AFTER_STEP`, and `DURING_STEP` returned the same final atom counts
on every motor input. Totals of the 13 single-call computation times were
49.111, 50.262, and 50.900 seconds, respectively; `NONE` took 21.102 seconds.
These measurements do not establish a consistent speed advantage. For
series12, `NONE` returned 4560 atoms in 5.828 seconds, while `FINAL` returned
2116 in 7.869 seconds, `AFTER_STEP` in 8.326, and `DURING_STEP` in 8.664.

The full run took 368.11 seconds including worker startup and result
counting. Saved measurements and output are in
`/tmp/logic1-motor-schedules-2026-10-22.json` and
`/tmp/logic1-motor-schedules-2026-10-22.log`. The new shared interface and
schedules passed the unit tests and module doctests; the changed Python
modules passed mypy with `--disallow-untyped-defs`.

## Regular-test performance and validation on 2026-10-22

`Simplification` is exported by `logic1.theories.RCF`. The public functions
accept `method="cartesian"` with `NONE`, `FINAL`, `AFTER_STEP`, or
`DURING_STEP`; nonzero schedules are currently unsupported by PyEDA.
The three simplifying implementations are siblings under a shared base.
Each public computation now creates a fresh normal-form instance. The module
type definitions are grouped at the top of `bnf.py`, and its unused
`BooleanFormula` import has been removed.

The 400-cube semantic-budget examples made the two recursive class doctests
take roughly four seconds each in the regular parallel suite. Their executable
examples now use 36 cubes and budgets 10, 30, and unbounded, producing 31, 19,
and one result cube. The focused run passed 57 tests in 0.65 seconds; each of
the two class doctests took 0.24 seconds. These are observed timings, not
performance thresholds. The larger experiment, including its default-budget
result and 760 unbounded pair checks, is preserved in `doc/notes/RCF/dnf.md`.

Commit-readiness checks in the Sage configuration passed:

- `make test`: typing passed for 19 stub files and 68 source files, followed
  by 660 passing regular tests in 13.43 seconds.
- `make test-doc`: all 13 API documentation tests passed in 3.06 seconds.
- Both changed DNF benchmark modules passed mypy with
  `--disallow-untyped-defs`.
- `git diff --check` was clean, and no references to the replaced method
  strings remained in source, API documentation, or benchmark modules.

The initial sandboxed test run could not create the local socket required by
parallel QE. Repeating the full check outside the sandbox passed; this was a
permission issue, not a DNF regression. The checks did not validate Flint.
The regular-check logs are in
`/tmp/logic1-precommit-test-unsandboxed-2026-10-22.log` and
`/tmp/logic1-precommit-doc-2026-10-22.log`.

An intermittent mypy `[has-type]` error for `__all__` in `logic1/__init__.py`
remains unresolved. It disappeared after cleaning or retrying in the reported
runs and did not recur in the commit-readiness checks. Switching RCF backends
while reusing the same cache is a plausible trigger, not an established cause.
Separate caches per backend and explicit `list[str]` annotations were
discussed but have not been implemented. Preserve a failing cache before
diagnosing with `--no-incremental --cache-dir=/dev/null`; do not treat a
successful retry as proof that the underlying issue has been fixed.

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
- Pytest collection finds exactly 26 `gsimplify` cases, 78 `simplify` cases,
  and 91 DNF cases (78 with PyEDA excluded).
- All 13 `radical=True` `gsimplify` results were computed and have the same
  output atom counts as `radical=False`.
- Both `gsimplify` radical settings passed for `series3` with benchmarking
  disabled.
- All 78 `simplify` cases passed with benchmarking disabled when the full
  option matrix was introduced. After the final ordering and ID changes, all
  six `series3` cases passed and collection still found 78 cases.
- The `gsimplify` and `simplify` modules pass mypy with
  `--disallow-untyped-defs`. The DNF module also passed that check after the
  full benchmark run.
- All six DNF configurations passed on `series3` in the full run.
- The full DNF benchmark was repeated after deterministic cube selection,
  with 71 results and the seven PyEDA timeouts listed above. Every successful
  result had an output atom count unchanged from the preceding run.
- The full DNF benchmark was repeated after changing initial Cartesian
  simplification to `explode_always=False, substitute=1`, with 71 results
  and the same seven PyEDA timeouts. The only changed output counts were
  early and subtree on series12 (2116 rather than 2100).
- The early and subtree variants had equal strict DNF sets before final
  semantic subsumption on every motor input after the preprocessing change.
- Both simplified-class doctests passed with `PYTHONHASHSEED` set to 0, 63,
  and 123, including their differing-output example.
- An AST formatting check found no multiline polynomial expressions.
- No full timed `gsimplify` or `simplify` pytest-benchmark run was performed
  after their latest matrix expansions.

Useful checks from the Logic1 package directory:

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache-motor conda run -n logic1_dev python -m py_compile benchmarks/theories/RCF/motor_series/motor_inputs.py benchmarks/theories/RCF/motor_series/test_gsimplify_motor.py benchmarks/theories/RCF/motor_series/test_simplify_motor.py benchmarks/theories/RCF/motor_series/test_dnf_motor.py
DOT_SAGE=/tmp/logic1-sage-full conda run -n logic1_dev mypy --disallow-untyped-defs benchmarks/theories/RCF/motor_series/test_gsimplify_motor.py benchmarks/theories/RCF/motor_series/test_simplify_motor.py benchmarks/theories/RCF/motor_series/test_dnf_motor.py
DOT_SAGE=/tmp/logic1-sage-full conda run -n logic1_dev pytest --collect-only -q benchmarks/theories/RCF/motor_series/test_gsimplify_motor.py
DOT_SAGE=/tmp/logic1-sage-full conda run -n logic1_dev pytest -q benchmarks/theories/RCF/motor_series/test_gsimplify_motor.py -k series3 --benchmark-disable
DOT_SAGE=/tmp/logic1-sage-full conda run -n logic1_dev pytest --collect-only -q benchmarks/theories/RCF/motor_series/test_simplify_motor.py
DOT_SAGE=/tmp/logic1-sage-full conda run -n logic1_dev pytest -q benchmarks/theories/RCF/motor_series/test_simplify_motor.py -k series3 --benchmark-disable
DOT_SAGE=/tmp/logic1-sage-bnf conda run --no-capture-output -n logic1_dev pytest --collect-only -q benchmarks/theories/RCF/motor_series/test_dnf_motor.py
DOT_SAGE=/tmp/logic1-sage-bnf conda run --no-capture-output -n logic1_dev pytest -q benchmarks/theories/RCF/motor_series/test_dnf_motor.py -k series3 --benchmark-only
DOT_SAGE=/tmp/logic1-sage-bnf conda run --no-capture-output -n logic1_dev pytest -q benchmarks/theories/RCF/motor_series/test_dnf_motor.py -k 'not pyeda' --benchmark-only --benchmark-json=/tmp/logic1-motor-schedules-2026-10-22.json --tb=short
```

## Current state

The shared input module is the source of truth for the formulas; each benchmark
module owns its function's options and output counts. The DNF counts are
observations from the completed run, with unknown counts for timed-out cases.
There are no known structural discrepancies. The original and reference files
remain unchanged.

A temporary generator exists at `/tmp/generate_factorized_motor_benchmark.py`,
but it predates the file split, latest docstrings, and explanatory comments. Do
not regenerate the input module from it without first preserving those edits.

After any structural change, re-run the exact `repr` comparison and pytest
collection described above.
