# SMT-LIB DNF benchmark notes

Updated: 2026-10-22

This document covers `benchmarks/theories/RCF/smtlib/` and all its descendants,
including `inputs/`, excluding sibling benchmark directories. It records
provenance, selection criteria, measurements, and completed verification for
that subtree. Operational instructions are in this directory's `AGENTS.md`;
`NOTES.md` is supporting documentation, not an automatically loaded agent
instruction file.

## Source and credit

The eight inputs are unmodified files from the QF_NRA part of the
[SMT-LIB](https://smt-lib.org/) [2025 non-incremental benchmark release](https://doi.org/10.5281/zenodo.16740866)
(version 2025.08.04). The release was curated by Mathias Preiner,
Hans-Jörg Schurr, Clark Barrett, Pascal Fontaine, Aina Niemetz, and
Cesare Tinelli. The files retain their original paths below `QF_NRA` in
`inputs/`, including their individual `:source` and `:license` fields.
Each selected file states the
[Creative Commons Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/).
No original `.smt2` file was modified.

The seven Uncu inputs credit Ali K. Uncu, Matthew England, and James H.
Davenport for generating the SMT-LIB files; their source fields also credit
Manuel Kauers's SUMCracker-ProveInequality and identify the underlying
inequality problems. The Ultimate input credits Matthias Heizmann and
the Ultimate program analysis framework, and cites the SV-COMP 2017 inputs
from which they were generated. The original source fields provide the
full publication details.

## Selection criteria

The first six candidates were selected by source structure before running
the methods. A pilot run found that both native methods reached 30 seconds
on two Ultimate cases. We retained `half-2` as a high-variable stress case,
removed the similar `count_by_k` case, and added three smaller Uncu inputs
to make the timing comparison useful across more workload sizes. The final
selection follows these criteria:

1. belong to QF_NRA and contain only real polynomial comparisons;
2. use conjunctions containing disjunctions, so DNF construction must
   distribute rather than merely copy an existing DNF;
3. use only the explicitly supported SMT-LIB operators in `smtlib_inputs.py`;
4. span small through substantial raw Cartesian products (4 to 2,048);
5. include two independently generated source families; and
6. are small enough to keep and inspect as original input files.

The raw product count applies the DNF size recurrence to the source Boolean
tree: an atom contributes one, `or` adds child counts, and `and` multiplies
them. It counts combinations before duplicate removal or any algebraic or
Boolean simplification. The count is a structural workload indicator, not
an expected output size. The selection is deliberately varied and is not a
statistical sample of all QF_NRA benchmarks.

| Case | Original SMT-LIB entry | Family | Input atoms | Raw products |
| --- | --- | --- | ---: | ---: |
| mitrinovic-4-14 | [Mitrinovic (4.14)](https://explore.smt-lib.org/benchmark/383983.html) | Uncu | 16 | 4 |
| mitrinovic-3-29 | [Mitrinovic (3.29)](https://explore.smt-lib.org/benchmark/383998.html) | Uncu | 20 | 8 |
| knopp-schur-b | [Knopp-Schur b](https://explore.smt-lib.org/benchmark/383961.html) | Uncu | 29 | 32 |
| half-2 | [half_2](https://explore.smt-lib.org/benchmark/386825.html) | Ultimate | 81 | 32 |
| issac10-ex4h | [ISSAC 2010 ex4h](https://explore.smt-lib.org/benchmark/383943.html) | Uncu | 39 | 128 |
| levin-rhsc | [LevinRHSc](https://explore.smt-lib.org/benchmark/383949.html) | Uncu | 46 | 256 |
| sumcracker-14d | [SUMCracker 14d](https://explore.smt-lib.org/benchmark/383952.html) | Uncu | 44 | 512 |
| sumcracker-14e | [SUMCracker 14e](https://explore.smt-lib.org/benchmark/383955.html) | Uncu | 54 | 2,048 |

## Translation and measurement

`smtlib_inputs.py` reads one original SMT-LIB file in each worker before
benchmark timing. This prevents variables from other cases entering the
worker's polynomial ring.
It accepts only real declarations, assertions, `and`, `or`, polynomial
addition, subtraction and multiplication, and the comparison operators
present in these files. Decimal numerals are converted exactly to rational
numbers, and every comparison `left OP right` becomes the equivalent
`left - right OP 0`, as required by Logic1's RCF simplifier. Sage does not
accept some Ultimate variable names containing `.`
and `~`; those names are renamed bijectively to safe names. Logic1 flattens
adjacent operators of the same kind when building `And` and `Or`. The same
resulting Logic1 formula is supplied to all configurations.

The main comparison has six configurations: Redlog DNF with both `bnfsac`
and `bnfsm` off or on, and `method="cartesian"` with `Simplification.NONE`,
`FINAL`, `AFTER_STEP`, or `DURING_STEP`. The default `NONE` retains ordinary
preprocessing and conservative final simplification. Other schedules use
`max_pair_checks=500`. Every Cartesian schedule preprocesses with
`explode_always=False, substitute=1`. In the historical measurements, early
means `DURING_STEP` and subtree means `AFTER_STEP`.
Each main-comparison case starts a fresh worker, times one call, and enforces
a 30-second wall limit on that call. It counts output atoms after timing.
Worker startup, SMT-LIB parsing, and output counting are outside the measured
interval. The timed call includes the small parent/worker pipe round trip,
as in the motor benchmark. Each worker has its own process group, which is
killed on timeout.
Startup and result counting have separate 30-second guards. A timeout is a
test failure with an unknown result size, not a zero-atom result. Timed runs
use one round, one iteration, and no warmup, with cases run sequentially.

The full five-method run on 2026-10-21 completed 36 of 40 computations.
All three Cartesian methods completed `half-2` and returned 280 atom
occurrences, compared with 288 from each Redlog method. Both Redlog methods
and both simplified Cartesian methods timed out on `sumcracker-14e`; the
Cartesian method completed it. The two measured tables are in the module
docstring of `test_dnf_smtlib.py`. Atom counts describe result size; this
benchmark does not perform a separate equivalence check. This historical
40-case run precedes the supplementary cases described below.

## Four-schedule comparison on 2026-10-22

The main six-configuration comparison plus the additional levin-rhsc case
ran 49 computations. There were 44 results and five 30-second computation
timeouts: Redlog off and on, `FINAL`, `AFTER_STEP`, and `DURING_STEP`, all on
sumcracker-14e. Plain Cartesian (`NONE`) returned 34,816 atoms for that input
in 14.397 seconds. Every other main case completed, and all three simplifying
schedules agreed on the atom counts of their completed results. The new
final-only schedule did not improve those counts over recursive reduction.

The additional levin-rhsc computation again returned 16 atoms with initial
implicit ranges disabled (0.203 seconds), compared with 88 for DURING_STEP
with the default setting (0.293 seconds). The module docstring records the
new size and timing tables and this additional comparison. `t/o` abbreviates
a computation timeout in the main tables; its result size is unknown.

The run took 359.02 seconds including worker startup and counting. Saved
measurements are in `/tmp/logic1-smtlib-schedules-2026-10-22.json`; the log
is `/tmp/logic1-smtlib-schedules-2026-10-22.log`. The historical PyEDA and
pair-budget measurements below were retained without recomputation. Only
their schedule labels and invocation examples were updated to the new API.

## Supplementary experiments

### Initial implicit ranges on levin-rhsc

The additional case uses `method="cartesian"` and `Simplification.DURING_STEP`, with
`implicit_ranges=False` only during initial simplification. It uses a
temporary replacement of `CartesianBooleanNormalForm.simplify` inside the
worker and restores it afterwards. Semantic implication checks and final
simplification retain their defaults. The computation limit remains
30 seconds.

On 2026-10-21, early returned 88 atom occurrences in 0.299 seconds with the
default setting and 16 in 0.209 seconds with initial implicit ranges disabled.
Enabling the option reduces the initial formula but collapses explicit
variable sign cases. The current simplifier cannot recover those signs from
product constraints, leaving six contradictory cubes. This is the only
observed case so far where enabling implicit ranges increases the final atom
count; other completed cases in the motor and SMT-LIB experiment had equal
or smaller counts with the option enabled. The default remains enabled.

Equivalence of the two levin-rhsc outputs was checked separately from timing.
Every default-output cube contains `V2 != 0`; splitting each cube on `V2 < 0`
and `V2 > 0` is therefore exhaustive. Simplifying this disjunction produced a
formula structurally identical to the output with initial implicit ranges
disabled. This check used the RCF simplifier, not an independent solver.

### Pair-check budgets on sumcracker-14e

The separate benchmark doubles `max_pair_checks` from 1 until the first
timeout, independently for early and subtree. It also has zero-budget
controls. The computation limit is 60 seconds, with the default initial
`implicit_ranges=True`; startup and counting guards remain 30 seconds.

Both sweeps completed budgets 1 through 1024 and first timed out at 2048.
Every completed run, including both zero-budget controls, returned 34,816
atom occurrences. The zero-budget times were 44.241 seconds for early and
41.600 for subtree; at budget 1024 they were 58.292 and 56.887 seconds.
Plain cartesian had returned the same count in 13.841 seconds in the main
comparison. Setting the budget to zero disables final semantic comparisons;
the local subsumption and cut steps still run, so it does not reproduce the
Cartesian algorithm. Increasing the budget gave no final size improvement
on this input. Selected budget results are in the module docstring table.

### Informational PyEDA results

A separate run on 2026-10-21 measured `method="pyeda"` with default options
and a 30-second computation limit on all eight inputs. All eight completed.
The module docstring contains an informational table of their atom counts
and times. These are not benchmark cases implemented in this file and do
not extend its main comparison. In particular, PyEDA returned
28,520 atoms for half-2 in 12.573 seconds and 34,816 for sumcracker-14e in
15.338 seconds.

## Collection and commands

### Latest validation and regular-test conventions

The schedule API and class refactoring passed the Sage commit-readiness
checks on 2026-10-22: 660 regular tests, 13 API documentation tests, and
the source and stub mypy checks. Both changed DNF benchmark modules also
passed mypy with `--disallow-untyped-defs`. These checks do not establish
Flint compatibility; that backend was not checked in this validation round.

The expensive 400-cube semantic-budget doctests have been replaced by
36-cube examples that still show improvements with increasing budgets.
Both recursive class doctests took 0.24 seconds in the focused run, which
passed 57 tests in 0.65 seconds. The larger experiment remains documented
in `doc/notes/RCF/dnf.md`, outside the regular test suite.

The historical PyEDA measurements and 60-second pair-budget sweeps were not
rerun for documentation or naming changes. The five sumcracker-14e timeouts
in the latest 49-case comparison remain recorded as timeouts with unknown
result sizes. No larger timeout was used to replace those observations.

An intermittent mypy inference error involving `logic1.__all__` remains a
separate unresolved issue. Backend switching and cache reuse are suspected,
but no cache-isolation or annotation fix has been applied. The latest Sage
checks passed without reproducing it.

### Collection and execution

The file currently collects 75 cases: 48 main comparisons, one additional
levin-rhsc case, and 26 pair-budget cases (13 budgets for each of two methods).
The module docstring keeps the main size and timing tables, supplementary
experiments, and informational PyEDA results separate. They record individual
measurements, not averaged performance estimates. Reuse these measurements
for presentation changes; preserve each run's date, options, and time limit.

Run from the repository root. Select the main comparison and additional
levin-rhsc case (49 cases) with:

```bash
pytest benchmarks/theories/RCF/smtlib/test_dnf_smtlib.py::test_dnf_smtlib --benchmark-only
```

Compare only the two DURING_STEP levin-rhsc cases with:

```bash
pytest benchmarks/theories/RCF/smtlib/test_dnf_smtlib.py -k levin-rhsc-during-step --benchmark-only
```

Run each doubling sweep separately, stopping at its first timeout:

```bash
pytest benchmarks/theories/RCF/smtlib/test_dnf_smtlib.py -k 'pair_checks and during-step and not 0-' --benchmark-only --exitfirst
pytest benchmarks/theories/RCF/smtlib/test_dnf_smtlib.py -k 'pair_checks and after-step and not 0-' --benchmark-only --exitfirst
```

Run the zero-budget controls separately:

```bash
pytest benchmarks/theories/RCF/smtlib/test_dnf_smtlib.py -k 'pair_checks and 0-' --benchmark-only
```
