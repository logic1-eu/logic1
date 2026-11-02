# Semantic Cut for RCF Normal Forms

## Status and recommendation

Investigation: 2026-11-02. No semantic-cut implementation was added to the
repository. Prototypes and measurement scripts were run under `/tmp`.

**Recommendation: defer implementation until more practical examples justify
the additional cost.** The rule is sound and gives reductions beyond current
merging and subsumption. Among the selected existing benchmarks, however,
only series3 improved: 38 to 29 atom occurrences, a 23.7% reduction, for
35.6% extra computation time (about 76 milliseconds). The other measured
runs took approximately 27–97% longer without improving their results.

The priorities are comprehensibility, maintainability, bounded resource use,
and useful reductions. Greater logical power alone does not establish a
better practical trade-off. Keep this investigation available for resumption
when additional application examples become available.

The existing design and historical measurements are described in
[dnf.md](dnf.md). This note records a proposed extension, not current behavior.

## Starting implementation

The investigation used `logic1/theories/RCF/bnf.py` after the `StrictDnf`
wrapper refactoring:

- `Cube` is an immutable, slotted wrapper around a private frozenset of
  polynomial constraints. It owns normalization, merging, implication checks,
  structural equality and hashing, and deterministic ordering.
- `Cube.merge` requires exactly one differing atom on each side, concerning
  the same polynomial. It unions their allowed signs. A full sign mask removes
  the differing constraint; a partial mask produces one weaker constraint.
- `StrictDnf` is a mutable, unhashable, slotted wrapper around a private set
  of cubes. Construction, `add`, and `update` do not simplify.
- `StrictDnf.insert` expects a normalized cube and an already syntactically
  reduced destination. It applies structural subsumption and `Cube.merge`,
  immediately deleting replaced inputs.
- `StrictDnf.semantic_subsumption` compares existing cubes using `Cube.implies`.
  It removes cubes but creates none. Exact membership and same-polynomial
  sign masks discharge simple implications before calling the RCF validity
  checker for the remaining atoms.
- `max_pair_checks=500` bounds the final semantic pass. Zero disables that
  pass; `None` removes the bound. It does not bound recursive construction or
  inexpensive syntactic reduction.
- `CartesianBnf` owns recursion and formula conversion. The simplifying
  subclasses select final, after-step, or during-step syntactic reduction;
  semantic subsumption remains a final pass.

All atom right-hand sides are zero after initial preprocessing. The current
preprocessor uses `explode_always=False` and `substitute=1`. Conservative
final simplification preserves DNF.

## Rule and correctness

Let two cubes have atom sets $C_1,C_2$. Define

$$
S=C_1\cap C_2,\qquad R_1=C_1\setminus S,\qquad R_2=C_2\setminus S.
$$

Write $s=\bigwedge S$ and $r_i=\bigwedge R_i$. Their disjunction is

$$
(s\land r_1)\lor(s\land r_2)\equiv s\land(r_1\lor r_2).
$$

Replacing both cubes by the common cube $S$ is therefore correct exactly when

$$
s\Rightarrow(r_1\lor r_2).
$$

Unlike the unconditional residual-validity condition in the introductory cut
discussion in `dnf.md`, this condition uses the shared atoms as assumptions.
The residuals may contain several atoms on different polynomials. Neither
input need imply the other. This is the additional capability beyond semantic
subsumption, which only deletes one input, and beyond `Cube.merge`, which
only compares two differing relations on the same polynomial.

The proposed operation only returns the common cube. It does not attempt to
synthesize arbitrary weaker polynomial constraints. Existing sign-mask merging
continues to handle partial unions such as `p < 0` with `p == 0`.

### Existing validity checker

The prototype computes `common`, `left_residual`, and `right_residual` as atom
sets, then makes this call through the `_is_valid` wrapper imported in `bnf.py`:

```python
residual = Or(And(*sorted(left_residual)), And(*sorted(right_residual)))
valid = _is_valid(residual, assume=sorted(common), explode_always=False)
replacement = Cube(common) if valid is True else None
```

Other validity-checker options retain their defaults, as in `Cube.implies`.
No quantifier elimination is proposed for this operation. Redlog QE was used
only for independent experimental verification.

The checker returns `True` if simplification establishes truth, `False` if
it establishes falsity, and `None` if inconclusive, all modulo the assumptions.
Only `True` authorizes replacement. Failure to prove validity must not be
treated as a proof of an invalid implication. In particular, `Cube.implies`
also returns false when its heuristic cannot prove an implication.

### Boundary cases

- An empty strict DNF represents false; it has no pairs to examine.
- An empty cube represents true. A successful cut with empty common part
  yields this cube. Check `replacement is not None`: testing its truth value
  would incorrectly reject the successful empty-cube result.
- An empty residual means structural subsumption already applies. Skip the
  semantic-cut call for that pair.
- Inconsistent shared assumptions make the implication vacuously true. Both
  original cubes and the common cube are then false, so replacement remains
  sound. The existing simplifier handles detected inconsistent assumptions
  by returning truth for the validity query.
- Normalization detects contradictions between relations on equal
  polynomials, but not every contradiction involving different polynomials.
  A missed contradiction or inconclusive validity query loses an opportunity;
  it does not authorize an unsafe deletion.

## Small examples surviving initial preprocessing

These examples use the default Cartesian FINAL computation and a pair budget
of 500. They were checked against the actual initial preprocessor and final
conservative simplifier, rather than only against isolated cube operations.

### Conditional cut across different polynomials

```python
f = And(Or(x < 0, y <= 0), Or(x + y <= 0, x == 0))
```

Initial preprocessing leaves the two disjunctions, apart from ordering. The
current final result is

```python
Or(And(y <= 0, x + y <= 0), And(x < 0, x + y <= 0))
```

The prototype returns `x + y <= 0`. The common constraint implies
`y <= 0 or x < 0`: its negation would require `y > 0` and `x >= 0`, contradicting
`x + y <= 0`. Neither cube implies the other, and the differing atoms concern
different polynomials. The residual disjunction is not unconditionally valid;
for example, `x = y = 1` falsifies it. The unconditional heuristic query
returned `None`.

The shared-budget prototype used two pair checks, one successful semantic
subsumption, and one successful semantic cut. With budget zero it retained
an additional cube `And(y <= 0, x == 0)`; budget one removed that cube; budget
two sufficed for the cut. This is a small correctness witness, not evidence
for selecting a practical default budget.

### A residual containing several atoms

```python
f = And(Or(x + y > 0, x - 1 != 0),
        Or(x - y == 0, x - 1 != 0),
        Or(x - 1 < 0, x - y == 0))
```

The initial preprocessor returns

```python
And(Or(y - 1 == 0, x - 1 != 0),
    Or(y + 1 > 0, x - 1 != 0),
    Or(x - y == 0, x - 1 < 0))
```

The current final result is

```python
Or(x - 1 < 0, And(y - 1 == 0, x - 1 == 0), And(y - 1 > 0, x - y == 0))
```

The prototype returns

```python
Or(x - y == 0, x - 1 < 0)
```

An internal successful cut has common constraint `x - y == 0` and residuals
`x - 1 != 0` and `And(y - 1 == 0, y + 1 > 0)`. Under `x == y`, either `x != 1`,
or `x == y == 1`, establishing the second residual. The prototype used four
pair checks and one successful cut, with no semantic subsumption deletions.

Removing the first disjunction from the original input gives a smaller
related witness with the same current and improved final results. Its
successful cut has single-atom residuals `y - 1 == 0` and `x - 1 != 0` under
`x - y == 0`. Retain the three-disjunction version when testing multi-atom
residuals.

## Measured shared-traversal design

The prototype temporarily replaced `StrictDnf.semantic_subsumption` in each
worker process. It left repository code unchanged. The principal variant
attempted cut after both implication checks failed. Its traversal was:

1. Copy the input collection. Initialize one pair counter to zero.
2. Sort the current cubes by `Cube.sort_key` and traverse unordered pairs.
   Skip cubes already deleted from the result.
3. Before examining a surviving pair, check the budget, then increment the
   counter once. A pair's implication and cut attempts share this one unit.
4. If the right cube implies the left, delete the right and continue. Otherwise,
   if the left implies the right, delete the left and leave its inner loop.
5. If neither implication succeeds, attempt semantic cut, skipping empty
   residuals. Only a proved implication produces a replacement.
6. On successful cut, delete both inputs immediately and call `insert` on
   the replacement. Restart the sorted pair traversal without resetting
   the counter.
7. Return when a traversal completes without a cut, or when the budget is
   exhausted. Leave the original collection unchanged.

The second variant attempted cut before the two implication checks. Both
variants retained the same deterministic ordering and immediate deletion.

### Processing a successful cut

The common part of normalized cubes is normalized: there is still at most
one relation per polynomial. Deleting cubes preserves the syntactic reduction
invariant of the destination. Consequently, `StrictDnf.insert` can handle the
replacement directly, supplying further structural subsumption and inexpensive
merging. A complete additional `syntactic_reduction` pass is unnecessary.

Insertion can change the replacement further or discard it because an
existing cube subsumes it. In either case the two original cubes can remain
deleted. New cubes can enable semantic comparisons with previously visited
cubes, so the simple prototype restarts after each successful cut. A deletion
from semantic subsumption alone creates no new cube and needs no restart.

This avoids a persistent pair cache, work queue, or second ordered collection.
The cost is repeated comparisons after successful cuts. Every revisit must
consume budget. Prioritizing subsumption and immediately deleting inputs can
lose other potential cuts; no ordering here guarantees a minimal DNF. The
order counterexample in `dnf.md` remains relevant. The measured preference
for trying subsumption first is empirical, not a dominance theorem.

### Termination, time, memory, and determinism

Every successful cut replaces two cubes by at most one; subsequent insertion
can only reduce the collection further. There are at most `n - 1` successful
cuts starting from `n` cubes. Even an unlimited run terminates. Retaining cut
inputs, with its potential exponential growth, is deliberately excluded.

With budget `B`, the shared traversal makes at most `3 * B` top-level validity
calls: two implication queries and one cut query per pair. Structural checks
can avoid some of those calls. A finite pair budget does not bound the cost
of an individual RCF simplification call, the Cartesian construction, or
inexpensive insertion. External timeouts are still necessary.

Without a budget, restarting can cause cubically many pair visits in the
initial cube count. Immediate deletion prevents growth in the stored DNF,
but it does not bound temporary allocations inside the validity checker.
Peak memory was not measured in this investigation.

Sorting cubes by the existing key and sorting residual atoms and assumptions
keeps decisions independent of set iteration order, assuming the existing
RCF simplifier is deterministic. No new cache was introduced.

## Measurements

### Provenance and scope

- Date: 2026-11-02; backend: Sage (`POLYLIB == "SAGE"`), Conda environment
  `logic1_dev`, with `DOT_SAGE=/tmp/logic1-sage-implicit`.
- Branch: `feature/native-dnf`; HEAD at recording time:
  `66a6f457012d2fc5fbfd16362c069fb5749bdd86`. The working tree contained
  existing uncommitted changes, including the completed wrapper refactoring;
  HEAD alone does not reproduce the measured source.
- SHA-256 of the measured, unchanged `logic1/theories/RCF/bnf.py`:
  `ca057426d30f3e8bbec998ebd78f2db73b9746492473fd817c48746ceee9e52f`.
- Each configuration ran once in a fresh sequential worker process, with
  `PYTHONHASHSEED=0`. For each input, the order was baseline, cut after
  subsumption, then cut before subsumption. Filtered runs followed separately.
- Timings used `time.perf_counter` around the complete `bnf.dnf` call. Input
  loading, interpreter startup, atom counting, and result serialization were
  outside the timed interval. The computation timeout was 30 seconds, with
  a 45-second parent-process guard. All selected calls completed.
- `method="cartesian"`, `max_pair_checks=500`, and the schedule shown below.
  Initial simplification used `explode_always=False`, `substitute=1`, and
  default implicit ranges. Final conservative simplification was unchanged.
- Atom counts are occurrences, computed as `len(list(formula.atoms()))`,
  not numbers of distinct polynomial constraints.
- Motor inputs came from `benchmarks/theories/RCF/motor_series/motor_inputs.py`.
  SMT-LIB inputs used `SMTLIB_CASES` and `load_input` from
  `benchmarks/theories/RCF/smtlib/smtlib_inputs.py`. See that directory's
  `NOTES.md` for source attribution and input provenance.
- The sample covers a smaller motor case, larger motor construction under
  two schedules, and existing SMT-LIB cases with distinct simplification
  behavior. It is exploratory, not a representative survey. sumcracker-14d
  supplies a larger tractable case; sumcracker-14e was not run here.
- No PyEDA runs, Flint measurements, repeated timing estimates, or peak-memory
  measurements were made. Existing historical tables were not recomputed.

### Main comparison: cut after subsumption

| Input | Schedule | Input atoms | Current atoms | With cut | Current seconds | With cut seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| series3 | FINAL | 94 | 38 | 29 | 0.214280 | 0.290480 |
| half-2 | FINAL | 81 | 280 | 280 | 0.751901 | 1.480434 |
| levin-rhsc | FINAL | 46 | 88 | 88 | 0.302375 | 0.545173 |
| sumcracker-14d | FINAL | 44 | 7,168 | 7,168 | 9.844661 | 15.497562 |
| series12 | FINAL | 2,176 | 2,116 | 2,116 | 7.816665 | 10.583355 |
| series12 | DURING_STEP | 2,176 | 2,116 | 2,116 | 8.763863 | 11.126567 |

Only series3 changed its result. Its single successful semantic cut had
seven common atoms and residual lengths one and three. Every other output
had the same SHA-256 digest of its formula representation as the baseline.

| Input | Schedule | Pair visits | Subsumption deletions | Cut queries | Successful cuts |
| --- | --- | ---: | ---: | ---: | ---: |
| series3 | FINAL | 18 | 2 | 16 | 1 |
| half-2 | FINAL | 6 | 0 | 6 | 0 |
| levin-rhsc | FINAL | 28 | 0 | 28 | 0 |
| sumcracker-14d | FINAL | 500 | 0 | 500 | 0 |
| series12 | FINAL | 500 | 224 | 276 | 0 |
| series12 | DURING_STEP | 500 | 224 | 276 | 0 |

Baseline counters in the temporary JSON are zeros because the original
method was not instrumented. They do not mean that the baseline made zero
comparisons. The above counters belong to the instrumented cut variant.

### Cut before subsumption

This order produced the same final formulas as cut after subsumption in
every selected case, with no additional successful cuts.

| Input | Schedule | Cut queries | Seconds |
| --- | --- | ---: | ---: |
| series3 | FINAL | 18 | 0.288608 |
| half-2 | FINAL | 6 | 1.472450 |
| levin-rhsc | FINAL | 28 | 0.552891 |
| sumcracker-14d | FINAL | 500 | 15.110513 |
| series12 | FINAL | 500 | 11.698850 |
| series12 | DURING_STEP | 500 | 12.469478 |

For series12 it made 500 cut queries instead of 276 and was slower. Small
timing differences elsewhere should not be interpreted as reliable advantages
from single runs. Trying cut after subsumption is the better starting point
for this sample.

### Residual-size filter

The filtered variant attempted cut after failed subsumption only when each
residual contained at most two atoms. It made no successful cuts and returned
the baseline formula on every input.

| Input | Schedule | Cut queries | Filtered pairs | Result atoms | Seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| series3 | FINAL | 0 | 13 | 38 | 0.210146 |
| half-2 | FINAL | 4 | 2 | 280 | 1.359688 |
| levin-rhsc | FINAL | 8 | 20 | 88 | 0.355373 |
| sumcracker-14d | FINAL | 13 | 487 | 7,168 | 9.576424 |
| series12 | FINAL | 1 | 275 | 2,116 | 7.997327 |
| series12 | DURING_STEP | 1 | 275 | 2,116 | 8.726941 |

The filter saves considerable work on some inputs, but discards the only
useful benchmark cut and still leaves substantial overhead on half-2. There
is no evidence for adopting this threshold. Any such filter sacrifices
completeness, not soundness, provided skipped pairs are simply left unchanged.

Skipping empty residuals is straightforward. Do not reject pairs merely
because the residuals concern different polynomials, or because their common
part is empty: both can support valid cuts. A sign-mask argument that ignores
the shared assumptions can also miss a conditional cut. No more elaborate
candidate index or speculative filter was implemented.

## Shared-budget bias and the next experimental design

A shared pair budget is simple but can confound comparisons. A successful
cut restarts traversal, and revisited pairs consume budget. This can prevent
subsumption checks that the original algorithm would have reached. Changed
cubes also change subsequent ordering. More powerful rules therefore need
not give a better final result under the same finite budget.

Attempting cut on an already counted pair does not itself use an additional
budget unit in the measured prototype. The potential loss comes from restarts
and changed traversal. It did not truncate subsumption in the reported
sample: cases other than series3 had no successful cuts and followed the
original subsumption traversal; series3 completed after 18 checks, below 500.

For the next controlled experiment, first run existing semantic subsumption
unchanged with its original budget, then run semantic cut with a separate
budget. This preserves the original subsumption opportunities and isolates
the additional cost and benefit. This design was proposed after the measured
experiment and has **not** been benchmarked or implemented.

The subsequent cut pass must still process structural consequences with
`insert` and revisit pairs after a replacement. If it also performs semantic
subsumption to exploit new cubes, those implication checks and all revisits
must be charged to the second budget. Do not invoke another unbudgeted
semantic pass after every successful cut. Preserve the first pass's result
when the second budget is zero.

Running subsumption first can delete inputs that would have supported a later
cut. Thus this is a controlled baseline-preserving experiment, not a claim
that two passes find all cuts or dominate a shared traversal. If useful new
examples emerge, compare both designs on those examples and report pair
visits, successful operations, and output sizes separately from elapsed time.

## Possible implementation if later justified

Keep semantic cut opt-in and final-only initially. Put the pair operation on
`Cube` and traversal on `StrictDnf`; do not add another recursive schedule,
phase class, pair cache, or retained witness pool. Keep the existing inexpensive
`merge` distinct from the validity-based operation. A combined semantic method
would need a name reflecting both subsumption and cut; final API names and
option names have not been agreed.

Copy the input collection, delete both cut inputs immediately, insert the
normalized common cube, and restart deterministically with the counter
preserved. Prefer this small algorithm until measurements justify a more
complicated work list. Apply the existing zero/finite/`None` budget conventions,
while keeping unlimited runs subject to external time limits in experiments.
No new production default for a cut budget has been selected.

Estimated additional implementation size, excluding docstrings and tests:

| Design | Additional lines |
| --- | ---: |
| Extend the existing traversal with a shared budget | 30–45 |
| Separate cut pass with its own budget | 50–70 |

Meaningful docstrings and tests were estimated to add another 60–100 lines.
These are planning estimates, not counts from a repository implementation.
The limited measured benefit currently does not justify that maintenance cost.

## Verification completed

- Independent Redlog QE established equivalence of each small input and its
  improved output, of the current and improved small outputs, and of their
  recorded local cut replacements. Universal closure was used, for example
  `redlog.qe(Equivalent(original, improved).all()) is T`.
- The successful local series3 replacement was independently verified with
  Redlog QE. The complete series3 input/output equivalence was not separately
  proved by QE; the verification concerns the new local transformation.
- Direct checks covered an empty common part (`x > 0` with `x < 1`, yielding
  truth), an inconclusive query (`x > 0` with `y > 0`), and inconsistent common
  assumptions (`x == 0` and `x != 0`). Successful replacements passed Redlog QE.
- Empty collections, a collection containing the empty cube, and input
  collection immutability were checked.
- A four-cube collection required two cuts:
  `{x+y<=0, y<=0}`, `{x+y<=0, x<0}`, `{u+v<=0, v<=0}`,
  `{u+v<=0, u<0}`. Budgets `0, 1, 2, 5, 500, None` produced respectively
  `0, 1, 2, 5, 5, 5` pair visits and `0, 1, 1, 2, 2, 2` successful cuts.
  Every result passed independent equivalence checking; finite budgets were
  respected across restarts.
- Determinism was checked in 44 runs: the first and multi-atom small witnesses
  at budgets `0, 1, 2, 500, None`, plus series3 at the default budget, across
  hash seeds `0, 1, 63, 314159`. Formula representations, pair counts, and cut
  counts matched across seeds. These checks used cut after subsumption.
- The repository diff before and after investigation matched. No production
  implementation or routine test suite was changed for the experiment.

These checks concern the Sage prototype. Prior Sage/Flint verification of the
wrapper refactoring, recorded in `dnf.md`, is not verification of semantic cut.

## Resuming the investigation

1. Read applicable `AGENTS.md` files, this note, `dnf.md`, and current `bnf.py`.
   Check whether the wrapper contracts, validity options, or traversal have
   changed. Preserve existing work; discussion alone does not authorize edits.
2. Collect practical examples where initial preprocessing, current merging,
   semantic subsumption, and conservative final simplification leave useful
   conditional cuts. Keep original formulas and both final outputs, not only
   extracted cube pairs.
3. Reconstruct the small pair operation and shared traversal described above
   if temporary files are gone. Reestablish the witnesses before larger runs.
4. Compare the unchanged baseline, the separately budgeted final extension,
   and, where informative, the measured shared-traversal design. Preserve
   baseline subsumption effort and explicitly account for any additional work.
5. Record the backend, source revision and local changes, options, pair
   budgets, timeouts, atom and cube counts, operation counts, and fresh-process
   timings. Use repeated runs for close timing comparisons. Measure peak memory
   if the new examples make temporary allocation a practical concern.
6. Verify every novel local replacement independently where feasible. Test
   restarts, budgets, input immutability, empty cases, and hash-seed stability.
   Check Flint separately before drawing conclusions about that backend.
7. Exclude PyEDA unless a new question specifically requires it. Reuse saved
   measurements for presentation changes. Decide whether recurring gains
   warrant implementation before adding permanent code or public options.

### Temporary artifacts from this investigation

These paths describe local artifacts, not durable dependencies of this note.
The examples, algorithm, settings, and tables above preserve the essential
information if `/tmp` is cleared.

| Path under `/tmp` | Contents |
| --- | --- |
| `semantic_cut.py` | Pair operation, monkeypatched shared traversal, counters, proof recording |
| `semantic-cut-bench.py` | Fresh-process baseline/after/before benchmark driver |
| `semantic-cut-bench.json` | Raw timings, atom counts, counters, output digests |
| `semantic-cut-filtered.py`, `semantic-cut-filtered.json` | Residual-size-filter driver and results |
| `semantic-cut-witness.py` | Seeded witness search (seed 438) |
| `semantic-cut-examples.py`, `semantic-cut-multi.py` | Witness minimization and independent checks |
| `semantic-cut-verify.py` | Boundary cases, series3 local QE proof, small budget checks |
| `semantic-cut-restarts.py` | Multiple cuts, budget accounting, immutability, empty cases |
| `semantic-cut-determinism.py` | Four-seed comparison |
| `semantic-cut-{case}-{schedule}-{mode}.pickle` | Input, output, and successful cut triples |
| `semantic-cut-before.diff`, `semantic-cut-after.diff` | Matching repository diffs around the investigation |

The drivers depend on the original workspace path in `semantic_cut.py`; adapt
it when resuming elsewhere. Baseline mode restores the original semantic
method. Modes `after` and `before` specify cut order; `filtered` selects the
after variant with the residual-size threshold. Do not mistake baseline JSON
counter zeros for measured zero work. Pickles are backend/source dependent;
prefer the original input modules when recreating an experiment.

Example commands from the repository root, if the temporary scripts remain:

```sh
DOT_SAGE=/tmp/logic1-sage-implicit conda run -n logic1_dev python /tmp/semantic-cut-bench.py
DOT_SAGE=/tmp/logic1-sage-implicit conda run -n logic1_dev python /tmp/semantic-cut-filtered.py
DOT_SAGE=/tmp/logic1-sage-implicit conda run -n logic1_dev python /tmp/semantic-cut-verify.py
DOT_SAGE=/tmp/logic1-sage-implicit conda run -n logic1_dev python /tmp/semantic-cut-restarts.py
DOT_SAGE=/tmp/logic1-sage-implicit conda run -n logic1_dev python /tmp/semantic-cut-determinism.py
```

Run these only to answer a new experimental question or reestablish a changed
baseline. Creating this note reused the saved results and reran no benchmarks.
