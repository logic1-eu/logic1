# Simplifying a strict DNF

These notes concern a strict DNF conceptually represented as
`set[frozenset[AtomicFormula]]`. Each inner frozenset is a conjunction
(a *cube*); the outer set is a disjunction of cubes. Thus, for a DNF `N`,

$$
\llbracket N \rrbracket
= \bigvee_{C \in N}\;\bigwedge_{a \in C} a.
$$

The empty outer set represents `F`. A DNF containing the empty frozenset
represents `T`, because the empty conjunction is true. In particular,
`set()` represents `F`, while `{frozenset()}` represents `T`.

In the implementation, `Cube` is an immutable wrapper around the inner
frozenset, and `StrictDnf` is a mutable wrapper around `set[Cube]`. The
corresponding Python values are `StrictDnf()` for `F` and
`StrictDnf((Cube(),))` for `T`.

The atoms are polynomial constraints. Equivalences with `T` or `F` below
mean validity or unsatisfiability over real closed fields, respectively.

## Subsumption

If cubes $C_1$ and $C_2$ satisfy $C_1 \subseteq C_2$, keep $C_1$ and delete $C_2$:

$$
(\bigwedge C_1) \lor (\bigwedge C_2) \equiv \bigwedge C_1.
$$

The smaller cube has fewer conjuncts and is therefore the weaker formula.
The rule also applies when $C_1$ is empty: a true cube subsumes every cube.

## Cut

For cubes $C_1$ and $C_2$, let $S=C_1\cap C_2$ contain their shared
constraints and let $R_i=C_i\setminus S$ contain the residual constraints
of $C_i$, for $i=1,2$. The two cubes can be replaced by $S$ if

$$
(\bigwedge R_1) \lor (\bigwedge R_2) \equiv T.
$$

Indeed,

$$
(\bigwedge C_1) \lor (\bigwedge C_2)
\equiv (\bigwedge S) \land
\bigl((\bigwedge R_1) \lor (\bigwedge R_2)\bigr)
\equiv \bigwedge S.
$$

The condition is a **disjunction** of the residual conjunctions. For
example, $\{a,b\}$ and $\{\bar a,b\}$ cut to $\{b\}$ when $\bar a$ is
the atomic complement of $a$. The validity check is not limited to
complementary atoms: for any polynomial $p$, $p\geq 0\lor p\leq 0$ is
valid, although the two constraints overlap when $p=0$.
Subsumption is the special case in which one residual cube is empty.

## Consensus

Cut and subsumption do not account for every useful two-cube
simplification. Suppose $a$ and $\bar a$ are complementary atomic
constraints and

$$
C_1=\{a\}\cup R_1,\qquad C_2=\{\bar a\}\cup R_2.
$$

The *consensus cube* $R_1\cup R_2$ may be added without changing the
DNF: whenever its conjunction holds, either $a$ or $\bar a$ holds, so
one of the original cubes holds. If $R_1\subseteq R_2$, the consensus
cube is $R_2$ and subsumes $C_2$. One can directly replace $C_2$ by
$R_2$ without increasing the number of cubes; the symmetric case is
analogous. For example,

$$
a\lor(\bar a\land b)\equiv a\lor b.
$$

Unrestricted consensus addition can grow the DNF. The direct
replacement above is available only when the consensus cube subsumes
an input cube. Complementary atoms make consensus sound, but do not
capture every relation between polynomial constraints.

## Order and iteration

Destructively applying subsumption first can lose a useful cut. Consider

$$
\{\{a\},\;\{a,b\},\;\{\bar a,b\},\;\{\bar b\}\},
$$

Take $a$ to be $x>0$ and $b$ to be $y>0$ for distinct real variables
$x,y$; $\bar a$ and $\bar b$ are their atomic complements. Subsumption removes
$\{a,b\}$, after which no remaining pair satisfies the cut condition.
If $\{a,b\}$ is first cut with $\{\bar a,b\}$, the result is $\{b\}$;
cutting this with $\{\bar b\}$ gives the empty cube, or `T`.

A cut can enable another cut, so cuts must be reconsidered after each
replacement. A practical greedy strategy is to perform nontrivial cuts
until none applies, then remove subsumed cubes. Greedy replacement can
still depend on which eligible pair is chosen. The one-sided consensus
rule can recover the simplification in the example even after the
initial subsumption, but it does not make rule order irrelevant in
general.

To avoid losing any cut derivable by these rules, add each cut result
without immediately deleting its inputs. Repeat until no new cube is
produced, then apply subsumption. Even with consensus, these local rules
do not guarantee a globally minimal DNF over real closed fields.

## Growth when cut inputs are retained

In cut-only saturation, every generated cube is an intersection of a
nonempty collection of the original cubes. With $m$ original cubes and
$k$ distinct atoms, there can be at most $\min(2^m-1,2^k)$ distinct
cubes. This bound can still be too large when formulas are large.

Exponential growth in $m$ is possible with polynomial constraints. Let
$a_i$ be the constraint $x\ne i$ for $i=1,\ldots,m$, let $r$ be $y>0$,
and start with the cubes

$$
C_i=\{r\}\cup\{a_j:j\ne i\}.
$$

At most one $a_i$ can be false at any value of $x$. Consequently, the
cut condition holds between the intersections generated from these
cubes. Retaining every result produces the distinct cubes

$$
\bigcap_{i\in I}C_i=\{r\}\cup\{a_j:j\notin I\}
\qquad (\varnothing\ne I\subseteq\{1,\ldots,m\}),
$$

of which there are $2^m-1$. Yet the DNF is equivalent to the single
cube $\{r\}$. Discovering that cube early allows a shortcut in this
example; unrestricted saturation has no general space guarantee.

## Recursive simplification strategy

The two recursive simplifying schedules separate inexpensive atomic
subsumption and cut during recursive DNF construction from a final
semantic subsumption pass. They keep the strict DNF as a set of
frozenset cubes and work with six polynomial relations against zero.

Each relation has a three-bit mask describing the possible negative,
zero, and positive signs of its polynomial. Within a cube, the
intersection of masks for the same polynomial combines conjuncts:
$p<0\land p\leq0$ becomes $p<0$, and an empty intersection rejects the
cube. For two cubes with identical remaining conjuncts, the union of
their differing masks combines disjuncts:

$$
(S\land p<0)\lor(S\land p=0)\equiv S\land p\leq0.
$$

A full mask removes the polynomial constraint. These operations use
only equality of normalized polynomials and bitwise operations. They
do not invoke a general RCF validity test.

`DuringStepSimplifyingCartesianBnf` inserts cubes into a running
strict DNF at each `Or` node. At each `And` node, it multiplies child
DNFs one at a time, with insertion after each product. Insertion removes a
structurally subsumed cube or merges two cubes as above; new merge
results are reconsidered until they can be inserted. This greedy rule
can discard a cube that might have supported a later cut, as shown in
"Order and iteration" above. It keeps the algorithm and its storage
small, but does not promise a minimal DNF.

`AfterStepSimplifyingCartesianBnf` uses the same atomic
rules but finishes the local Cartesian expansion of each `And` or `Or`
node before reducing its cubes. Its recursive calls return already
reduced child DNFs. This matches the placement of Redlog's
`cl_strict!-gdnf` simplification around `cl_strict!-gdnf1`: the latter
constructs a complete local product, and the wrapper simplifies it
before returning to the parent. The two Logic1 classes use identical
cube rules. Both traverse cube sets by a stable key: cube size, then the
sorted atoms. Insertion chooses the lowest-key applicable cube for
subsumption or merging. It finds that cube in one scan rather than sorting
the result set at every insertion. The greedy choices therefore do not
depend on Python set iteration order. Greedy reduction still does not
guarantee a minimal DNF; a different choice priority can yield another
equivalent result.

After recursive construction, semantic subsumption compares the
remaining cubes. A cube $C_1$ is removable when its conjunction implies
the conjunction of another cube $C_2$. Exact atoms and same-polynomial
sign masks are checked first. For the remaining atoms of $C_2$, the
RCF simplifier tests validity under the assumptions in $C_1$. This
also finds implications between constraints on different polynomials.
An inconclusive result never justifies removing a cube.

`max_pair_checks: int | None = 500` limits cube pairs examined in this
final semantic pass for either method. Zero disables that pass but retains
all recursive atomic rules; `None` removes the limit. The bound does not
cap the time of one RCF simplification call or the Cartesian products.
The default was chosen after measuring the 13 motor series: a larger limit made
the previous final-stage search exceed 30 seconds, while 500 checks
preserved the main semantic gains in the recursive method.

### Interface and schedules (2026-10-22)

The public `dnf` and `cnf` functions separate the algorithm, selected with
`method="cartesian"` or `method="pyeda"`, from the simplification schedule.
`Simplification` is exported by `logic1.theories.RCF` and has these values:

| Member | Value | DNF-specific reductions |
| --- | ---: | --- |
| `NONE` | 0 | None; preserve ordinary preprocessing and conservative final simplification |
| `FINAL` | 1 | Reduce the complete strict DNF once |
| `AFTER_STEP` | 2 | Reduce each completed Boolean node |
| `DURING_STEP` | 3 | Reduce cubes and partial products while constructing each node |

A step constructs the DNF at one `And` or `Or` node. The integer values
identify schedules, not increasing guarantees of quality or speed. Since
2026-10-23, the defaults are `method="cartesian"` and `Simplification.FINAL`
(previously PyEDA and `NONE`). Nonzero schedules require `method="cartesian"`;
unsupported combinations are rejected. Plain Cartesian construction remains
useful because the recursive reductions can cost time without reducing the
result. `FINAL` provides the previously missing experiment: plain recursive
construction followed by one dedicated reduction of the completed strict DNF.
Select `Simplification.NONE` explicitly for plain Cartesian or PyEDA calls.

The current implementation separates values and computation:

- `Cube` owns a conjunction's immutable atom set, ordering, normalization,
  merging, and implication checks.
- `StrictDnf` owns a mutable set of cubes. Its constructor, `add`, and `update`
  collect without simplifying. `insert` requires a normalized cube and an
  already reduced destination, which it mutates using subsumption and cut.
  `syntactic_reduction` and `semantic_subsumption` return new collections.
- `CartesianBnf` owns recursion and formula conversion. Its finishing hook
  is the identity for plain Cartesian computation.
- `SimplifyingCartesianBnf` owns budget validation and storage and implements
  final semantic subsumption. The final, after-step, and during-step classes
  inherit from it and select when syntactic reduction runs. The final-only
  finishing hook runs both passes; the two recursive schedules inherit the
  common semantic finishing hook.

`Cube.sort_key` determines reconstruction, greedy insertion, and bounded
comparison order. `StrictDnf` has structural equality and is unhashable.
Its set is owned independently of any input container; immutable cubes may
be shared. A strict DNF can be unsimplified, so none of these values stores
an implicit simplification policy. Semantic counters start anew on each call.

Earlier on 2026-11-02, `StrictDnfSimplificationMixin` supplied collection
operations to three classes using independent multiple inheritance. Moving
those operations into `StrictDnf` eliminated the mixin. The common Cartesian
base now handles configuration and final processing without duplicating
constructors or hooks. Historical measurements of the intermediate design
are retained below.

`simplify` and `conservative_simplify` continue to call the general RCF
simplifier before construction and after formula reconstruction, respectively.
Deterministic formula reconstruction is shared by all Cartesian schedules.
The refactoring preserves the rules, budgets, and measured schedules; the
historical benchmark data below is unchanged.

All three simplifying schedules use `max_pair_checks=500` for final semantic
subsumption; zero skips that pass and `None` removes the limit. `NONE` and
PyEDA require the default budget because neither uses this pass. CNF uses the
DNF of the negation. The former witness pool and general pairwise validity-cut
search are not part of these implementations.

In the measurements below, **early** means `DURING_STEP` and **subtree** means
`AFTER_STEP`. Those historical labels and dates are retained when describing
the original experiments. The benchmark modules use the current schedule
names for new measurements.

### Mixin refactoring verification (2026-11-02)

Checked with the Sage backend in the `logic1_dev` Conda environment, using
`DOT_SAGE=/tmp/logic1-sage-implicit`:

- Focused BNF unit tests and module doctests: 129 passed.
- `make test`: mypy passed for 19 stub files and 68 source files; all 688
  regular tests and doctests passed in 20.05 seconds.
- `make test-doc`, run separately: all 13 API documentation tests passed
  in 2.93 seconds.
- Reversed-parent tests cover all three concrete schedules, DNF and CNF,
  budgets of 0, 1, 500, and `None`, and examples requiring structural and
  semantic simplification.
- Python code and docstring line lengths and `git diff --check` passed.

The initial sandboxed full run failed because the parallel QE test's
multiprocessing manager could not create a local socket. The full rerun
with socket access passed. Flint was not checked. No benchmarks were rerun,
and the existing benchmark measurements and tables were retained.

The full-run log is `/tmp/logic1-bnf-mixin-test-unsandboxed-2026-11-02.log`;
the API documentation log is `/tmp/logic1-bnf-mixin-doc-2026-11-02.log`.
These temporary paths record provenance and are not permanent artifacts.

The subsequent terminology cleanup on 2026-11-02 renamed the batch operation
to `_syntactic_reduction` and added two cases covering batch and incremental
normalization and contradiction rejection. With the same Sage setup,
`make test` passed mypy and all 690 tests in 20.43 seconds; the separate
`make test-doc` run passed all 13 tests in 2.97 seconds. The logs are
`/tmp/logic1-bnf-syntactic-test-unsandboxed-2026-11-02.log` and
`/tmp/logic1-bnf-syntactic-doc-2026-11-02.log`. The rules, schedules, budgets,
and parent-order independence are unchanged. Flint was not checked.

### Ordering cleanup and reproducibility check (2026-11-02)

The cube-ordering lambda in Cartesian reconstruction and the mixin's static
method were consolidated into one module-level `_cube_key` function. All
existing sorting was retained; no caches, ordered representations, or new
configuration options were introduced.

A one-off experiment compared exact result representations before and after
this cleanup with the Sage backend. The RCF simplifier was assumed stable
and left unchanged; PyEDA was outside the investigation. The six inputs were
`T`, `F`, the six-cube cut witness, a product of two two-cube DNFs, a 3-by-3
grid requiring semantic subsumption, and a quantified example.

Each run covered DNF and CNF, all four Cartesian schedules, and budgets of
0, 1, 500, and `None` for the three simplifying schedules. `NONE` used its
supported default budget of 500. This gave 156 conversions per process.
Hash seeds 0, 1, 63, and 314159 produced four distinct atom-set iteration
orders. All exact outputs agreed across seeds, and all post-cleanup outputs
matched the pre-cleanup baseline: 1,248 conversions in total. The witnesses
produced three distinct results across schedules and three across budgets,
so the checks exercised choices that can affect the result.

This is evidence for the checked cases, not a proof for arbitrary inputs.
Flint was not checked. Sorting costs were not measured, and no benchmark
timings or tables were updated. The temporary driver is
`/tmp/logic1-bnf-repro-2026-11-02.py`; its JSON records are named
`/tmp/logic1-bnf-repro-{before,after}-seed{seed}-2026-11-02.json`.

After the cleanup, Sage `make test` passed mypy and all 690 tests; the
separate `make test-doc` run passed all 13 tests.

### Cube wrapper (2026-11-02)

The module-level ordering helper was an intermediate arrangement. `Cube` now
owns `sort_key`, `normalize`, `merge`, `implies`, and the sign-mask mappings.
It is a frozen, slotted dataclass containing one private
`frozenset[AtomicFormula]`. Equality is structural, and hashing delegates
directly to the frozenset. Construction removes duplicate atoms but does not
normalize constraints. `conjoin` and `combine` also leave normalization
explicit, preserving the four construction schedules.

The mixin retains insertion, batch syntactic reduction, bounded semantic
subsumption, and option initialization. Cartesian reconstruction and all
collection processing use `Cube.sort_key`. The three concrete classes still
inherit directly from the two independent parents. No sorting was removed,
and no caches or additional options were introduced.

#### Representation investigation

Before implementation, isolated wrapper and frozenset-subclass prototypes
moved the same operations into `Cube`. Both passed mypy, and both matched
the current public outputs on 156 small cases for each of two hash seeds:
936 checked outputs across the current representation and both prototypes.
The comparison covered DNF and CNF, all schedules, and the selected semantic
budgets. Strict construction also agreed on series3 and series12 for all four
schedules. Complete series12 DNF calls agreed for NONE and DURING_STEP.

Measured with Sage and Python 3.14.10, the wrapper adds 40 bytes per cube,
excluding the stored frozenset and referenced atoms. For the measured series12
strict DNFs, that added 29,840 bytes with NONE (746 cubes) and 15,760 bytes
with DURING_STEP (394 cubes). These are container measurements, not
whole-process peak memory. The subclass had no retained-container overhead
but required adapters because inherited set operations return plain
frozensets. A wrapper was chosen for its explicit interface and encapsulation.

The prototype study's three-fresh-process median times, in seconds, were:

| Series12 schedule | Current frozensets | Wrapper prototype | Subclass prototype |
| --- | ---: | ---: | ---: |
| NONE | 6.168 | 6.191 | 6.164 |
| DURING_STEP | 8.572 | 8.602 | 8.545 |

Recursive construction alone was approximately 3–5% slower with the wrapper
on series12; the observed complete-call differences were within run-to-run
variation. The temporary investigation script is
`/tmp/logic1-cube-study-2026-11-02.py`; detailed measurements are in
`/tmp/logic1-cube-micro-2026-11-02.json` and
`/tmp/logic1-cube-integrated-2026-11-02.json`.

#### Final implementation verification

The implemented wrapper passed mypy and all 705 regular tests and doctests
with Sage. The separate API documentation run passed all 13 tests. New tests
cover structural equality and hashing, deduplication, immutable storage,
pickle round trips, conjunction without automatic normalization, and the
individual normalization, merge, and implication operations.

A 624-conversion check covered the same six small inputs, all schedules and
supported selected budgets, DNF and CNF, and hash seeds 0, 1, 63, and 314159.
All exact outputs matched across seeds and matched the saved pre-wrapper
baseline. Records are
`/tmp/logic1-bnf-repro-cube-seed{seed}-2026-11-02.json`.

The final code was compared with a saved pre-wrapper source in fresh workers,
using three timed calls per representation and schedule on series12. Worker
startup, input construction, and output counting were outside timing; the
calls retained their ordinary preprocessing, final simplification, and
default semantic budget. Median times in seconds were:

| Series12 schedule | Pre-wrapper baseline | Implemented wrapper | Output atoms |
| --- | ---: | ---: | ---: |
| NONE | 6.143 | 6.207 | 4560 |
| DURING_STEP | 8.571 | 8.588 | 2116 |

All exact result representations matched. The observed median differences
were about 1.0% and 0.2%, respectively, smaller than the run-to-run variation.
These selected cases do not establish performance bounds for other inputs.
The records are `/tmp/logic1-cube-final-timings-2026-11-02.json`.
Neither the RCF simplifier nor PyEDA was modified. PyEDA was excluded from
the representation investigation and timed comparisons. Flint was not
checked. Existing benchmark-module tables retain their historical data.

### StrictDnf wrapper (2026-11-02)

An isolated prototype replaced the outer `set[Cube]` alias with a mutable,
slotted wrapper and moved insertion, syntactic reduction, and semantic
subsumption into it. Its batch methods preserve inputs. Insertion works
on the private set directly, avoiding delegated set operations inside its
comparison loop. No rules, sorting, schedules, defaults, or budgets changed.
The wrapper adds 40 bytes per DNF container on Python 3.14.10; this is not a
measurement of whole-process peak memory. It does not duplicate cubes.

The prototype matched 624 small-case output comparisons across hash seeds
0, 1, 63, and 314159 on each of Sage and Flint. These cover DNF and CNF, all
schedules, selected budgets, and witnesses distinguishing schedules and
budgets. Ownership, structural equality, nonmutation by batch passes, and
unhashability were also checked.

Complete series12 DNF calls used the default semantic budget of 500, hash
seed 0, three fresh-process runs per configuration, and alternating execution
order. Input creation, startup, and output counting were outside timing.
Median times in seconds and output atom counts were:

| Schedule | Sage current | Sage prototype | Flint current | Flint prototype | Atoms, all |
| --- | ---: | ---: | ---: | ---: | ---: |
| NONE | 6.289 | 6.218 | 7.475 | 7.425 | 4560 |
| FINAL | 7.850 | 7.781 | 18.563 | 18.515 | 2116 |
| AFTER_STEP | 8.064 | 8.047 | 20.429 | 20.563 | 2116 |
| DURING_STEP | 8.542 | 8.598 | 23.145 | 23.272 | 2116 |

Every exact result representation agreed. Construction alone was timed on a
preprocessed matrix, excluding the final semantic pass and reconstruction,
using five alternating repetitions in one process. Median seconds were:

| Schedule | Sage current | Sage prototype | Flint current | Flint prototype |
| --- | ---: | ---: | ---: | ---: |
| NONE | 0.003251 | 0.003636 | 0.003172 | 0.003517 |
| AFTER_STEP | 2.013391 | 2.013776 | 12.383674 | 12.420979 |
| DURING_STEP | 2.469750 | 2.474261 | 14.987632 | 15.358939 |

FINAL shares NONE's construction. There is small absolute overhead in plain
construction and a 2.5% during-step increase in the Flint construction sample.
Complete-call differences were at most about 1.1% in magnitude. These selected
measurements establish no general performance bounds. PyEDA was excluded.

Flint was selected in a temporary package copy; every complete-call worker
asserted `POLYLIB == "FLINT"`. The repository's selector remained on Sage.
Raw investigation records are `/tmp/strictdnf-timings.json`,
`/tmp/strictdnf-construction.json`, `/tmp/strictdnf-flint-timings.json`, and
`/tmp/strictdnf-flint-construction.json`. These tables describe the prototype
comparison and do not replace the benchmark modules' historical tables.

#### Implementation verification

The implemented wrapper passed Sage mypy checks (19 stub files and 68 source
files), all 697 regular tests and doctests, and the separate 13 API
documentation checks. The focused BNF suite passed 138 checks with both Sage
and Flint. The earlier 24 reversed-parent tests became 12 inherited-budget
and finishing tests; three collection tests and one doctest were added.
The existing schedule and semantic-budget witnesses remain in place.

The final source matched all 624 saved small-case outputs on each backend
across the four hash seeds, for 1,248 comparisons. Records are
`/tmp/strictdnf-final-{SAGE,FLINT}-seed{seed}.json`. Flint validation used a
temporary package copy with the final BNF source and the Flint selector;
the full repository-wide typing and test run used Sage.

Selected final-code series12 checks used one fresh-process call per row,
hash seed 0 and budget 500. Every exact output matched its saved baseline:

| Backend | Schedule | Seconds | Result atoms |
| --- | --- | ---: | ---: |
| Sage | NONE | 6.158 | 4560 |
| Sage | DURING_STEP | 8.394 | 2116 |
| Flint | NONE | 7.445 | 4560 |
| Flint | DURING_STEP | 23.025 | 2116 |

These are individual verification runs, not replacements for the repeated
prototype comparison above. Raw records are
`/tmp/strictdnf-final-timings.json`. PyEDA was not benchmarked.

### Large semantic-budget illustration

The former class doctests used a 20-by-20 grid with 400 cubes to demonstrate
the effect of budgets below, at, and above the default of 500:

```python
from logic1.firstorder import And, Or
from logic1.theories.RCF import VV
from logic1.theories.RCF.bnf import AfterStepSimplifyingCartesianBnf

x, y = VV.get('x', 'y')
g = Or(*(And(x > i, y > j) for i in range(20) for j in range(20)))
for checks in (100, 500, None):
    result = AfterStepSimplifyingCartesianBnf(max_pair_checks=checks).dnf(g)
    cubes = len(result.args) if isinstance(result, Or) else 1
    print(checks, cubes, len(list(result.atoms())))
```

Initial simplification retains all 400 cubes, and distinct polynomials prevent
structural subsumption. Both recursive schedules gave these recorded results:

| Pair budget | Result cubes | Result atoms |
| --- | ---: | ---: |
| 100 | 341 | 682 |
| 500 | 141 | 282 |
| `None` | 1 | 2 |

The unbounded pass examined 760 pairs. Previously recorded standalone call
times were approximately 0.5, 0.9, and 1.1 seconds, respectively. These are
historical measurements, not performance expectations. Under the parallel
regular test suite, each class doctest was reported at roughly four seconds.

On 2026-10-22, the executable class doctests were reduced to a 6-by-6 grid
with budgets 10, 30, and unbounded. They retain a meaningful quality comparison
(31, 19, and one result cube), while keeping the larger experiment here,
outside the regular test suite.

## Motor-series comparison

With initial Cartesian simplification set to `explode_always=False,
substitute=1`, the two classes again produced the same strict DNF sets on
all 13 motor series, before final semantic subsumption. Their final output
atom counts were also identical on every series. On series12 both results
changed from 2100 to 2116 atom occurrences; the other counts stayed the
same. These inputs provide no evidence that inserting cubes during product
generation improves the result over reducing each completed subtree.

The 2026-10-21 full run used one timed call per case. The subtree method was
faster on six series and slower on seven. On the largest case, series12,
early insertion took 8.735 seconds and subtree reduction took 8.174
seconds. The preceding run measured 7.845 and 8.071 seconds respectively,
illustrating run-to-run variation. These measurements do not establish a
reliable speed advantage for either method. Peak memory use was not
measured; completing a local product before reduction can require a larger
temporary set, and sorting cube sets creates temporary lists.

### Final-only comparison on 2026-10-22

The four Cartesian schedules were measured on all 13 motor inputs, alongside
Redlog off and on; PyEDA was excluded. All 78 cases completed within their
30-second computation limits. Final-only, after-step, and during-step
returned identical final atom counts on every motor input. Their total
single-call computation times were 49.111, 50.262, and 50.900 seconds,
respectively, compared with 21.102 seconds for plain Cartesian construction.
For series12, plain Cartesian returned 4560 atoms, while every simplifying
schedule returned 2116. The module docstring of `test_dnf_motor.py` records
all result sizes and times. The earlier strict-DNF equality checks above
concern the two recursive schedules; this run compared final atom counts.

### SMT-LIB comparison on 2026-10-22

On the eight SMT-LIB inputs, all three simplifying schedules returned the
same atom counts for every completed main case. On sumcracker-14e, plain
Cartesian returned 34,816 atoms in 14.397 seconds, while all three simplifying
schedules and both Redlog configurations exceeded 30 seconds. Thus the
final-only schedule also fails to improve this input within the main time
limit. No result size was obtained for those timed-out computations.
The separate 60-second budget experiments described in the SMT-LIB notes
concern the two recursive schedules and were not rerun for this refactoring.

## Implementation and maintenance trade-offs

Early insertion reduces cubes during `And` products and `Or` unions, so
it can avoid storing a complete local Cartesian product and can stop
when a partial result becomes `T` or `F`. That possible peak-memory
advantage has not been measured. Its construction method needs separate
branches for `And` and `Or`, intermediate products, insertion at each
step, and early exits. Repeated reduction of partial products can also
cost time.

Subtree reduction has a shorter construction method: build a node's
strict DNF, then reduce its cubes. It follows Redlog's simplification
placement and is easier to explain. It must, however, materialize the
complete local product, including cubes that reduction will discard.
Before the 2026-10-22 refactoring, the subtree subclass inherited the
reduction helpers from the early class and explicitly called
`CartesianBnf._strict_dnf`. That call recursively dispatched
back to the subtree subclass. The current sibling classes instead inherit
the helpers from a shared base; after-step uses `super()._strict_dnf`.
If only one simplifying schedule is eventually retained, its helpers could
be moved into that subclass. While comparing three schedules, the shared
base keeps each construction algorithm separate without duplicating rules.

The six-cube example in the class docstrings gives two cubes with early
insertion and four with subtree reduction, both at the default
`max_pair_checks=500`. This shows that reduction order can affect
quality, but does not establish that early insertion always gives the
smaller DNF. The motor series give equal strict DNFs and no clear
timing winner. The implementations share reduction helpers, so keeping
both adds little duplicate core code. It does require separate schedules
and corresponding documentation, tests, and benchmarks.

The historical evidence does not justify maintaining both as permanent
public schedules. Final-only is now included in the comparison before
choosing which schedules to retain. If one method is chosen chiefly for comprehensibility
and maintenance, subtree reduction is simpler. Because large formulas
and DNF growth are central here, peak memory should be measured on
inputs with large local products before removing early insertion.

## Dual CNF rules

For a CNF with the same set structure, each inner frozenset is a
*disjunction* (a clause), and the outer set is their conjunction.
Subsumption still keeps the subset: if $L_1\subseteq L_2$, then
$(\bigvee L_1)\land(\bigvee L_2)\equiv\bigvee L_1$.

For clauses $L_1$ and $L_2$, define $S=L_1\cap L_2$ and
$R_i=L_i\setminus S$, for $i=1,2$. The dual cut replaces the two
clauses by $S$ when

$$
(\bigvee R_1)\land(\bigvee R_2)\equiv F,
$$

because

$$
(\bigvee L_1)\land(\bigvee L_2)
\equiv (\bigvee S)\lor
\bigl((\bigvee R_1)\land(\bigvee R_2)\bigr)
\equiv \bigvee S.
$$

For CNF, the empty outer set represents `T`, while a CNF containing an
empty clause represents `F`.
