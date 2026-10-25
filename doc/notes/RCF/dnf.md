# Simplifying a strict DNF

These notes concern a strict DNF represented as
`set[frozenset[AtomicFormula]]`. Each inner frozenset is a conjunction
(a *cube*); the outer set is a disjunction of cubes. Thus, for a DNF `N`,

$$
\llbracket N \rrbracket
= \bigvee_{C \in N}\;\bigwedge_{a \in C} a.
$$

The empty outer set represents `F`. A DNF containing the empty frozenset
represents `T`, because the empty conjunction is true. In particular,
`set()` represents `F`, while `{frozenset()}` represents `T`.

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

`DuringStepSimplifyingCartesianBooleanNormalForm` inserts cubes into a running
strict DNF at each `Or` node. At each `And` node, it multiplies child
DNFs one at a time, with insertion after each product. Insertion removes a
structurally subsumed cube or merges two cubes as above; new merge
results are reconsidered until they can be inserted. This greedy rule
can discard a cube that might have supported a later cut, as shown in
"Order and iteration" above. It keeps the algorithm and its storage
small, but does not promise a minimal DNF.

`AfterStepSimplifyingCartesianBooleanNormalForm` uses the same atomic
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

`SimplifyingCartesianBooleanNormalForm` is the shared base for
`FinalSimplifyingCartesianBooleanNormalForm`,
`AfterStepSimplifyingCartesianBooleanNormalForm`, and
`DuringStepSimplifyingCartesianBooleanNormalForm`. It provides the reduction
rules, deterministic ordering, and bounded final semantic pass. The final-only
class inherits plain Cartesian recursion and extends the final pass with cube
normalization, structural subsumption, and cut. The other classes select the
placement of those reductions through `_strict_dnf`. They are siblings;
after-step no longer inherits the during-step construction algorithm.

All three simplifying schedules use `max_pair_checks=500` for final semantic
subsumption; zero skips that pass and `None` removes the limit. `NONE` and
PyEDA require the default budget because neither uses this pass. CNF uses the
DNF of the negation. The former witness pool and general pairwise validity-cut
search are not part of these implementations.

In the measurements below, **early** means `DURING_STEP` and **subtree** means
`AFTER_STEP`. Those historical labels and dates are retained when describing
the original experiments. The benchmark modules use the current schedule
names for new measurements.

### Large semantic-budget illustration

The former class doctests used a 20-by-20 grid with 400 cubes to demonstrate
the effect of budgets below, at, and above the default of 500:

```python
from logic1.firstorder import And, Or
from logic1.theories.RCF import VV
from logic1.theories.RCF.bnf import AfterStepSimplifyingCartesianBooleanNormalForm

x, y = VV.get('x', 'y')
g = Or(*(And(x > i, y > j) for i in range(20) for j in range(20)))
for checks in (100, 500, None):
    result = AfterStepSimplifyingCartesianBooleanNormalForm(max_pair_checks=checks).dnf(g)
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
`CartesianBooleanNormalForm._strict_dnf`. That call recursively dispatched
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
