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

The two simplified Cartesian methods separate inexpensive atomic
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

`SimplifyingCartesianBooleanNormalForm` inserts cubes into a running
strict DNF at each `Or` node. At each `And` node, it multiplies child
DNFs one at a time, with insertion after each product. Insertion removes a
structurally subsumed cube or merges two cubes as above; new merge
results are reconsidered until they can be inserted. This greedy rule
can discard a cube that might have supported a later cut, as shown in
"Order and iteration" above. It keeps the algorithm and its storage
small, but does not promise a minimal DNF.

`SubtreeSimplifyingCartesianBooleanNormalForm` uses the same atomic
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

Both classes are in `RCF/bnf.py`. The public `dnf` and `cnf` functions
select them with `method="cartesian-simplified"` and
`method="cartesian-subtree-simplified"`, respectively; `cnf` uses the
DNF of the negated input. The base Cartesian implementation remains
available with `method="cartesian"`. The former witness pool and general
pairwise validity-cut search are not part of either simplified method.

## Motor-series comparison

With initial Cartesian simplification set to `explode_always=False,
substitute=1`, the two classes again produced the same strict DNF sets on
all 13 motor series, before final semantic subsumption. Their final output
atom counts were also identical on every series. On series12 both results
changed from 2100 to 2116 atom occurrences; the other counts stayed the
same. These inputs provide no evidence that inserting cubes during product
generation improves the result over reducing each completed subtree.

The latest full run used one timed call per case. The subtree method was
faster on six series and slower on seven. On the largest case, series12,
early insertion took 8.735 seconds and subtree reduction took 8.174
seconds. The preceding run measured 7.845 and 8.071 seconds respectively,
illustrating run-to-run variation. These measurements do not establish a
reliable speed advantage for either method. Peak memory use was not
measured; completing a local product before reduction can require a larger
temporary set, and sorting cube sets creates temporary lists.

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
The current subclass inherits the reduction helpers from the early
class and explicitly calls `CartesianBooleanNormalForm._strict_dnf`.
That call recursively dispatches back to the subclass, which makes the
inheritance less obvious than the algorithm itself. If subtree reduction
were the only method, it could instead be a direct subclass of
`CartesianBooleanNormalForm` with those helpers moved into it. Removing
subtree is the smaller immediate edit; retaining subtree alone would
remove the longer early construction method after that refactoring.

The six-cube example in the class docstrings gives two cubes with early
insertion and four with subtree reduction, both at the default
`max_pair_checks=500`. This shows that reduction order can affect
quality, but does not establish that early insertion always gives the
smaller DNF. The motor series give equal strict DNFs and no clear
timing winner. The implementations share reduction helpers, so keeping
both adds little duplicate core code. It does require two public method
names and corresponding documentation, tests, and benchmarks.

The present evidence does not justify maintaining both as permanent
public methods. If one method is chosen chiefly for comprehensibility
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
