# Implicit ranges and the size of a computed DNF

The SMT-LIB benchmark levin-rhsc produces a DNF with 88 atom occurrences
when initial simplification uses `implicit_ranges=True`, but only 16 when
it uses `False`. This occurs with cartesian, early, and subtree.

The cause is an interaction between preprocessing and later inference.
Implicit ranges reduce the initial formula more effectively, including
collapsing explicit negative and positive variable cases into a
disequality. After distribution, the remaining product constraints still
imply the relevant variable signs, but the current interval engine cannot
recover them. Keeping the original cases makes those signs explicit and
allows the later simplification to eliminate more contradictions.

The two final formulas are equivalent. The issue concerns simplification
quality, and does not indicate an unsound interval deduction.

## Scope of the comparison

The measurements below were checked on 2026-10-21. The input is
`DDC_ProveIneq_ISSAC05_LevinRHSc.smt2` in
`benchmarks/theories/RCF/smtlib/inputs/20220314-Uncu/` under
`Rational_Function_Proof_Calls_With_Disjunctive_Denominator_Simplifications/`.
The SMT-LIB benchmark directory's `NOTES.md` records its source and credits.

The experimental setting affects only
`CartesianBooleanNormalForm.simplify`, called at the beginning of `_dnf`.
Both variants use `explode_always=False` and `substitute=1` there.
Semantic implication checks and `conservative_simplify` retain their
defaults, including `implicit_ranges=True`. Early and subtree use
`max_pair_checks=500` unless stated otherwise.

Here early denotes `SimplifyingCartesianBooleanNormalForm`, and subtree
denotes `SubtreeSimplifyingCartesianBooleanNormalForm`, both in `bnf.py`.

An atom count means occurrences, including repetitions in different
cubes. A strict DNF is a set of cubes; its atom count is the sum of their
cardinalities.

## What the option does

`Options.implicit_ranges` in `simplify.py` documents the use of explicit
variable bounds, algebraic properties such as non-negativity of squares,
and interval arithmetic to derive bounds on complex terms. Disabling the
option is described as reducing computation time at the price of less
simplification.

The relevant implementation has three parts:

1. `_Knowledge.get` intersects a term's explicit range with its implicit
   range when the option is enabled.
2. `_Knowledge._term_as_range` evaluates an expanded polynomial using
   interval arithmetic. For each monomial, it obtains bounds on its
   individual variables from `self.dict_`.
3. `InternalRepresentation.extract` also uses implicit ranges to avoid
   emitting constraints already implied by the available variable bounds.

An explicit range on a compound term is available when that exact term
is queried. It is not propagated backwards to its variables, nor generally
used as a building block when evaluating a different compound term.

In this example, the documented expectation holds for the initial
simplification: enabling the option reduces the formula from 43 to 26
atom occurrences. It does not imply monotonicity of the size produced by
the subsequent DNF computation.

## Where the counts diverge

Early and subtree have identical counts at every recorded stage:

| Stage | Initial option `True`: atoms | Initial option `False`: atoms |
| --- | ---: | ---: |
| Original input | 46 | 46 |
| Initial simplification | 26 | 43 |
| Recursive DNF construction | 88 | 832 |
| Semantic subsumption | 88 | 26 |
| Final simplification | **88** | **16** |

With the option enabled, recursive construction produces eight cubes,
each containing eleven atoms. All eight survive the remaining stages.
With the option disabled initially, recursive construction produces 64
cubes with thirteen atoms each. Semantic subsumption retains two cubes;
final simplification reduces each of them to eight atoms.

Plain cartesian shows the same final discrepancy without semantic
subsumption. Its strict DNF has 16 cubes and 184 atoms with the option
enabled initially, versus 256 cubes and 3,584 atoms with it disabled.
Final simplification produces eight cubes and 88 atoms, versus two cubes
and 16 atoms, respectively.

## The lost case distinction

The input requires

$$
V_1 \geq 0, \qquad 0 < x \leq 1.
$$

Some denominator sign conditions explicitly distinguish `V2 < 0` from
`V2 > 0`. With implicit ranges enabled, preprocessing can remove other
constraints within those cases and combine the alternatives into
`V2 != 0`. This is a correct simplification.

Other denominator conditions still distinguish signs of the products

$$
p_0 = V_2x, \qquad
p_2 = (V_1+2)V_2x, \qquad
p_4 = (V_1+4)V_2x.
$$

Since $V_1+2$, $V_1+4$, and $x$ are positive, all three products have the
same sign as $V_2$. The enabled result nevertheless contains all eight
strict sign combinations of $p_0$, $p_2$, and $p_4$. Only `---` and `+++`
can be consistent; the other six cubes are contradictory.

The interval engine cannot recover the sign of `V2` from `V2*x > 0`
and `x > 0`. Consequently, it cannot establish the sign of the expanded
polynomial `V1*V2*x + 2*V2*x` from those assumptions. Direct checks give:

| Assumptions | Simplification of `(V1 + 2)*V2*x > 0` |
| --- | --- |
| `V1 >= 0`, `x > 0`, `V2*x > 0` | The atom remains |
| `V1 >= 0`, `x > 0`, `V2 > 0` | `T` |

With implicit ranges disabled initially, the explicit `V2 < 0` and
`V2 > 0` cases survive until distribution. The resulting cubes provide
the variable bounds needed by the later interval computations. These
computations detect contradictory product signs and remove redundant
product constraints.

No logical information was lost by preprocessing. The equivalent
representation makes some consequences harder for the available
inference rules to derive.

## A small reproducer

The same effect occurs with two variables. Define

$$
S(p,q) = (p<0 \land q<0) \lor (p>0 \land q>0)
$$

and consider

$$
\varphi = x>0 \land S(b,bx) \land S(bx,bx+b).
$$

Under $x>0$, both $x$ and $x+1$ are positive. Thus $b$, $bx$, and
$bx+b=b(x+1)$ have the same sign, and

$$
\varphi \equiv x>0 \land b\neq0.
$$

With implicit ranges enabled initially, `S(b, b*x)` simplifies to
`b != 0`. The final DNF is

$$
\begin{aligned}
&(x>0 \land b\neq0 \land bx<0 \land bx+b<0)\\
{}\lor{}\;&(x>0 \land b\neq0 \land bx>0 \land bx+b>0),
\end{aligned}
$$

with eight atom occurrences. With implicit ranges disabled initially,
the final DNF is

$$
(x>0 \land b<0) \lor (x>0 \land b>0),
$$

with four atom occurrences. Both are equivalent to the two-atom formula
above; neither reported result is claimed to be minimal.

The following code reproduces the counts using the current early method:

```python
from logic1.firstorder import And, Or
from logic1.theories.RCF import VV
from logic1.theories.RCF.bnf import SimplifyingCartesianBooleanNormalForm
from logic1.theories.RCF.simplify import simplify


class WithoutInitialImplicitRanges(SimplifyingCartesianBooleanNormalForm):
    def simplify(self, f):
        return simplify(f, explode_always=False, substitute=1,
                        implicit_ranges=False)


def same_sign(p, q):
    return Or(And(p < 0, q < 0), And(p > 0, q > 0))


x, b = VV.get('x', 'b')
f = And(x > 0, same_sign(b, b*x), same_sign(b*x, b*x + b))

for normal_form in (SimplifyingCartesianBooleanNormalForm(),
                    WithoutInitialImplicitRanges()):
    pre = normal_form.simplify(f)
    result = normal_form.dnf(f)
    print(len(list(pre.atoms())), len(list(result.atoms())))
```

Output columns are initial simplified atoms and final DNF atoms:

```text
6 8
9 4
```

## Checks on levin-rhsc

The following checks distinguish the inference limitation from a budget
effect or a difference between the DNF construction methods:

- Every cube of the enabled result explicitly contains `V2 != 0`.
  Splitting each cube by `V2 < 0` and `V2 > 0` is therefore exhaustive.
  Simplifying the resulting disjunction produces exactly the same formula
  as the disabled run, using structural formula equality for the check.
- For each of the six mixed product-sign combinations, both added `V2`
  sign cases simplify to `F`. For `---` and `+++`, only the corresponding
  `V2` sign case remains, with eight atoms.
- The enabled run checks all 28 unordered pairs of its eight cubes and
  removes none. Setting `max_pair_checks=None` gives the same result.
- The disabled run checks 63 pairs and removes 62 cubes. It also reaches
  the same 16-atom final result with `max_pair_checks=0`, because final
  simplification can eliminate the contradictory cases itself.
- Applying `conservative_simplify` again changes neither final result.
- Plain cartesian produces the same two final sizes, as recorded above.

## Implications for the simplifier

The immediate limitation is the absence of propagation from product
signs to variable signs. A targeted improvement would derive `b > 0`
from `x > 0` and `b*x > 0`, with the corresponding rules for other signs.
Context-sensitive cancellation of factors with known signs is another
way to expose the same information. Either approach would need to handle
zero and weak inequalities correctly and make inferred bounds available
to subsequent simplification.

Such a change should be evaluated for both inference cost and its effect
on later computations. The sign split used above is diagnostic evidence;
unrestricted splitting would risk the DNF growth that preprocessing is
intended to prevent. Increasing `max_pair_checks` or repeating the
existing final simplification does not address this case.

The documentation's statement about more simplification describes the
additional deductions enabled by interval arithmetic. It should not be
read as a guarantee that every later computation on the simplified
formula produces fewer atoms. In particular, a locally smaller formula
can remove explicit cases on which later heuristic inference depends.
