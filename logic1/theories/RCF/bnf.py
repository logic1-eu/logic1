"""Boolean normal forms over real closed fields.

The Cartesian methods distribute conjunction over disjunction directly. The
``cartesian-simplified`` method simplifies during construction, while
``cartesian-subtree-simplified`` simplifies each completed subtree. The
``pyeda`` method uses a Boolean abstraction.
"""
from collections.abc import Mapping
import itertools
from typing import Final, Literal, NewType, TypeGuard

from logic1.firstorder import And, Or, Not, _F, _T, BooleanFormula
from logic1 import abc
from logic1.theories.RCF.term import Term, Variable
from logic1.theories.RCF.atomic import AtomicFormula, Eq, Ge, Gt, Le, Lt, Ne
from logic1.theories.RCF.simplify import is_valid as _is_valid, simplify as _simplify
from logic1.theories.RCF.types import Formula, Number


AndOr = NewType("AndOr", Formula)


def is_and_or(f: Formula) -> TypeGuard[AndOr]:
    if isinstance(f, (_F, _T, AtomicFormula)):
        return True
    if isinstance(f, (And, Or)):
        return all(is_and_or(arg) for arg in f.args)
    return False


class CartesianBooleanNormalForm:

    def cnf(self, f: Formula) -> Formula:
        """Compute a conjunctive normal form. If ``f`` contains quantifiers,
        then the result is a prenex normal form whose matrix is in CNF.
        """
        return self.conservative_simplify(Not(self._dnf(Not(f))).to_nnf())

    def dnf(self, f: Formula) -> Formula:
        """Compute a disjunctive normal form. If ``f`` contains quantifiers,
        then the result is a prenex normal form whose matrix is in DNF.
        """
        return self.conservative_simplify(self._dnf(f))

    def _dnf(self, f: Formula) -> Formula:
        f = self.simplify(f.to_pnf())
        matrix, prefix = f.matrix()
        assert is_and_or(matrix)
        dnf = self._dnf_and_or(matrix)
        return dnf.quantify(prefix)

    def _dnf_and_or(self, f: AndOr) -> Formula:
        sdnf = self._strict_dnf(f)
        dnf = Or(*(And(*(atom for atom in conj)) for conj in sdnf))
        return dnf

    def conservative_simplify(self, f: Formula) -> Formula:
        return _simplify(f, explode_always=False)

    def simplify(self, f: Formula) -> Formula:
        """Preprocess without unconditional splitting or monomial substitution.

        Value substitution remains enabled. Monomial substitution can create
        extra disjuncts before Cartesian distribution.
        """
        return _simplify(f, explode_always=False, substitute=1)

    def _strict_dnf(self, f: AndOr) -> set[frozenset[AtomicFormula]]:
        if isinstance(f, _F):
            return set()
        if isinstance(f, _T):
            return {frozenset()}
        if isinstance(f, AtomicFormula):
            return {frozenset((f,))}
        if isinstance(f, And):
            strict_dnfs = (self._strict_dnf(arg) for arg in f.args)
            result = set()
            for tuple in itertools.product(*strict_dnfs):
                result.add(frozenset.union(*tuple))
            return result
        if isinstance(f, Or):
            result = set()
            for arg in f.args:
                result.update(self._strict_dnf(arg))
            return result
        raise ValueError(f"Unexpected formula type: {type(f)}")


type Cube = frozenset[AtomicFormula]
type StrictDNF = set[Cube]


class SimplifyingCartesianBooleanNormalForm(CartesianBooleanNormalForm):
    """Reduce cubes as they are produced by Cartesian distribution.

    The inherited DNF computation first prenexes and simplifies the input.
    Its matrix contains only ``And``, ``Or``, truth values, and atomic
    relations against zero. A strict DNF is a set of cubes, each a
    frozenset of atoms interpreted as a conjunction.

    At an ``Or`` node, insert child cubes into a running result and stop
    if the empty cube makes it true. At an ``And`` node, fold in child
    DNFs one at a time. Each pair of cubes produces a union of atoms;
    normalize it when both factors are nonempty, then try to insert it
    into the next result. Stop if that result becomes false. Cube
    normalization intersects the allowed signs of relations on the same
    polynomial and rejects contradictions. Insertion removes structurally
    subsumed cubes and repeatedly merges cubes that differ in one relation
    on the same polynomial. These recursive steps use no RCF validity
    checks. Cube traversal and greedy comparison choices follow
    :meth:`_cube_key`, independent of Python set iteration order.

    After construction, a semantic pass examines at most
    ``max_pair_checks`` cube pairs (500 by default; zero skips the pass;
    ``None`` is unbounded). It checks implications first by exact atoms
    and sign masks, then by RCF simplification under assumptions. The
    resulting DNF is converted to a formula and conservatively
    simplified. Inherited CNF computation uses the DNF of the negation.

    Early insertion can avoid storing a full local Cartesian product when
    many cubes collapse, and can stop on ``T`` or ``F`` during recursion.
    Its repeated subset and merge checks can also cost time. Greedy
    deletion can prevent a later cut, so minimality is not guaranteed.
    On the 13 motor series it produced the same strict DNFs as
    :class:`SubtreeSimplifyingCartesianBooleanNormalForm`, with no
    consistent timing advantage. See ``doc/notes/RCF/dnf.md``.

    In the following input, the initial simplification retains six cubes.
    Both classes cut on ``d``. Early insertion also cuts on ``a`` and then
    ``b``, producing a cube that subsumes the remaining ``c`` cube. The
    completed-subtree class subsumes an input to those cuts first. Both
    calls use the default ``max_pair_checks=500``:

    >>> from logic1.theories.RCF import VV
    >>> a, b, c, d, e = VV.get('a', 'b', 'c', 'd', 'e')
    >>> f = Or(And(a > 0, c > 0), And(a > 0, b > 0, c > 0),
    ...        And(a <= 0, b > 0, c > 0), And(b <= 0, c > 0),
    ...        And(d > 0, e > 0), And(d <= 0, e > 0))
    >>> normal_form = SimplifyingCartesianBooleanNormalForm()
    >>> len(normal_form.simplify(f.to_pnf()).args)
    6
    >>> normal_form.dnf(f)
    Or(e > 0, c > 0)
    >>> other = SubtreeSimplifyingCartesianBooleanNormalForm()
    >>> for clause in other.dnf(f).args:
    ...     print(clause)
    e > 0
    c > 0 and b <= 0
    c > 0 and b > 0 and a <= 0
    c > 0 and a > 0

    A grid of polynomial constraints shows the quality and cost of the
    semantic pair budget. Initial simplification retains all 400 cubes;
    distinct polynomials also prevent structural subsumption. Higher
    budgets find more semantic implications. On one machine, the three
    calls took about 0.5, 0.9, and 1.1 seconds, respectively. The
    unbounded pass examines 760 cube pairs. The output columns are
    budget, cubes, and atoms:

    >>> x, y = VV.get('x', 'y')
    >>> g = Or(*(And(x > i, y > j) for i in range(20) for j in range(20)))
    >>> len(normal_form.simplify(g.to_pnf()).args)
    400
    >>> BNF = SimplifyingCartesianBooleanNormalForm
    >>> for checks in (100, 500, None):
    ...     result = BNF(max_pair_checks=checks).dnf(g)
    ...     cubes = len(result.args) if isinstance(result, Or) else 1
    ...     print(checks, cubes, len(list(result.atoms())))
    100 341 682
    500 141 282
    None 1 2
    """

    max_pair_checks: int | None

    # The bits represent negative, zero, and positive signs of the left-hand
    # side. _dnf simplifies every atom to an equation or inequality against 0.
    _SIGN_MASK: Final[Mapping[type[AtomicFormula], int]] = {
        Eq: 0b010, Ne: 0b101, Lt: 0b001, Le: 0b011, Gt: 0b100, Ge: 0b110}
    _MASK_OP: Final[Mapping[int, type[AtomicFormula]]] = {
        0b001: Lt, 0b010: Eq, 0b011: Le, 0b100: Gt, 0b101: Ne, 0b110: Ge}

    def __init__(self, *, max_pair_checks: int | None = 500) -> None:
        """Limit final semantic cube comparisons; ``None`` is unbounded.
        """
        if max_pair_checks is not None:
            if isinstance(max_pair_checks, bool) or not isinstance(max_pair_checks, int):
                raise TypeError("max_pair_checks must be an integer or None")
            if max_pair_checks < 0:
                raise ValueError("max_pair_checks must be nonnegative or None")
        self.max_pair_checks = max_pair_checks

    @staticmethod
    def _cube_key(cube: Cube) -> tuple[int, tuple[AtomicFormula, ...]]:
        """Order cubes first by size, then by their sorted atoms.
        """
        return len(cube), tuple(sorted(cube))

    def _dnf_and_or(self, f: AndOr) -> Formula:
        """Build a DNF and apply final semantic subsumption.
        """
        strict_dnf = self._simplify_strict_dnf(self._strict_dnf(f))
        return Or(*(And(*sorted(cube)) for cube in sorted(strict_dnf, key=self._cube_key)))

    def _implies(self, premise: Cube, conclusion: Cube) -> bool:
        """Check whether a cube implies another by RCF simplification.
        """
        premise_atoms = sorted(premise)
        remaining = []
        for atom in sorted(conclusion):
            if atom in premise:
                continue
            if any(other.lhs == atom.lhs
                   and self._SIGN_MASK[other.op] & self._SIGN_MASK[atom.op]
                   == self._SIGN_MASK[other.op] for other in premise_atoms):
                continue
            remaining.append(atom)
        if not remaining:
            return True
        return _is_valid(And(*remaining), assume=premise_atoms, explode_always=False) is True

    def _insert(self, result: StrictDNF, cube: Cube) -> None:
        """Insert a cube, choosing the lowest-key applicable comparison.

        A single scan makes the same choice as sorting the result set without
        allocating a sorted copy at each insertion. Repeat after each change.
        """
        if frozenset() in result:
            return
        while True:
            if not cube:
                result.clear()
                result.add(cube)
                return
            selected: Cube | None = None
            selected_key: tuple[int, tuple[AtomicFormula, ...]] | None = None
            selected_merge: Cube | None = None
            selected_discard = False
            for other in result:
                if other <= cube:
                    merged = None
                    discard = True
                elif cube <= other:
                    merged = None
                    discard = False
                else:
                    merged = self._merge(cube, other)
                    if merged is None:
                        continue
                    discard = False
                key = self._cube_key(other)
                if selected_key is None or key < selected_key:
                    selected = other
                    selected_key = key
                    selected_merge = merged
                    selected_discard = discard
            if selected is None:
                result.add(cube)
                return
            if selected_discard:
                return
            result.remove(selected)
            if selected_merge is not None:
                cube = selected_merge

    @classmethod
    def _merge(cls, left: Cube, right: Cube) -> Cube | None:
        """Merge cubes differing only in one relation on the same polynomial.
        """
        different = left ^ right
        if len(left) != len(right) or len(different) != 2:
            return None
        atom_left, atom_right = tuple(different)
        if atom_left.lhs != atom_right.lhs:
            return None
        common = left & right
        mask = cls._SIGN_MASK[atom_left.op] | cls._SIGN_MASK[atom_right.op]
        if mask == 0b111:
            return common
        return common | {cls._MASK_OP[mask](atom_left.lhs, 0)}

    def _simplify_cube(self, cube: Cube) -> Cube | None:
        """Intersect relations on each polynomial, rejecting contradictions.
        """
        signs: dict[Term, int] = {}
        for atom in cube:
            mask = signs.get(atom.lhs, 0b111) & self._SIGN_MASK[atom.op]
            if not mask:
                return None
            signs[atom.lhs] = mask
        return frozenset(self._MASK_OP[mask](lhs, 0) for lhs, mask in signs.items())

    def _simplify_strict_dnf(self, strict_dnf: StrictDNF) -> StrictDNF:
        """Remove semantically subsumed cubes from a reduced strict DNF.

        Compare each pair once. A successful implication removes the stronger
        cube. The budget applies to pairs examined here, not to cheap atomic
        simplification during recursive construction.
        """
        cubes = sorted(strict_dnf, key=self._cube_key)
        result = set(strict_dnf)
        pair_checks = 0
        for i, left in enumerate(cubes):
            if left not in result:
                continue
            for right in cubes[i + 1:]:
                if right not in result:
                    continue
                if self.max_pair_checks is not None and pair_checks >= self.max_pair_checks:
                    return result
                pair_checks += 1
                if self._implies(right, left):
                    result.remove(right)
                elif self._implies(left, right):
                    result.remove(left)
                    break
        return result

    def _strict_dnf(self, f: AndOr) -> StrictDNF:
        """Distribute recursively, reducing each intermediate strict DNF.
        """
        if isinstance(f, (AtomicFormula, _T, _F)):
            return super()._strict_dnf(f)
        if isinstance(f, Or):
            result: StrictDNF = set()
            for arg in f.args:
                for cube in sorted(self._strict_dnf(arg), key=self._cube_key):
                    self._insert(result, cube)
                if frozenset() in result:
                    break
            return result
        if isinstance(f, And):
            result = {frozenset()}
            for arg in f.args:
                child = sorted(self._strict_dnf(arg), key=self._cube_key)
                product: StrictDNF = set()
                for left in sorted(result, key=self._cube_key):
                    for right in child:
                        combined = left | right
                        cube = self._simplify_cube(combined) if left and right else combined
                        if cube is not None:
                            self._insert(product, cube)
                result = product
                if not result:
                    break
            return result
        raise ValueError(f"Unexpected formula type: {type(f)}")


class SubtreeSimplifyingCartesianBooleanNormalForm(SimplifyingCartesianBooleanNormalForm):
    """Reduce each subtree after its local Cartesian expansion is complete.

    Recursion returns reduced strict DNFs for the children. An ``Or`` node
    first unions their cubes; an ``And`` node first forms their full
    Cartesian product. Only after that node's raw strict DNF exists does
    this class simplify each cube and insert it into a reduced set.
    Inherited rules combine relations on the same polynomial, reject
    contradictory cubes, remove structural subsumption, and repeatedly
    merge cubes differing in one relation. As in the parent class, the
    final semantic pass examines at most ``max_pair_checks`` cube pairs.
    The reduced strict DNF is converted to a formula and conservatively
    simplified. Cube traversal and inherited insertion use the same stable
    key as the parent class. Inherited CNF computation uses the DNF of the
    negation.

    This ordering matches the placement of Redlog's recursive
    simplification, though the rules used here are different. It avoids
    reducing partial products during an ``And`` node's expansion and gives
    each node a simpler reduction schedule. The complete local
    product must be stored before redundant or contradictory cubes are
    removed, which can increase peak memory and construction time. On
    the 13 motor series it gave exactly the same strict DNFs as the
    parent class; one-call timings gave neither method a clear advantage.
    See ``doc/notes/RCF/dnf.md`` for the rules and measurements.

    The initial simplification leaves all six cubes below. Both classes
    cut on ``d``; the parent class also cuts on ``a`` and ``b`` before
    subsumption can remove the needed inputs. These calls use the default
    ``max_pair_checks=500``:

    >>> from logic1.theories.RCF import VV
    >>> a, b, c, d, e = VV.get('a', 'b', 'c', 'd', 'e')
    >>> f = Or(And(a > 0, c > 0), And(a > 0, b > 0, c > 0),
    ...        And(a <= 0, b > 0, c > 0), And(b <= 0, c > 0),
    ...        And(d > 0, e > 0), And(d <= 0, e > 0))
    >>> normal_form = SubtreeSimplifyingCartesianBooleanNormalForm()
    >>> len(normal_form.simplify(f.to_pnf()).args)
    6
    >>> for clause in normal_form.dnf(f).args:
    ...     print(clause)
    e > 0
    c > 0 and b <= 0
    c > 0 and b > 0 and a <= 0
    c > 0 and a > 0
    >>> SimplifyingCartesianBooleanNormalForm().dnf(f)
    Or(e > 0, c > 0)

    The same grid shows the quality and cost of the semantic pair budget
    for this class. Initial simplification retains all 400 cubes. On one
    machine, budgets 100, 500, and unbounded took about 0.5, 0.9, and
    1.1 seconds, respectively. The unbounded pass examines 760 cube
    pairs. The output columns are budget, cubes, and atoms:

    >>> x, y = VV.get('x', 'y')
    >>> g = Or(*(And(x > i, y > j) for i in range(20) for j in range(20)))
    >>> len(normal_form.simplify(g.to_pnf()).args)
    400
    >>> BNF = SubtreeSimplifyingCartesianBooleanNormalForm
    >>> for checks in (100, 500, None):
    ...     result = BNF(max_pair_checks=checks).dnf(g)
    ...     cubes = len(result.args) if isinstance(result, Or) else 1
    ...     print(checks, cubes, len(list(result.atoms())))
    100 341 682
    500 141 282
    None 1 2
    """

    def _strict_dnf(self, f: AndOr) -> StrictDNF:
        """Reduce a node after constructing its full Cartesian result.

        The base method calls ``self._strict_dnf`` for its children, so each
        child has already been reduced before the parent is expanded.
        """
        strict_dnf = CartesianBooleanNormalForm._strict_dnf(self, f)
        if isinstance(f, (AtomicFormula, _T, _F)):
            return strict_dnf
        result: StrictDNF = set()
        for cube in sorted(strict_dnf, key=self._cube_key):
            simplified = self._simplify_cube(cube)
            if simplified is not None:
                self._insert(result, simplified)
        return result


class PyedaBooleanNormalForm(abc.bnf.BooleanNormalForm[AtomicFormula, Term, Variable, Number]):
    """Implements the abstract methods :meth:`simplify
    <.abc.bnf.BooleanNormalForm.simplify>` and :meth:`final_simplify
    <.abc.bnf.BooleanNormalForm.final_simplify>` of its super class
    :class:`.abc.bnf.BooleanNormalForm`. Inherits the methods :meth:`cnf
    <.abc.bnf.BooleanNormalForm.cnf>` and :meth:`dnf
    <.abc.bnf.BooleanNormalForm.dnf>`, which should be called via the functions
    :func:`.cnf` and :func:`.dnf`, respectively.
    """

    def simplify(self, f: Formula) -> Formula:
        """Implement :meth:`.abc.bnf.BooleanNormalForm.simplify`.
        """
        return _simplify(f)

    def conservative_simplify(self, f: Formula) -> Formula:
        """Simplify while preserving the computed normal form.
        """
        return _simplify(f, explode_always=False)

    def final_simplify(self, f: Formula) -> Formula:
        """Implement the hook required by :class:`.abc.bnf.BooleanNormalForm`.
        """
        return self.conservative_simplify(f)


_cartesian_bnf = CartesianBooleanNormalForm()

_pyeda_bnf = PyedaBooleanNormalForm()
"""One instance of the PyedaBooleanNormalForm class used for both CNF and DNF
computations.
"""


def cnf(f: Formula,
        method: Literal["cartesian", "cartesian-simplified",
                        "cartesian-subtree-simplified", "pyeda"] = "pyeda",
        *, max_pair_checks: int | None = 500) -> Formula:
    """Compute a CNF. The limit applies to the simplified Cartesian methods.
    """
    if method == "cartesian-simplified":
        return SimplifyingCartesianBooleanNormalForm(max_pair_checks=max_pair_checks).cnf(f)
    if method == "cartesian-subtree-simplified":
        return SubtreeSimplifyingCartesianBooleanNormalForm(max_pair_checks=max_pair_checks).cnf(f)
    if max_pair_checks != 500:
        raise ValueError("Simplification limits require a simplified Cartesian method")
    if method == "cartesian":
        return _cartesian_bnf.cnf(f)
    if method == "pyeda":
        return _pyeda_bnf.cnf(f)
    else:
        raise ValueError(f"Unknown method: {method}")


def dnf(f: Formula,
        method: Literal["cartesian", "cartesian-simplified",
                        "cartesian-subtree-simplified", "pyeda"] = "pyeda",
        *, max_pair_checks: int | None = 500) -> Formula:
    """Compute a DNF. The limit applies to the simplified Cartesian methods.
    """
    if method == "cartesian-simplified":
        return SimplifyingCartesianBooleanNormalForm(max_pair_checks=max_pair_checks).dnf(f)
    if method == "cartesian-subtree-simplified":
        return SubtreeSimplifyingCartesianBooleanNormalForm(max_pair_checks=max_pair_checks).dnf(f)
    if max_pair_checks != 500:
        raise ValueError("Simplification limits require a simplified Cartesian method")
    if method == "cartesian":
        return _cartesian_bnf.dnf(f)
    if method == "pyeda":
        return _pyeda_bnf.dnf(f)
    else:
        raise ValueError(f"Unknown method: {method}")
