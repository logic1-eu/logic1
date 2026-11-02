"""Boolean normal forms over real closed fields.

The default ``cartesian`` method distributes conjunction over disjunction
directly and simplifies the completed strict DNF with ``Simplification.FINAL``.
Other schedules select no additional simplification or simplification after
or during each recursive step. The ``pyeda`` method uses a Boolean abstraction
and requires ``Simplification.NONE``.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import IntEnum
import itertools
from typing import ClassVar, Final, Literal, NewType, TypeGuard

from logic1.firstorder import And, Or, Not, _F, _T
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


@dataclass(frozen=True, slots=True, init=False)
class Cube:
    """An immutable conjunction of RCF atoms, stored as a frozenset.

    In Boolean logic, a cube is a conjunction of literals, dual to a clause,
    which is a disjunction of literals. A DNF is a disjunction of cubes,
    just as a CNF is a conjunction of clauses. Unlike a minterm, a cube need
    not mention every propositional variable. The name comes from viewing
    partial Boolean assignments as faces of the Boolean hypercube.

    Here a cube contains polynomial constraints, with negation expressed by
    complementary atomic relations such as ``x > 0`` and ``x <= 0``. Its
    atoms need not be logically independent, and their conjunction can be
    contradictory. The name describes the Boolean structure, not the
    geometry of the solution set in real space.

    Equality and hashing are structural: equal cubes contain the same atoms.
    The empty cube represents truth. Construction removes duplicate atoms
    but leaves normalization explicit, so construction schedules retain
    control over when constraints are combined or contradictions detected.
    The methods assume that atom right-hand sides have been simplified to 0.

    The dataclass option ``slots=True`` stores the reference to ``_atoms``
    directly in each instance, without an attribute dictionary (``__dict__``).
    This reduces memory overhead when many cubes are created. The option
    ``frozen=True`` prevents ordinary reassignment of ``_atoms`` after
    construction; the frozenset itself prevents changes to its contents.

    >>> from logic1.theories.RCF import VV
    >>> x = VV['x']
    >>> Cube((x > 0, x >= 0)).normalize() == Cube((x > 0,))
    True
    >>> Cube((x > 0,)).merge(Cube((x <= 0,))) == Cube()
    True
    >>> Cube((x - 1 > 0,)).implies(Cube((x > 0,)))
    True
    """

    _atoms: frozenset[AtomicFormula]

    # The bits represent negative, zero, and positive polynomial signs.
    _SIGN_MASK: ClassVar[Final[Mapping[type[AtomicFormula], int]]] = {
        Eq: 0b010, Ne: 0b101, Lt: 0b001, Le: 0b011, Gt: 0b100, Ge: 0b110}
    _MASK_OP: ClassVar[Final[Mapping[int, type[AtomicFormula]]]] = {
        0b001: Lt, 0b010: Eq, 0b011: Le, 0b100: Gt, 0b101: Ne, 0b110: Ge}

    def __hash__(self) -> int:
        """Return the hash of the underlying immutable atom set.
        """
        return hash(self._atoms)

    def __init__(self, atoms: Iterable[AtomicFormula] = ()) -> None:
        """Freeze the supplied atoms without normalizing their constraints.
        """
        object.__setattr__(self, '_atoms', frozenset(atoms))

    def __iter__(self) -> Iterator[AtomicFormula]:
        """Iterate over the atoms in unspecified order.
        """
        return iter(self._atoms)

    def __le__(self, other: Cube) -> bool:
        """Check structural inclusion of this cube's atoms in ``other``.
        """
        return self._atoms <= other._atoms

    def __len__(self) -> int:
        """Return the number of distinct atoms in this cube.
        """
        return len(self._atoms)

    @classmethod
    def combine(cls, cubes: Iterable[Cube]) -> Cube:
        """Return a cube representing the conjunction of all supplied cubes.

        Take the union of their atom sets: an atom required by any input
        must hold in the combined conjunction. Repeated atoms occur once.
        With no inputs, return the empty cube, representing truth, the
        identity for conjunction.

        Leave the input cubes unchanged. Do not combine different relations
        or detect contradictions; the construction schedule decides when to
        call ``normalize``. This operation builds each conjunction produced
        by the ordinary Cartesian product of child DNFs.
        """
        sets = tuple(cube._atoms for cube in cubes)
        return cls(frozenset.union(*sets)) if sets else cls()

    def conjoin(self, other: Cube) -> Cube:
        """Return a cube representing this cube conjoined with ``other``.

        Form the union of the two atom sets, removing duplicate atoms and
        leaving both inputs unchanged. This is the two-input counterpart
        of ``combine``, used when building Cartesian products incrementally.

        Normalization remains explicit. For example, conjoining ``x > 0``
        with ``x <= 0`` retains both atoms; a subsequent ``normalize`` call
        detects the contradiction.
        """
        return Cube(self._atoms | other._atoms)

    def implies(self, other: Cube) -> bool:
        """Try to establish that this cube logically implies ``other``.

        Every atom in ``other`` must follow from this cube's conjunction.
        First discharge atoms that occur explicitly here or follow from
        one of this cube's relations on the same polynomial. Compare their
        allowed signs: for example, ``x > 0`` implies ``x >= 0`` because
        positive is a subset of nonnegative.

        Ask the RCF simplifier to establish the conjunction of the remaining
        atoms under all of this cube's assumptions. This can also establish
        implications between different polynomials, such as ``x - 1 > 0``
        implying ``x > 0``. Traverse atoms in sorted order for reproducibility.

        Return true only when the implication is established. False means
        that this heuristic check could not prove it, and does not certify
        a counterexample. In a DNF, a true result allows this stronger cube
        to be removed when the weaker ``other`` cube is also present.
        """
        premise_atoms = sorted(self._atoms)
        remaining = []
        for atom in sorted(other._atoms):
            if atom in self._atoms:
                continue
            if any(premise.lhs == atom.lhs
                   and self._SIGN_MASK[premise.op] & self._SIGN_MASK[atom.op]
                   == self._SIGN_MASK[premise.op] for premise in premise_atoms):
                continue
            remaining.append(atom)
        if not remaining:
            return True
        return _is_valid(And(*remaining), assume=premise_atoms, explode_always=False) is True

    def merge(self, other: Cube) -> Cube | None:
        r"""Try to express the disjunction of two cubes as a single cube.

        Apply the restricted cut rule ``(C /\ A) \/ (C /\ B)`` equals
        ``C /\ (A \/ B)``. The cubes must have the same number of atoms
        and differ in exactly one atom each. Those two atoms must compare
        the same polynomial with zero; all other atoms form the common
        conjunction ``C``.

        Union the signs allowed by the two differing relations. For example,
        ``x < 0`` and ``x == 0`` combine to ``x <= 0``. If the union covers
        negative, zero, and positive, omit the relation altogether: merging
        ``C /\ x > 0`` with ``C /\ x <= 0`` yields just ``C``. An empty
        common conjunction then yields the empty cube, representing truth.

        Return a new cube equivalent to the disjunction, leaving both inputs
        unchanged. Return ``None`` when this rule does not apply; no general
        RCF validity check or search for other possible merges is performed.
        """
        different = self._atoms ^ other._atoms
        if len(self._atoms) != len(other._atoms) or len(different) != 2:
            return None
        atom_left, atom_right = tuple(different)
        if atom_left.lhs != atom_right.lhs:
            return None
        common = self._atoms & other._atoms
        mask = self._SIGN_MASK[atom_left.op] | self._SIGN_MASK[atom_right.op]
        if mask == 0b111:
            return Cube(common)
        return Cube(common | {self._MASK_OP[mask](atom_left.lhs, 0)})

    def normalize(self) -> Cube | None:
        """Combine constraints on equal polynomials within this conjunction.

        Group atoms by their polynomial left-hand sides. Each relation
        permits some of the signs negative, zero, and positive. Since all
        atoms must hold, intersect the permitted signs within each group
        and replace the group by the single relation expressing that
        intersection. For example, ``x > 0`` together with ``x >= 0`` becomes
        just ``x > 0``.

        If an intersection is empty, return ``None``: the cube is
        contradictory, as with ``x > 0`` together with ``x <= 0``. Otherwise
        return a new cube, leaving this one unchanged. An empty input yields
        the empty cube, representing truth.

        This uses only relations on equal polynomials and makes no general
        RCF validity calls. The returned cube can still be contradictory
        through constraints on different polynomials. Both batch and
        incremental syntactic reduction use this operation.
        """
        signs: dict[Term, int] = {}
        for atom in self._atoms:
            mask = signs.get(atom.lhs, 0b111) & self._SIGN_MASK[atom.op]
            if not mask:
                return None
            signs[atom.lhs] = mask
        return Cube(self._MASK_OP[mask](lhs, 0) for lhs, mask in signs.items())

    def sort_key(self) -> tuple[int, tuple[AtomicFormula, ...]]:
        """Return a key for deterministic traversal and comparison choices.

        The first component is the number of atoms, so shorter conjunctions
        precede longer ones. Break ties lexicographically using the tuple of
        atoms in their existing RCF ordering. The key depends on the stored
        atoms, not on the iteration order of their frozenset.

        Formula reconstruction, greedy subsumption and cut, and the bounded
        semantic pass share this ordering. Their choices can affect which
        equivalent DNF is obtained, so a stable traversal order matters.
        This method neither normalizes nor changes the cube.
        """
        return len(self._atoms), tuple(sorted(self._atoms))


class Simplification(IntEnum):
    """Choose when to simplify strict DNFs during normal form computation.

    A step constructs the DNF at one ``And`` or ``Or`` node. ``NONE`` keeps
    ordinary preprocessing and conservative final simplification. ``FINAL``
    simplifies the complete strict DNF once. ``AFTER_STEP`` simplifies each
    node's completed result; ``DURING_STEP`` simplifies cubes and partial
    products as they are generated. The latter three also run final semantic
    subsumption.

    Integer values identify schedules, not increasing quality or speed.
    """

    NONE = 0
    FINAL = 1
    AFTER_STEP = 2
    DURING_STEP = 3


class StrictDnf:
    """A mutable disjunction of immutable cubes, stored in a private set.

    Equality compares the stored cubes structurally; instances are unhashable.
    An empty collection represents false, and an empty cube represents true.
    Construction, ``add``, and ``update`` only collect cubes. In particular,
    a strict DNF need not be normalized or free of redundant cubes.

    ``insert`` mutates an already syntactically reduced collection by adding
    a normalized cube with structural subsumption and cut. The batch methods
    ``syntactic_reduction`` and ``semantic_subsumption`` return new collections
    without changing their input. They share immutable cubes where possible.
    Schedules in the Cartesian classes determine when these operations run.
    See ``doc/notes/RCF/dnf.md`` for the rules and experimental findings.

    >>> from logic1.theories.RCF import VV
    >>> x, y = VV.get('x', 'y')
    >>> raw = StrictDnf((Cube((x > 0, y == 0)), Cube((x <= 0, y == 0))))
    >>> raw.syntactic_reduction() == StrictDnf((Cube((y == 0,)),))
    True
    >>> len(raw)
    2
    """

    __slots__ = ('_cubes',)
    _cubes: set[Cube]

    def __contains__(self, cube: Cube) -> bool:
        """Check structural membership in the stored cube set.
        """
        return cube in self._cubes

    def __eq__(self, other: object) -> bool:
        """Compare cube sets without checking logical equivalence.
        """
        if not isinstance(other, StrictDnf):
            return NotImplemented
        return self._cubes == other._cubes

    def __init__(self, cubes: Iterable[Cube] = ()) -> None:
        """Own a fresh set of the supplied cubes without simplifying it.
        """
        self._cubes = set(cubes)

    def __iter__(self) -> Iterator[Cube]:
        """Iterate over cubes in unspecified order.
        """
        return iter(self._cubes)

    def __len__(self) -> int:
        """Return the number of distinct stored cubes.
        """
        return len(self._cubes)

    def add(self, cube: Cube) -> None:
        """Add a cube without normalization, subsumption, or cut.
        """
        self._cubes.add(cube)

    def insert(self, cube: Cube) -> None:
        """Insert a normalized cube using structural subsumption and cut.

        The destination must already be syntactically reduced. This method
        mutates it, preserving that property. It does not normalize cubes.

        Choose the lowest-key applicable comparison. A single scan makes the
        same choice as sorting the result set without allocating a sorted
        copy at each insertion. Repeat after each change.
        """
        result = self._cubes
        if Cube() in result:
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
                    merged = cube.merge(other)
                    if merged is None:
                        continue
                    discard = False
                key = other.sort_key()
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

    def semantic_subsumption(self, max_pair_checks: int | None) -> StrictDnf:
        """Remove cubes subsumed by semantic implication checks.

        Compare each pair once. A successful implication removes the stronger
        cube. Return a new collection, leaving this one unchanged. The caller
        supplies a validated nonnegative pair budget, or ``None`` for no
        limit. Zero skips comparisons. The counter starts anew on each call.
        The budget does not apply to syntactic reduction during construction.
        """
        cubes = sorted(self._cubes, key=Cube.sort_key)
        output = StrictDnf(self._cubes)
        result = output._cubes
        pair_checks = 0
        for i, left in enumerate(cubes):
            if left not in result:
                continue
            for right in cubes[i + 1:]:
                if right not in result:
                    continue
                if max_pair_checks is not None and pair_checks >= max_pair_checks:
                    return output
                pair_checks += 1
                if right.implies(left):
                    result.remove(right)
                elif left.implies(right):
                    result.remove(left)
                    break
        return output

    def syntactic_reduction(self) -> StrictDnf:
        """Apply syntactic reduction to an existing strict DNF.

        Return a new collection without changing this one. Normalize each
        cube and insert it using structural subsumption and cut. Incremental
        construction instead calls ``Cube.normalize`` and ``insert`` directly,
        avoiding a complete intermediate collection.
        """
        result = StrictDnf()
        for cube in sorted(self._cubes, key=Cube.sort_key):
            normalized = cube.normalize()
            if normalized is not None:
                result.insert(normalized)
        return result

    def update(self, other: StrictDnf) -> None:
        """Add another collection's cubes without simplifying either input.
        """
        self._cubes.update(other._cubes)


class CartesianBnf:

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
        """Prenex and preprocess a formula, then convert its matrix to DNF.

        Restore the quantifier prefix after converting the quantifier-free
        matrix. The public methods apply conservative final simplification.
        """
        f = self.simplify(f.to_pnf())
        matrix, prefix = f.matrix()
        assert is_and_or(matrix)
        dnf = self._dnf_and_or(matrix)
        return dnf.quantify(prefix)

    def _dnf_and_or(self, f: AndOr) -> Formula:
        """Convert an And/Or matrix to DNF through its strict representation.

        Finish the constructed strict DNF, then reconstruct its formula in
        deterministic order. The default finishing hook leaves it unchanged.
        """
        strict_dnf = self._finish_strict_dnf(self._strict_dnf(f))
        cubes = sorted(strict_dnf, key=Cube.sort_key)
        return Or(*(And(*sorted(cube)) for cube in cubes))

    def _finish_strict_dnf(self, strict_dnf: StrictDnf) -> StrictDnf:
        """Return the completed strict DNF without additional simplification.

        Subclasses select the remaining simplification passes here.
        """
        return strict_dnf

    def conservative_simplify(self, f: Formula) -> Formula:
        """Simplify a formula, preserving DNF and CNF when present.

        An input in DNF remains in DNF; an input in CNF remains in CNF.
        For a prenex formula, this applies to its quantifier-free matrix.

        Unconditional splitting is disabled with ``explode_always=False``.
        """
        return _simplify(f, explode_always=False)

    def simplify(self, f: Formula) -> Formula:
        """Preprocess without unconditional splitting or monomial substitution.

        Value substitution remains enabled. Monomial substitution can create
        extra disjuncts before Cartesian distribution.
        """
        return _simplify(f, explode_always=False, substitute=1)

    def _strict_dnf(self, f: AndOr) -> StrictDnf:
        """Construct a strict DNF as a set of immutable cubes.

        An ``And`` node forms the Cartesian product of its children's DNFs;
        an ``Or`` node unions them. Sets remove duplicate cubes and atoms.
        False is the empty set; true is the set containing the empty cube.
        Recursive calls dispatch to subclasses through ``self._strict_dnf``.
        """
        if isinstance(f, _F):
            return StrictDnf()
        if isinstance(f, _T):
            return StrictDnf((Cube(),))
        if isinstance(f, AtomicFormula):
            return StrictDnf((Cube((f,)),))
        if isinstance(f, And):
            strict_dnfs = (self._strict_dnf(arg) for arg in f.args)
            result = StrictDnf()
            for cubes in itertools.product(*strict_dnfs):
                result.add(Cube.combine(cubes))
            return result
        if isinstance(f, Or):
            result = StrictDnf()
            for arg in f.args:
                result.update(self._strict_dnf(arg))
            return result
        raise ValueError(f"Unexpected formula type: {type(f)}")


class SimplifyingCartesianBnf(CartesianBnf):
    """Share configuration and final semantic subsumption across schedules.

    The concrete subclasses choose when syntactic reduction runs. This base
    stores the validated comparison budget and finishes a constructed DNF
    with bounded semantic subsumption. Collection algorithms belong to
    ``StrictDnf``; cube algorithms belong to ``Cube``. No multiple inheritance
    is needed. This class is infrastructure for the three concrete schedules.
    """

    max_pair_checks: int | None

    def __init__(self, *, max_pair_checks: int | None = 500) -> None:
        """Validate and store the semantic pair budget; ``None`` is unbounded.

        All three concrete schedules inherit this constructor.
        """
        if max_pair_checks is not None:
            if isinstance(max_pair_checks, bool) or not isinstance(max_pair_checks, int):
                raise TypeError("max_pair_checks must be an integer or None")
            if max_pair_checks < 0:
                raise ValueError("max_pair_checks must be nonnegative or None")
        self.max_pair_checks = max_pair_checks

    def _finish_strict_dnf(self, strict_dnf: StrictDnf) -> StrictDnf:
        """Apply the final semantic pass with this computation's pair budget.
        """
        return strict_dnf.semantic_subsumption(self.max_pair_checks)


class FinalSimplifyingCartesianBnf(SimplifyingCartesianBnf):
    """Simplify only after constructing the complete strict DNF.

    Inherit ordinary Cartesian recursion without simplifying child DNFs or
    partial products. Apply batch syntactic reduction to the completed strict
    DNF once, then run bounded semantic subsumption.
    Ordinary preprocessing and conservative final simplification also run.

    This schedule avoids repeated simplification during recursion, but retains
    the entire unsimplified strict DNF in memory. It can be useful when plain
    Cartesian construction is fast and only the completed result needs DNF
    simplification. It does not guarantee a smaller result or a faster
    computation than the other schedules. See ``doc/notes/RCF/dnf.md``.

    Initial simplification retains the six cubes below. The final-only
    schedule subsumes the second cube, cuts on ``d``, and keeps four cubes.
    The during-step schedule instead uses the second cube for further cuts
    and returns two. Both calls use the default semantic pair budget of 500:

    >>> from logic1.theories.RCF import VV
    >>> a, b, c, d, e = VV.get('a', 'b', 'c', 'd', 'e')
    >>> f = Or(And(a > 0, c > 0), And(a > 0, b > 0, c > 0),
    ...        And(a <= 0, b > 0, c > 0), And(b <= 0, c > 0),
    ...        And(d > 0, e > 0), And(d <= 0, e > 0))
    >>> normal_form = FinalSimplifyingCartesianBnf()
    >>> len(normal_form.simplify(f.to_pnf()).args)
    6
    >>> for clause in normal_form.dnf(f).args:
    ...     print(clause)
    e > 0
    c > 0 and b <= 0
    c > 0 and b > 0 and a <= 0
    c > 0 and a > 0
    >>> DuringStepSimplifyingCartesianBnf().dnf(f)
    Or(e > 0, c > 0)
    """

    def _finish_strict_dnf(self, strict_dnf: StrictDnf) -> StrictDnf:
        """Apply batch syntactic reduction, then bounded semantic subsumption.
        """
        return strict_dnf.syntactic_reduction().semantic_subsumption(self.max_pair_checks)


class AfterStepSimplifyingCartesianBnf(SimplifyingCartesianBnf):
    """Simplify each subtree after its local Cartesian expansion is complete.

    Recursion returns simplified strict DNFs for the children. An ``Or`` node
    first unions their cubes; an ``And`` node first forms their full
    Cartesian product. Only after that node's raw strict DNF exists does
    this class apply batch syntactic reduction: normalize each cube and use
    structural subsumption and cut as it inserts the cube into a new set.
    The collection rules combine relations on the same polynomial, reject
    contradictory cubes, remove structural subsumption, and repeatedly
    merge cubes differing in one relation. As in the during-step class, the
    final semantic pass examines at most ``max_pair_checks`` cube pairs.
    The simplified strict DNF is converted to a formula and conservatively
    simplified. Cube traversal and collection insertion use the same stable
    key as the during-step class. Inherited CNF computation uses the DNF of the
    negation.

    This ordering matches the placement of Redlog's recursive
    simplification, though the rules used here are different. It avoids
    simplifying partial products during an ``And`` node's expansion and gives
    each node a simpler schedule. The complete local
    product must be stored before redundant or contradictory cubes are
    removed, which can increase peak memory and construction time. On
    the 13 motor series it gave exactly the same strict DNFs as the
    during-step class; one-call timings gave neither method a clear advantage.
    See ``doc/notes/RCF/dnf.md`` for the rules and measurements.

    The initial simplification leaves all six cubes below. Both classes
    cut on ``d``; the during-step class also cuts on ``a`` and ``b`` before
    subsumption can remove the needed inputs. These calls use the default
    ``max_pair_checks=500``:

    >>> from logic1.theories.RCF import VV
    >>> a, b, c, d, e = VV.get('a', 'b', 'c', 'd', 'e')
    >>> f = Or(And(a > 0, c > 0), And(a > 0, b > 0, c > 0),
    ...        And(a <= 0, b > 0, c > 0), And(b <= 0, c > 0),
    ...        And(d > 0, e > 0), And(d <= 0, e > 0))
    >>> normal_form = AfterStepSimplifyingCartesianBnf()
    >>> len(normal_form.simplify(f.to_pnf()).args)
    6
    >>> for clause in normal_form.dnf(f).args:
    ...     print(clause)
    e > 0
    c > 0 and b <= 0
    c > 0 and b > 0 and a <= 0
    c > 0 and a > 0
    >>> DuringStepSimplifyingCartesianBnf().dnf(f)
    Or(e > 0, c > 0)

    A small grid shows how the semantic pair budget affects result quality.
    Initial simplification retains all 36 cubes. Larger budgets find more
    implications. The output columns are budget, cubes, and atoms. The
    larger 400-cube experiment is recorded in ``doc/notes/RCF/dnf.md``:

    >>> x, y = VV.get('x', 'y')
    >>> g = Or(*(And(x > i, y > j) for i in range(6) for j in range(6)))
    >>> len(normal_form.simplify(g.to_pnf()).args)
    36
    >>> BNF = AfterStepSimplifyingCartesianBnf
    >>> for checks in (10, 30, None):
    ...     result = BNF(max_pair_checks=checks).dnf(g)
    ...     cubes = len(result.args) if isinstance(result, Or) else 1
    ...     print(checks, cubes, len(list(result.atoms())))
    10 31 62
    30 19 38
    None 1 2
    """

    def _strict_dnf(self, f: AndOr) -> StrictDnf:
        """Simplify a node after constructing its full Cartesian result.

        The inherited construction calls ``self._strict_dnf`` for children,
        so each child is simplified before the parent is expanded.
        """
        strict_dnf = CartesianBnf._strict_dnf(self, f)
        if isinstance(f, (AtomicFormula, _T, _F)):
            return strict_dnf
        return strict_dnf.syntactic_reduction()


class DuringStepSimplifyingCartesianBnf(SimplifyingCartesianBnf):
    """Simplify cubes as they are produced by Cartesian distribution.

    The inherited DNF computation first prenexes and simplifies the input.
    Its matrix contains only ``And``, ``Or``, truth values, and atomic
    relations against zero. A strict DNF is a set of cubes, each a
    ``Cube`` representing a conjunction of atoms.

    Syntactic reduction is incremental. At an ``Or`` node, insert child cubes
    into a running result and stop if the empty cube makes it true. At an
    ``And`` node, fold in child DNFs one at a time. Each pair of cubes produces
    a union of atoms; normalize it when both factors are nonempty, then try to
    insert it into the next result. Stop if that result becomes false. Cube
    normalization intersects the allowed signs of relations on the same
    polynomial and rejects contradictions. Insertion removes structurally
    subsumed cubes and repeatedly merges cubes that differ in one relation on
    the same polynomial. These recursive steps use no RCF validity checks. Cube
    traversal and greedy comparison choices follow :meth:`Cube.sort_key`,
    independent of Python set iteration order.

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
    :class:`AfterStepSimplifyingCartesianBnf`, with no
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
    >>> normal_form = DuringStepSimplifyingCartesianBnf()
    >>> len(normal_form.simplify(f.to_pnf()).args)
    6
    >>> normal_form.dnf(f)
    Or(e > 0, c > 0)
    >>> other = AfterStepSimplifyingCartesianBnf()
    >>> for clause in other.dnf(f).args:
    ...     print(clause)
    e > 0
    c > 0 and b <= 0
    c > 0 and b > 0 and a <= 0
    c > 0 and a > 0

    A small grid of polynomial constraints shows how the semantic pair
    budget affects result quality. Initial simplification retains all 36 cubes;
    distinct polynomials also prevent structural subsumption. Higher
    budgets find more semantic implications. The output columns are budget,
    cubes, and atoms. The larger 400-cube experiment is recorded in
    ``doc/notes/RCF/dnf.md``:

    >>> x, y = VV.get('x', 'y')
    >>> g = Or(*(And(x > i, y > j) for i in range(6) for j in range(6)))
    >>> len(normal_form.simplify(g.to_pnf()).args)
    36
    >>> BNF = DuringStepSimplifyingCartesianBnf
    >>> for checks in (10, 30, None):
    ...     result = BNF(max_pair_checks=checks).dnf(g)
    ...     cubes = len(result.args) if isinstance(result, Or) else 1
    ...     print(checks, cubes, len(list(result.atoms())))
    10 31 62
    30 19 38
    None 1 2
    """

    def _strict_dnf(self, f: AndOr) -> StrictDnf:
        """Distribute recursively, simplifying each intermediate strict DNF.
        """
        if isinstance(f, (AtomicFormula, _T, _F)):
            return CartesianBnf._strict_dnf(self, f)
        if isinstance(f, Or):
            result = StrictDnf()
            for arg in f.args:
                for cube in sorted(self._strict_dnf(arg), key=Cube.sort_key):
                    result.insert(cube)
                if Cube() in result:
                    break
            return result
        if isinstance(f, And):
            result = StrictDnf((Cube(),))
            for arg in f.args:
                child = sorted(self._strict_dnf(arg), key=Cube.sort_key)
                product = StrictDnf()
                for left in sorted(result, key=Cube.sort_key):
                    for right in child:
                        combined = left.conjoin(right)
                        cube = combined.normalize() if left and right else combined
                        if cube is not None:
                            product.insert(cube)
                result = product
                if not result:
                    break
            return result
        raise ValueError(f"Unexpected formula type: {type(f)}")


class PyedaBnf(abc.bnf.BooleanNormalForm[AtomicFormula, Term, Variable, Number]):
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
        """Simplify a formula, preserving DNF and CNF when present.

        An input in DNF remains in DNF; an input in CNF remains in CNF.
        For a prenex formula, this applies to its quantifier-free matrix.

        Unconditional splitting is disabled with ``explode_always=False``.
        """
        return _simplify(f, explode_always=False)

    def final_simplify(self, f: Formula) -> Formula:
        """Implement the hook required by :class:`.abc.bnf.BooleanNormalForm`.
        """
        return self.conservative_simplify(f)


def _normal_form(method: Literal["cartesian", "pyeda"], simplification: Simplification | int,
                 max_pairs: int | None) -> CartesianBnf | PyedaBnf:
    """Select the algorithm and validate its simplification schedule.
    """
    if isinstance(simplification, bool) or not isinstance(simplification, int):
        raise TypeError("simplification must be a Simplification member or an integer")
    schedule = Simplification(simplification)
    if method not in ("cartesian", "pyeda"):
        raise ValueError(f"Unknown method: {method}")
    if schedule is Simplification.NONE:
        if max_pairs != 500:
            raise ValueError("Comparison limits require a simplification schedule other than NONE")
        return CartesianBnf() if method == "cartesian" else PyedaBnf()
    if method != "cartesian":
        raise ValueError("Simplification schedules are not supported by the pyeda method")
    classes = {
        Simplification.FINAL: FinalSimplifyingCartesianBnf,
        Simplification.AFTER_STEP: AfterStepSimplifyingCartesianBnf,
        Simplification.DURING_STEP: DuringStepSimplifyingCartesianBnf}
    return classes[schedule](max_pair_checks=max_pairs)


def cnf(f: Formula, method: Literal["cartesian", "pyeda"] = "cartesian",
        *, simplification: Simplification | int = Simplification.FINAL,
        max_pair_checks: int | None = 500) -> Formula:
    """Compute a CNF with the selected algorithm and simplification schedule.

    See :func:`dnf` for the schedules and semantic comparison budget.

    Distribute a disjunction over a conjunction using the defaults:

    >>> from logic1.theories.RCF import VV
    >>> x, y, z = VV.get('x', 'y', 'z')
    >>> f = Or(And(x > 0, y > 0), z > 0)
    >>> cnf(f)
    And(Or(z > 0, y > 0), Or(z > 0, x > 0))
    """
    return _normal_form(method, simplification, max_pair_checks).cnf(f)


def dnf(f: Formula, method: Literal["cartesian", "pyeda"] = "cartesian",
        *, simplification: Simplification | int = Simplification.FINAL,
        max_pair_checks: int | None = 500) -> Formula:
    """Compute a DNF, preserving a quantifier prefix when present.

    The defaults select Cartesian construction and ``Simplification.FINAL``.
    ``simplification`` accepts a :class:`Simplification` member or its integer
    value. ``NONE`` preserves ordinary Cartesian construction. ``FINAL``
    simplifies the complete strict DNF once; ``AFTER_STEP`` simplifies after
    each Boolean node; ``DURING_STEP`` simplifies during its construction.
    These schedules are currently supported only by ``method="cartesian"``.
    Use ``simplification=Simplification.NONE`` with ``method="pyeda"``.

    ``max_pair_checks`` limits the final semantic pass for the three
    simplifying schedules: 500 by default, zero to disable, or ``None`` for
    no limit. Other schedules require its default value. All methods retain
    ordinary preprocessing and conservative final simplification.

    Distribute a conjunction over a disjunction using the defaults:

    >>> from logic1.theories.RCF import VV
    >>> x, y, z = VV.get('x', 'y', 'z')
    >>> f = And(Or(x > 0, y > 0), z > 0)
    >>> dnf(f)
    Or(And(z > 0, y > 0), And(z > 0, x > 0))
    """
    return _normal_form(method, simplification, max_pair_checks).dnf(f)
