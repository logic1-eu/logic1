"""Boolean normal forms over real closed fields.

The public ``dnf`` and ``cnf`` functions default to Cartesian construction
with ``Simplification.FINAL``. The Cartesian pipeline first converts the input
to prenex form and preprocesses it with the RCF simplifier. It distributes
conjunctions over disjunctions to construct a strict DNF of the quantifier-free
matrix, applying the selected simplification schedule. Finally, it reconstructs
the formula, restores the quantifier prefix, and conservatively simplifies the
result while preserving DNF. CNF is obtained through the DNF of the negation.

A strict DNF is a disjunction of cubes, each a conjunction of polynomial
constraints. ``Cube`` owns an immutable atom set and the operations on one
conjunction. ``StrictDnf`` owns a mutable set of cubes and the operations on
their disjunction. Creating a ``Cube`` removes repeated atoms; creating a
``StrictDnf`` removes identical cubes. Beyond duplicate removal, neither
constructor performs simplification. The computation schedule determines when
further simplification runs.

DNF-specific simplification has two parts. Syntactic reduction uses fixed
rules: combine constraints on equal polynomials, remove cubes subsumed by
atom inclusion, and merge cubes differing in one relation on the same
polynomial. Semantic subsumption additionally uses RCF implication checks to
remove cubes whose solutions are covered by another disjunct. These greedy
procedures preserve equivalence but do not guarantee a minimal DNF.

For ``method="cartesian"``, the ``simplification`` keyword argument of
``dnf`` and ``cnf`` selects the implementation class. ``Simplification.NONE``
selects ``CartesianBnf`` and skips DNF-specific simplification.
``Simplification.FINAL`` selects ``FinalSimplifyingCartesianBnf``, which
constructs the complete strict DNF before applying syntactic reduction.

The other two options apply the syntactic rules within recursive construction.
A call processing an ``And`` or ``Or`` node in the preprocessed input's syntax
tree constructs a strict DNF for the subformula rooted at that node.
``Simplification.AFTER_STEP`` selects ``AfterStepSimplifyingCartesianBnf``.
Under this option, the call completes that construction, reduces the resulting
strict DNF, and returns it. ``Simplification.DURING_STEP`` selects
``DuringStepSimplifyingCartesianBnf``. Under this option, the call applies the
rules as it generates cubes and partial products.

All three simplifying subclasses apply semantic subsumption once, after the
complete strict DNF has been constructed and syntactically reduced. The
``max_pair_checks`` keyword argument of ``dnf`` and ``cnf`` limits this pass
to 500 cube pairs by default. It does not limit syntactic reduction or the
time spent on an individual implication check. The common base class
``SimplifyingCartesianBnf`` provides the final semantic pass and validates
the limit. Ordinary preprocessing and conservative final simplification
remain active under every schedule.

See ``doc/notes/RCF/dnf.md`` for the design and experimental findings.

The ``pyeda`` method uses a Boolean abstraction and requires
``Simplification.NONE``.
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
    """Check recursively for And, Or, RCF atoms, and the two truth values.

    Reject quantifiers and other Boolean operators, including explicit Not.
    """
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
    The empty cube represents truth. Construction only removes duplicate
    atoms; ``normalize`` must be called separately to combine constraints.
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

        Union their atom sets, removing duplicates. With no inputs, return
        the empty cube, representing truth. Leave the inputs unchanged and
        perform no normalization or contradiction checks. Cartesian
        construction uses this operation for each tuple of child cubes.
        """
        sets = tuple(cube._atoms for cube in cubes)
        return cls(frozenset.union(*sets)) if sets else cls()

    def conjoin(self, other: Cube) -> Cube:
        """Return a cube representing this cube conjoined with ``other``.

        Union the atom sets without changing either input. This is the
        two-input counterpart of ``combine``. No normalization is performed:
        conjoining ``x > 0`` with ``x <= 0`` retains both atoms until a
        subsequent ``normalize`` call detects the contradiction.
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

        Factor ``(C /\ A) \/ (C /\ B)`` as ``C /\ (A \/ B)``. Each
        cube must have exactly one atom absent from the other, and these
        two atoms must compare the same polynomial with zero. The remaining
        atoms form the common conjunction ``C``.

        Union the signs allowed by the two differing relations. For example,
        ``x < 0`` and ``x == 0`` combine to ``x <= 0``. If the union covers
        negative, zero, and positive, omit the relation altogether: merging
        ``C /\ x > 0`` with ``C /\ x <= 0`` yields just ``C``. An empty
        common conjunction then yields the empty cube, representing truth.
        This special case is the cut to the common conjunction.

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
        through constraints on different polynomials.
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
        atoms in their existing RCF ordering. This makes reconstruction and
        greedy comparison choices independent of set iteration order.
        """
        return len(self._atoms), tuple(sorted(self._atoms))


class Simplification(IntEnum):
    """Select when the Cartesian method applies DNF-specific simplification.

    The ``simplification`` argument of ``dnf`` and ``cnf`` accepts these
    members or their integer values. A step is a recursive call constructing
    the strict DNF for an ``And`` or ``Or`` subformula of the preprocessed
    input. ``AFTER_STEP`` reduces that call's result before returning it;
    ``DURING_STEP`` applies the syntactic rules as the call generates cubes
    and partial products. ``FINAL`` applies syntactic reduction only after
    constructing the complete strict DNF.

    These three schedules also apply semantic subsumption once, after the
    complete construction. ``NONE`` skips both DNF-specific parts. All
    schedules retain preprocessing and conservative final simplification.
    Larger integer values do not guarantee better or faster results.
    """

    NONE = 0
    FINAL = 1
    AFTER_STEP = 2
    DURING_STEP = 3


class StrictDnf:
    """A mutable disjunction of immutable cubes, stored in a private set.

    Equality compares the stored cubes structurally; instances are unhashable.
    An empty collection represents false; a collection containing the empty
    cube represents true. Construction, ``add``, and ``update`` collect cubes
    without simplification beyond duplicate removal. Other redundancies
    and unnormalized cubes may remain.

    ``insert`` mutates an already syntactically reduced collection by adding
    a normalized cube with structural subsumption and cut. The batch methods
    ``syntactic_reduction`` and ``semantic_subsumption`` return new collections
    without changing their input: the former uses fixed atomic rules, the
    latter implication checks.
    Schedules in the Cartesian classes determine when these operations run.

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
        """Own a fresh set of the supplied cubes, removing duplicates only.
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
        """Add one conjunction as another disjunct of this DNF.

        Add the cube to this collection's set, with no change if it is
        already present. Apply no further simplification: adding the empty
        cube, representing truth, leaves other cubes in the set.
        """
        self._cubes.add(cube)

    def insert(self, cube: Cube) -> None:
        """Add a disjunct while removing structural redundancy and merging.

        Require a normalized input cube and a syntactically reduced
        collection: at most one relation per polynomial in each cube, and
        no pair eligible for structural subsumption or ``Cube.merge``.
        Mutate the collection to represent its disjunction with the input,
        preserving these properties.

        A cube subsumes another when its atom set is a subset of the other's:
        for example, ``x > 0`` subsumes ``x > 0 and y == 0``. For a selected
        pair, discard the incoming cube if the existing one subsumes it;
        otherwise remove the existing cube if the incoming one subsumes it.

        Also try ``Cube.merge`` on pairs differing in one relation on the
        same polynomial. For example, the disjunction of ``x < 0 and y == 0``
        and ``x >= 0 and y == 0`` becomes ``y == 0``. Replace both inputs
        by the merged cube and reconsider it against the remaining cubes.

        At each iteration, choose the applicable comparison with the lowest
        ``Cube.sort_key``. A scan selects it without allocating a sorted
        copy. Store the candidate when no rule applies. The empty cube
        represents truth and replaces the entire collection;
        a collection already containing it needs no change.

        These rules use atom inclusion and polynomial sign masks, without
        general RCF validity checks. Greedy deletion can prevent a later
        merge, so deterministic choices do not guarantee a minimal DNF.
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
        """Remove cubes proved to imply another cube in the disjunction.

        A cube that implies another adds no solutions to their disjunction.
        ``Cube.implies`` can establish this beyond atom-set inclusion, as
        with ``x - 1 > 0`` implying ``x > 0``. Inconclusive checks never
        justify deletion.

        Copy the collection and sort its cubes by ``Cube.sort_key``. Visit
        each surviving pair at most once, first testing whether the later
        cube implies the earlier one, then the converse if necessary.
        Delete a cube as soon as an implication is established and skip it
        in subsequent comparisons. No new cubes are created, so deletion
        does not require revisiting earlier pairs.

        Return the new collection, leaving this one unchanged. The caller
        supplies a validated nonnegative ``max_pair_checks``, or ``None``
        for no limit. Count each visited pair once, even if both implication
        directions are tried. Zero returns an unchanged copy; exhaustion
        returns the result obtained so far. The counter starts anew on each
        call. This bounds pair visits, not the cost of an individual RCF
        check or the syntactic reduction performed before this final pass.
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
        """Simplify the cubes and their disjunction using fixed atomic rules.

        Build a new collection, processing the input cubes in deterministic
        ``Cube.sort_key`` order. First call ``Cube.normalize`` to combine
        constraints on equal polynomials within each conjunction and discard
        any cubes it detects as contradictory.

        Insert each remaining cube into the new collection with ``insert``.
        This removes conjunctions subsumed by atom inclusion and merges
        conjunctions differing in one relation on the same polynomial.
        An empty cube absorbs all other disjuncts. These operations use no
        general RCF validity checks; relationships between different
        polynomials may leave contradictions or redundancies undetected.

        Return the syntactically reduced collection without changing this
        one. For incremental construction, the same rules can be applied
        by calling ``Cube.normalize`` and ``insert`` as cubes are generated.
        """
        result = StrictDnf()
        for cube in sorted(self._cubes, key=Cube.sort_key):
            normalized = cube.normalize()
            if normalized is not None:
                result.insert(normalized)
        return result

    def update(self, other: StrictDnf) -> None:
        """Extend this DNF with all disjuncts of ``other``.

        Union the cube sets, removing duplicates but performing no further
        simplification. Leave ``other`` unchanged; the collections retain
        separate sets but may share immutable cubes. Updating a collection
        with itself has no effect.
        """
        self._cubes.update(other._cubes)


class CartesianBnf:
    """Compute normal forms by distributing conjunction over disjunction.

    This class preprocesses the input, recursively constructs a ``StrictDnf``,
    and converts it back to a formula, handling quantifiers and final
    simplification. CNF computation uses the DNF of the negation. The option
    ``Simplification.NONE`` selects this class for the Cartesian method.

    Construction handles truth values and duplicates through set operations.
    Further simplification belongs to ``Cube`` and ``StrictDnf``.
    Subclasses override ``_strict_dnf`` to simplify during recursion or
    ``_finish_strict_dnf`` to simplify the completed collection before
    reconstruction. All Cartesian subclasses share the reconstruction of
    a formula from the computed strict DNF.
    """

    def cnf(self, f: Formula) -> Formula:
        """Compute a conjunctive normal form. If ``f`` contains quantifiers,
        then the result is a prenex normal form whose matrix is in CNF.

        Compute the DNF of the negation, then negate that result. Pushing
        negations inward exchanges conjunctions and disjunctions, producing
        CNF. Apply conservative simplification to preserve this form.
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

        Construct a collection of cubes, then call ``_finish_strict_dnf``
        for any simplification selected by the subclass. Reconstruct each
        cube as a conjunction of atoms and the collection as their
        disjunction. Sort cubes and atoms for deterministic output.
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

        Recursively convert each child to a DNF. At an ``Or`` node, collect
        all child cubes. At an ``And`` node, choose one cube from each child
        DNF in every possible combination and unite their atom sets. This
        Cartesian product distributes conjunction over disjunction.

        An atom becomes a cube containing that atom. Sets remove duplicates;
        no normalization, pairwise subsumption, or merging is performed here.
        False is the empty set; true is the set containing the empty cube.
        Recursive calls use ``self._strict_dnf``, allowing subclasses to
        simplify at each node rather than only after the full construction.
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
    """A base class for Cartesian DNF computation with semantic subsumption.

    This base validates and stores ``max_pair_checks``, which limits the
    number of cube pairs examined by semantic subsumption. Its default
    ``_finish_strict_dnf`` hook delegates to ``StrictDnf.semantic_subsumption``
    using that limit. Subclasses choose when to apply syntactic reduction.
    """

    max_pair_checks: int | None

    def __init__(self, *, max_pair_checks: int | None = 500) -> None:
        """Validate and store the budget for the final semantic pass.

        Accept a nonnegative integer or ``None`` for no limit; reject booleans
        and other types. Zero skips semantic comparisons. The budget does
        not limit syntactic reduction or the cost of one validity check.
        """
        if max_pair_checks is not None:
            if isinstance(max_pair_checks, bool) or not isinstance(max_pair_checks, int):
                raise TypeError("max_pair_checks must be an integer or None")
            if max_pair_checks < 0:
                raise ValueError("max_pair_checks must be nonnegative or None")
        self.max_pair_checks = max_pair_checks

    def _finish_strict_dnf(self, strict_dnf: StrictDnf) -> StrictDnf:
        """Remove semantically redundant cubes before formula reconstruction.

        Delegate to ``StrictDnf.semantic_subsumption`` with the stored pair
        budget, starting a fresh counter for this call. Return a new
        collection, leaving the supplied strict DNF unchanged.
        """
        return strict_dnf.semantic_subsumption(self.max_pair_checks)


class FinalSimplifyingCartesianBnf(SimplifyingCartesianBnf):
    """Apply DNF-specific simplification after constructing the full DNF.

    This class is selected by ``simplification=Simplification.FINAL`` in
    calls to ``dnf`` and ``cnf``. The inherited Cartesian recursion constructs
    the complete strict DNF before syntactic reduction and semantic
    subsumption are applied.

    Reducing only the completed strict DNF avoids repeated reduction during
    recursion, but requires retaining the entire unreduced strict DNF in
    memory.

    In the following example, initial simplification leaves ``f`` unchanged.
    The final-only computation removes ``And(c > 0, b > 0, a > 0)``, which
    is subsumed by ``And(c > 0, a > 0)``. It merges the two cubes containing
    ``e`` into ``e > 0``, since ``d > 0`` and ``d <= 0`` cover all real values
    of ``d``. Four cubes remain.

    The computation uses the default ``max_pair_checks=500``, permitting
    at most 500 cube-pair comparisons in final semantic subsumption.

    >>> from logic1.theories.RCF import VV
    >>> a, b, c, d, e = VV.get('a', 'b', 'c', 'd', 'e')
    >>> f = Or(And(e > 0, d <= 0),
    ...        And(e > 0, d > 0),
    ...        And(c > 0, b <= 0),
    ...        And(c > 0, b > 0, a <= 0),
    ...        And(c > 0, b > 0, a > 0),
    ...        And(c > 0, a > 0))

    >>> final_bnf = FinalSimplifyingCartesianBnf()
    >>> final_bnf.simplify(f) == f
    True
    >>> final_bnf.dnf(f)
    Or(e > 0, And(c > 0, b <= 0), And(c > 0, b > 0, a <= 0), And(c > 0, a > 0))

    For the input shown in :func:`dnf`, ``_strict_dnf`` constructs eight cubes
    with 20 atom occurrences. Syntactic reduction processes them in
    ``Cube.sort_key`` order, so two-atom cubes precede three-atom cubes.
    Consequently, ``And(c > 0, a > 0)`` subsumes
    ``And(c > 0, b > 0, a > 0)`` before the latter can be used in a cut.
    The corresponding deletion occurs for ``d`` in place of ``c``.

    The deleted cube could have been cut with
    ``And(c > 0, b > 0, a <= 0)`` to produce ``And(c > 0, b > 0)``. A further
    cut with ``And(c > 0, b <= 0)`` would then yield ``c > 0``. The same
    opportunity is lost for ``d``. Subsumption preserves equivalence, but
    deleting a cube can prevent later cuts.

    Six cubes with 14 atom occurrences remain. No further syntactic
    reduction applies, and final semantic subsumption removes nothing.
    Increasing ``max_pair_checks`` does not help: none of these cubes implies
    another, and the semantic pass does not combine cubes into replacements.
    """

    def _finish_strict_dnf(self, strict_dnf: StrictDnf) -> StrictDnf:
        """Apply batch syntactic reduction, then bounded semantic subsumption.

        First normalize cubes and apply structural subsumption and merging
        through ``syntactic_reduction``. Then use ``semantic_subsumption``
        to remove further redundant cubes by implication checks.

        Only the second pass uses ``max_pair_checks``; a zero budget still
        performs syntactic reduction. Return a new collection for formula
        reconstruction, leaving the supplied strict DNF unchanged.
        """
        return strict_dnf.syntactic_reduction().semantic_subsumption(self.max_pair_checks)


class AfterStepSimplifyingCartesianBnf(SimplifyingCartesianBnf):
    """Reduce a subformula's strict DNF before returning it from recursion.

    This class is selected by ``simplification=Simplification.AFTER_STEP`` in
    calls to ``dnf`` and ``cnf``. A recursive call constructs the strict DNF
    for an ``And`` or ``Or`` subformula of the preprocessed input, then
    syntactically reduces that DNF before returning it. Semantic subsumption
    runs once, after the complete strict DNF has been constructed.

    Completing an ``And`` call's Cartesian product before reduction avoids
    repeated reduction of partial products, but requires retaining the
    entire local product in memory. This placement of syntactic reduction
    matches Redlog's recursive simplification, though the rules differ.

    Using this class in the example of
    :class:`FinalSimplifyingCartesianBnf` reproduces the same result.

    In the following example, reduction within a disjunction shrinks the
    Cartesian product needed by the enclosing conjunction. The formula ``f``
    has ten atom occurrences and is unchanged by initial simplification.

    The four cubes of the second disjunction merge to
    ``And(y <= 0, x <= 0)``. The merges combine the signs allowed by
    ``x == 0`` and ``x < 0`` into ``x <= 0``, and likewise for ``y``.
    The outer ``And`` therefore constructs two cubes instead of eight.
    One contains both ``y > 0`` and ``y <= 0`` and is discarded during
    normalization. One cube with three atoms remains; final semantic
    subsumption has no pair to check.

    >>> from logic1.theories.RCF import VV
    >>> x, y, z = VV.get('x', 'y', 'z')
    >>> f = And(Or(z > 0, y > 0),
    ...         Or(And(y == 0, x == 0),
    ...            And(y == 0, x < 0),
    ...            And(y < 0, x == 0),
    ...            And(y < 0, x < 0)))

    >>> after_bnf = AfterStepSimplifyingCartesianBnf()
    >>> after_bnf.simplify(f) == f
    True
    >>> after_bnf.dnf(f)
    And(z > 0, y <= 0, x <= 0)

    For the input shown in :func:`dnf`, this class improves on
    :class:`FinalSimplifyingCartesianBnf`: the result has four cubes with
    eight atom occurrences instead of six cubes with 14 atom occurrences.

    The recursive call for the nested disjunction reduces its two cubes to
    ``b > 0`` by cut. The enclosing conjunction therefore produces
    ``And(c > 0, b > 0)``. The outer ``Or`` collects seven cubes with 16 atom
    occurrences before its own reduction.

    Reduction in ``Cube.sort_key`` order cuts ``And(c > 0, b > 0)`` with
    ``And(c > 0, b <= 0)`` to produce ``c > 0``. This cube then subsumes
    ``And(c > 0, a > 0)`` when it is considered. For ``d``,
    ``And(d > 0, a > 0)`` subsumes ``And(d > 0, b > 0, a > 0)`` before
    the latter can be cut with ``And(d > 0, b > 0, a <= 0)``.

    Final semantic subsumption leaves this result unchanged, even with
    ``max_pair_checks=None``. The pass cannot create replacement cubes, so it
    cannot recover the lost cut.

    The following example demonstrates how raising ``max_pair_checks``
    allows semantic subsumption to remove more cubes. The formula ``g``
    defined below has 36 cubes and 72 atom occurrences; initial simplification
    leaves it unchanged. The dictionary records cube counts, atom occurrences,
    and the frequencies of cube sizes. Its keys identify the comparison
    limits; ``None`` removes the limit.

    >>> from collections import Counter
    >>> x, y = VV.get('x', 'y')
    >>> g = Or(*(And(y - j > 0, x - i > 0)
    ...          for j in range(5, -1, -1) for i in range(5, -1, -1)))
    >>> after_bnf.simplify(g) == g
    True
    >>> results = {}
    >>> for checks in (10, 30, None):
    ...     after_bnf = AfterStepSimplifyingCartesianBnf(max_pair_checks=checks)
    ...     result = after_bnf.dnf(g)
    ...     cubes = result.args if isinstance(result, Or) else (result,)
    ...     sizes = [len(list(cube.atoms())) for cube in cubes]
    ...     results[str(checks)] = {'cubes': len(sizes), 'atoms': sum(sizes),
    ...                             'atoms_per_cube': dict(Counter(sizes))}
    >>> results
    {'10': {'cubes': 31, 'atoms': 62, 'atoms_per_cube': {2: 31}},
     '30': {'cubes': 19, 'atoms': 38, 'atoms_per_cube': {2: 19}},
     'None': {'cubes': 1, 'atoms': 2, 'atoms_per_cube': {2: 1}}}
    >>> result
    And(y > 0, x > 0)

    The dictionary gives the following table. An ``atoms_per_cube`` value of
    ``{2: 31}`` means 31 cubes with two atoms each.

    +---------------------+-------+-------+----------------+
    | ``max_pair_checks`` | Cubes | Atoms | Atoms per cube |
    +=====================+=======+=======+================+
    |                  10 |    31 |    62 |        {2: 31} |
    +---------------------+-------+-------+----------------+
    |                  30 |    19 |    38 |        {2: 19} |
    +---------------------+-------+-------+----------------+
    |                None |     1 |     2 |         {2: 1} |
    +---------------------+-------+-------+----------------+

    A larger 400-cube experiment is recorded in ``doc/notes/RCF/dnf.md``.
    """

    def _strict_dnf(self, f: AndOr) -> StrictDnf:
        """Simplify a node after constructing its full Cartesian result.

        Use Cartesian construction to collect an ``Or`` node's child cubes
        or form an ``And`` node's complete product. Its recursive calls use
        this method, so child DNFs are already reduced. Apply
        ``syntactic_reduction`` only after the current node's full result
        exists, normalizing cubes and applying structural subsumption and
        merging. Atoms and truth values need no such pass.
        """
        strict_dnf = CartesianBnf._strict_dnf(self, f)
        if isinstance(f, (AtomicFormula, _T, _F)):
            return strict_dnf
        return strict_dnf.syntactic_reduction()


class DuringStepSimplifyingCartesianBnf(SimplifyingCartesianBnf):
    """Simplify cubes as they are produced by Cartesian distribution.

    This class is selected by ``simplification=Simplification.DURING_STEP``
    in calls to ``dnf`` and ``cnf``. A recursive call for an ``And`` or ``Or``
    subformula of the preprocessed input uses ``StrictDnf.insert`` to reduce
    its DNF as cubes are added. An ``And`` call normalizes newly conjoined
    cubes and reduces partial products before multiplying by the next
    child's DNF. Semantic subsumption runs once, after the complete strict
    DNF has been constructed.

    Reducing partial products can avoid retaining a full unreduced Cartesian
    product. Repeated subsumption and merge checks can cost time, and greedy
    deletion can prevent later cuts. An ``Or`` call stops when its result
    contains the empty cube, representing true; an ``And`` call stops when
    a partial product is empty, representing false.

    The following example uses the same input as the example in
    :class:`FinalSimplifyingCartesianBnf`. This class improves the result
    from four cubes to two.

    Initial simplification leaves ``f`` unchanged. The two cubes containing
    ``e`` cut to ``e > 0``. The two three-atom cubes containing ``c`` are
    inserted before ``And(c > 0, a > 0)``. Their cut gives
    ``And(c > 0, b > 0)``, which then cuts with ``And(c > 0, b <= 0)`` to
    give ``c > 0``. The last cube, ``And(c > 0, a > 0)``, is subsumed when
    it arrives.

    The computation uses the default ``max_pair_checks=500``, permitting
    at most 500 cube-pair comparisons in final semantic subsumption.

    >>> from logic1.theories.RCF import VV
    >>> a, b, c, d, e = VV.get('a', 'b', 'c', 'd', 'e')
    >>> f = Or(And(e > 0, d <= 0),
    ...        And(e > 0, d > 0),
    ...        And(c > 0, b <= 0),
    ...        And(c > 0, b > 0, a <= 0),
    ...        And(c > 0, b > 0, a > 0),
    ...        And(c > 0, a > 0))

    >>> during_bnf = DuringStepSimplifyingCartesianBnf()
    >>> during_bnf.simplify(f) == f
    True
    >>> during_bnf.dnf(f)
    Or(e > 0, c > 0)

    For the input shown in :func:`dnf`, this class improves on
    :class:`AfterStepSimplifyingCartesianBnf`: the result has two cubes with
    two atom occurrences instead of four cubes with eight atom occurrences.

    At the outer ``Or``, ``And(d > 0, b <= 0)`` is inserted first. The two
    three-atom cubes containing ``d`` arrive next and cut to
    ``And(d > 0, b > 0)``. A further cut produces ``d > 0``, which subsumes
    ``And(d > 0, a > 0)`` when it arrives.

    The recursive call for the nested disjunction reduces its two cubes to
    ``b > 0`` by cut. The enclosing conjunction therefore produces
    ``And(c > 0, b > 0)``. Inserting this cube into the outer ``Or`` allows
    a cut with the retained ``And(c > 0, b <= 0)``, producing ``c > 0``.
    This cube then subsumes the already retained ``And(c > 0, a > 0)``.

    The running DNF of the outer ``Or`` contains at most three cubes. Its
    final two cubes represent ``Or(d > 0, c > 0)``. Final semantic subsumption
    leaves this result unchanged, even with ``max_pair_checks=None``:
    neither cube implies the other.

    Using this class in the ``max_pair_checks`` example of
    :class:`AfterStepSimplifyingCartesianBnf` reproduces the same results.
    """

    def _strict_dnf(self, f: AndOr) -> StrictDnf:
        """Distribute recursively, simplifying each intermediate strict DNF.

        At an ``Or`` node, insert child cubes into a running collection.
        Each insertion applies structural subsumption and merging. Stop
        processing children when the result contains the empty cube.

        At an ``And`` node, start with the empty cube, representing truth,
        and multiply the running DNF by one child DNF at a time. Conjoin
        each pair of cubes, normalize their combined constraints, and
        insert results not rejected as contradictory by ``normalize``.
        Conjoining with an empty cube needs no normalization. Stop if a
        product is empty, representing false.

        Thus each partial product is reduced before multiplication by the
        next child, avoiding storage of the full unreduced product. Traverse
        cubes in sorted order to make greedy simplification reproducible.
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
    """Validate the options of ``dnf`` and ``cnf`` and return an instance.

    Convert ``simplification`` to a ``Simplification`` member, rejecting
    booleans. For ``method="cartesian"``, select ``CartesianBnf`` for
    ``Simplification.NONE`` or the corresponding simplifying subclass for
    the other values. ``method="pyeda"`` selects ``PyedaBnf`` and requires
    ``Simplification.NONE``.

    ``max_pairs`` receives the public ``max_pair_checks`` argument. With
    ``Simplification.NONE``, it must equal 500 and is unused. Otherwise, pass
    it to the selected constructor, which validates and stores the limit on
    cube pairs examined by final semantic subsumption.
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
    """Compute a CNF, with a prenex quantifier prefix when needed.

    See :func:`dnf` for the options and examples. CNF computation uses
    duality: negate the input, compute a DNF of that negation, then negate
    the result and move negations inward.

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
    """Compute a DNF, with a prenex quantifier prefix when needed.

    The ``method`` option selects either distribution of conjunctions over
    disjunctions (``"cartesian"``) or Boolean abstraction and Espresso
    minimization (``"pyeda"``). Both methods apply RCF simplification before
    construction and preserve DNF during final simplification.

    For ``method="cartesian"``, the ``simplification`` option controls
    when to simplify the DNFs produced during construction.
    ``Simplification.NONE`` omits this additional simplification.
    ``Simplification.FINAL`` simplifies only after constructing the complete
    DNF. ``Simplification.AFTER_STEP`` simplifies the recursively constructed
    DNF of every ``And`` or ``Or`` subformula. ``Simplification.DURING_STEP``
    simplifies each DNF during its recursive construction, whenever a new
    conjunction is added.

    The three values other than ``Simplification.NONE`` also enable a final
    semantic subsumption pass using implication checks. The
    ``max_pair_checks`` option limits the number of cube pairs examined in
    this pass, to 500 by default. Set it to ``0`` to disable the pass or
    ``None`` to remove the pair limit. This option limits neither syntactic
    reduction nor the time spent on an individual implication check.

    With ``simplification=Simplification.NONE``, ``max_pair_checks`` must
    retain its default value and is unused. ``method="pyeda"`` requires
    ``simplification=Simplification.NONE``.

    Different values of ``simplification`` can produce different DNFs. The
    formula ``f`` defined below has seven outer disjuncts and 19 atom
    occurrences. It is not in DNF: distributing its nested disjunction
    without reduction produces eight cubes with 20 atom occurrences.
    Preprocessing leaves ``f`` unchanged. All three computations use the
    default ``max_pair_checks=500``.

    >>> from logic1.theories.RCF import VV
    >>> a, b, c, d = VV.get('a', 'b', 'c', 'd')
    >>> f = Or(And(d > 0, b <= 0),
    ...        And(d > 0, b > 0, a <= 0),
    ...        And(d > 0, b > 0, a > 0),
    ...        And(d > 0, a > 0),
    ...        And(c > 0, b <= 0),
    ...        And(c > 0, a > 0),
    ...        And(c > 0, Or(And(b > 0, a <= 0), And(b > 0, a > 0))))
    >>> cartesian_bnf = CartesianBnf()
    >>> cartesian_bnf.simplify(f) == f
    True

    The algorithm uses subsumption and cut to merge two disjuncts into one
    without changing their combined solution set.

    With ``Simplification.FINAL``, shorter cubes subsume
    ``And(c > 0, b > 0, a > 0)`` and ``And(d > 0, b > 0, a > 0)`` before
    either can merge with its counterpart containing ``a <= 0``. Six cubes
    remain, containing 14 atom occurrences.

    >>> dnf(f, simplification=Simplification.FINAL)
    Or(And(d > 0, b <= 0),
       And(d > 0, b > 0, a <= 0),
       And(d > 0, a > 0),
       And(c > 0, b <= 0),
       And(c > 0, b > 0, a <= 0),
       And(c > 0, a > 0))

    With ``Simplification.AFTER_STEP``, the nested disjunction first reduces
    to ``b > 0``. The resulting ``And(c > 0, b > 0)`` then merges with
    ``And(c > 0, b <= 0)`` to give ``c > 0``. Four cubes remain, containing
    eight atom occurrences.

    >>> dnf(f, simplification=Simplification.AFTER_STEP)
    Or(c > 0,
       And(d > 0, b <= 0),
       And(d > 0, b > 0, a <= 0),
       And(d > 0, a > 0))

    With ``Simplification.DURING_STEP``, the two three-atom cubes containing
    ``d > 0`` merge before the shorter ``And(d > 0, a > 0)`` is inserted.
    A further merge with ``And(d > 0, b <= 0)`` gives ``d > 0``. The result
    contains two cubes and two atom occurrences.

    >>> dnf(f, simplification=Simplification.DURING_STEP)
    Or(d > 0, c > 0)
    """
    return _normal_form(method, simplification, max_pair_checks).dnf(f)
