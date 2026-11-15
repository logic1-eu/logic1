"""Boolean normal forms over real closed fields.

The public :func:`dnf` and :func:`cnf` functions default to Cartesian
construction with simplification schedule :attr:`Simplification.FINAL`. The
Cartesian pipeline first converts the input to prenex form and preprocesses it
with the RCF simplifier. It distributes conjunctions over disjunctions to
construct a strict DNF of the quantifier-free matrix, applying the selected
simplification schedule. Finally, it reconstructs the formula, restores the
quantifier prefix, and conservatively simplifies the result while preserving
DNF. CNF is obtained through the DNF of the negation.

A strict DNF is a disjunction of cubes, each a conjunction of polynomial
constraints. :class:`Cube` owns an immutable set of atoms and provides
operations on their conjunction. :class:`StrictDnf` owns a mutable set of cubes
and provides operations on their disjunction. Creating a :class:`Cube` removes
repeated atoms; creating a :class:`StrictDnf` removes identical cubes. Beyond
duplicate removal, neither constructor performs simplification. The computation
schedule determines when further simplification runs.

DNF-specific simplification has two parts:

1. Syntactic reduction uses three rules. Structural subsumption removes a cube
   when another cube's atom set is a subset of its own. Normalization combines
   constraints on equal polynomials within a cube. Merging replaces two cubes
   differing in one relation on the same polynomial by a single cube
   representing their disjunction.

2. Semantic subsumption additionally uses RCF implication checks to remove cubes
   whose solutions are covered by another disjunct. These greedy procedures
   preserve equivalence but do not guarantee a minimal DNF.

For ``method="cartesian"``, the ``simplification`` keyword argument of
:func:`dnf` and :func:`cnf` selects the simplification schedule for syntactic
reduction from :class:`Simplification`.

The three simplifying schedules (:attr:`~Simplification.FINAL`,
:attr:`~Simplification.AFTER_STEP`, :attr:`~Simplification.DURING_STEP`) equally
apply semantic subsumption once, after the complete strict DNF has been
constructed and syntactically reduced.

The ``max_pair_checks`` keyword argument of :func:`dnf` and :func:`cnf` limits
semantic subsumption to 500 cube pairs by default. It does not limit syntactic
reduction or the time spent on an individual implication check. Ordinary
preprocessing and conservative final simplification remain active under every
schedule.

The historical ``pyeda`` method uses a Boolean abstraction and ignores
``simplification`` and ``max_pair_checks``.
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
"""An :class:`.And`-:class:`.Or`-combination of RCF atoms and truth values.
"""


def is_and_or(f: Formula) -> TypeGuard[AndOr]:
    """A type guard for :class:`AndOr` formulas.

    Check ``f`` recursively for :class:`.And`, :class:`.Or`,
    :class:`.RCF.atomic.AtomicFormula`, :class:`._F`, and :class:`._T`. Reject
    quantifiers and other Boolean operators, including explicit :class:`.Not`.
    """
    if isinstance(f, (_F, _T, AtomicFormula)):
        return True
    if isinstance(f, (And, Or)):
        return all(is_and_or(arg) for arg in f.args)
    return False


class Simplification(IntEnum):
    """Select the schedule when to apply syntactic reduction with the Cartesian
    method.

    The ``simplification`` argument of :func:`dnf` and :func:`cnf` accepts the
    enum members of this class as well as their integer values.

    A STEP is a recursive call constructing the strict DNF for an :class:`.And`
    or :class:`.Or` subformula of the pre-simplified input. :attr:`AFTER_STEP`
    reduces that call's result before returning it; :attr:`DURING_STEP` applies the
    syntactic reduction rules as the call generates cubes and partial products;
    :attr:`FINAL` applies syntactic reduction only after constructing the complete
    strict DNF.

    These three schedules equally apply semantic subsumption once, after the
    complete strict DNF has been constructed and syntactically reduced, while
    :attr:`NONE` skips also this semantic subsumption step. However, all schedules
    retain preprocessing and conservative final simplification, both based on
    :func:`.RCF.simplify.simplify`.

    Larger integer values of :class:`Simplification` do not guarantee better or
    faster results. As a rule of thumb, higher values need more time and less
    space.
    """

    NONE = 0
    """Skip syntactic reduction and semantic subsumption. Selects the
    implementation subclass :class:`CartesianBnf`.
    """

    FINAL = 1
    """Construct the complete strict DNF before applying syntactic reduction.
    Selects the implementation subclass :class:`FinalCartesianBnf`.
    """

    AFTER_STEP = 2
    """Construct the strict DNF for an :class:`.And` or :class:`.Or` subformula,
    then apply syntactic reduction before returning it. Selects the
    implementation subclass :class:`AfterStepCartesianBnf`.
    """

    DURING_STEP = 3
    """Construct the strict DNF for an :class:`.And` or :class:`.Or` subformula,
    applying syntactic reduction as cubes and partial products are generated.
    Selects the implementation subclass :class:`DuringStepCartesianBnf`.
    """


@dataclass(frozen=True, slots=True, init=False)
class Cube:
    r"""An immutable representation of a conjunction of RCF atoms.

    In Boolean logic, a cube is a conjunction of literals, dual to a clause,
    which is a disjunction of literals. A DNF is a disjunction of cubes, just as
    a CNF is a conjunction of clauses. Unlike a minterm, a cube need not mention
    every propositional variable. The name comes from viewing partial Boolean
    assignments as faces of the Boolean hypercube.

    Here a cube stores polynomial constraints in a frozenset. Negation is
    expressed by complementary relations such as :math:`x > 0` and
    :math:`x \le 0`. Its atoms need not be logically independent, and their
    conjunction can be contradictory.

    Equality and hashing are structural: equal cubes contain the same atoms. The
    ``<=`` operator compares atom sets by inclusion. The empty cube represents
    truth. The methods assume that atom right-hand sides have been
    simplified to 0.

    Construction collects atoms; :meth:`combine` and :meth:`conjoin` unite atom
    sets. These operations remove duplicates but leave normalization explicit.
    :meth:`normalize` combines relations on the same polynomial and detects
    contradictions within those groups. :meth:`merge` tries to represent the
    disjunction of two cubes as one cube. :meth:`implies` tries to establish
    logical implication; failure to prove it is inconclusive. These operations
    leave existing cubes unchanged.

    >>> from logic1.theories.RCF import VV
    >>> x = VV['x']
    >>> Cube((x > 0, x >= 0)).normalize() == Cube((x > 0,))
    True
    >>> Cube((x > 0,)).merge(Cube((x <= 0,))) == Cube()
    True
    >>> Cube((x - 1 > 0,)).implies(Cube((x > 0,)))
    True

    The class is declared as::

      @dataclass(frozen=True, slots=True, init=False)
      class Cube:
        ...

    The dataclass option ``slots=True`` stores the reference to ``_atoms``
    directly in each instance, without an attribute dictionary (``__dict__``).
    This reduces memory overhead when many cubes are created. The option
    ``frozen=True`` prevents ordinary reassignment of ``_atoms`` after
    construction; the frozenset itself prevents changes to its contents.
    """

    _atoms: frozenset[AtomicFormula]
    """The immutable set of atoms forming this conjunction.
    """

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
        construction in :meth:`CartesianBnf.conjoin_dnfs` uses this operation
        for each tuple of child cubes.
        """
        sets = tuple(cube._atoms for cube in cubes)
        return cls(frozenset.union(*sets)) if sets else cls()

    def conjoin(self, other: Cube) -> Cube:
        r"""Return a cube representing the conjunction of this cube with
        ``other``.

        Union the atom sets without changing either input. This is the two-input
        counterpart of :meth:`combine`. No normalization is performed:
        conjoining :math:`x > 0` with :math:`x \le 0` retains both atoms until a
        subsequent :meth:`normalize` call detects the contradiction. """
        return Cube(self._atoms | other._atoms)

    def implies(self, other: Cube) -> bool:
        r"""Try to establish that this cube logically implies ``other``.

        Every atom in ``other`` must follow from this cube's conjunction.
        First discharge atoms that occur explicitly here or follow from
        one of this cube's relations on the same polynomial. Compare their
        allowed signs: for example, :math:`x > 0` implies :math:`x \ge 0`
        because positive is a subset of nonnegative.

        Ask the RCF simplifier to establish the conjunction of the remaining
        atoms under all of this cube's assumptions. This can also establish
        implications between different polynomials, such as :math:`x - 1 > 0`
        implying :math:`x > 0`. Traverse atoms in sorted order for
        reproducibility.

        Return :obj:`True` only when the implication is established.
        :obj:`False` means that this heuristic check could not prove it and does
        not certify a counterexample. In a DNF, a :obj:`True` result allows this
        stronger cube to be removed when the weaker ``other`` cube is also
        present.
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
        r"""Try to express the disjunction between this cube and ``other`` as a
        single cube.

        Each cube must have exactly one atom absent from the other. Call these
        atoms :math:`A` and :math:`B`, and let :math:`C` be the conjunction of
        the common atoms. Both :math:`A` and :math:`B` must compare the same
        polynomial with zero.

        Factor :math:`(C \land A) \lor (C \land B)` as
        :math:`C \land (A \lor B)`. Combine :math:`A` and :math:`B` by taking
        the union of their allowed signs. For example, :math:`x < 0` and
        :math:`x = 0` combine to the single atom :math:`x \le 0`.

        If the union contains all three signs (negative, zero, and positive),
        :math:`A \lor B` is true and the result is just :math:`C`. This is the
        cut to the common conjunction: for example, :math:`C \land (x > 0)`
        and :math:`C \land (x \le 0)` merge to :math:`C`. If no atoms are common
        to the two cubes, this cut yields the empty cube, representing truth.

        Return a new cube equivalent to the disjunction, leaving both inputs
        unchanged. Return :obj:`None` when this rule does not apply; no general
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
        r"""Combine constraints on equal polynomials within this cube.

        Group atoms by their polynomial left-hand sides. Each relation
        permits some of the signs negative, zero, and positive. Since all
        atoms must hold, intersect the permitted signs within each group
        and replace the group by the single relation expressing that
        intersection. For example, :math:`x > 0` together with :math:`x \ge 0`
        becomes just :math:`x > 0`.

        If an intersection is empty, return ``None`` because the cube is
        contradictory, as with :math:`x > 0` together with :math:`x \le 0`.
        Otherwise return a new cube, leaving this one unchanged. An empty
        input yields the empty cube, representing truth.

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

        The first component is the number of atoms, so cubes with fewer atoms
        precede those with more atoms. Break ties lexicographically using the
        tuple of atoms in their existing RCF ordering. This makes reconstruction
        and greedy comparison choices independent of set iteration order.
        """
        return len(self._atoms), tuple(sorted(self._atoms))


class StrictDnf:
    """A mutable representation of a DNF as a private set of immutable cubes.

    Each cube represents a conjunction of atoms; the set represents their
    disjunction.

    Equality compares the stored cubes structurally; instances are unhashable.
    An instance with no cubes represents false; one containing the empty cube
    represents true. Construction, :meth:`add`, and :meth:`update` collect cubes
    without simplification beyond duplicate removal. Other redundancies and
    unnormalized cubes may remain.

    :meth:`insert` mutates an already syntactically reduced instance by adding
    a normalized cube with structural subsumption and cut. The batch methods
    :meth:`syntactic_reduction` and :meth:`semantic_subsumption` return new
    instances without changing their input: the former uses fixed atomic
    rules, the latter implication checks. Schedules in the Cartesian classes
    determine when these operations run.

    Construction retains both cubes below; :meth:`syntactic_reduction` merges
    them into a single cube representing ``y == 0``. The final equality verifies
    that the original instance still contains exactly the two input cubes.

    >>> from logic1.theories.RCF import VV
    >>> x, y = VV.get('x', 'y')
    >>> cubes = (Cube((x > 0, y == 0)), Cube((x <= 0, y == 0)))
    >>> strict_dnf = StrictDnf(cubes)
    >>> strict_dnf.syntactic_reduction() == StrictDnf((Cube((y == 0,)),))
    True
    >>> strict_dnf == StrictDnf(cubes)
    True

    :attr:`__slots__` stores the reference to :attr:`_cubes` directly in each
    instance, without a ``__dict__`` attribute dictionary. This reduces
    memory overhead but does not make instances immutable. Each instance owns
    its mutable set; the immutable cubes in it may be shared with other
    instances.
    """

    __slots__ = ('_cubes',)
    """Store only the ``_cubes`` reference, without an instance dictionary.
    """

    _cubes: set[Cube]
    """The independently owned mutable set of cubes forming this disjunction.
    """

    def __contains__(self, cube: Cube) -> bool:
        """Return whether a structurally equal cube is stored.
        """
        return cube in self._cubes

    def __eq__(self, other: object) -> bool:
        """Compare the stored cube sets by structural equality, not logical
        equivalence.
        """
        if not isinstance(other, StrictDnf):
            return NotImplemented
        return self._cubes == other._cubes

    def __init__(self, cubes: Iterable[Cube] = ()) -> None:
        """Collect the supplied cubes in a new set, removing duplicates only.
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
        """Add ``cube`` to ``self``, ignoring structural duplicates.

        Neither normalize the cube nor remove existing cubes. In particular,
        adding the empty cube makes the represented DNF true but leaves all
        other cubes stored in ``self``.
        """
        self._cubes.add(cube)

    def insert(self, cube: Cube) -> None:
        """Add ``cube`` to the represented DNF using subsumption and merging.

        ``cube`` must be normalized, and ``self`` must already be
        syntactically reduced, as returned by :meth:`syntactic_reduction`.
        These preconditions are not checked. The method modifies ``self`` in
        place to represent the disjunction of its original cubes with ``cube``,
        leaving ``self`` syntactically reduced.

        The initial candidate is ``cube``. A cube structurally subsumes another
        when its atom set is a subset of the other's. The algorithm compares
        the candidate with stored cubes using these rules:

        - If a stored cube subsumes the candidate, discard the candidate and
          stop.
        - If the candidate subsumes a stored cube, remove the stored cube and
          continue.
        - Otherwise, try :meth:`Cube.merge`, which combines cubes differing in
          one relation on the same polynomial. On success, remove the stored
          cube and continue with the merged cube as the new candidate.

        At each iteration, the method selects the stored cube with the lowest
        :meth:`Cube.sort_key` among those eligible for subsumption or a
        successful merge. It repeats the scan after each removal or merge; a
        changed candidate may allow further reductions. If no rule applies,
        it stores the candidate.

        An empty candidate represents truth and replaces all stored cubes.
        If ``self`` contains the empty cube, no other cubes are stored, since
        ``self`` is assumed to be syntactically reduced. The method then leaves
        ``self`` unchanged.

        No general RCF validity checks are performed. Immediate deletion can
        prevent later merges, so the deterministic result need not be minimal.
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
        """Return a new instance with cubes removed by semantic subsumption.

        A cube that implies another adds no solutions to their disjunction and
        can be removed. :meth:`Cube.implies` can establish implications beyond
        atom-set inclusion, such as :math:`x - 1 > 0` implying :math:`x > 0`.
        Inconclusive checks do not justify deletion.

        The new instance initially contains the same cubes as ``self``. The
        method orders the cubes by :meth:`Cube.sort_key` and compares each cube
        with the later cubes in that order, skipping removed cubes. It first
        tests whether the later cube implies the earlier one, then tests the
        converse if necessary. A successful proof removes the implying cube
        from the new instance. If neither direction is proved, both cubes
        remain. No cubes are created or modified, so earlier pairs need not be
        revisited. The original instance ``self`` remains unchanged.

        ``max_pair_checks`` must be a nonnegative integer or :obj:`None`; this
        precondition is not checked here. Each visited pair counts once, even
        if both implication directions are tried. The counter starts at zero
        on each call. :obj:`None` permits all pairs to be considered; zero
        returns a copy of ``self`` without sorting or implication checks.
        When the limit is reached, return the result obtained so far. The limit
        bounds pair visits, not the time spent on an individual implication
        check.
        """
        if max_pair_checks == 0:
            return StrictDnf(self._cubes)
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
        """Return a new instance representing a syntactically reduced DNF.

        The method starts with an empty ``StrictDnf`` instance and processes
        the cubes stored in ``self`` in :meth:`Cube.sort_key` order. Each cube
        is first normalized with :meth:`Cube.normalize`, which combines
        relations on the same polynomial. If normalization detects a
        contradiction, the cube is omitted. Otherwise, the normalized cube is
        immediately added to the new instance using :meth:`insert`.

        Insertion uses structural subsumption and merging as needed, leaving the
        new instance syntactically reduced. If the empty cube is inserted or
        produced by merging, it becomes the only stored cube and the result
        represents truth.

        The returned instance represents a DNF equivalent to the original;
        ``self`` remains unchanged. No general RCF validity checks are
        performed. Contradictions and redundancies that depend on relationships
        between different polynomials may remain.
        """
        result = StrictDnf()
        for cube in sorted(self._cubes, key=Cube.sort_key):
            normalized = cube.normalize()
            if normalized is not None:
                result.insert(normalized)
        return result

    def update(self, other: StrictDnf) -> None:
        """Add all cubes from ``other`` to ``self``, ignoring structural
        duplicates.

        The set owned by ``self`` is updated without normalization, subsumption,
        or merging. ``other`` remains unchanged. Distinct instances retain
        separate owned sets but may share immutable cubes. Updating ``self``
        with itself has no effect.
        """
        self._cubes.update(other._cubes)


class CartesianBnf:
    """Compute normal forms by distributing conjunction over disjunction.

    This class is selected by ``method="cartesian"`` together with
    ``simplification=Simplification.NONE`` in calls to the module-level
    :func:`dnf` and :func:`cnf` functions.

    The method :meth:`~CartesianBnf.dnf` converts the input to prenex form and
    preprocesses it with :meth:`simplify`. The quantifier-free matrix is
    recursively converted to a :class:`StrictDnf` instance, then reconstructed
    as a formula. The quantifier prefix is restored before
    :meth:`conservative_simplify` is applied. The method
    :meth:`~CartesianBnf.cnf` obtains CNF by computing the DNF of the negated
    input and negating the result.

    The method :meth:`strict_dnf` recursively traverses the input. It calls
    the combination methods :meth:`conjoin_dnfs` for conjunctions and
    :meth:`disjoin_dnfs` for disjunctions. They form the Cartesian product and
    union, respectively, of the strict DNFs of the immediate arguments of the
    conjunction or disjunction being processed. Both combination methods
    remove duplicates without further simplification. Before formula
    reconstruction, :meth:`finish_strict_dnf` applies semantic subsumption
    to the completed strict DNF. With a zero limit, it returns an equal,
    independent copy without sorting or implication checks.

    The methods :meth:`conjoin_dnfs`, :meth:`disjoin_dnfs`, and
    :meth:`finish_strict_dnf` are extension points for DNF-specific
    simplification. A subclass may override any one or more of them while
    sharing the traversal in :meth:`strict_dnf`.

    Overrides of the combination methods receive child DNFs lazily in input
    order, each already processed using the subclass's methods. They must
    leave child DNFs unchanged and accept an iterable that supports only one
    traversal. They may stop requesting children once the result is
    determined. The combination methods do not traverse formulas.
    """

    max_pair_checks: int | None
    """Limit on cube pairs examined per semantic subsumption pass.
    """

    def __init__(self, *, max_pair_checks: int | None) -> None:
        """Validate ``max_pair_checks`` and store it on ``self``.

        Accept a nonnegative integer or :obj:`None` for no limit. Zero disables
        implication checks. Raise :exc:`TypeError` for booleans and other
        unsupported types, or :exc:`ValueError` for negative integers.

        The limit counts cube pairs, not individual implication checks. It
        does not limit syntactic reduction or the time spent on one check.
        """
        if max_pair_checks is not None:
            if isinstance(max_pair_checks, bool) or not isinstance(max_pair_checks, int):
                raise TypeError("max_pair_checks must be an integer or None")
            if max_pair_checks < 0:
                raise ValueError("max_pair_checks must be nonnegative or None")
        self.max_pair_checks = max_pair_checks


    def cnf(self, f: Formula) -> Formula:
        """Compute a conjunctive normal form of ``f``.

        If ``f`` contains quantifiers, the result is in prenex form with a CNF
        matrix.

        Call :meth:`_dnf_from_formula` on the negation of ``f``, then negate the
        result and push negations inward to obtain CNF. Finally,
        :meth:`conservative_simplify` simplifies the result while preserving
        CNF.
        """
        return self.conservative_simplify(Not(self._dnf_from_formula(Not(f))).to_nnf())

    def conjoin_dnfs(self, children: Iterable[StrictDnf]) -> StrictDnf:
        """Conjoin the strict DNFs in ``children``.

        Form the Cartesian product of ``children``. For each tuple, construct
        a cube whose atom set is the union of the atom sets of the cubes in
        that tuple. Return a new :class:`StrictDnf` containing these cubes.
        The set representations of :class:`Cube` and :class:`StrictDnf`
        inherently identify repeated atoms and equal cubes. No normalization,
        subsumption, or merging is performed.

        With no children, return the singleton containing the empty cube,
        representing truth.

        This exact cube set is the contract used by after-step reduction.
        """
        result = StrictDnf()
        for cubes in itertools.product(*children):
            result.add(Cube.combine(cubes))
        return result

    def conservative_simplify(self, f: Formula) -> Formula:
        """Simplify ``f`` while preserving DNF and CNF.

        If ``f`` is in DNF, the result must remain in DNF. The same requirement
        holds for CNF. If ``f`` is prenex, it applies to the quantifier-free
        matrix. The correctness of :meth:`~CartesianBnf.dnf` and
        :meth:`~CartesianBnf.cnf` depends on this guarantee, which must be
        preserved when changing this method.

        Technically, the function :func:`.RCF.simplify.simplify` is called with
        ``explode_always=False``, disabling unconditional splitting of atoms.
        """
        return _simplify(f, explode_always=False)

    def disjoin_dnfs(self, children: Iterable[StrictDnf]) -> StrictDnf:
        """Disjoin the strict DNFs in ``children``.

        Form the union of the cube sets of the strict DNFs in ``children``.
        Return a new :class:`StrictDnf` containing these cubes. The set
        representation of :class:`StrictDnf` inherently identifies equal cubes.
        No normalization, subsumption, or merging is performed.

        With no children, return the empty strict DNF, representing false.

        This exact cube set is the contract used by after-step reduction.
        """
        result = StrictDnf()
        for child in children:
            result.update(child)
        return result

    def dnf(self, f: Formula) -> Formula:
        """Compute a disjunctive normal form of ``f``.

        If ``f`` contains quantifiers, the result is in prenex form with a DNF
        matrix.

        The method constructs a DNF with :meth:`_dnf_from_formula` and applies
        :meth:`conservative_simplify` to the result, preserving DNF.
        """
        return self.conservative_simplify(self._dnf_from_formula(f))

    def _dnf_from_and_or(self, f: AndOr) -> Formula:
        """Construct a DNF formula from the prepared And/Or formula ``f``.

        The caller supplies ``f`` after initial simplification with
        :meth:`simplify`. All atomic right-hand sides must be zero. The type
        :class:`AndOr` describes the Boolean structure but does not encode
        this additional requirement.

        The method constructs a :class:`StrictDnf` instance with
        :meth:`strict_dnf` and passes it to :meth:`finish_strict_dnf`. Each cube
        in the returned instance becomes a conjunction of its atoms; the result
        is the disjunction of those conjunctions.

        For deterministic output, reconstruction sorts cubes by
        :meth:`Cube.sort_key` and atoms by their RCF ordering.
        """
        strict_dnf = self.finish_strict_dnf(self.strict_dnf(f))
        cubes = sorted(strict_dnf, key=Cube.sort_key)
        return Or(*(And(*sorted(cube)) for cube in cubes))

    def _dnf_from_formula(self, f: Formula) -> Formula:
        """Construct a DNF from arbitrary ``f`` without final simplification.

        Apply :meth:`~logic1.firstorder.formula.Formula.to_pnf` and then
        :meth:`simplify` to ``f``. Pass the resulting quantifier-free part, of
        type :class:`AndOr`, to :meth:`_dnf_from_and_or`. Restore any quantifier
        prefix afterward.

        The public methods :meth:`~CartesianBnf.dnf` and
        :meth:`~CartesianBnf.cnf` share this construction. Each applies
        :meth:`conservative_simplify` after obtaining the requested normal form,
        so CNF is simplified after negation rather than at the intermediate DNF
        stage.
        """
        f = self.simplify(f.to_pnf())
        matrix, prefix = f.matrix()
        assert is_and_or(matrix)
        dnf = self._dnf_from_and_or(matrix)
        return dnf.quantify(prefix)

    def finish_strict_dnf(self, strict_dnf: StrictDnf) -> StrictDnf:
        """Apply semantic subsumption to ``strict_dnf``.

        Call :meth:`StrictDnf.semantic_subsumption` with the limit in
        :attr:`max_pair_checks`. A cube is removed only if its conjunction is
        proved to imply that of another cube. No cubes are created.

        The pair counter starts at zero for each call. Return a new
        :class:`StrictDnf` instance representing an equivalent DNF, leaving
        ``strict_dnf`` unchanged.
        """
        return strict_dnf.semantic_subsumption(self.max_pair_checks)

    def simplify(self, f: Formula) -> Formula:
        """Preprocess ``f`` for Cartesian distribution.

        The RCF function :func:`~logic1.theories.RCF.simplify.simplify` is
        called with two options chosen for separate reasons.

        Setting ``explode_always=False`` disables unconditional splitting of
        atoms. Splitting product or square-sum constraints can introduce
        disjunctions inside conjunctions, increasing the number of combinations
        generated by Cartesian distribution.

        Setting ``substitute=1`` retains value substitution but disables
        monomial substitution. Context-dependent substitution can turn equal
        atoms in different branches into different atoms, hiding opportunities
        for structural subsumption and cut. On the SMT-LIB benchmark
        ``half-2``, enabling monomial substitution increased the strict DNF from
        4 to 400 cubes with either value of ``explode_always``. Whether lost
        structural matches explain this increase remains unproven. The
        experiment and a separate example demonstrating a lost cut are recorded
        in ``doc/notes/RCF/dnf.md``.
        """
        return _simplify(f, explode_always=False, substitute=1)

    def strict_dnf(self, f: AndOr) -> StrictDnf:
        """Traverse ``f`` and construct its strict DNF.

        An atom yields one singleton cube, false yields no cubes, and true
        yields the empty cube. For every processed conjunction or disjunction,
        pass the DNFs of its immediate arguments to :meth:`conjoin_dnfs` or
        :meth:`disjoin_dnfs`, respectively. Use the returned DNF unchanged in
        the enclosing computation.

        Child DNFs are supplied lazily in argument order. Subclasses implement
        their simplification schedule through these two methods, sharing this
        traversal. An implementation change must preserve these calls and
        their order, including evaluation of children only when requested.
        """
        if isinstance(f, _F):
            return StrictDnf()
        if isinstance(f, _T):
            return StrictDnf((Cube(),))
        if isinstance(f, AtomicFormula):
            return StrictDnf((Cube((f,)),))
        if isinstance(f, And):
            children = (self.strict_dnf(arg) for arg in f.args)
            return self.conjoin_dnfs(children)
        if isinstance(f, Or):
            children = (self.strict_dnf(arg) for arg in f.args)
            return self.disjoin_dnfs(children)
        raise ValueError(f"Unexpected formula type: {type(f)}")


class FinalCartesianBnf(CartesianBnf):
    """Apply syntactic reduction and semantic subsumption after construction.

    This class is selected by ``method="cartesian"`` together with
    ``simplification=Simplification.FINAL`` in calls to the module-level
    :func:`dnf` and :func:`cnf` functions.

    The methods inherited from :class:`CartesianBnf` construct the complete
    strict DNF without DNF-specific simplification. The override of
    :meth:`finish_strict_dnf` then applies :meth:`StrictDnf.syntactic_reduction`
    followed by :meth:`StrictDnf.semantic_subsumption` before formula
    reconstruction. This avoids reducing intermediate DNFs, but requires storing
    the entire unreduced strict DNF in memory.

    In the following example, initial simplification leaves the six-disjunct
    formula ``f`` unchanged. During syntactic reduction, structural subsumption
    removes the cube for ``And(c > 0, b > 0, a > 0)`` because its atom set is a
    superset of the atom set of the cube for ``And(c > 0, a > 0)``. A cut merges
    the two cubes containing ``e > 0`` into one cube representing ``e > 0``,
    since their remaining atoms ``d > 0`` and ``d <= 0`` are complementary.
    Four cubes remain. No cubes are removed by semantic subsumption in this
    example.

    >>> from logic1.theories.RCF import VV
    >>> a, b, c, d, e = VV.get('a', 'b', 'c', 'd', 'e')
    >>> f = Or(And(e > 0, d <= 0),
    ...        And(e > 0, d > 0),
    ...        And(c > 0, b <= 0),
    ...        And(c > 0, b > 0, a <= 0),
    ...        And(c > 0, b > 0, a > 0),
    ...        And(c > 0, a > 0))
    >>> final_cartesian_bnf = FinalCartesianBnf(max_pair_checks=500)
    >>> final_cartesian_bnf.simplify(f) == f
    True
    >>> final_cartesian_bnf.dnf(f)
    Or(e > 0, And(c > 0, b <= 0), And(c > 0, b > 0, a <= 0), And(c > 0, a > 0))
    """

    def finish_strict_dnf(self, strict_dnf: StrictDnf) -> StrictDnf:
        """Apply batch syntactic reduction, then bounded semantic subsumption.

        First apply :meth:`StrictDnf.syntactic_reduction`. Second call
        :meth:`CartesianBnf.finish_strict_dnf` to remove further redundant
        cubes by semantic subsumption.

        Only the second pass uses ``max_pair_checks``; a zero budget still
        performs syntactic reduction. Return a new collection for formula
        reconstruction, leaving the supplied strict DNF unchanged.
        """
        return super().finish_strict_dnf(strict_dnf.syntactic_reduction())


class AfterStepCartesianBnf(CartesianBnf):
    """Syntactically reduce the DNF of each conjunction or disjunction.

    This class is selected by ``method="cartesian"`` together with
    ``simplification=Simplification.AFTER_STEP`` in calls to the module-level
    :func:`dnf` and :func:`cnf` functions.

    The overrides of :meth:`conjoin_dnfs` and :meth:`disjoin_dnfs` apply
    :meth:`StrictDnf.syntactic_reduction` after the inherited methods have
    constructed the complete Cartesian product or union. The traversal in
    :meth:`~CartesianBnf.strict_dnf` therefore uses reduced child DNFs when
    processing enclosing conjunctions and disjunctions. The inherited method
    :meth:`~CartesianBnf.finish_strict_dnf` applies
    :meth:`StrictDnf.semantic_subsumption` once, before formula reconstruction.

    Reducing child DNFs can shrink subsequent Cartesian products. Completing
    each product before reduction avoids repeated reduction of partial
    products, but requires storing the entire unreduced product in memory.
    This placement of syntactic reduction matches Redlog's recursive
    simplification, though the rules differ.

    The example in :class:`FinalCartesianBnf` gives the same result with this
    class. In the following example, ``f`` has ten atom occurrences and is
    unchanged by initial simplification. During syntactic reduction, the four
    cubes of the second disjunction merge into the cube for
    ``And(y <= 0, x <= 0)``. The merges combine ``x == 0`` and ``x < 0`` into
    ``x <= 0``, and likewise for ``y``. The enclosing conjunction therefore
    produces two cubes instead of eight. Normalization discards the cube
    containing both ``y > 0`` and ``y <= 0``. One cube with three atoms
    remains. No cubes are removed by semantic subsumption in this example.

    >>> from logic1.theories.RCF import VV
    >>> x, y, z = VV.get('x', 'y', 'z')
    >>> f = And(Or(z > 0, y > 0),
    ...         Or(And(y == 0, x == 0),
    ...            And(y == 0, x < 0),
    ...            And(y < 0, x == 0),
    ...            And(y < 0, x < 0)))
    >>> after_step_cartesian_bnf = AfterStepCartesianBnf(max_pair_checks=500)
    >>> after_step_cartesian_bnf.simplify(f) == f
    True
    >>> after_step_cartesian_bnf.dnf(f)
    And(z > 0, y <= 0, x <= 0)

    The next example demonstrates how raising ``max_pair_checks`` allows
    semantic subsumption to remove more cubes. The formula ``g`` defined below
    has 36 cubes and 72 atom occurrences; initial simplification leaves it
    unchanged. The dictionary records cube counts, atom occurrences, and the
    frequencies of cube sizes. Its keys identify the comparison limits; ``None``
    removes the limit.

    >>> from collections import Counter
    >>> from itertools import product
    >>> P = product(range(5, -1, -1), repeat=2)
    >>> g = Or(*(And(y - i > 0, x - j > 0) for i, j in P))
    >>> g
    Or(And(y - 5 > 0, x - 5 > 0), And(y - 5 > 0, x - 4 > 0),
       And(y - 5 > 0, x - 3 > 0), And(y - 5 > 0, x - 2 > 0),
       And(y - 5 > 0, x - 1 > 0), And(y - 5 > 0, x > 0),
       And(y - 4 > 0, x - 5 > 0), And(y - 4 > 0, x - 4 > 0),
       And(y - 4 > 0, x - 3 > 0), And(y - 4 > 0, x - 2 > 0),
       And(y - 4 > 0, x - 1 > 0), And(y - 4 > 0, x > 0),
       And(y - 3 > 0, x - 5 > 0), And(y - 3 > 0, x - 4 > 0),
       And(y - 3 > 0, x - 3 > 0), And(y - 3 > 0, x - 2 > 0),
       And(y - 3 > 0, x - 1 > 0), And(y - 3 > 0, x > 0),
       And(y - 2 > 0, x - 5 > 0), And(y - 2 > 0, x - 4 > 0),
       And(y - 2 > 0, x - 3 > 0), And(y - 2 > 0, x - 2 > 0),
       And(y - 2 > 0, x - 1 > 0), And(y - 2 > 0, x > 0),
       And(y - 1 > 0, x - 5 > 0), And(y - 1 > 0, x - 4 > 0),
       And(y - 1 > 0, x - 3 > 0), And(y - 1 > 0, x - 2 > 0),
       And(y - 1 > 0, x - 1 > 0), And(y - 1 > 0, x > 0),
       And(y > 0, x - 5 > 0), And(y > 0, x - 4 > 0),
       And(y > 0, x - 3 > 0), And(y > 0, x - 2 > 0),
       And(y > 0, x - 1 > 0), And(y > 0, x > 0))
    >>> after_step_cartesian_bnf.simplify(g) == g
    True
    >>> statistics = []
    >>> for checks in (10, 30, None):
    ...     after_step_bnf = AfterStepCartesianBnf(max_pair_checks=checks)
    ...     result = after_step_bnf.dnf(g)
    ...     cubes = result.args if isinstance(result, Or) else (result,)
    ...     sizes = [len(list(cube.atoms())) for cube in cubes]
    ...     statistics.append((checks, dict(Counter(sizes)), sum(sizes)))
    >>> statistics
    [(10, {2: 31}, 62), (30, {2: 19}, 38), (None, {2: 1}, 2)]
    >>> result
    And(y > 0, x > 0)

    The statistics give the following table:

    +---------------------+----------------------------+-------------+
    | ``max_pair_checks`` | Atoms per cube             | Total atoms |
    +=====================+============================+=============+
    |                  10 | 31 cubes with 2 atoms each |          62 |
    +---------------------+----------------------------+-------------+
    |                  30 | 19 cubes with 2 atoms each |          38 |
    +---------------------+----------------------------+-------------+
    |                None | 1 cube with 2 atoms        |           2 |
    +---------------------+----------------------------+-------------+
    """

    def conjoin_dnfs(self, children: Iterable[StrictDnf]) -> StrictDnf:
        """Conjoin the strict DNFs in ``children``, then reduce the result.

        Use :meth:`CartesianBnf.conjoin_dnfs` to construct the strict DNF from
        the Cartesian product of their cube sets. Apply
        :meth:`StrictDnf.syntactic_reduction` to the complete result before
        returning it.
        """
        return CartesianBnf.conjoin_dnfs(self, children).syntactic_reduction()

    def disjoin_dnfs(self, children: Iterable[StrictDnf]) -> StrictDnf:
        """Disjoin the strict DNFs in ``children``, then reduce the result.

        Use :meth:`CartesianBnf.disjoin_dnfs` to construct the strict DNF from
        the union of their cube sets. Apply
        :meth:`StrictDnf.syntactic_reduction` to the complete result before
        returning it.
        """
        return CartesianBnf.disjoin_dnfs(self, children).syntactic_reduction()


class DuringStepCartesianBnf(CartesianBnf):
    """Simplify cubes as they are produced by Cartesian distribution.

    This class is selected by ``simplification=Simplification.DURING_STEP``
    in calls to ``dnf`` and ``cnf``. A recursive call for an ``And`` or ``Or``
    subformula of the preprocessed input uses ``StrictDnf.insert`` to reduce
    its DNF as cubes are added. An ``And`` call normalizes newly conjoined
    cubes and reduces partial products before multiplying by the next
    child's DNF. Semantic subsumption runs once, after the complete strict
    DNF has been constructed.

    Reducing partial products can avoid retaining a full unreduced Cartesian
    product. Repeated structural subsumption and merge checks can cost time,
    and greedy deletion can prevent later cuts. An ``Or`` call stops when its
    result contains the empty cube, representing true; an ``And`` call stops
    when a partial product is empty, representing false.

    The ``max_pair_checks`` example in :class:`AfterStepCartesianBnf` gives the
    same results with this class. The following example uses the same input as
    the example in :class:`FinalCartesianBnf`. This class improves the result
    from four cubes to two. Initial simplification leaves ``f`` unchanged.
    During syntactic reduction, the two cubes containing the atom ``e > 0``
    cut to ``e > 0``. The two three-atom cubes containing the atom ``c > 0``
    are inserted before ``And(c > 0, a > 0)``. Their cut gives
    ``And(c > 0, b > 0)``, which then cuts with ``And(c > 0, b <= 0)`` to
    give ``c > 0``. The last cube, ``And(c > 0, a > 0)``, is subsumed when
    it arrives. Two cubes remain. No cubes are removed by semantic subsumption
    in this example.

    >>> from logic1.theories.RCF import VV
    >>> a, b, c, d, e = VV.get('a', 'b', 'c', 'd', 'e')
    >>> f = Or(And(e > 0, d <= 0),
    ...        And(e > 0, d > 0),
    ...        And(c > 0, b <= 0),
    ...        And(c > 0, b > 0, a <= 0),
    ...        And(c > 0, b > 0, a > 0),
    ...        And(c > 0, a > 0))
    >>> during_step_cartesian_bnf = DuringStepCartesianBnf(max_pair_checks=500)
    >>> during_step_cartesian_bnf.simplify(f) == f
    True
    >>> during_step_cartesian_bnf.dnf(f)
    Or(e > 0, c > 0)
    """

    def conjoin_dnfs(self, children: Iterable[StrictDnf]) -> StrictDnf:
        """Conjoin the strict DNFs in ``children``, reducing as cubes are added.

        Start with a strict DNF containing the empty cube, representing truth.
        For each strict DNF in ``children``, pair its cubes with those of the
        accumulated result. Conjoin each pair and normalize the resulting
        cube. Use :meth:`StrictDnf.insert` to build the next result from the
        surviving cubes. Sort both cube sets by :meth:`Cube.sort_key` for
        deterministic insertion order.

        Return the final result. If an intermediate result is empty, return
        it without evaluating further ``children``.
        """
        result = StrictDnf((Cube(),))
        for child_dnf in children:
            child = sorted(child_dnf, key=Cube.sort_key)
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

    def disjoin_dnfs(self, children: Iterable[StrictDnf]) -> StrictDnf:
        """Disjoin the strict DNFs in ``children``, reducing as cubes are added.

        Start with an empty strict DNF and insert the cubes of each strict
        DNF in ``children`` using :meth:`StrictDnf.insert`. Sort each child's
        cubes by :meth:`Cube.sort_key` for deterministic insertion order.

        Return the final result. If the result contains the empty cube after
        processing a child, it represents truth; return it without evaluating
        further ``children``.
        """
        result = StrictDnf()
        for child_dnf in children:
            for cube in sorted(child_dnf, key=Cube.sort_key):
                result.insert(cube)
            if Cube() in result:
                break
        return result


class PyedaBnf(abc.bnf.BooleanNormalForm[AtomicFormula, Term, Variable, Number]):
    """Implements the abstract methods
    :meth:`~.abc.bnf.BooleanNormalForm.simplify` and
    :meth:`~.abc.bnf.BooleanNormalForm.final_simplify` of its super class
    :class:`.abc.bnf.BooleanNormalForm`. Inherits the methods
    :meth:`~.abc.bnf.BooleanNormalForm.cnf` and
    :meth:`~.abc.bnf.BooleanNormalForm.dnf`, which should be called via the
    functions :func:`.cnf` and :func:`.dnf`, respectively.
    """

    def simplify(self, f: Formula) -> Formula:
        """Implement :meth:`.abc.bnf.BooleanNormalForm.simplify`.
        """
        return _simplify(f)

    def conservative_simplify(self, f: Formula) -> Formula:
        """Simplify ``f`` while preserving DNF and CNF.

        If ``f`` is in DNF or CNF, preserve that normal form. If ``f`` is
        prenex, this applies to its quantifier-free matrix.

        Unconditional splitting is disabled with ``explode_always=False``.
        """
        return _simplify(f, explode_always=False)

    def final_simplify(self, f: Formula) -> Formula:
        """Implement the hook required by :class:`.abc.bnf.BooleanNormalForm`.
        """
        return self.conservative_simplify(f)


def _normal_form(method: Literal["cartesian", "pyeda"], simplification: Simplification | int,
                 max_pairs: int | None) -> CartesianBnf | PyedaBnf:
    """Select an implementation for ``dnf`` and ``cnf``.

    ``method="pyeda"`` returns ``PyedaBnf`` without inspecting the other
    options. For ``method="cartesian"``, convert ``simplification`` to a
    ``Simplification`` member, rejecting booleans. Select ``CartesianBnf`` for
    ``Simplification.NONE`` or the corresponding simplifying subclass for
    the other values.

    ``max_pairs`` receives the public ``max_pair_checks`` argument. With
    ``Simplification.NONE``, it must equal 500 and is unused; construct
    ``CartesianBnf(max_pair_checks=0)`` to disable implication checks.
    Otherwise, pass ``max_pairs`` to the selected constructor, which
    validates and stores the limit on cube pairs examined.
    """
    if method == "pyeda":
        return PyedaBnf()
    if method != "cartesian":
        raise ValueError(f"Unknown method: {method}")
    if isinstance(simplification, bool) or not isinstance(simplification, int):
        raise TypeError("simplification must be a Simplification member or an integer")
    schedule = Simplification(simplification)
    if schedule is Simplification.NONE:
        if max_pairs != 500:
            raise ValueError("Comparison limits require a simplification schedule other than NONE")
        return CartesianBnf(max_pair_checks=0)
    classes = {
        Simplification.FINAL: FinalCartesianBnf,
        Simplification.AFTER_STEP: AfterStepCartesianBnf,
        Simplification.DURING_STEP: DuringStepCartesianBnf}
    return classes[schedule](max_pair_checks=max_pairs)


def cnf(f: Formula, method: Literal["cartesian", "pyeda"] = "cartesian",
        *, simplification: Simplification | int = Simplification.FINAL,
        max_pair_checks: int | None = 500) -> Formula:
    """Compute a CNF of ``f``, with a prenex quantifier prefix when needed.

    CNF computation uses duality: negate ``f``, compute a DNF of that
    negation, then negate the result and move negations inward.

    >>> from logic1.firstorder import And, Or, Equivalent
    >>> from logic1.theories.RCF import VV
    >>> x, y, z = VV.get('x', 'y', 'z')
    >>> f = Or(And(x > 0, y > 0), z > 0)
    >>> cnf(f)
    And(Or(z > 0, y > 0), Or(z > 0, x > 0))
    >>> g = Equivalent(x == 0, y == 0)
    >>> cnf(g)
    And(Or(y == 0, x != 0), Or(y != 0, x == 0))

    See :func:`dnf` for the documentation of the options. Find further examples
    below.
    """
    return _normal_form(method, simplification, max_pair_checks).cnf(f)


def dnf(f: Formula, method: Literal["cartesian", "pyeda"] = "cartesian",
        *, simplification: Simplification | int = Simplification.FINAL,
        max_pair_checks: int | None = 500) -> Formula:
    """Compute a DNF of ``f``, with a prenex quantifier prefix when needed.

    The ``method`` option selects either distribution of conjunctions over
    disjunctions (``"cartesian"``) or Boolean abstraction and Espresso
    minimization (``"pyeda"``). Both methods apply
    :func:`.RCF.simplify.simplify` before and after DNF construction.

    For ``method="cartesian"``, the ``simplification`` option controls the
    schedule when to apply syntactic reduction pass to DNFs recursively produced
    during construction. See :class:`Simplification` for the available
    options.

    The ``max_pair_checks`` option limits the number of cube pairs examined in
    this subsequent semantic subsumption pass, to 500 by default. Set it to
    ``0`` to disable the pass or ``None`` to remove the pair limit. This option
    limits neither syntactic reduction nor the time spent on an individual
    implication check. For the Cartesian method with
    ``simplification=Simplification.NONE``, ``max_pair_checks`` must retain
    its default value and is unused.

    For ``method="pyeda"``, both ``simplification`` and ``max_pair_checks`` are
    ignored.

    >>> from logic1.firstorder import And, Or, Equivalent
    >>> from logic1.theories.RCF import VV
    >>> x, y, z = VV.get('x', 'y', 'z')
    >>> f = And(Or(x > 0, y > 0), z > 0)
    >>> dnf(f)
    Or(And(z > 0, y > 0), And(z > 0, x > 0))
    >>> g = Equivalent(x == 0, y == 0)
    >>> dnf(g)
    Or(And(y == 0, x == 0), And(y != 0, x != 0))

    .. seealso::

      :ref:`Simplification <api-RCF-simplify>`
        for the RCF simplifier used before and after normal form construction,
        and for final semantic subsumption in the simplifying schedules.
    """
    return _normal_form(method, simplification, max_pair_checks).dnf(f)