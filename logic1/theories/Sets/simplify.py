"""
This module implements *deep simplification* through the generation and
propagation of internal representations during recursion. It is an adaptation
of the *standard simplifier* from [DolzmannSturm-1997]_ tailored to the
specific requirements of Sets.
"""
from __future__ import annotations

from typing import Iterable, Iterator, Never, Optional, Self

from ... import abc
from dataclasses import dataclass, field

from ...firstorder import And, _F, Or, _T
from .atomic import AtomicFormula, C, C_, Eq, Index, Ne, oo, Variable
from .typing import Formula

from ...support.tracing import trace  # noqa


@dataclass(unsafe_hash=True)
class UnionFind:
    _parents: dict[Variable, Variable] = field(default_factory=dict)

    def copy(self) -> UnionFind:
        return UnionFind(self._parents.copy())

    def find(self, v: Variable) -> Variable:
        try:
            root = self.find(self._parents[v])
        except KeyError:
            return v
        self._parents[v] = root
        return root

    def union(self, v1: Variable, v2: Variable) -> None:
        root1 = self.find(v1)
        root2 = self.find(v2)
        if root1 == root2:
            return
        if Variable.sort_key(root1) <= Variable.sort_key(root2):
            self._parents[root2] = root1
        else:
            self._parents[root1] = root2

    def equations(self) -> Iterator[Eq]:
        for y in self._parents:
            yield Eq(self.find(y), y)


@dataclass
class InternalRepresentation(
        abc.simplify.InternalRepresentation[AtomicFormula, Variable, Variable, Never]):
    """Implements the abstract methods
    :meth:`add() <.abc.simplify.InternalRepresentation.add>`,
    :meth:`extract() <.abc.simplify.InternalRepresentation.extract>`, and
    :meth:`next_() <.abc.simplify.InternalRepresentation.next_>` of it super
    class :class:`.abc.simplify.InternalRepresentation`. Required by
    :class:`.Sets.simplify.Simplify` for instantiating the type variable
    :data:`.abc.simplify.ρ` of :class:`.abc.simplify.Simplify`.
    """
    _options: abc.simplify.Options
    _min_card: Index = 1
    _max_card: Index = oo
    _equations: UnionFind = field(default_factory=UnionFind)
    _inequations: set[Ne] = field(default_factory=set)

    def add(self, gand: type[And | Or], atoms: Iterable[AtomicFormula]) -> abc.simplify.RESTART:
        """Implements the abstract method :meth:`.abc.simplify.InternalRepresentation.add`.
        """
        for atom in atoms:
            if gand is Or:
                atom = atom.to_complement()
            # Collect information
            match atom:
                case C(index=n):
                    if n > self._min_card:
                        self._min_card = n
                    if self._min_card > self._max_card:
                        raise InternalRepresentation.Inconsistent()
                case C_(index=n):
                    if n - 1 < self._max_card:
                        self._max_card = n - 1
                    if self._min_card > self._max_card:
                        raise InternalRepresentation.Inconsistent()
                case Eq(lhs=lhs, rhs=rhs):
                    self._equations.union(lhs, rhs)
                case Ne(lhs=lhs, rhs=rhs):
                    self._inequations.add(atom)
                case _:
                    assert False
            for ne in self._inequations:
                if self._equations.find(ne.lhs) == self._equations.find(ne.rhs):
                    raise InternalRepresentation.Inconsistent()
            for ne in self._inequations:
                if self._equations.find(ne.lhs) == self._equations.find(ne.rhs):
                    raise InternalRepresentation.Inconsistent()
        return abc.simplify.RESTART.OTHERS

    def extract(self, gand: type[And | Or], ref: InternalRepresentation) -> list[AtomicFormula]:
        """Implements the abstract method :meth:`.abc.simplify.InternalRepresentation.extract`.
        """
        def canonicalize(ne: Ne) -> Ne:
            ne_subs = ne.subs({ne.lhs: self._equations.find(ne.lhs),
                               ne.rhs: self._equations.find(ne.rhs)})
            if Variable.sort_key(ne_subs.lhs) <= Variable.sort_key(ne_subs.rhs):
                return ne_subs
            else:
                return Ne(ne_subs.rhs, ne_subs.lhs)

        L: list[AtomicFormula] = []
        if self._min_card > ref._min_card:
            L.append(C(self._min_card))
        if self._max_card < ref._max_card:
            L.append(C_(self._max_card + 1))
        for eq in self._equations.equations():
            if ref._equations.find(eq.lhs) != ref._equations.find(eq.rhs):
                L.append(eq)
        canonical_ref_inequations = {canonicalize(ne) for ne in ref._inequations}
        for ne in self._inequations:
            ne = canonicalize(ne)
            if ne not in canonical_ref_inequations:
                L.append(ne)
        if gand is Or:
            L = [atom.to_complement() for atom in L]
        return L

    def next_(self, remove: Optional[Variable] = None) -> Self:
        """Implements the abstract method :meth:`.abc.simplify.next_`.
        """
        if remove is None:
            equations = self._equations.copy()
            inequations = self._inequations.copy()
        else:
            equations = UnionFind()
            for eq in self._equations.equations():
                if remove not in eq.fvars():
                    equations.union(eq.lhs, eq.rhs)
            inequations = {ne for ne in self._inequations if remove not in ne.fvars()}
        return self.__class__(self._options, self._min_card, self._max_card,
                              equations, inequations)


@dataclass(frozen=True)
class Simplify(abc.simplify.Simplify[
        AtomicFormula, Variable, Variable, Never, InternalRepresentation, abc.simplify.Options]):
    """Deep simplification in the style of [DolzmannSturm-1997]_. Implements
    the abstract methods :meth:`create_initial_representation
    <.abc.simplify.Simplify.create_initial_representation>` and :meth:`simpl_at
    <.abc.simplify.Simplify.simpl_at>` of its super class
    :class:`.abc.simplify.Simplify`.

    The simplifier should be called via :func:`.simplify`, as described below.
    In addition, this class inherits :meth:`.abc.simplify.Simplify.is_valid`,
    which should be called via :func:`.is_valid`, as described below.
    """

    _options: abc.simplify.Options = field(default_factory=abc.simplify.Options)

    def create_initial_representation(self, assume=Iterable[AtomicFormula]) \
            -> InternalRepresentation:
        """Implements the abstract method
        :meth:`.abc.simplify.Simplify.create_initial_representation`.
        """
        ir = InternalRepresentation(self._options)
        for atom in assume:
            simplified_atom = self.simpl_at(atom, And)
            match simplified_atom:
                case AtomicFormula():
                    ir.add(And, [simplified_atom])
                case _T():
                    continue
                case _F():
                    raise ir.Inconsistent()
                case _:
                    assert False, simplified_atom
        return ir

    def simpl_at(self, atom: AtomicFormula, context: Optional[type[And] | type[Or]]) -> Formula:
        """Implements the abstract method
        :meth:`.abc.simplify.Simplify.simpl_at`.
        """
        return atom.simplify()


def simplify(f: Formula, assume: Iterable[AtomicFormula] = []) -> Formula:
    return Simplify().simplify(f, assume)


def is_valid(f: Formula, assume: Iterable[AtomicFormula] = []) -> Optional[bool]:
    return Simplify().is_valid(f, assume)
