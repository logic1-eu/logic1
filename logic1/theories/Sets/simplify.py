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


@dataclass(frozen=True)
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
    _ref_min_card: Index
    _ref_max_card: Index
    _ref_equations: UnionFind
    _ref_inequations: set[Ne]

    _cur_min_card: Index
    _cur_max_card: Index
    _cur_equations: UnionFind
    _cur_inequations: set[Ne]

    def __init__(self) -> None:
        self._ref_min_card = 1
        self._ref_max_card = oo
        self._ref_equations = UnionFind()
        self._ref_inequations = set()
        self._cur_min_card = 1
        self._cur_max_card = oo
        self._cur_equations = UnionFind()
        self._cur_inequations = set()

    def __repr__(self) -> str:
        return (f'InternalRepresentation([{self._ref_min_card}..{self._ref_max_card}], '
                f'{self._ref_equations}, {self._ref_inequations} | '
                f'[{self._cur_min_card}..{self._cur_max_card}], '
                f'{self._cur_equations}, {self._cur_inequations})')

    def add(self, gand: type[And | Or], atoms: Iterable[AtomicFormula]) -> abc.simplify.Restart:
        """Implements the abstract method :meth:`.abc.simplify.InternalRepresentation.add`.
        """
        for atom in atoms:
            if gand is Or:
                atom = atom.to_complement()
            # Collect information
            match atom:
                case C(index=n):
                    if n > self._cur_min_card:
                        self._cur_min_card = n
                    if self._cur_min_card > self._cur_max_card:
                        raise InternalRepresentation.Inconsistent()
                case C_(index=n):
                    if n - 1 < self._cur_max_card:
                        self._cur_max_card = n - 1
                    if self._cur_min_card > self._cur_max_card:
                        raise InternalRepresentation.Inconsistent()
                case Eq(lhs=lhs, rhs=rhs):
                    self._cur_equations.union(lhs, rhs)
                case Ne(lhs=lhs, rhs=rhs):
                    self._cur_inequations.add(atom)
                case _:
                    assert False
            for ne in self._cur_inequations:
                if self._cur_equations.find(ne.lhs) == self._cur_equations.find(ne.rhs):
                    raise InternalRepresentation.Inconsistent()
            for ne in self._cur_inequations:
                if self._cur_equations.find(ne.lhs) == self._cur_equations.find(ne.rhs):
                    raise InternalRepresentation.Inconsistent()
        return abc.simplify.Restart.OTHERS

    def extract(self, gand: type[And | Or]) -> list[AtomicFormula]:
        """Implements the abstract method :meth:`.abc.simplify.InternalRepresentation.extract`.
        """
        def canonicalize(ne: Ne) -> Ne:
            ne_subs = ne.subs({ne.lhs: self._cur_equations.find(ne.lhs),
                               ne.rhs: self._cur_equations.find(ne.rhs)})
            if Variable.sort_key(ne_subs.lhs) <= Variable.sort_key(ne_subs.rhs):
                return ne_subs
            else:
                return Ne(ne_subs.rhs, ne_subs.lhs)

        L: list[AtomicFormula] = []
        if self._cur_min_card > self._ref_min_card:
            L.append(C(self._cur_min_card))
        if self._cur_max_card < self._ref_max_card:
            L.append(C_(self._cur_max_card + 1))
        for eq in self._cur_equations.equations():
            if self._ref_equations.find(eq.lhs) != self._ref_equations.find(eq.rhs):
                L.append(eq)
        canonical_ref_inequations = {canonicalize(ne) for ne in self._ref_inequations}
        for ne in self._cur_inequations:
            ne = canonicalize(ne)
            if ne not in canonical_ref_inequations:
                L.append(ne)
        if gand is Or:
            L = [atom.to_complement() for atom in L]
        return L

    def next_(self, remove: Optional[Variable] = None) -> Self:
        """Implements the abstract method :meth:`.abc.simplify.next_`.
        """
        result = self.__class__()
        result._ref_min_card = self._cur_min_card
        result._ref_max_card = self._cur_max_card
        if remove is None:
            result._ref_equations = self._cur_equations.copy()
            result._ref_inequations = self._cur_inequations.copy()
        else:
            result._ref_equations = UnionFind()
            for eq in self._cur_equations.equations():
                if remove not in eq.fvars():
                    result._ref_equations.union(eq.lhs, eq.rhs)
            result._ref_inequations = {ne for ne in self._cur_inequations
                                       if remove not in ne.fvars()}
        result._cur_min_card = result._ref_min_card
        result._cur_max_card = result._ref_max_card
        result._cur_equations = result._ref_equations.copy()
        result._cur_inequations = result._ref_inequations.copy()
        return result


class Simplify(
        abc.simplify.Simplify[AtomicFormula, Variable, Variable, Never, InternalRepresentation]):
    """Deep simplification in the style of [DolzmannSturm-1997]_. Implements
    the abstract methods :meth:`create_initial_representation
    <.abc.simplify.Simplify.create_initial_representation>` and :meth:`simpl_at
    <.abc.simplify.Simplify.simpl_at>` of its super class
    :class:`.abc.simplify.Simplify`.

    The simplifier should be called via :func:`.simplify`, as described below.
    In addition, this class inherits :meth:`.abc.simplify.Simplify.is_valid`,
    which should be called via :func:`.is_valid`, as described below.
    """

    def create_initial_representation(self, assume=Iterable[AtomicFormula]) \
            -> InternalRepresentation:
        """Implements the abstract method
        :meth:`.abc.simplify.Simplify.create_initial_representation`.
        """
        ir = InternalRepresentation()
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


simplify = Simplify().simplify


is_valid = Simplify().is_valid
