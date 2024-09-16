"""
This module implements *deep simplification* through the generation and
propagation of internal theories during recursion. It is an adaptation of the
*standard simplifier* from [DolzmannSturm-1997]_ tailored to the specific
requirements of Sets.
"""
from typing import Iterable, Never, Optional, Self

from ... import abc

from ...firstorder import And, Or
from .atomic import AtomicFormula, C, C_, Eq, Index, Ne, oo, Variable
from .typing import Formula

from ...support.tracing import trace  # noqa


class Theory(abc.simplify.Theory[AtomicFormula, Variable, Variable, Never]):
    """Implements the abstract methods :meth:`add() <.abc.simplify.Theory.add>`,
    :meth:`extract() <.abc.simplify.Theory.extract>`, and :meth:`next_()
    <.abc.simplify.Theory.next_>` of it super class
    :class:`.abc.simplify.Theory`. Required by
    :class:`.Sets.simplify.Simplify` for instantiating the type variable
    :data:`.abc.simplify.θ` of :class:`.abc.simplify.Simplify`.
    """
    _ref_min_card: Index
    _ref_max_card: Index
    _ref_equations: list[set[Variable]]
    _ref_inequations: set[Ne]

    _cur_min_card: Index
    _cur_max_card: Index
    _cur_equations: list[set[Variable]]
    _cur_inequations: set[Ne]

    def __init__(self) -> None:
        self._ref_min_card = 1
        self._ref_max_card = oo
        self._ref_equations = []
        self._ref_inequations = set()
        self._cur_min_card = 1
        self._cur_max_card = oo
        self._cur_equations = []
        self._cur_inequations = set()

    def __repr__(self) -> str:
        return (f'Theory([{self._ref_min_card}..{self._ref_max_card}], '
                f'{self._ref_equations}, {self._ref_inequations} | '
                f'[{self._cur_min_card}..{self._cur_max_card}], '
                f'{self._cur_equations}, {self._cur_inequations})')

    def add(self, gand: type[And | Or], atoms: Iterable[AtomicFormula]) -> None:
        """Implements the abstract method :meth:`.abc.simplify.Theory.add`.
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
                        raise Theory.Inconsistent()
                case C_(index=n):
                    if n - 1 < self._cur_max_card:
                        self._cur_max_card = n - 1
                    if self._cur_min_card > self._cur_max_card:
                        raise Theory.Inconsistent()
                case Eq(lhs=lhs, rhs=rhs):
                    equations = []
                    Q = {lhs, rhs}
                    for P in self._cur_equations:
                        if lhs in P or rhs in P:
                            # This happens at most twice.
                            Q = Q.union(P)
                        else:
                            equations.append(P)
                    equations.append(Q)
                    self._cur_equations = equations
                case Ne(lhs=lhs, rhs=rhs):
                    self._cur_inequations.add(atom)
                case _:
                    assert False
            for ne in self._cur_inequations:
                for P in self._cur_equations:
                    if ne.lhs in P and ne.rhs in P:
                        raise Theory.Inconsistent()

            # Create substitution from the equations
            sigma = dict()
            for P in self._cur_equations:
                x = min(P, key=Variable.sort_key)
                for y in P:
                    sigma[y] = x
            # Substitute into inequations
            inequations = set()
            for ne in self._cur_inequations:
                ne_subs = ne.subs(sigma)
                if ne_subs.lhs == ne_subs.rhs:
                    raise Theory.Inconsistent()
                if Variable.sort_key(ne_subs.lhs) <= Variable.sort_key(ne_subs.rhs):
                    inequations.add(ne_subs)
                else:
                    inequations.add(Ne(ne_subs.rhs, ne_subs.lhs))
            self._cur_inequations = inequations

    def extract(self, gand: type[And | Or]) -> list[AtomicFormula]:
        """Implements the abstract method :meth:`.abc.simplify.Theory.extract`.
        """
        L: list[AtomicFormula] = []
        if self._cur_min_card > self._ref_min_card:
            L.append(C(self._cur_min_card))
        if self._cur_max_card < self._ref_max_card:
            L.append(C_(self._cur_max_card + 1))
        for P in self._cur_equations:
            x = min(P, key=Variable.sort_key)
            for y in P:
                if x != y:
                    for Q in self._ref_equations:
                        if x in Q and y in Q:
                            break
                    else:
                        L.append(Eq(x, y))
        for ne in self._cur_inequations:
            if ne not in self._ref_inequations:
                L.append(ne)
        if gand is Or:
            L = [atom.to_complement() for atom in L]
        return L

    def next_(self, remove: Optional[Variable] = None) -> Self:
        """Implements the abstract method :meth:`.abc.simplify.Theory.next_`.
        """
        if remove is not None:
            raise NotImplementedError
        theory_next = self.__class__()
        theory_next._ref_min_card = self._cur_min_card
        theory_next._ref_max_card = self._cur_max_card
        theory_next._ref_equations = self._cur_equations.copy()
        theory_next._ref_inequations = self._cur_inequations.copy()
        theory_next._cur_min_card = self._cur_min_card
        theory_next._cur_max_card = self._cur_max_card
        theory_next._cur_equations = self._cur_equations.copy()
        theory_next._cur_inequations = self._cur_inequations.copy()
        return theory_next


class Simplify(abc.simplify.Simplify[AtomicFormula, Variable, Variable, Never, Theory]):
    """Deep simplification in the style of [DolzmannSturm-1997]_. Implements
    the abstract methods :meth:`create_initial_theory
    <.abc.simplify.Simplify.create_initial_theory>` and :meth:`simpl_at
    <.abc.simplify.Simplify.simpl_at>` of its super class
    :class:`.abc.simplify.Simplify`.

    The simplifier should be called via :func:`.simplify`, as described below.
    In addition, this class inherits :meth:`.abc.simplify.Simplify.is_valid`,
    which should be called via :func:`.is_valid`, as described below.
    """

    def create_initial_theory(self) -> Theory:
        """Implements the abstract method
        :meth:`.abc.simplify.Simplify.create_initial_theory`.
        """
        return Theory()

    def simpl_at(self,
                 atom: AtomicFormula,
                 context: Optional[type[And] | type[Or]]) -> Formula:
        """Implements the abstract method
        :meth:`.abc.simplify.Simplify.simpl_at`.
        """
        return atom.simplify()


simplify = Simplify().simplify


is_valid = Simplify().is_valid
