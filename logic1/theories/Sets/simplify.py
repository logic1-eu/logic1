from sympy import default_sort_key, oo
from sympy.core.numbers import Infinity, Integer
from typing import Iterable, Optional, Self, TypeAlias

from ... import abc

from ...firstorder import And, AtomicFormula, Formula, Or
from .sets import C, C_, Eq, Ne, Term, Variable
from .pnf import pnf

from ...support.tracing import trace  # noqa


class Theory(abc.simplify.Theory):

    _ref_min_card: Integer
    _ref_max_card: Integer | Infinity
    _ref_equations: list[set[Term]]
    _ref_inequations: set[Ne]

    _cur_min_card: Integer
    _cur_max_card: Integer | Infinity
    _cur_equations: list[set[Term]]
    _cur_inequations: set[Ne]

    def __init__(self) -> None:
        self._ref_min_card = Integer(1)
        self._ref_max_card = oo
        self._ref_equations = []
        self._ref_inequations = set()
        self._cur_min_card = Integer(1)
        self._cur_max_card = oo
        self._cur_equations = []
        self._cur_inequations = set()

    def __repr__(self) -> str:
        return (f'Theory([{self._ref_min_card}..{self._ref_max_card}], '
                f'{self._ref_equations}, {self._ref_inequations} | '
                f'[{self._cur_min_card}..{self._cur_max_card}], '
                f'{self._cur_equations}, {self._cur_inequations})')

    def add(self, gand: type[And] | type[Or], atoms: Iterable[AtomicFormula]) -> None:
        for atom in atoms:
            if gand is Or:
                atom = atom.to_complement()
            # Collect information
            match atom:
                case C(index=n):
                    if n > self._cur_min_card:
                        self._cur_min_card = n
                    if self._cur_min_card > self._cur_max_card:
                        raise Theory.Inconsistent
                case C_(index=n):
                    if n - 1 < self._cur_max_card:
                        self._cur_max_card = n - 1
                    if self._cur_min_card > self._cur_max_card:
                        raise Theory.Inconsistent
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
                    self._cur_inequations = self._cur_inequations.union({atom})
                case _:
                    assert False
            for ne in self._cur_inequations:
                for P in self._cur_equations:
                    if ne.lhs in P and ne.rhs in P:
                        raise Theory.Inconsistent
            # # Create substitution from the equations
            # sigma = dict()
            # for P in self._cur_equations:
            #     x = max(P, key=default_sort_key)
            #     for y in P:
            #         if y is not x:
            #             sigma[y] = x
            # # Substitute into inequations
            # inequations = set()
            # for ne in self._cur_inequations:
            #     ne_subs = ne.subs(sigma)
            #     match ne_subs.lhs.compare(ne_subs.rhs):
            #         case 0:
            #             raise Theory.Inconsistent
            #         case 1:
            #             inequations.update({Ne(ne_subs.rhs, ne_subs.lhs)})
            #         case -1:
            #             inequations.update({ne_subs})
            # self._cur_inequations = inequations

    def extract(self, gand: type[And] | type[Or]) -> list[AtomicFormula]:
        L: list[AtomicFormula] = []
        if self._cur_min_card > self._ref_min_card:
            L.append(C(self._cur_min_card))
        if self._cur_max_card < self._ref_max_card:
            L.append(C_(self._cur_max_card + 1))
        for P in self._cur_equations:
            x = min(P, key=default_sort_key)
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
        if remove is not None:
            raise NotImplementedError
        theory_next = self.__class__()
        theory_next._ref_min_card = self._cur_min_card
        theory_next._ref_max_card = self._cur_max_card
        theory_next._ref_equations = self._cur_equations
        theory_next._ref_inequations = self._cur_inequations
        theory_next._cur_min_card = self._cur_min_card
        theory_next._cur_max_card = self._cur_max_card
        theory_next._cur_equations = self._cur_equations
        theory_next._cur_inequations = self._cur_inequations
        return theory_next


class Simplify(abc.simplify.Simplify['Theory']):

    AtomicSortKey: TypeAlias = tuple[int, Integer] | tuple[int, Term, Term]
    SortKey: TypeAlias = tuple[int, int, int, tuple[AtomicSortKey, ...]]

    def __call__(self, f: Formula, assume: list[AtomicFormula] = []) -> Formula:
        try:
            return self.simplify(f, assume)
        except Simplify.NotInPnf:
            return self.simplify(pnf(f), assume)

    def _simpl_at(self, f: AtomicFormula) -> Formula:
        return f.simplify()

    def _Theory(self) -> Theory:
        return Theory()


simplify = Simplify()
