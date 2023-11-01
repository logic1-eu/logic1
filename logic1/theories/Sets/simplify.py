# from functools import lru_cache
from typing import Iterable, Optional, Self

from ... import abc

from ...firstorder.formula import Formula
from ...firstorder.boolean import And, Or
from ...firstorder.atomic import AtomicFormula

from .sets import C, C_, Eq, Ne, Term, Variable

from sympy import default_sort_key, oo
from sympy.core.numbers import Infinity

from ...support.tracing import trace  # noqa


class Theory(abc.simplify.Theory):

    _ref_min_card: int
    _ref_max_card: int | Infinity
    _ref_equations: list[set[Term]]
    _ref_inequations: set[Ne]

    _cur_min_card: int
    _cur_max_card: int | Infinity
    _cur_equations: list[set[Term]]
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

    # @trace()
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
        # print(self)

    # @trace()
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

    def __call__(self, f: Formula) -> Formula:
        return self.simplify(f)

    def _simpl_at(self, f: AtomicFormula) -> Formula:
        return f.simplify()

    def sort_atoms(self, atoms: list[AtomicFormula]) -> None:
        atoms.sort(key=Simplify._sort_key_at)

    def sort_others(self, others: list[Formula]) -> None:
        others.sort(key=Simplify._sort_key)

    @staticmethod
    def _sort_key(f: Formula) ->\
            tuple[int, int, int, tuple[tuple[int, int] | tuple[int, Term, Term], ...]]:
        assert isinstance(f, (And, Or))
        atom_sort_keys = tuple(Simplify._sort_key_at(a) for a in f.atoms())
        return (f.depth(), len(f.args), len(atom_sort_keys), atom_sort_keys)

    @staticmethod
    def _sort_key_at(f: AtomicFormula) -> tuple[int, int] | tuple[int, Term, Term]:
        match f:
            case C():
                return (0, f.index)
            case C_():
                return (1, f.index)
            case Eq():
                return (2, default_sort_key(f.lhs), default_sort_key(f.rhs))
            case Ne():
                return (3, default_sort_key(f.lhs), default_sort_key(f.rhs))
            case _:
                assert False

    def _Theory(self) -> Theory:
        return Theory()


simplify = Simplify()
