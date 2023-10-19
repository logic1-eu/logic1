import more_itertools

from sympy import ordered
from typing import Any, Generic, Iterable, Iterator, Optional, Self, TypeVar

from abc import ABC, abstractmethod

from logic1.firstorder.atomic import AtomicFormula
from logic1.firstorder.boolean import Equivalent, Implies, And, Or, Not
from logic1.firstorder.formula import Formula
from logic1.firstorder.quantified import QuantifiedFormula
from logic1.firstorder.truth import TruthValue, T

# About Generic:
# https://stackoverflow.com/q/74103528/
# https://peps.python.org/pep-0484/

TH = TypeVar('TH', bound='Theory')


class Theory(ABC):

    class Inconsistent(Exception):
        pass

    @abstractmethod
    def __init__(self, th: Optional[Self] = None) -> None:
        ...

    @abstractmethod
    def add(self, gand: type[And] | type[Or], atoms: Iterable[AtomicFormula]) -> None:
        ...

    @abstractmethod
    def extract(self, gand: type[And] | type[Or]) -> Iterable[AtomicFormula]:
        ...

    @abstractmethod
    def next_(self, remove: Any = None) -> Self:
        ...


class Simplify(ABC, Generic[TH]):

    def simplify(self, f: Formula, assume: list[AtomicFormula] = [], implicit_not: bool = False)\
            -> Formula:
        """
        Deep simplification according to [DS95].

        [DS95] A. Dolzmann, T. Sturm. Simplification of Quantifier-Free
               Formulae over Ordered Fields J. Symb. Comput. 24(2):209–231,
               1997. Open access at doi:10.1006/jsco.1997.0123
        """
        th = self._Theory()
        try:
            th.add(And, assume)
        except th.Inconsistent:
            return T
        match f:
            case AtomicFormula():
                return self._simplify(And(f), th, implicit_not)
            case _:
                return self._simplify(f, th, implicit_not)

    def _simplify(self, f: Formula, th: TH, implicit_not: bool) -> Formula:
        match f:
            # This method is expected to be time critical. I am putting the
            # most frequent cases first.
            case AtomicFormula():
                return self._simpl_at(f, implicit_not)
            case And() | Or():
                return self._simpl_and_or(f, th, implicit_not)
            case TruthValue():
                if implicit_not:
                    return f.dual_func()
                return f
            case Not(arg=arg):
                return self._simplify(arg, th, not implicit_not)
            case QuantifiedFormula(func=qua, var=var, arg=arg):
                simplified_arg = self._simplify(arg, th.next_(remove=var), implicit_not)
                if var not in simplified_arg.get_vars().free:
                    return simplified_arg
                return qua(var, simplified_arg)
            case Implies(lhs=lhs, rhs=rhs):
                return self._simplify(Or(Not(lhs), rhs), th, implicit_not)
            case Equivalent(lhs=lhs, rhs=rhs):
                tmp = And(Implies(lhs, rhs), Implies(rhs, lhs))
                return self._simplify(tmp, th, implicit_not)
            case _:
                raise NotImplementedError(f'Simplify does not know {f.func}')

    def _simpl_and_or(self, f: And | Or, th: TH, implicit_not: bool) -> Formula:

        def split(args: Iterable[Formula]) -> tuple[set[Formula], Iterator[AtomicFormula]]:
            """
            Returns iterators over non-atoms and atoms contained in
            :data:`args`, in that order.
            """
            def f(arg):
                return isinstance(arg, AtomicFormula)

            i1, i2 = more_itertools.partition(f, args)
            return set(i1), i2  # type: ignore
            # mypy would incorrectly derive that i2 is only Iterable[Formula].

        gand = f.func if not implicit_not else f.dual_func
        others, atoms = split(f.args)
        _1 = map(lambda arg: self._simpl_at(arg, implicit_not), atoms)
        new_others, atoms = split(_1)
        others = others.union(new_others)
        try:
            th.add(gand, atoms)
        except th.Inconsistent:
            return gand.definite_func()
        simplified_others: set[Formula] = set()
        while others:
            arg = others.pop()
            simplified_arg = self._simplify(arg, th.next_(), implicit_not)
            match simplified_arg:
                case gand.definite_func():
                    return simplified_arg
                case gand.neutral_func():
                    new_others = set()
                    new_atoms: Iterable[AtomicFormula] = ()
                case gand.func():  # MyPy does not accept gand() as a pattern here.
                    new_others, new_atoms = split(simplified_arg.args)
                case AtomicFormula():
                    new_others = set()
                    new_atoms = (simplified_arg,)
                case gand.dual_func() | QuantifiedFormula():
                    new_others = {simplified_arg}
                    new_atoms = ()
                case _:
                    # Implies and Equivalent have been recursively translated
                    # in to AndOr.
                    assert False
            if new_atoms:
                try:
                    th.add(gand, new_atoms)
                except th.Inconsistent:
                    return gand.definite_func()
                others = others.union(simplified_others)
                simplified_others = new_others
            else:
                simplified_others = simplified_others.union(new_others)
        return gand(*th.extract(gand), *ordered(simplified_others))

    @abstractmethod
    def _simpl_at(self, f: AtomicFormula, implicit_not: bool) -> Formula:
        # Doe not receive the theory by design.
        ...

    @abstractmethod
    def _Theory(self) -> TH:
        ...
