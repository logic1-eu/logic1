import more_itertools

from typing import Any, Generic, Iterable, Iterator, Optional, Self, TypeVar

from abc import ABC, abstractmethod

from logic1.firstorder.atomic import AtomicFormula
from logic1.firstorder.boolean import Equivalent, Implies, And, Or, Not
from logic1.firstorder.formula import Formula
from logic1.firstorder.quantified import QuantifiedFormula
from logic1.firstorder.truth import TruthValue, T

from ..support.tracing import trace  # noqa

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

    class NotInPnf(Exception):
        pass

    _develop: int

    def simplify(self, f: Formula, assume: list[AtomicFormula] = []) -> Formula:
        """
        Deep simplification according to [DS95].

        [DS95] A. Dolzmann, T. Sturm. Simplification of Quantifier-Free
               Formulae over Ordered Fields J. Symb. Comput. 24(2):209–231,
               1997. Open access at doi:10.1006/jsco.1997.0123
        """
        try:
            th = self._Theory()
            try:
                th.add(And, assume)
            except th.Inconsistent:
                return T
            match f:
                case AtomicFormula():
                    return self._simpl_nnf(And(f), th)
                case _:
                    return self._simpl_pnf(f, th)
        except Simplify.NotInPnf:
            return self.simplify(f.to_pnf(), assume)

    def _simpl_pnf(self, f: Formula, th: TH) -> Formula:
        match f:
            case QuantifiedFormula(func=Q, var=var, arg=arg):
                simplified_arg = self._simpl_pnf(arg, th.next_(remove=var))
                if var not in simplified_arg.get_vars().free:
                    return simplified_arg
                return Q(var, simplified_arg)
            case _:
                return self._simpl_nnf(f, th)

    def _simpl_nnf(self, f: Formula, th: TH) -> Formula:
        match f:
            case AtomicFormula():
                return self._simpl_at(f)
            case And() | Or():
                return self._simpl_and_or(f, th)
            case TruthValue():
                return f
            case Not() | Implies() | Equivalent():
                raise Simplify.NotInPnf
            case _:
                raise NotImplementedError(f'Simplify does not know {f.func!r}')

    def _simpl_and_or(self, f: And | Or, th: TH) -> Formula:

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

        gand = f.func
        others, atoms = split(f.args)
        simplified_atoms = (self._simpl_at(atom) for atom in atoms)
        new_others, atoms = split(simplified_atoms)
        others = others.union(new_others)
        try:
            th.add(gand, atoms)
        except th.Inconsistent:
            return gand.definite_func()
        simplified_others: set[Formula] = set()
        while others:
            arg = others.pop()
            simplified_arg = self._simpl_nnf(arg, th.next_())
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
        final_atoms = list(th.extract(gand))
        self.sort_atoms(final_atoms)
        final_others = list(simplified_others)
        self.sort_others(final_others)
        return gand(*final_atoms, *final_others)

    @abstractmethod
    def _simpl_at(self, f: AtomicFormula) -> Formula:
        # Does not receive the theory, by design.
        ...

    @abstractmethod
    def sort_atoms(self, atoms: list[AtomicFormula]) -> None:
        ...

    @abstractmethod
    def sort_others(self, others: list[Formula]) -> None:
        ...

    @abstractmethod
    def _Theory(self) -> TH:
        ...
