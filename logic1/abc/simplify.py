import more_itertools

from abc import ABC, abstractmethod
from typing import Any, cast, Generic, Iterable, Optional, Self, TypeVar

from ..firstorder import (All, And, AtomicFormula, Ex, _F, Formula, Or, pnf,
                          _T, T)

from ..support.tracing import trace  # noqa

# About Generic:
# https://stackoverflow.com/q/74103528/
# https://peps.python.org/pep-0484/

AT = TypeVar('AT', bound='AtomicFormula')
TH = TypeVar('TH', bound='Theory')


class Theory(ABC, Generic[AT]):

    class Inconsistent(Exception):
        pass

    @abstractmethod
    def __init__(self) -> None:
        ...

    @abstractmethod
    def add(self, gand: type[And | Or], atoms: Iterable[AT]) -> None:
        ...

    @abstractmethod
    def extract(self, gand: type[And | Or]) -> Iterable[AT]:
        ...

    @abstractmethod
    def next_(self, remove: Any = None) -> Self:
        ...


class Simplify(ABC, Generic[AT, TH]):

    @property
    @abstractmethod
    def class_AT(self) -> type[AT]:
        ...

    @property
    @abstractmethod
    def class_TH(self) -> type[TH]:
        ...

    @property
    @abstractmethod
    def TH_kwargs(self) -> dict[str, bool]:
        ...

    def simplify(self, f: Formula, assume: Optional[list[AT]]) -> Formula:
        """
        Deep simplification according to [DS95].

        [DS95] A. Dolzmann, T. Sturm. Simplification of Quantifier-Free
               Formulae over Ordered Fields J. Symb. Comput. 24(2):209–231,
               1997. Open access at doi:10.1006/jsco.1997.0123
        """
        if assume is None:
            assume = []
        th = self.class_TH(**self.TH_kwargs)
        try:
            th.add(And, assume)
        except th.Inconsistent:
            return T
        th = th.next_()
        f = pnf(f)
        quantifiers = []
        while isinstance(f, (Ex, All)):
            th = th.next_(remove=f.var)
            quantifiers.append((f.func, f.var))
            f = f.arg
        f = self._simpl_nnf(f, th)
        free_vars = set(f.fvars())
        for Q, var in reversed(quantifiers):
            if var in free_vars:
                f = Q(var, f)
        return f

    def _simpl_nnf(self, f: Formula, th: TH) -> Formula:
        match f:
            case And() | Or():
                return self._simpl_and_or(f, th)
            case self.class_AT():
                # Build a trivial binary And in order to apply th. Unary And
                # does not exist.
                return self._simpl_and_or(And(f, T), th)
            case _F() | _T():
                return f
            case _:
                raise NotImplementedError(f'Simplify does not know {f.func!r}')

    def _simpl_and_or(self, f: Formula, th: TH) -> Formula:
        """
        `f` must be in negation normal form (NNF).
        """

        def split(args: Iterable[Formula]) -> tuple[set[Formula], set[AT]]:
            """
            Returns the set of non-atoms and an iterator of atoms contained in
            :data:`args`, in that order.
            """
            def is_AT(f: Formula) -> bool:
                if isinstance(f, self.class_AT):
                    return True
                assert not isinstance(f, AtomicFormula), (type(f), f)
                return False

            i1, i2 = more_itertools.partition(is_AT, args)
            return set(i1), cast(set[AT], set(i2))

        gand = f.func
        others, atoms = split(f.args)
        simplified_atoms = (self.simpl_at(atom, f.func) for atom in atoms)
        new_others, atoms = split(simplified_atoms)
        others = others.union(new_others)
        try:
            th.add(gand, atoms)
        except th.Inconsistent:
            return gand.definite_func()

        simplified_others: set[Formula] = set()
        while others:
            arg = others.pop()
            simplified_arg = self._simpl_and_or(arg, th.next_())
            match simplified_arg:
                case gand.definite_func():
                    return simplified_arg
                case gand.neutral_func():
                    new_others = set()
                    new_atoms: Iterable[AT] = ()
                case gand.func():
                    new_others, new_atoms = split(simplified_arg.args)
                case self.class_AT():
                    new_others = set()
                    new_atoms = (simplified_arg,)
                case gand.dual_func():
                    new_others = {simplified_arg}
                    new_atoms = ()
                case _:
                    raise NotImplementedError(f'unknown operator {simplified_arg.func} in {f}')
            if new_atoms:
                try:
                    th.add(gand, new_atoms)  # Can save resimp if th does not change
                except th.Inconsistent:
                    return gand.definite_func()
                others = others.union(simplified_others)
                simplified_others = new_others
            else:
                simplified_others = simplified_others.union(new_others)
        final_atoms = list(th.extract(gand))
        final_atoms.sort()
        final_others = list(simplified_others)
        final_others.sort()
        return gand(*final_atoms, *final_others)

    @abstractmethod
    def simpl_at(self,
                 atom: AT,
                 context: Optional[type[And] | type[Or]]) -> Formula:
        # Does not receive the theory, by design.
        ...


class IsValid(ABC, Generic[AT]):

    def is_valid(self, f: Formula, assume: Optional[list[AT]]) -> Optional[bool]:
        if assume is None:
            assume = []
        match self._simplify(f, assume):
            case _T():
                return True
            case _F():
                return False
            case _:
                return None

    @abstractmethod
    def _simplify(self, f: Formula, assume: list[AT]) -> Formula:
        ...
