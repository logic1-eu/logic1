import more_itertools

from abc import abstractmethod
from typing import Any, cast, Generic, Iterable, Optional, Self, TypeVar

from ..firstorder import All, And, AtomicFormula, Ex, _F, Formula, Or, _T
from ..firstorder.formula import α, τ, χ

from ..support.tracing import trace  # noqa

# About Generic:
# https://stackoverflow.com/q/74103528/
# https://peps.python.org/pep-0484/

θ = TypeVar('θ', bound='Theory')


class Theory(Generic[α, τ, χ]):

    class Inconsistent(Exception):
        pass

    @abstractmethod
    def __init__(self) -> None:
        ...

    @abstractmethod
    def add(self, gand: type[And[α, τ, χ] | Or[α, τ, χ]], atoms: Iterable[α]) -> None:
        ...

    @abstractmethod
    def extract(self, gand: type[And[α, τ, χ] | Or[α, τ, χ]]) -> Iterable[α]:
        ...

    @abstractmethod
    def next_(self, remove: Any = None) -> Self:
        ...


class Simplify(Generic[α, τ, χ, θ]):

    @property
    @abstractmethod
    def class_AT(self) -> type[α]:
        ...

    @property
    @abstractmethod
    def class_TH(self) -> type[θ]:
        ...

    @property
    @abstractmethod
    def TH_kwargs(self) -> dict[str, bool]:
        ...

    def simplify(self, f: Formula[α, τ, χ], assume: Optional[list[α]]) -> Formula[α, τ, χ]:
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
            return _T()
        th = th.next_()
        f = f.to_pnf()
        quantifiers = []
        while isinstance(f, (Ex, All)):
            th = th.next_(remove=f.var)
            quantifiers.append((f.op, f.var))
            f = f.arg
        f = self._simpl_nnf(f, th)
        free_vars = set(f.fvars())
        for Q, var in reversed(quantifiers):
            if var in free_vars:
                f = Q(var, f)
        return f

    def _simpl_nnf(self, f: Formula[α, τ, χ], th: θ) -> Formula[α, τ, χ]:
        match f:
            case And() | Or():
                return self._simpl_and_or(f, th)
            case _F() | _T():
                return f
            case self.class_AT():
                # Build a trivial binary And in order to apply th. Unary And
                # does not exist.
                return self._simpl_and_or(And(f, _T()), th)
            case _:
                raise NotImplementedError(f'Simplify does not know {f.op!r}')

    def _simpl_and_or(self, f: And[α, τ, χ] | Or[α, τ, χ], th: θ) -> Formula[α, τ, χ]:
        """
        `f` must be in negation normal form (NNF).
        """

        def split(args: Iterable[Formula[α, τ, χ]]) -> tuple[set[Formula[α, τ, χ]], set[α]]:
            """
            Returns the set of non-atoms and an iterator of atoms contained in
            :data:`args`, in that order.
            """
            def is_AT(f: Formula[α, τ, χ]) -> bool:
                if isinstance(f, self.class_AT):
                    return True
                assert not isinstance(f, AtomicFormula), (type(f), f)
                return False

            i1, i2 = more_itertools.partition(is_AT, args)
            return set(i1), cast(set[α], set(i2))

        gand = f.op
        others, atoms = split(f.args)
        simplified_atoms = (self.simpl_at(atom, f.op) for atom in atoms)
        new_others, atoms = split(simplified_atoms)
        others = others.union(new_others)
        try:
            th.add(gand, atoms)
        except th.Inconsistent:
            return gand.definite_element()
        simplified_others: set[Formula] = set()
        while others:
            arg = others.pop()
            simplified_arg = self._simpl_nnf(arg, th.next_())
            if isinstance(simplified_arg, gand.definite()):
                return simplified_arg
            elif isinstance(simplified_arg, gand.neutral()):
                new_others = set()
                new_atoms: Iterable[α] = ()
            elif isinstance(simplified_arg, gand):
                new_others, new_atoms = split(simplified_arg.args)
            elif isinstance(simplified_arg, self.class_AT):
                new_others = set()
                new_atoms = (simplified_arg,)
            elif isinstance(simplified_arg, gand.dual()):
                new_others = {simplified_arg}
                new_atoms = ()
            else:
                raise NotImplementedError(f'unknown operator {simplified_arg.op} in {f}')
            if new_atoms:
                try:
                    th.add(gand, new_atoms)  # Can save resimp if th does not change
                except th.Inconsistent:
                    return gand.definite_element()
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
                 atom: α,
                 context: Optional[type[And[α, τ, χ]] | type[Or[α, τ, χ]]]) -> Formula[α, τ, χ]:
        # Does not receive the theory, by design.
        ...


class IsValid(Generic[α, τ, χ]):

    def is_valid(self, f: Formula[α, τ, χ], assume: Optional[list[α]]) -> Optional[bool]:
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
    def _simplify(self, f: Formula[α, τ, χ], assume: list[α]) -> Formula[α, τ, χ]:
        ...
