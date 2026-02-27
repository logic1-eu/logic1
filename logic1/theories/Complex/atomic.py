from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import ClassVar, Final, Generic, Never, Self, TypeVar

from gmpy2 import mpq

from ... import firstorder


τ = TypeVar('τ', bound='Term')

@dataclass
class VariableSet(firstorder.atomic.VariableSet['Variable']):
    
    _names: set[str] = field(default_factory=set)
     
    @property
    def stack(self) -> list[set[str]]:
        return [self._names]

    def __getitem__(self, index: str) -> Variable:
        """Implements abstract method
        :meth:`.firstorder.atomic.VariableSet.__getitem__`.
        """
        if not isinstance(index, str):
            raise ValueError(f'expecting string as index; {index} is {type(index)}')
        self._names.add(index)
        return Variable(index)

    def __repr__(self) -> str:
        s = ', '.join(str(g) for g in (*self._names, '...'))
        return f'{{{s}}}'

    def fresh(self, suffix: str = '') -> Variable:
        """Return a fresh variable, by default from the sequence G0001, G0002,
        ..., G9999, G10000, ... This naming convention is inspired by Lisp's
        gensym(). If the optional argument :data:`suffix` is specified, the
        sequence G0001<suffix>, G0002<suffix>, ... is used instead.
        """
        i = 1
        v = f'G{i:04d}{suffix}'
        while v in self._names:
            i += 1
            v = f'G{i:04d}{suffix}'
        self._names.add(v)
        return Variable(v)
    
    def pop(self) -> None:
        raise NotImplementedError()

    def push(self) -> None:
        raise NotImplementedError()
    
    def reset(self) -> None:
        self._names = set()


VV: Final = VariableSet()


@dataclass
class SortKey(Generic[τ]):

    term: τ

    def __eq__(self, other: Self) -> bool:  # type: ignore[override]
        raise NotImplementedError()

    def __ge__(self, other: Self) -> bool:
        raise NotImplementedError()

    def __gt__(self, other: Self) -> bool:
        raise NotImplementedError()

    def __hash__(self) -> int:
        raise NotImplementedError()

    def __le__(self, other: Self) -> bool:
        raise NotImplementedError()

    def __lt__(self, other: Self) -> bool:
        raise NotImplementedError()

    def __ne__(self, other: Self) -> bool:  # type: ignore[override]
        raise NotImplementedError()


class Term(firstorder.Term['Term', 'Variable', Never, SortKey['Term']]):
    """
    Abstract class representing a node of the AST
    """
    
    def __add__(self, other: object) -> Term:
        raise NotImplementedError()

    def __eq__(self, other: Term | int) -> Eq:  # type: ignore[override]
        raise NotImplementedError()

    def __ge__(self, other: Term | int) -> Ge | Le:
        raise NotImplementedError()

    def __gt__(self, other: Term | int) -> Gt | Lt:
        raise NotImplementedError()

    def __hash__(self) -> int:
        raise NotImplementedError()

    def __init__(self, arg: float | Fraction | int | mpq) -> None:
        raise NotImplementedError()

    def __le__(self, other: Term | int | mpq) -> Ge | Le:
        raise NotImplementedError()

    def __lt__(self, other: Term | int | mpq) -> Gt | Lt:
        raise NotImplementedError()

    def __mul__(self, other: object) -> Term:
        raise NotImplementedError()

    def __ne__(self, other: Term | int | mpq) -> Ne: # type: ignore[override]
        raise NotImplementedError()

    def __neg__(self) -> Term:
        raise NotImplementedError()

    def __pow__(self, other: object) -> Term:
        raise NotImplementedError()

    def __repr__(self) -> str:
        raise NotImplementedError()

    def __radd__(self, other: object) -> Term:
        assert not isinstance(object, Term)
        raise NotImplementedError()

    def __rmul__(self, other: object) -> Term:
        assert not isinstance(object, Term)
        raise NotImplementedError()

    def __rsub__(self, other: object) -> Term:
        assert not isinstance(object, Term)
        raise NotImplementedError()

    def __sub__(self, other: object) -> Term:
        raise NotImplementedError()

    def __truediv__(self, other: object) -> Term:
        raise NotImplementedError()

    def __xor__(self, other: object) -> Term:
        raise NotImplementedError(
            "Use ** for exponentiation, not '^', which means xor "
            "in Python, and has the wrong precedence")

    def as_latex(self) -> str:
        """LaTeX representation as a string. Implements the abstract method
        :meth:`.firstorder.atomic.Term.as_latex`.
        """
        raise NotImplementedError()

    def is_constant(self) -> bool:
        """Return :obj:`True` if this term is constant.
        """
        raise NotImplementedError()

    def is_variable(self) -> bool:
        """Return :obj:`True` if this term is a variable.
        """
        raise NotImplementedError()

    def normalize(self) -> Term:
        raise NotImplementedError()

    def sort_key(self) -> SortKey[Self]:
        """A sort key suitable for ordering instances of this class. Implements
        the abstract method :meth:`.firstorder.atomic.Term.sort_key`.
        """
        return SortKey(self)

    def vars(self) -> Iterator[Variable]:
        """An iterator that yields each variable of this term once. Implements
        the abstract method :meth:`.firstorder.atomic.Term.vars`.
        """
        raise NotImplementedError()

    
class Add(Term):
    pass


class Mul(Term):
    pass


class Pow(Term):
    pass


class Variable(Term, firstorder.Variable['Variable', int, SortKey['Variable']]):

    name: str
    VV: ClassVar[VariableSet] = VV

    def __init__(self, name: str) -> None:
        self.name = name

    def fresh(self) -> Variable:
        """Returns a variable that has not been used so far. Implements
        abstract method :meth:`.firstorder.atomic.Variable.fresh`.
        """
        return self.VV.fresh(suffix=f'_{str(self)}')


class Constant(Term):
    pass


class Re(Term):
    pass


class Im(Term):
    pass


class AtomicFormula(firstorder.AtomicFormula['AtomicFormula', 'Term', 'Variable', Never]):

    @property
    def lhs(self) -> Term:
        """The left hand side term of an atomic formula.
        """
        return self.args[0]

    @property
    def rhs(self) -> Term:
        """The right hand side term of an atomic formula.
        """
        return self.args[1]
    
    def __bool__(self) -> bool:
        raise NotImplementedError()

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError()

    def __hash__(self) -> int:
        raise NotImplementedError()

    def __init__(self, lhs: Term, rhs: Term):
        raise NotImplementedError()

    def __le__(self, other: Formula) -> bool:
        """Returns `True` if this atomic formula should be sorted before or is
        equal to other. Implements abstract method
        :meth:`.firstorder.atomic.AtomicFormula.__le__`.
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        raise NotImplementedError()

    def __str__(self) -> str:
        """String representation of this atomic formula. Implements the
        abstract method :meth:`.firstorder.atomic.AtomicFormula.__str__`.
        """
        SYMBOL: Final = {Eq: '==', Ne: '!=', Ge: '>=', Le: '<=', Gt: '>', Lt: '<'}
        SPACING: Final = ' '
        return f'{self.lhs}{SPACING}{SYMBOL[self.op]}{SPACING}{self.rhs}'

    def as_latex(self) -> str:
        """Latex representation as a string. Implements the abstract method
        :meth:`.firstorder.atomic.AtomicFormula.as_latex`.
        """
        SYMBOL: Final = {
            Eq: '=', Ne: '\\neq', Ge: '\\geq', Le: '\\leq', Gt: '>', Lt: '<'}
        SPACING: Final = ' '
        return f'{self.lhs.as_latex()}{SPACING}{SYMBOL[self.op]}{SPACING}{self.rhs.as_latex()}'

    def bvars(self, quantified: frozenset[Variable] = frozenset()) -> Iterator[Variable]:
        """Iterate over occurrences of variables that are elements of
        `quantified`. Yield each such variable once for each term that it
        occurs in. Implements the abstract method
        :meth:`.firstorder.atomic.AtomicFormula.bvars`.
        """
        raise NotImplementedError()

    def fvars(self, quantified: frozenset[Variable] = frozenset()) -> Iterator[Variable]:
        """Iterate over occurrences of variables that are *not* elements of
        `quantified`. Yield each such variable once for each term that it
        occurs in. Implements the abstract method
        :meth:`.firstorder.atomic.AtomicFormula.fvars`.
        """
        raise NotImplementedError()

    def simplify(self) -> Formula:
        """Fast basic simplification. The result is equivalent to self.
        Implements the abstract method :meth:`.firstorder.atomic.AtomicFormula.simplify`.
        """
        raise NotImplementedError()
        
    def subs(self, sigma: Mapping[Variable, Term]) -> Self:
        """Formal simultaneous term substitution into the two argument terms of
        the atomic formula. Implements the abstract method
        :meth:`.firstorder.atomic.AtomicFormula.subs`.
        """
        raise NotImplementedError()



class Eq(AtomicFormula):
    pass

class Ne(AtomicFormula):
    pass


class Ge(AtomicFormula):
    pass


class Le(AtomicFormula):
    pass


class Gt(AtomicFormula):
    pass


class Lt(AtomicFormula):
    pass


from .typing import Formula