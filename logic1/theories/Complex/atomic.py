from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import ClassVar, Final, Generic, Never, Self, TypeVar

from gmpy2 import mpq, mpz

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

    PRECEDENCE = {
        'Add': 10,
        'Neg': 10,
        'Mul': 20,
        'Pow': 30,
        'Re': 100,
        'Im': 100,
        'Variable': 100,
        'Constant': 100
    }

    @property
    def op(self) -> type[Self]:
        return type(self)
    
    @property
    def args(self) -> tuple[object, ...]:
        raise NotImplementedError()
    
    @property
    def _precedence(self) -> int:
        return Term.PRECEDENCE.get(self.op.__name__, 0)
    
    def __add__(self, other: object) -> Term:
        if not isinstance(other, Term):
            return self + Constant(other)
        return Add(self, other)

    def __eq__(self, other: object) -> Eq:  # type: ignore[override]
        if not isinstance(other, Term):
            return self == Constant(other)
        return Eq(self, other)

    def __ge__(self, other: object) -> Ge:
        if not isinstance(other, Term):
            return self >= Constant(other)
        return Ge(self, other)

    def __gt__(self, other: object) -> Gt:
        if not isinstance(other, Term):
            return self > Constant(other)
        return Gt(self, other)

    def __hash__(self) -> int:
        return hash((tuple(str(cls) for cls in self.op.mro()), self.args))

    def __le__(self, other: object) -> Le:
        if not isinstance(other, Term):
            return self <= Constant(other)
        return Le(self, other)

    def __lt__(self, other: object) -> Lt:
        if not isinstance(other, Term):
            return self < Constant(other)
        return Lt(self, other)

    def __mul__(self, other: object) -> Term:
        if not isinstance(other, Term):
            return self * Constant(other)
        return Mul(self, other)

    def __ne__(self, other: object) -> Ne:  # type: ignore[override]
        if not isinstance(other, Term):
            return self != Constant(other)
        return Ne(self, other)

    def __neg__(self) -> Term:
        return Neg(self)

    def __pow__(self, other: int) -> Term:
        return Pow(self, other)

    def __repr__(self) -> str:
        return f'{self.op.__name__}({', '.join(map(repr, self.args))})'

    def __radd__(self, other: object) -> Term:
        assert not isinstance(other, Term)
        return Constant(other) + self

    def __rmul__(self, other: object) -> Term:
        assert not isinstance(other, Term)
        return Constant(other) * self

    def __rsub__(self, other: object) -> Term:
        assert not isinstance(other, Term)
        return Constant(other) - self
    
    def __str__(self) -> str:
        return f'{self.op.__name__}({', '.join(map(str, self.args))})'

    def __sub__(self, other: object) -> Term:
        if not isinstance(other, Term):
            return self - Constant(other)
        return self + (-other)

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
        return isinstance(self, Constant)

    def is_variable(self) -> bool:
        """Return :obj:`True` if this term is a variable.
        """
        return isinstance(self, Variable)

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

    precedence: int = 10

    _args: tuple[Term, ...]

    @property
    def args(self) -> tuple[Term, ...]:
        return self._args

    def __init__(self, *args: Term) -> None:
        super().__init__()
        args_flat = []
        for arg in args:
            if isinstance(arg, Add):
                args_flat.extend(list(arg.args))
            else:
                args_flat.append(arg)
        self._args = tuple(args_flat)

    def __new__(cls, *args: Term):
        if not args:
            return Constant(0)
        if len(args) == 1:
            return args[0]
        return super().__new__(cls)
    
    def __str__(self) -> str:
        result = []
        for arg in self.args:
            if arg._precedence > self._precedence:
                result.append(f'{arg}')
            else:
                result.append(f'({arg})')
        return " + ".join(result)


class Mul(Term):

    _args: tuple[Term, ...]

    @property
    def args(self) -> tuple[Term, ...]:
        return self._args

    def __init__(self, *args: Term) -> None:
        super().__init__()
        args_flat = []
        for arg in args:
            if isinstance(arg, Mul):
                args_flat.extend(list(arg.args))
            else:
                args_flat.append(arg)
        self._args = tuple(args_flat)

    def __new__(cls, *args: Term):
        if not args:
            return Constant(1)
        if len(args) == 1:
            return args[0]
        return super().__new__(cls)
    
    def __str__(self) -> str:
        result = []
        for arg in self.args:
            if arg._precedence > self._precedence:
                result.append(f'{arg}')
            else:
                result.append(f'({arg})')
        return "*".join(result)


class Pow(Term):
    
    base: Term
    exponent: int

    @property
    def args(self) -> tuple[Term, int]:
        return (self.base, self.exponent)
    
    def __init__(self, base: Term, exponent: int) -> None:
        if exponent < 0:
            raise ValueError('Negative exponents are not allowed!')
        self.base = base
        self.exponent = exponent

    def __str__(self) -> str:
        if self._precedence > self.base._precedence:
            return f'({self.base})^{self.exponent}'
        else:
            return f'{self.base}^{self.exponent}'


class Variable(Term, firstorder.Variable['Variable', int, SortKey['Variable']]):

    name: str
    VV: ClassVar[VariableSet] = VV

    @property
    def args(self) -> tuple[str]:
        return (self.name,)

    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name

    def fresh(self) -> Variable:
        """Returns a variable that has not been used so far. Implements
        abstract method :meth:`.firstorder.atomic.Variable.fresh`.
        """
        return self.VV.fresh(suffix=f'_{str(self)}')


class Constant(Term):
    
    real: mpq
    imag: mpq

    @property
    def args(self) -> tuple[mpq, mpq]:
        return (self.real, self.imag)
    
    @property
    def _precedence(self) -> int:
        if self.is_real():
            if self.real < 0:
                return Term.PRECEDENCE['Neg']
            else:
                return Term.PRECEDENCE['Constant']
        elif self.is_imaginary():
            if self.imag == 1:
                return Term.PRECEDENCE['Variable']
            elif self.imag == -1:
                return Term.PRECEDENCE['Neg']
            else:
                return Term.PRECEDENCE['Mul']
        else:
            return Term.PRECEDENCE['Add']

    def __init__(self, *args: object) -> None:
        super().__init__()
        if len(args) == 1:
            if isinstance(args[0], complex):
                self.real = mpq(args[0].real)
                self.imag = mpq(args[0].imag)
            else:
                self.real = Constant._real_to_mpq(args[0])
                self.imag = mpq(0)
        elif len(args) == 2:
            self.real = Constant._real_to_mpq(args[0])
            self.imag = Constant._real_to_mpq(args[1])
        else:
            raise ValueError(f'Cannot contruct constant from {type(args)}!')
        
    def __str__(self) -> str:
        if self.is_real():
            return str(self.real)
        elif self.is_imaginary():
            return Constant._imag_str(self.imag)
        elif self.imag < 0:
            return f'{str(self.real)} - {Constant._imag_str(-self.imag)}'
        else:
            return f'{str(self.real)} + {Constant._imag_str(self.imag)}'
        
    @staticmethod
    def _imag_str(x: mpq) -> str:
        assert x.denominator > 0
        denom = '/' + str(x.denominator) if x.denominator != 1 else ''
        if x.numerator == 0:
            return '0'
        elif x.numerator == 1:
            return 'i' + denom
        elif x.numerator == -1:
            return '-i' + denom
        else:
            return str(x.numerator) + 'i' + denom
        
    def is_imaginary(self) -> bool:
        return self.real == mpq(0)

    def is_real(self) -> bool:
        return self.imag == mpq(0)

    @staticmethod
    def _real_to_mpq(value: object) -> mpq:
        if isinstance(value, (int, float)):
            return mpq(value)
        elif isinstance(value, Fraction):
            return mpq(value.numerator, value.denominator)
        elif isinstance(value, mpq):
            return value
        else:
            raise ValueError(f'Cannot contruct mpq from {type(value)}!')
        
        
class UnaryOperation(Term):

    arg: Term

    @property
    def args(self) -> tuple[Term]:
        return (self.arg,)
    
    def __init__(self, arg: object) -> None:
        if isinstance(arg, Term):
            self.arg = arg
        else:
            self.arg = Constant(arg)


class Neg(UnaryOperation):
    
    def __str__(self) -> str:
        if self._precedence > self.arg._precedence:
            return f'-({self.arg})'
        else:
            return f'-{self.arg}'


class Re(UnaryOperation):
    
    def __str__(self) -> str:
        return f'Re({self.arg})'
    

class Im(UnaryOperation):
    
    def __str__(self) -> str:
        return f'Im({self.arg})'


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

    def __init__(self, lhs: object, rhs: object):
        self.args = (
            lhs if isinstance(lhs, Term) else Constant(lhs), 
            rhs if isinstance(rhs, Term) else Constant(rhs)
        )

    def __le__(self, other: Formula) -> bool:
        """Returns `True` if this atomic formula should be sorted before or is
        equal to other. Implements abstract method
        :meth:`.firstorder.atomic.AtomicFormula.__le__`.
        """
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
        for v in self.lhs.vars():
            if v in quantified:
                yield v
        for v in self.rhs.vars():
            if v in quantified:
                yield v

    @classmethod
    def complement(cls) -> type[AtomicFormula]:
        """Complement relation. Implements the abstract method
        :meth:`.firstorder.atomic.AtomicFormula.complement`.

        .. seealso::
          Inherited method :meth:`.firstorder.atomic.AtomicFormula.to_complement`
          """
        return {Eq: Ne, Ne: Eq, Le: Gt, Lt: Ge, Ge: Lt, Gt: Le}[cls]

    def fvars(self, quantified: frozenset[Variable] = frozenset()) -> Iterator[Variable]:
        """Iterate over occurrences of variables that are *not* elements of
        `quantified`. Yield each such variable once for each term that it
        occurs in. Implements the abstract method
        :meth:`.firstorder.atomic.AtomicFormula.fvars`.
        """
        for v in self.lhs.vars():
            if v not in quantified:
                yield v
        for v in self.rhs.vars():
            if v not in quantified:
                yield v

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