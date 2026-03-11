from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
import functools
import operator
from typing import TYPE_CHECKING, ClassVar, Final, Generic, Optional, Self, TypeVar

from gmpy2 import mpq

from logic1 import firstorder
from logic1.theories.Complex.typing import Number, _NUMBER_TYPES

α = TypeVar('α')
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
@functools.total_ordering
class SortKey(Generic[τ]):

    term: τ

    @property
    def op(self) -> type[Term]:
        return self.term.op
    
    @property
    def args(self) -> tuple[object, ...]:
        return tuple(SortKey(arg) if isinstance(arg, Term) else arg for arg in self.term.args)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SortKey):
            return False
        if self.term is other.term:
            return True
        return self.op == other.op and self.args == other.args

    def __hash__(self) -> int:
        return hash(self.term)

    def __le__(self, other: SortKey) -> bool:
        ORDER = (Add, Mul, Neg, Conj, Pow, Im, Re, Variable, _I, Rational)
        assert self.op in ORDER and other.op in ORDER
        if self.op == other.op:
            return self.args <= other.args
        else:
            return ORDER.index(self.op) < ORDER.index(other.op)
        

class Term(firstorder.Term['Term', 'Variable', Number, SortKey['Term']]):
    """
    Class representing a node of the AST
    """

    @property
    def op(self) -> type[Self]:
        return type(self)
    
    @property
    def args(self) -> tuple[object, ...]:  # type: ignore[empty-body]
        ...
    
    def __add__(self, other: Number | Term) -> Add:
        if isinstance(other, Term):
            return Add(self, other)
        return self + Term.from_number(other)
        
    def __eq__(self, other: Number | Term) -> Eq:  # type: ignore[override]
        if isinstance(other, Term):
            return Eq(self, other)
        return self == Term.from_number(other)
        
    def __ge__(self, other: Number | Term) -> Ge:
        if isinstance(other, Term):
            return Ge(self, other)
        return self >= Term.from_number(other)
        
    def __gt__(self, other: Number | Term) -> Gt:
        if isinstance(other, Term):
            return Gt(self, other)
        return self > Term.from_number(other)
        
    def __hash__(self) -> int:
        return hash((tuple(str(cls) for cls in self.op.mro()), self.args))
    
    @abstractmethod
    def __init__(self, *args: object) -> None:
        ...

    def __invert__(self) -> Conj:
        return Conj(self)

    def __le__(self, other: Number | Term) -> Le:
        if isinstance(other, Term):
            return Le(self, other)
        return self <= Term.from_number(other)

    def __lt__(self, other: Number | Term) -> Lt:
        if isinstance(other, Term):
            return Lt(self, other)
        return self < Term.from_number(other)
        
    def __mul__(self, other: Number | Term) -> Mul:
        if isinstance(other, Term):
            return Mul(self, other)
        return self * Term.from_number(other)
        
    def __ne__(self, other: Number | Term) -> Ne:  # type: ignore[override]
        if isinstance(other, Term):
            return Ne(self, other)
        return self != Term.from_number(other)

    def __neg__(self) -> Neg:
        return Neg(self)

    def __pow__(self, other: int) -> Pow:
        return Pow(self, other)

    def __radd__(self, other: Number | Term) -> Add:
        assert not isinstance(other, Term)
        return Term.from_number(other) + self
    
    def __repr__(self) -> str:
        return self.accept(ReprFormatter())

    def __rmul__(self, other: Number | Term) -> Mul:
        assert not isinstance(other, Term)
        return Term.from_number(other) * self

    def __rsub__(self, other: Number | Term) -> Add:
        assert not isinstance(other, Term)
        return Term.from_number(other) - self
    
    def __str__(self) -> str:
        return self.accept(StrFormatter())

    def __sub__(self, other: Number | Term) -> Add:
        if isinstance(other, Term):
            return self + (-other)
        return self - Term.from_number(other)
        
    def __truediv__(self, other: Number | Term) -> Term:
        if isinstance(other, Term):
            try:
                a, b = other.eval_constant()
            except ValueError:
                raise ValueError('Cannot divide by a non-constant term')
            if a == mpq(0) and b == mpq(0):
                raise ZeroDivisionError('Division by zero')
            a, b = (a / (a * a + b * b), -b / (a * a + b * b))
            return Mul(self, Term.from_real_imag(a, b))
        return self / Term.from_number(other)

    def __xor__(self, other: Number | Term) -> Term:
        raise NotImplementedError(
            "Use ** for exponentiation, not '^', which means xor "
            "in Python, and has the wrong precedence")
    
    @abstractmethod
    def accept(self, visitor: TermVisitor[α]) -> α:
        """Accept a visitor."""
        ...

    def as_latex(self) -> str:
        """LaTeX representation as a string. Implements the abstract method
        :meth:`.firstorder.atomic.Term.as_latex`.
        """
        return self.accept(LatexFormatter())
    
    def _dump(self) -> str:
        """Dump this term as a string that can be evaluated to reconstruct the term. This is used for debugging.
        """
        args = []
        for arg in self.args:
            if isinstance(arg, Term):
                args.append(arg._dump())
            else:
                args.append(repr(arg))
        return f'{self.op.__name__}({", ".join(args)})'
    
    def eval_constant(self) -> tuple[mpq, mpq]:
        """Evaluate this term if it is constant, and return the real and
        imaginary part as a pair of rational numbers. Raises ValueError if
        this term is not constant.
        """
        return self.accept(Evaluator())
        
    @staticmethod
    def from_real_imag(real: mpq, imag: mpq) -> Term:
        """Construct a term from the given real and imaginary part.
        """
        if imag == mpq(0):
            return Rational(real)
        elif real == mpq(0):
            if imag == mpq(1):
                return _I()
            elif imag == mpq(-1):
                return Neg(_I())
            else:
                return Mul(Rational(imag), _I())
        else:
            return Add(Rational(real), Mul(Rational(imag), _I()))
    
    @staticmethod
    def from_number(value: Number) -> Term:
        """Construct a term from the given number.
        """
        if isinstance(value, (int, float)):
            return Rational(mpq(value))
        elif isinstance(value, Fraction):
            return Rational(mpq(value.numerator, value.denominator))
        elif isinstance(value, mpq):
            return Rational(value)
        elif isinstance(value, complex):
            return Term.from_real_imag(mpq(value.real), mpq(value.imag))
        else:
            number_types = ', '.join(c.__name__ for c in _NUMBER_TYPES)
            raise ValueError(f'expected one of {number_types}; {value} is {type(value)}')

    def is_constant(self) -> bool:
        """Return :obj:`True` if this term is constant.
        """
        try:
            self.eval_constant()
            return True
        except ValueError:
            return False

    def is_variable(self) -> bool:
        """Return :obj:`True` if this term is a variable.
        """
        return isinstance(self, Variable)

    def normalize(self) -> Term:
        return self.accept(Normalizer())
    
    def normalize_complex(self) -> Term:
        return self.accept(ComplexNormalizer())
    
    def normalize_weak(self) -> Term:
        return self.accept(WeakNormalizer())
    
    def _repr_latex_(self) -> str:
        result = f'$\\displaystyle {self.as_latex()}$'
        if len(result) > 5000:
            raise ValueError('Latex output too long')
        return result 

    def sort_key(self) -> SortKey[Self]:
        """A sort key suitable for ordering instances of this class. Implements
        the abstract method :meth:`.firstorder.atomic.Term.sort_key`.
        """
        return SortKey(self)

    def subs(self, sigma: Mapping[Variable, Number | Term]) -> Term:
        """Formal term substitution. Returns the result of
        substituting each variable `v` in `sigma` with
        `sigma[v]`.
        """
        return self.accept(VariableSubstitutor(sigma))

    def vars(self) -> Iterator[Variable]:
        """An iterator that yields each variable of this term once. Implements
        the abstract method :meth:`.firstorder.atomic.Term.vars`.
        """
        if isinstance(self, Variable):
            yield self
        else:
            vars: set[Variable] = set()
            for arg in self.args:
                if isinstance(arg, Term):
                    vars.update(arg.vars())
            yield from vars


class Rational(Term):
    
    value: mpq

    @property
    def args(self) -> tuple[mpq]:
        return (self.value,)

    def __init__(self, value: mpq) -> None:
        self.value = value

    def accept(self, visitor: TermVisitor[α]) -> α:
        return visitor.visit_rational(self)


class _I(Term):

    _instance: Optional[_I] = None
    
    @property
    def args(self) -> tuple[()]:
        return ()
    
    def __init__(self) -> None:
        pass

    def accept(self, visitor: TermVisitor[α]) -> α:
        return visitor.visit_i(self)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    
I: Final = _I()


class Variable(Term, firstorder.Variable['Variable', Number, SortKey['Variable']]):

    name: str
    VV: ClassVar[VariableSet] = VV

    @property
    def args(self) -> tuple[str]:
        return (self.name,)

    def __init__(self, name: str) -> None:
        self.name = name

    def accept(self, visitor: TermVisitor[α]) -> α:
        return visitor.visit_variable(self)

    def fresh(self) -> Variable:
        """Returns a variable that has not been used so far. Implements
        abstract method :meth:`.firstorder.atomic.Variable.fresh`.
        """
        return self.VV.fresh(suffix=f'_{str(self)}')
    
    
class MonoidalOperation(Term):

    _args: tuple[Term, ...]
    identity: Term

    @property
    def args(self) -> tuple[Term, ...]:
        return self._args

    def __init__(self, *args: Term) -> None:
        args_flat = []
        for arg in args:
            if isinstance(arg, self.__class__):
                args_flat.extend(list(arg.args))
            else:
                args_flat.append(arg)
        self._args = tuple(args_flat)

    def __new__(cls, *args: Term):
        if not args:
            return cls.identity
        if len(args) == 1:
            return args[0]
        return super().__new__(cls)
    

class Add(MonoidalOperation):

    identity: Rational = Rational(mpq(0))

    def accept(self, visitor: TermVisitor[α]) -> α:
        return visitor.visit_add(self)


class Mul(MonoidalOperation):

    identity: Rational = Rational(mpq(1))

    def accept(self, visitor: TermVisitor[α]) -> α:
        return visitor.visit_mul(self)


class Pow(Term):
    
    base: Term
    exponent: int

    @property
    def args(self) -> tuple[Term, int]:
        return (self.base, self.exponent)
    
    def __init__(self, base: Term, exponent: int) -> None:
        if not isinstance(exponent, int) or exponent < 0:
            raise TypeError('Exponent must be a non-negative integer')
        self.base = base
        self.exponent = exponent

    def accept(self, visitor: TermVisitor[α]) -> α:
        return visitor.visit_pow(self)

        
class UnaryOperation(Term):

    arg: Term

    @property
    def args(self) -> tuple[Term]:
        return (self.arg,)
    
    def __init__(self, arg: Number | Term) -> None:
        if isinstance(arg, Term):
            self.arg = arg
        else:
            self.arg = Term.from_number(arg)


class Neg(UnaryOperation):

    def accept(self, visitor: TermVisitor[α]) -> α:
        return visitor.visit_neg(self)
    

class Conj(UnaryOperation):

    def accept(self, visitor: TermVisitor[α]) -> α:
        return visitor.visit_conj(self)


class FunctionSymbol(UnaryOperation):
    pass


class Re(FunctionSymbol):

    def accept(self, visitor: TermVisitor[α]) -> α:
        return visitor.visit_re(self)   


class Im(FunctionSymbol):

    def accept(self, visitor: TermVisitor[α]) -> α:
        return visitor.visit_im(self)


class TermVisitor(Generic[α]):

    @abstractmethod
    def visit_rational(self, num: Rational) -> α:
        ...

    @abstractmethod
    def visit_i(self, i: _I) -> α:
        ...

    @abstractmethod
    def visit_variable(self, var: Variable) -> α:
        ...

    @abstractmethod
    def visit_add(self, add: Add) -> α:
        ...

    @abstractmethod
    def visit_mul(self, mul: Mul) -> α:
        ...

    @abstractmethod
    def visit_pow(self, pow: Pow) -> α:
        ...

    @abstractmethod
    def visit_neg(self, neg: Neg) -> α:
        ...

    @abstractmethod
    def visit_conj(self, conj: Conj) -> α:
        ...

    @abstractmethod
    def visit_re(self, re: Re) -> α:
        ...

    @abstractmethod
    def visit_im(self, im: Im) -> α:
        ...


class IdentityTermVisitor(TermVisitor[Term]):
    """Visitor that returns the same term, but with all subterms visited. Useful as a base class for other visitors.
    """
    
    def visit_rational(self, num: Rational) -> Term:
        return num
    
    def visit_i(self, i: _I) -> Term:
        return i
    
    def visit_variable(self, var: Variable) -> Term:
        return var
    
    def visit_add(self, add: Add) -> Term:
        return Add(*[arg.accept(self) for arg in add.args])
    
    def visit_mul(self, mul: Mul) -> Term:
        return Mul(*[arg.accept(self) for arg in mul.args])

    def visit_pow(self, pow: Pow) -> Term:
        return Pow(pow.base.accept(self), pow.exponent)

    def visit_neg(self, neg: Neg) -> Term:
        return Neg(neg.arg.accept(self))

    def visit_conj(self, conj: Conj) -> Term:
        return Conj(conj.arg.accept(self))

    def visit_re(self, re: Re) -> Term:
        return Re(re.arg.accept(self))

    def visit_im(self, im: Im) -> Term:
        return Im(im.arg.accept(self))
    

class VariableSubstitutor(IdentityTermVisitor):
    """Visitor that substitutes variables according to a given mapping.
    """
    
    mapping: Mapping[Variable, Number | Term]

    def __init__(self, mapping: Mapping[Variable, Number | Term]) -> None:
        self.mapping = mapping
    
    def visit_variable(self, var: Variable) -> Term:
        value = self.mapping.get(var, var)
        if isinstance(value, Term):
            return value
        else:
            return Term.from_number(value)
    

from logic1.theories.Complex.atomic import Eq, Ge, Gt, Le, Lt, Ne
from logic1.theories.Complex.format import LatexFormatter, ReprFormatter, StrFormatter
from logic1.theories.Complex.simplify import ComplexNormalizer, Evaluator, Normalizer, WeakNormalizer