from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, ClassVar, Final, Generic, Never, Optional, Self, TypeVar

from gmpy2 import mpq

from logic1 import firstorder

if TYPE_CHECKING:
    from logic1.theories.Complex.typing import Formula

α = TypeVar('α')
τ = TypeVar('τ', bound='Term')

type Number = int | float | Fraction | mpq | complex
_NUMBER_TYPES: Final = (int, float, Fraction, mpq, complex)


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

    # Variable(x), Re(x), Im(x), abs(x), Varriable(y), ... Rational, I, ...

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
    def visit_re(self, re: Re) -> α:
        ...

    @abstractmethod
    def visit_im(self, im: Im) -> α:
        ...


class Term(firstorder.Term['Term', 'Variable', Never, SortKey['Term']]):
    """
    Class representing a node of the AST
    """

    @property
    def op(self) -> type[Self]:
        return type(self)
    
    @property
    def args(self) -> tuple[object, ...]:  # type: ignore[empty-body]
        ...
    
    def __add__(self, other: Number | Term) -> Term:
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

    def __le__(self, other: Number | Term) -> Le:
        if isinstance(other, Term):
            return Le(self, other)
        return self <= Term.from_number(other)

    def __lt__(self, other: Number | Term) -> Lt:
        if isinstance(other, Term):
            return Lt(self, other)
        return self < Term.from_number(other)
        
    def __mul__(self, other: Number | Term) -> Term:
        if isinstance(other, Term):
            return Mul(self, other)
        return self * Term.from_number(other)
        
    def __ne__(self, other: Number | Term) -> Ne:  # type: ignore[override]
        if isinstance(other, Term):
            return Ne(self, other)
        return self != Term.from_number(other)

    def __neg__(self) -> Term:
        return Neg(self)

    def __pow__(self, other: int) -> Term:
        return Pow(self, other)

    def __radd__(self, other: Number | Term) -> Term:
        assert not isinstance(other, Term)
        return Term.from_number(other) + self
    
    def __repr__(self) -> str:
        return self.accept(ReprFormatter())

    def __rmul__(self, other: Number | Term) -> Term:
        assert not isinstance(other, Term)
        return Term.from_number(other) * self

    def __rsub__(self, other: Number | Term) -> Term:
        assert not isinstance(other, Term)
        return Term.from_number(other) - self
    
    def __str__(self) -> str:
        return self.accept(StrFormatter())

    def __sub__(self, other: Number | Term) -> Term:
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
        if isinstance(self, Variable):
            return False
        return all(isinstance(arg, Term) and arg.is_constant() for arg in self.args)

    def is_variable(self) -> bool:
        """Return :obj:`True` if this term is a variable.
        """
        return isinstance(self, Variable)

    def normalize(self) -> Term:
        return self.accept(Normalizer())

    def sort_key(self) -> SortKey[Self]:
        """A sort key suitable for ordering instances of this class. Implements
        the abstract method :meth:`.firstorder.atomic.Term.sort_key`.
        """
        return SortKey(self)

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


class Variable(Term, firstorder.Variable['Variable', int, SortKey['Variable']]):

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


class Re(UnaryOperation):

    def accept(self, visitor: TermVisitor[α]) -> α:
        return visitor.visit_re(self)   


class Im(UnaryOperation):

    def accept(self, visitor: TermVisitor[α]) -> α:
        return visitor.visit_im(self)


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

    def __init__(self, lhs: Number | Term, rhs: Number | Term):
        self.args = (
            lhs if isinstance(lhs, Term) else Term.from_number(lhs), 
            rhs if isinstance(rhs, Term) else Term.from_number(rhs)
        )

    def __le__(self, other: Formula) -> bool:
        """Returns `True` if this atomic formula should be sorted before or is
        equal to other. Implements abstract method
        :meth:`.firstorder.atomic.AtomicFormula.__le__`.
        """
        raise NotImplementedError()
    
    def __repr__(self) -> str:
        SYMBOL: Final = {Eq: '==', Ne: '!=', Ge: '>=', Le: '<=', Gt: '>', Lt: '<'}
        SPACING: Final = ' '
        return f'{repr(self.lhs)}{SPACING}{SYMBOL[self.op]}{SPACING}{repr(self.rhs)}'

    def __str__(self) -> str:
        """String representation of this atomic formula. Implements the
        abstract method :meth:`.firstorder.atomic.AtomicFormula.__str__`.
        """
        SYMBOL: Final = {Eq: '==', Ne: '!=', Ge: '>=', Le: '<=', Gt: '>', Lt: '<'}
        SPACING: Final = ' '
        return f'{str(self.lhs)}{SPACING}{SYMBOL[self.op]}{SPACING}{str(self.rhs)}'

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
    
    def _dump(self) -> str:
        return f'{self.op.__name__}({self.lhs._dump()}, {self.rhs._dump()})'

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


from logic1.theories.Complex.format import LatexFormatter, ReprFormatter, StrFormatter
from logic1.theories.Complex.simplify import Evaluator, Normalizer