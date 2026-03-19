from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
import functools
from typing import TYPE_CHECKING, ClassVar, Final, Generic, Optional, Self, TypeVar

from gmpy2 import mpq

from logic1 import firstorder
from logic1.theories.Complex.types import _RATIONAL_NUMBER_TYPES, Number, _NUMBER_TYPES, RationalNumber

α = TypeVar('α')
τ = TypeVar('τ', bound='Term')


@dataclass
class VariableSet(firstorder.atomic.VariableSet['Variable']):
    
    _names: set[str] = field(default_factory=set)
     
    @property
    def stack(self) -> list[set[str]]:
        return [self._names]

    def __getitem__(self, index: str) -> Variable:
        """Implements the abstract method
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
    """A sort key for terms. Implements the abstract class
    :class:`.firstorder.atomic.Term.SortKey`.
    """

    term: τ
    """The term for which this is a sort key.
    """

    @property
    def op(self) -> type[Term]:
        """The operation of the underlying term.
        """
        return self.term.op
    
    @property
    def args(self) -> tuple[object, ...]:
        """The arguments of the underlying term, where each argument
        that is itself a term is replaced by its sort key.
        """
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
        

class Term(firstorder.Term['Term', 'Variable', Number, SortKey]):
    """An expression consisting of complex variables, rational numbers,
    the imaginary unit, arithmetic operations, complex conjugation, and
    real and imaginary part. Implements the abstract class
    :class:`.firstorder.atomic.Term` for the theory of complex numbers.
    """

    @property
    def op(self) -> type[Self]:
        """The operator of this term, which is represented by the
        class of this term.
        """
        return type(self)
    
    @property
    def args(self) -> tuple[object, ...]:  # type: ignore[empty-body]
        """The arguments of this term. Note that `self == self.op(*self.args)`.
        """
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
        """String representation of this term that can be evaluated to
        reconstruct the term. For a more human-readable string
        representation, use :meth:`.Term.__str__` or :meth:`.Term.as_latex`.
        """
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
        """Division is defined as multiplication by the inverse. If the
        other term is not constant, this method raises a ValueError.
        """
        if isinstance(other, Term):
            try:
                a, b = other.eval()
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
    
    def lc(self) -> Term:
        """Returns the leading constant of this term.
        """
        if self.is_constant():
            return self
        elif isinstance(self, Add):
            return self.args[0].lc()
        elif isinstance(self, Neg):
            return -self.arg.lc()
        elif isinstance(self, Mul) and self.args[0].is_constant():
            return self.args[0] * Mul(*self.args[1:]).lc()
        elif isinstance(self, Mul) and isinstance(self.args[0], Neg):
            return -Mul(*self.args[1:]).lc()
        else:
            return Rational(1)

    def _dump(self) -> str:
        """Dump this term as a string that can be evaluated to
        reconstruct the term. This is used for debugging.
        """
        args = []
        for arg in self.args:
            if isinstance(arg, Term):
                args.append(arg._dump())
            else:
                args.append(repr(arg))
        return f'{self.op.__name__}({", ".join(args)})'
    
    def eval(self, variables: dict[Variable, Number] = dict()) -> tuple[mpq, mpq]:
        """Evaluate this term to a pair of rational numbers
        representing the real and imaginary part of this term, given an
        assignment of the variables in this term to rational numbers.
        Raises a ValueError if this term contains any variables that are
        not in the given assignment.
        """
        return self.accept(ConstantEvaluator(variables))
        
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
        if isinstance(value, _RATIONAL_NUMBER_TYPES):
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
            self.eval()
            return True
        except ValueError:
            return False
        
    def is_imaginary(self) -> bool:
        """Return :obj:`True` if this term is imaginary, i.e., its real
        part is zero.
        """
        return Re(self).is_zero()

    def is_real(self) -> bool:
        """Return :obj:`True` if this term is real, i.e., its imaginary
        part is zero.
        """
        return Im(self).is_zero()

    def is_variable(self) -> bool:
        """Return :obj:`True` if this term is a variable.
        """
        return isinstance(self, Variable)

    def is_zero(self) -> bool:
        """Return :obj:`True` if this term is equivalent to zero.
        """
        try:
            a, b = self.normalize_complex().eval()
            return a == mpq(0) and b == mpq(0)
        except ValueError:
            return False

    def normalize(self) -> Term:
        return self.accept(Normalizer())
    
    def normalize_complex(self) -> Term:
        return self.accept(ComplexNormalizer())
    
    def normalize_weak(self) -> Term:
        return self.accept(WeakNormalizer())
    
    def _repr_latex_(self) -> str:
        """LaTeX representation for Jupyter notebooks.
        """
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
    """A rational number, represented as a gmpy2.mpq. Implements the
    abstract class :class:`.firstorder.atomic.Term`.
    """
    
    value: mpq

    @property
    def args(self) -> tuple[mpq]:
        """The arguments of this term, which is just the rational number itself."""
        return (self.value,)

    def __init__(self, value: RationalNumber) -> None:
        """Initialize this rational term with the given value. The
        value must be non-negative, otherwise this term is represented
        as a negation of a rational term with a non-negative value.
        """
        if isinstance(value, (int, float)):
            self.value = mpq(value)
        elif isinstance(value, Fraction):
            self.value = mpq(value.numerator, value.denominator)
        elif isinstance(value, mpq):
            self.value = value
        else:
            number_types = ', '.join(c.__name__ for c in _RATIONAL_NUMBER_TYPES)
            raise ValueError(f'expected one of {number_types}; {value} is {type(value)}')
        assert self.value >= mpq(0)

    def __new__(cls, value: RationalNumber):
        if value < 0:
            return Neg(Rational(-value))
        else:
            return super().__new__(cls)
        

    def accept(self, visitor: TermVisitor[α]) -> α:
        """Implements the abstract method :meth:`.Term.accept`."""
        return visitor.visit_rational(self)


class _I(Term):
    """The imaginary unit. This is a singleton class, and the only
    instance is `I`. Implements the abstract class
    :class:`.firstorder.atomic.Term`.
    """

    _instance: Optional[_I] = None
    """The singleton instance of this class.
    """
    
    @property
    def args(self) -> tuple[()]:
        """The imaginary unit has no arguments.
        """
        return ()
    
    def __init__(self) -> None:
        """Initialize the imaginary unit. This is a singleton class, so
        this method should not be called directly. Use `I` instead.
        """
        pass

    def accept(self, visitor: TermVisitor[α]) -> α:
        """Implements the abstract method :meth:`.Term.accept`.
        """
        return visitor.visit_i(self)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    
I: Final = _I()
"""The singleton instance of the imaginary unit.
"""


class Variable(Term, firstorder.Variable['Variable', Number, SortKey['Variable']]):
    """A variable, represented by a string name. Implements the
    abstract class :class:`.firstorder.atomic.Term` and
    :class:`.firstorder.atomic.Variable`.
    """

    name: str
    """The name of this variable.
    """

    VV: ClassVar[VariableSet] = VV
    """The variable set containing all existing complex variables. See
    :class:`.VariableSet` for details.
    """
    
    @property
    def args(self) -> tuple[str]:
        """The arguments of this term, which is just the name of the
        variable itself.
        """
        return (self.name,)

    def __init__(self, name: str) -> None:
        """Initialize this variable with the given name.
        """
        self.name = name  # TODO: VV register here?

    def accept(self, visitor: TermVisitor[α]) -> α:
        """Implements the abstract method :meth:`.Term.accept`.
        """
        return visitor.visit_variable(self)

    def fresh(self) -> Variable:
        """Returns a variable that has not been used so far. Implements
        abstract method :meth:`.firstorder.atomic.Variable.fresh`.
        """
        return self.VV.fresh(suffix=f'_{str(self)}')
    
    
class MonoidalOperation(Term):
    """A monoidal operation, which is an associative operation with an
    identity element. This is a base class for addition and
    multiplication. Implements parts the abstract class :class:`.Term`.
    """

    _args: tuple[Term, ...]
    """The arguments of this term.
    """

    identity: ClassVar[Term]
    """The identity element of this operation. This should be
    overridden by subclasses.
    """

    @property
    def args(self) -> tuple[Term, ...]:
        """The arguments of this term.
        """
        return self._args

    def __init__(self, *args: Term) -> None:
        """Initialize this monoidal operation with the given arguments.
        If any of the arguments is itself a monoidal operation of the
        same type, it is flattened into the arguments of this term.
        """
        args_flat = []
        for arg in args:
            if isinstance(arg, self.__class__):
                args_flat.extend(list(arg.args))
            else:
                args_flat.append(arg)
        self._args = tuple(args_flat)

    def __new__(cls, *args: Term):
        """Create a new instance of this monoidal operation with the
        given arguments. If no arguments are given, return the identity
        element. If only one argument is given, return that argument.
        """
        if not args:
            return cls.identity
        if len(args) == 1:
            return args[0]
        return super().__new__(cls)
    

class Add(MonoidalOperation):
    """Addition. Implements the abstract class :class:`.MonoidalOperation`.
    """

    identity: ClassVar[Rational] = Rational(mpq(0))
    """The identity element of addition, which is the rational number 0.
    """

    def accept(self, visitor: TermVisitor[α]) -> α:
        """Implements the abstract method :meth:`.Term.accept`.
        """
        return visitor.visit_add(self)


class Mul(MonoidalOperation):
    """Multiplication. Implements the abstract class
    :class:`.MonoidalOperation`.
    """

    identity: ClassVar[Rational] = Rational(mpq(1))
    """The identity element of multiplication, which is the rational number 1.
    """

    def accept(self, visitor: TermVisitor[α]) -> α:
        """Implements the abstract method :meth:`.Term.accept`.
        """
        return visitor.visit_mul(self)


class Pow(Term):
    """Exponentiation. Implements the abstract class :class:`.Term`.
    """
    
    base: Term
    """The base of this power term.
    """
    
    exponent: int
    """The exponent of this power term. Must be a non-negative integer.
    """

    @property
    def args(self) -> tuple[Term, int]:
        """The arguments of this term, which is the base and the exponent.
        """
        return (self.base, self.exponent)
    
    def __init__(self, base: Term, exponent: int) -> None:
        """Initialize this power term with the given base and exponent.
        The exponent must be a non-negative integer.
        """
        if not isinstance(exponent, int) or exponent < 0:
            raise TypeError('Exponent must be a non-negative integer')
        self.base = base
        self.exponent = exponent

    def accept(self, visitor: TermVisitor[α]) -> α:
        """Implements the abstract method :meth:`.Term.accept`.
        """
        return visitor.visit_pow(self)

        
class UnaryOperation(Term):
    """A unary operation, which is an operation with one argument.
    This is a base class for negation, conjugation, real part, and
    imaginary part. Implements parts the abstract class :class:`.Term`.
    """

    arg: Term
    """The single argument of this unary operation."""

    @property
    def args(self) -> tuple[Term]:
        """The arguments of this term, which is just the single
        argument of this unary operation.
        """
        return (self.arg,)
    
    def __init__(self, arg: Number | Term) -> None:
        """Initialize this unary operation with the given argument,
        which can be either a term or a number.
        If it is a number, it is converted to a term using
        :meth:`.Term.from_number`.
        """
        if isinstance(arg, Term):
            self.arg = arg
        else:
            self.arg = Term.from_number(arg)


class Neg(UnaryOperation):
    """Negation. Implements the abstract class :class:`.UnaryOperation`.
    """

    def accept(self, visitor: TermVisitor[α]) -> α:
        """Implements the abstract method :meth:`.Term.accept`.
        """
        return visitor.visit_neg(self)
    

class Conj(UnaryOperation):
    """Complex conjugation. Implements the abstract class
    :class:`.UnaryOperation`.
    """

    def accept(self, visitor: TermVisitor[α]) -> α:
        """Implements the abstract method :meth:`.Term.accept`.
        """
        return visitor.visit_conj(self)


class Re(UnaryOperation):
    """Real part. Implements the abstract class :class:`.UnaryOperation`.
    """

    def accept(self, visitor: TermVisitor[α]) -> α:
        """Implements the abstract method :meth:`.Term.accept`.
        """
        return visitor.visit_re(self)   


class Im(UnaryOperation):
    """Imaginary part. Implements the abstract class :class:`.UnaryOperation`.
    """

    def accept(self, visitor: TermVisitor[α]) -> α:
        """Implements the abstract method :meth:`.Term.accept`.
        """
        return visitor.visit_im(self)


class TermVisitor(Generic[α]):
    """Visitor for terms. This is used to implement various operations
    on terms, such as normalization, evaluation, etc.
    """

    @abstractmethod
    def visit_rational(self, num: Rational) -> α:
        """Visit a rational term.
        """
        ...

    @abstractmethod
    def visit_i(self, i: _I) -> α:
        """Visit the imaginary unit.
        """
        ...

    @abstractmethod
    def visit_variable(self, var: Variable) -> α:
        """Visit a variable.
        """
        ...

    @abstractmethod
    def visit_add(self, add: Add) -> α:
        """Visit an addition term.
        """
        ...

    @abstractmethod
    def visit_mul(self, mul: Mul) -> α:
        """Visit a multiplication term.
        """
        ...

    @abstractmethod
    def visit_pow(self, pow: Pow) -> α:
        """Visit a power term.
        """
        ...

    @abstractmethod
    def visit_neg(self, neg: Neg) -> α:
        """Visit a negation term.
        """
        ...

    @abstractmethod
    def visit_conj(self, conj: Conj) -> α:
        """Visit a conjugation term.
        """
        ...

    @abstractmethod
    def visit_re(self, re: Re) -> α:
        """Visit a real part term.
        """
        ...

    @abstractmethod
    def visit_im(self, im: Im) -> α:
        """Visit an imaginary part term.
        """
        ...


class IdentityTermVisitor(TermVisitor[Term]):
    """Visitor that returns the same term, but with all subterms
    visited. Useful as a base class for other visitors.
    """
    
    def visit_rational(self, num: Rational) -> Term:
        """Return the same rational term. Implements the abstract
        method :meth:`.TermVisitor.visit_rational`.
        """
        return num
    
    def visit_i(self, i: _I) -> Term:
        """Return the imaginary unit. Implements the abstract method
        :meth:`.TermVisitor.visit_i`.
        """
        return i
    
    def visit_variable(self, var: Variable) -> Term:
        """Return the same variable. Implements the abstract method
        :meth:`.TermVisitor.visit_variable`.
        """
        return var
    
    def visit_add(self, add: Add) -> Term:
        """Return the same addition term, but with all arguments
        visited. Implements the abstract method
        :meth:`.TermVisitor.visit_add`.
        """
        return Add(*[arg.accept(self) for arg in add.args])
    
    def visit_mul(self, mul: Mul) -> Term:
        """Return the same multiplication term, but with all arguments
        visited. Implements the abstract method
        :meth:`.TermVisitor.visit_mul`.
        """
        return Mul(*[arg.accept(self) for arg in mul.args])

    def visit_pow(self, pow: Pow) -> Term:
        """Return the same power term, but with the base visited.
        Implements the abstract method :meth:`.TermVisitor.visit_pow`.
        """
        return Pow(pow.base.accept(self), pow.exponent)

    def visit_neg(self, neg: Neg) -> Term:
        """Return the same negation term, but with the argument
        visited. Implements the abstract method
        :meth:`.TermVisitor.visit_neg`.
        """
        return Neg(neg.arg.accept(self))

    def visit_conj(self, conj: Conj) -> Term:
        """Return the same conjugation term, but with the argument
        visited. Implements the abstract method
        :meth:`.TermVisitor.visit_conj`.
        """
        return Conj(conj.arg.accept(self))

    def visit_re(self, re: Re) -> Term:
        """Return the same real part term, but with the argument
        visited. Implements the abstract method
        :meth:`.TermVisitor.visit_re`.
        """
        return Re(re.arg.accept(self))

    def visit_im(self, im: Im) -> Term:
        """Return the same imaginary part term, but with the argument
        visited. Implements the abstract method
        :meth:`.TermVisitor.visit_im`.
        """
        return Im(im.arg.accept(self))
    

class VariableSubstitutor(IdentityTermVisitor):
    """Visitor that substitutes variables according to a given
    mapping. See also :meth:`.Term.subs`.
    """
    
    mapping: Mapping[Variable, Number | Term]

    def __init__(self, mapping: Mapping[Variable, Number | Term]) -> None:
        """Initialize the substitutor with the given mapping
        containing either terms or numbers.
        """
        self.mapping = mapping
    
    def visit_variable(self, var: Variable) -> Term:
        """Return the substituted term for the given variable, or the
        variable itself if not found in the mapping.
        """
        value = self.mapping.get(var, var)
        if isinstance(value, Term):
            return value
        else:
            return Term.from_number(value)
    

from logic1.theories.Complex.atomic import Eq, Ge, Gt, Le, Lt, Ne
from logic1.theories.Complex.format import LatexFormatter, ReprFormatter, StrFormatter
from logic1.theories.Complex.normalize import ComplexNormalizer, ConstantEvaluator, Normalizer, WeakNormalizer