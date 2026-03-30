from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import total_ordering
from typing import ClassVar, Final, Generic, Never, Optional, Self, TypeVar

from gmpy2 import mpq

from logic1.theories.Complex.types import _RATIONAL_NUMBER_TYPES, Number, _NUMBER_TYPES, RationalNumber


α = TypeVar('α')
η = TypeVar('η', bound='AST')


@dataclass
@total_ordering
class SortKey(Generic[η]):
    """A sort key for AST nodes.
    """

    ast: η
    """The AST for which this is a sort key.
    """

    @property
    def op(self) -> type[AST]:
        """The operation of the underlying AST node.
        """
        return self.ast.op
    
    @property
    def args(self) -> tuple[object, ...]:
        """The arguments of the underlying AST node, where each argument
        that is itself an AST node is replaced by its sort key.
        """
        return tuple(SortKey(arg) if isinstance(arg, AST) else arg for arg in self.ast.args)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SortKey):
            return False
        if self.ast is other.ast:
            return True
        return self.op == other.op and self.args == other.args

    def __hash__(self) -> int:
        return hash(self.ast)

    def __le__(self, other: SortKey) -> bool:
        ORDER = (Rat, _I, Var, Conj, Re, Im, Pow, Neg, Mul, Add)
        assert self.op in ORDER and other.op in ORDER
        if self.op == other.op:
            return self.args <= other.args
        else:
            return ORDER.index(self.op) < ORDER.index(other.op)
        

class AST:
   
    @property
    def op(self) -> type[Self]:
        """The operation of this AST node, which is just the class of this node.
        """
        return type(self)
    
    @property
    def args(self) -> tuple[object, ...]:  # type: ignore[empty-body]
        """The arguments of this AST node. This should be overridden by
        subclasses to return the appropriate arguments so that 
        `self == self.op(*self.args)`.
        """
        ...
    
    def __add__(self, other: Number | AST) -> Add:
        if isinstance(other, AST):
            return Add(self, other)
        return self + AST.from_number(other)
        
    def __eq__(self, other: object) -> bool:
        if isinstance(other, AST):
            return self.sort_key() == other.sort_key()
        return False

    def __ge__(self, other: AST) -> bool:
        return self.sort_key() >= other.sort_key()
        
    def __gt__(self, other: AST) -> bool:
        return self.sort_key() > other.sort_key()

    def __hash__(self) -> int:
        return hash((tuple(str(cls) for cls in self.op.mro()), self.args))
    
    @abstractmethod
    def __init__(self, *args: object) -> None:
        ...

    def __invert__(self) -> Conj:
        return Conj(self)

    def __le__(self, other: AST) -> bool:
        return self.sort_key() <= other.sort_key()

    def __lt__(self, other: AST) -> bool:
        return self.sort_key() < other.sort_key()

    def __mul__(self, other: Number | AST) -> Mul:
        if isinstance(other, AST):
            return Mul(self, other)
        return self * AST.from_number(other)
        
    def __ne__(self, other: object) -> bool:
        if isinstance(other, AST):
            return self.sort_key() != other.sort_key()
        return True

    def __neg__(self) -> Neg:
        return Neg(self)

    def __pow__(self, other: int) -> Pow:
        return Pow(self, other)

    def __radd__(self, other: Number | AST) -> Add:
        assert not isinstance(other, AST)
        return AST.from_number(other) + self
    
    def __repr__(self) -> str:
        """String representation of this AST node that can be evaluated to
        reconstruct the node. For a more human-readable string
        representation, use :meth:`.AST.__str__` or :meth:`.AST.as_latex`.
        """
        return self.accept(ReprFormatter())

    def __rmul__(self, other: Number | AST) -> Mul:
        assert not isinstance(other, AST)
        return AST.from_number(other) * self

    def __rsub__(self, other: Number | AST) -> Add:
        assert not isinstance(other, AST)
        return AST.from_number(other) - self
    
    def __str__(self) -> str:
        return self.accept(StrFormatter())

    def __sub__(self, other: Number | AST) -> Add:
        if isinstance(other, AST):
            return self + (-other)
        return self - AST.from_number(other)

    def __truediv__(self, other: Number | AST) -> AST:
        """Division is defined as multiplication by the inverse. If the
        other AST node is not constant, this method raises a ValueError.
        """
        if isinstance(other, AST):
            try:
                a, b = other.eval()
            except ValueError:
                raise ValueError('Cannot divide by a non-constant AST node')
            if a == mpq(0) and b == mpq(0):
                raise ZeroDivisionError('Division by zero')
            a, b = (a / (a * a + b * b), -b / (a * a + b * b))
            return Mul(self, AST.from_real_imag(a, b))
        return self / AST.from_number(other)

    def __xor__(self, other: Never) -> AST:
        raise NotImplementedError(
            "Use ** for exponentiation, not '^', which means xor "
            "in Python, and has the wrong precedence")
    
    @abstractmethod
    def accept(self, visitor: ASTVisitor[α]) -> α:
        """Accept a visitor."""
        ...

    def as_latex(self) -> str:
        """LaTeX representation as a string.
        """
        return self.accept(LatexFormatter())
    
    def lc(self) -> AST:
        """Returns the left-most coefficient of this AST node.
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
            return Rat(1)

    def eval(self) -> tuple[mpq, mpq]:
        """Evaluate this AST node as a complex number and return the real and 
        imaginary part as a tuple of mpq. If this AST node is not constant, this 
        method raises a ValueError.
        """
        return self.accept(ConstantEvaluator())
        
    @staticmethod
    def from_real_imag(real: mpq, imag: mpq) -> AST:
        """Construct a AST node from a given real and imaginary part.

        >>> AST.from_real_imag(mpq(2), mpq(0))
        2
        >>> AST.from_real_imag(mpq(0), mpq(1))
        I
        >>> AST.from_real_imag(mpq(0), mpq(-1))
        -I
        >>> AST.from_real_imag(mpq(0), mpq(3))
        3 * I
        >>> AST.from_real_imag(mpq(2), mpq(1))
        2 + I
        >>> AST.from_real_imag(mpq(2), mpq(-1))
        2 - I
        >>> AST.from_real_imag(mpq(2), mpq(3))
        2 + 3 * I
        """
        if imag == mpq(0):
            return Rat(real)
        elif real == mpq(0):
            if imag == mpq(1):
                return _I()
            elif imag == mpq(-1):
                return Neg(_I())
            else:
                return Mul(Rat(imag), _I())
        else:
            if imag == mpq(1):
                return Add(Rat(real), _I())
            elif imag == mpq(-1):
                return Add(Rat(real), Neg(_I()))
            else:
                return Add(Rat(real), Mul(Rat(imag), _I()))
    
    @staticmethod
    def from_number(value: Number) -> AST:
        """Construct a AST node from a given number.

        >>> AST.from_number(2)
        2
        >>> AST.from_number(3.5)
        7/2
        >>> AST.from_number(Fraction(1, 3))
        1/3
        >>> AST.from_number(mpq(1, 4))
        1/4
        >>> AST.from_number(2 + 3j)
        2 + 3 * I
        >>> AST.from_number("x")
        Traceback (most recent call last):
          ...
        ValueError: expected one of int, float, Fraction, mpq, complex; x is <class 'str'>
        """
        if isinstance(value, _RATIONAL_NUMBER_TYPES):
            return Rat(value)
        elif isinstance(value, complex):
            return AST.from_real_imag(mpq(value.real), mpq(value.imag))
        else:
            number_types = ', '.join(c.__name__ for c in _NUMBER_TYPES)
            raise ValueError(f'expected one of {number_types}; {value} is {type(value)}')

    def is_constant(self) -> bool:
        """Return :obj:`True` if this AST node is constant.

        >>> x = Var('x')
        >>> (x + 2).is_constant()
        False
        >>> (2 * I).is_constant()
        True
        """
        try:
            self.eval()
            return True
        except ValueError:
            return False
        
    def is_variable(self) -> bool:
        """Return :obj:`True` if this AST node is a variable.

        >>> x = Var('x')
        >>> (x + 2).is_variable()
        False
        >>> x.is_variable()
        True
        >>> I.is_variable()
        False
        """
        return isinstance(self, Var)

    def is_zero(self) -> bool:
        """Return :obj:`True` if this AST node is the rational number zero.

        >>> x = Var('x')
        >>> (x + 2).is_zero()
        False
        >>> Rat(0).is_zero()
        True
        """
        return isinstance(self, Rat) and self.value == mpq(0)

    def normalize(self) -> AST:
        """Return a unique normal form of this AST node
        """
        return self.accept(ComplexNormalizer())
    
    def _repr_latex_(self) -> str:
        """LaTeX representation for Jupyter notebooks.
        """
        result = f'$\\displaystyle {self.as_latex()}$'
        if len(result) > 5000:
            raise ValueError('Latex output too long')
        return result 

    def sort_key(self) -> SortKey[Self]:
        """A sort key suitable for comparing AST nodes.
        """
        return SortKey(self)

    def subs(self, sigma: Mapping[Var, Number | AST]) -> AST:
        """Formal substitution of variables in this AST node according to the 
        given mapping. The mapping can map variables to either numbers or AST 
        nodes.
        """
        return self.accept(VariableSubstitutor(sigma))


class Rat(AST):
    """A rational number, represented as a mpq. Implements the abstract class 
    :class:`.AST`.
    """
    
    value: mpq

    @property
    def args(self) -> tuple[mpq]:
        """The arguments of this AST node, which is just the rational number 
        itself.
        """
        return (self.value,)

    def __init__(self, value: RationalNumber) -> None:
        """Initialize this rational number with the given value. The
        value must be non-negative, otherwise this AST node is represented
        as a negation of a rational number with a non-negative value.
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
            return Neg(Rat(-value))
        else:
            return super().__new__(cls)
        

    def accept(self, visitor: ASTVisitor[α]) -> α:
        """Implements the abstract method :meth:`.AST.accept`."""
        return visitor.visit_rat(self)


class _I(AST):
    """The imaginary unit. This is a singleton class, and the only instance 
    is `I`. Implements the abstract class :class:`.AST`.
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

    def accept(self, visitor: ASTVisitor[α]) -> α:
        """Implements the abstract method :meth:`.AST.accept`.
        """
        return visitor.visit_i(self)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    
I: Final = _I()
"""The singleton instance of the imaginary unit.
"""


class Var(AST):
    """A variable, represented by a string name. Implements the
    abstract class :class:`.AST`.
    """

    name: str
    """The name of this variable.
    """
    
    @property
    def args(self) -> tuple[str]:
        """The arguments of this variable, which are just the name of the
        variable itself.
        """
        return (self.name,)

    def __init__(self, name: str) -> None:
        """Initialize this variable with the given name.
        """
        self.name = name

    def accept(self, visitor: ASTVisitor[α]) -> α:
        """Implements the abstract method :meth:`.AST.accept`.
        """
        return visitor.visit_var(self)

    
class MonoidalOperation(AST):
    """A monoidal operation, which is an associative operation with an
    identity element. This is a base class for addition and
    multiplication. Implements parts the abstract class :class:`.AST`.
    """

    _args: tuple[AST, ...]
    """The arguments of this AST node.
    """

    identity: ClassVar[AST]
    """The identity element of this operation. This should be
    overridden by subclasses.
    """

    @property
    def args(self) -> tuple[AST, ...]:
        """The arguments of this AST node.
        """
        return self._args

    def __init__(self, *args: AST) -> None:
        """Initialize this monoidal operation with the given arguments.
        If any of the arguments is itself a monoidal operation of the same 
        type, then the arguments are flattened into this operation.
        """
        args_flat = []
        for arg in args:
            if isinstance(arg, self.__class__):
                args_flat.extend(list(arg.args))
            else:
                args_flat.append(arg)
        self._args = tuple(args_flat)

    def __new__(cls, *args: AST):
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

    identity: ClassVar[Rat] = Rat(0)
    """The identity element of addition, which is the rational number 0.
    """

    def accept(self, visitor: ASTVisitor[α]) -> α:
        """Implements the abstract method :meth:`.AST.accept`.
        """
        return visitor.visit_add(self)


class Mul(MonoidalOperation):
    """Multiplication. Implements the abstract class
    :class:`.MonoidalOperation`.
    """

    identity: ClassVar[Rat] = Rat(1)
    """The identity element of multiplication, which is the rational number 1.
    """

    def accept(self, visitor: ASTVisitor[α]) -> α:
        """Implements the abstract method :meth:`.AST.accept`.
        """
        return visitor.visit_mul(self)


class Pow(AST):
    """Exponentiation. Implements the abstract class :class:`.AST`.
    """
    
    base: AST
    """The base of this power node.
    """
    
    exponent: int
    """The exponent of this power node. Must be a non-negative integer.
    """

    @property
    def args(self) -> tuple[AST, int]:
        """The arguments of this node, which is the base and the exponent.
        """
        return (self.base, self.exponent)
    
    def __init__(self, base: AST, exponent: int) -> None:
        """Initialize this power node with the given base and exponent.
        The exponent must be a non-negative integer.
        """
        if not isinstance(exponent, int) or exponent < 0:
            raise TypeError('Exponent must be a non-negative integer')
        self.base = base
        self.exponent = exponent

    def accept(self, visitor: ASTVisitor[α]) -> α:
        """Implements the abstract method :meth:`.AST.accept`.
        """
        return visitor.visit_pow(self)

        
class UnaryOperation(AST):
    """A unary operation, which is an operation with one argument.
    This is a base class for negation, conjugation, real part, and
    imaginary part. Implements parts the abstract class :class:`.AST`.
    """

    arg: AST
    """The single argument of this unary operation."""

    @property
    def args(self) -> tuple[AST]:
        """The arguments of this AST node, which is just the single
        argument of this unary operation.
        """
        return (self.arg,)
    
    def __init__(self, arg: Number | AST) -> None:
        """Initialize this unary operation with the given argument,
        which can be either an AST node or a number.
        If it is a number, it is converted to a AST node using
        :meth:`.AST.from_number`.
        """
        if isinstance(arg, AST):
            self.arg = arg
        else:
            self.arg = AST.from_number(arg)


class Neg(UnaryOperation):
    """Negation. Implements the abstract class :class:`.UnaryOperation`.
    """

    def accept(self, visitor: ASTVisitor[α]) -> α:
        """Implements the abstract method :meth:`.AST.accept`.
        """
        return visitor.visit_neg(self)
    

class Conj(UnaryOperation):
    """Complex conjugation. Implements the abstract class
    :class:`.UnaryOperation`.
    """

    def accept(self, visitor: ASTVisitor[α]) -> α:
        """Implements the abstract method :meth:`.AST.accept`.
        """
        return visitor.visit_conj(self)


class Re(UnaryOperation):
    """Real part. Implements the abstract class :class:`.UnaryOperation`.
    """

    def accept(self, visitor: ASTVisitor[α]) -> α:
        """Implements the abstract method :meth:`.AST.accept`.
        """
        return visitor.visit_re(self)   


class Im(UnaryOperation):
    """Imaginary part. Implements the abstract class :class:`.UnaryOperation`.
    """

    def accept(self, visitor: ASTVisitor[α]) -> α:
        """Implements the abstract method :meth:`.AST.accept`.
        """
        return visitor.visit_im(self)


class ASTVisitor(Generic[α]):
    """Visitor for AST nodes. This is used to implement various operations
    on AST nodes, such as normalization, evaluation, etc.
    """

    @abstractmethod
    def visit_rat(self, num: Rat) -> α:
        """Visit a rational number.
        """
        ...

    @abstractmethod
    def visit_i(self, i: _I) -> α:
        """Visit the imaginary unit.
        """
        ...

    @abstractmethod
    def visit_var(self, var: Var) -> α:
        """Visit a variable.
        """
        ...

    @abstractmethod
    def visit_add(self, add: Add) -> α:
        """Visit an addition node.
        """
        ...

    @abstractmethod
    def visit_mul(self, mul: Mul) -> α:
        """Visit a multiplication node.
        """
        ...

    @abstractmethod
    def visit_pow(self, pow: Pow) -> α:
        """Visit a power node.
        """
        ...

    @abstractmethod
    def visit_neg(self, neg: Neg) -> α:
        """Visit a negation node.
        """
        ...

    @abstractmethod
    def visit_conj(self, conj: Conj) -> α:
        """Visit a conjugation node.
        """
        ...

    @abstractmethod
    def visit_re(self, re: Re) -> α:
        """Visit a real part node.
        """
        ...

    @abstractmethod
    def visit_im(self, im: Im) -> α:
        """Visit an imaginary part node.
        """
        ...


class IdentityASTVisitor(ASTVisitor[AST]):
    """Visitor that returns the same AST node, but with all children
    visited. Useful as a base class for other visitors.
    """
    
    def visit_rat(self, num: Rat) -> AST:
        """Return the same rational number. Implements the abstract
        method :meth:`.ASTVisitor.visit_rat`.
        """
        return num
    
    def visit_i(self, i: _I) -> AST:
        """Return the imaginary unit. Implements the abstract method
        :meth:`.ASTVisitor.visit_i`.
        """
        return i
    
    def visit_var(self, var: Var) -> AST:
        """Return the same variable. Implements the abstract method
        :meth:`.ASTVisitor.visit_var`.
        """
        return var
    
    def visit_add(self, add: Add) -> AST:
        """Return the same addition node, but with all arguments
        visited. Implements the abstract method :meth:`.ASTVisitor.visit_add`.
        """
        return Add(*[arg.accept(self) for arg in add.args])
    
    def visit_mul(self, mul: Mul) -> AST:
        """Return the same multiplication node, but with all arguments
        visited. Implements the abstract method :meth:`.ASTVisitor.visit_mul`.
        """
        return Mul(*[arg.accept(self) for arg in mul.args])

    def visit_pow(self, pow: Pow) -> AST:
        """Return the same power node, but with the base visited.
        Implements the abstract method :meth:`.ASTVisitor.visit_pow`.
        """
        return Pow(pow.base.accept(self), pow.exponent)

    def visit_neg(self, neg: Neg) -> AST:
        """Return the same negation node, but with the argument
        visited. Implements the abstract method :meth:`.ASTVisitor.visit_neg`.
        """
        return Neg(neg.arg.accept(self))

    def visit_conj(self, conj: Conj) -> AST:
        """Return the same conjugation node, but with the argument
        visited. Implements the abstract method :meth:`.ASTVisitor.visit_conj`.
        """
        return Conj(conj.arg.accept(self))

    def visit_re(self, re: Re) -> AST:
        """Return the same real part node, but with the argument
        visited. Implements the abstract method :meth:`.ASTVisitor.visit_re`.
        """
        return Re(re.arg.accept(self))

    def visit_im(self, im: Im) -> AST:
        """Return the same imaginary part node, but with the argument
        visited. Implements the abstract method :meth:`.ASTVisitor.visit_im`.
        """
        return Im(im.arg.accept(self))
    

class VariableSubstitutor(IdentityASTVisitor):
    """Visitor that substitutes variables according to a given mapping. See 
    also :meth:`.AST.subs`.
    """
    
    mapping: Mapping[Var, Number | AST]

    def __init__(self, mapping: Mapping[Var, Number | AST]) -> None:
        """Initialize the substitutor with the given mapping containing either 
        AST nodes or numbers.
        """
        self.mapping = mapping
    
    def visit_variable(self, var: Var) -> AST:
        """Return the substituted AST node for the given variable, or the
        variable itself if not found in the mapping.
        """
        value = self.mapping.get(var, var)
        if isinstance(value, AST):
            return value
        else:
            return AST.from_number(value)
    

from logic1.theories.Complex.format import LatexFormatter, ReprFormatter, StrFormatter
from logic1.theories.Complex.normalize import ComplexNormalizer, ConstantEvaluator