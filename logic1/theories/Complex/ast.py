"""Abstract syntax trees for complex expressions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import total_ordering
from typing import Callable, ClassVar, Final, Generic, Never, Optional, Self

from gmpy2 import mpq

from logic1.theories.Complex.types import α, η, _RATIONAL_NUMBER_TYPES, Number, _NUMBER_TYPES, RationalNumber


class AST(ABC):
    """Abstract base class for all AST nodes that implements basic functionality.
    AST nodes can be constructed using the constructors of the subclasses or
    using arithmetic operators.

    >>> ast = Re(Add(Var('z'), Mul(Rat(2), _I())))
    >>> ast
    Re(Add(Var('z'), Mul(Rat(mpq(2,1)), _I())))
    >>> print(ast)
    Re(z + 2 * i)

    >>> z = Var('z')
    >>> ast = (Im(z) - I)**2
    >>> ast
    Pow(Add(Im(Var('z')), Neg(_I())), 2)
    >>> print(ast)
    (Im(z) - i)^2

    .. seealso::
        :class:`.Rat`, :class:`._I`, :class:`.Var`, :class:`.Add`, :class:`.Mul`,
        :class:`.Neg`, :class:`.Pow`, :class:`.Re`, :class:`.Im`, :class:`.Conj`
    """

    @property
    def op(self) -> type[Self]:
        """The operation of this AST node, which is just the class of this node.

        >>> z = Var('z')
        >>> z.op
        <class 'logic1.theories.Complex.ast.Var'>
        """
        return type(self)

    @property
    def args(self) -> tuple[object, ...]:
        """The arguments of this AST node. This should be overridden by
        subclasses to return the appropriate arguments so that
        :code:`self == self.op(*self.args)`.

        >>> z = Var('z')
        >>> z.args
        ('z',)
        >>> z.op(*z.args)
        Var('z')
        """
        return ()

    def __add__(self, other: Number | AST) -> Add:
        """Construct an addition node from this AST node and another AST node
        or number.

        >>> z = Var('z')
        >>> z + 2
        Add(Var('z'), Rat(mpq(2,1)))
        """
        if isinstance(other, AST):
            return Add(self, other)
        return self + AST.from_number(other)

    def __eq__(self, other: object) -> bool:
        """Comparison of two AST nodes based on :class:`.SortKey`.
        """
        if isinstance(other, AST):
            return self.sort_key() == other.sort_key()
        return False

    def __ge__(self, other: AST) -> bool:
        """Comparison of two AST nodes based on :class:`.SortKey`.
        """
        return self.sort_key() >= other.sort_key()

    def __gt__(self, other: AST) -> bool:
        """Comparison of two AST nodes based on :class:`.SortKey`.
        """
        return self.sort_key() > other.sort_key()

    def __hash__(self) -> int:
        """Return the hash value of this AST node.
        """
        return hash((tuple(str(cls) for cls in self.op.mro()), self.args))

    @abstractmethod
    def __init__(self, *args: object) -> None:
        """This abstract base class is not supposed to have instances itself.
        """
        ...

    def __invert__(self) -> Conj:
        """Construct a conjugation node from this AST node.

        >>> z = Var('z')
        >>> ~z
        Conj(Var('z'))
        """
        return Conj(self)

    def __le__(self, other: AST) -> bool:
        """Comparison of two AST nodes based on :class:`.SortKey`.
        """
        return self.sort_key() <= other.sort_key()

    def __lt__(self, other: AST) -> bool:
        """Comparison of two AST nodes based on :class:`.SortKey`.
        """
        return self.sort_key() < other.sort_key()

    def __mul__(self, other: Number | AST) -> Mul:
        """Construct a multiplication node from this AST node and another
        AST node or number.

        >>> z = Var('z')
        >>> z * 2
        Mul(Var('z'), Rat(mpq(2,1)))
        """
        if isinstance(other, AST):
            return Mul(self, other)
        return self * AST.from_number(other)

    def __ne__(self, other: object) -> bool:
        """Comparison of two AST nodes based on :class:`.SortKey`.
        """
        if isinstance(other, AST):
            return self.sort_key() != other.sort_key()
        return True

    def __neg__(self) -> Neg:
        """Construct a negation node from this AST node.

        >>> z = Var('z')
        >>> -z
        Neg(Var('z'))
        """
        return Neg(self)

    def __pow__(self, other: int) -> Pow:
        """Construct a power node from this AST node and an exponent.

        >>> z = Var('z')
        >>> z ** 2
        Pow(Var('z'), 2)
        """
        return Pow(self, other)

    def __radd__(self, other: Number | AST) -> Add:
        """Construct an addition node from a number as left summand and this AST
        node as right summand. All other cases are handled by :meth:`__add__`.

        >>> z = Var('z')
        >>> 2 + z
        Add(Rat(mpq(2,1)), Var('z'))
        """
        assert not isinstance(other, AST)
        return AST.from_number(other) + self

    def __repr__(self) -> str:
        """Return a string representation of this AST node that can be evaluated
        to reconstruct the node. For a more human-readable string
        representation, use :meth:`__str__` or :meth:`as_latex`.

        >>> z = Var('z')
        >>> repr(z + 2)
        "Add(Var('z'), Rat(mpq(2,1)))"
        """
        return f"{self.__class__.__name__}({', '.join(repr(arg) for arg in self.args)})"

    def __rmul__(self, other: Number | AST) -> Mul:
        """Construct a multiplication node from a number as left factor and
        this AST node as right factor. All other cases are handled by
        :meth:`__mul__`.

        >>> z = Var('z')
        >>> 2 * z
        Mul(Rat(mpq(2,1)), Var('z'))
        """
        assert not isinstance(other, AST)
        return AST.from_number(other) * self

    def __rsub__(self, other: Number | AST) -> Add:
        """Construct an addition node from a number as left summand and the
        negation of this AST node as right summand. All other cases are handled
        by :meth:`__sub__`.

        >>> z = Var('z')
        >>> 2 - z
        Add(Rat(mpq(2,1)), Neg(Var('z')))
        """
        assert not isinstance(other, AST)
        return AST.from_number(other) - self

    def __rtruediv__(self, other: Number | AST) -> AST:
        """Construct an AST node representing the division of a number by this
        AST node. Division is defined as multiplication by the inverse.
        Raise a :class:`ValueError` if this AST node is not constant.
        All other cases are handled by :meth:`__truediv__`.

        >>> 1 / I
        Mul(Rat(mpq(1,1)), Neg(_I()))
        >>> z = Var('z')
        >>> 1 / z
        Traceback (most recent call last):
          ...
        ValueError: Cannot divide by a non-constant AST node
        """
        assert not isinstance(other, AST)
        return AST.from_number(other) / self

    def __str__(self) -> str:
        """Return a human-readable string representation of this AST node.

        >>> z = Var('z')
        >>> str(Add(z, Rat(mpq(2))))
        'z + 2'
        """
        return self.accept(StrFormatter())

    def __sub__(self, other: Number | AST) -> Add:
        """Construct an addition node from this AST node and the negation of
        another AST node or number.

        >>> z = Var('z')
        >>> z - 2
        Add(Var('z'), Neg(Rat(mpq(2,1))))
        """
        if isinstance(other, AST):
            return self + (-other)
        return self - AST.from_number(other)

    def __truediv__(self, other: Number | AST) -> AST:
        """Construct an AST node representing the division of this AST node by
        another AST node or number. Division is defined as multiplication by
        the inverse. Raise a :class:`ValueError` if the other AST node is not
        constant.

        >>> z = Var('z')
        >>> z / 2
        Mul(Var('z'), Rat(mpq(1,2)))
        >>> print(z / (1 + I))
        z * (1/2 + -1/2 * i)
        >>> I / z
        Traceback (most recent call last):
          ...
        ValueError: Cannot divide by a non-constant AST node
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
        """Raise a :class:`NotImplementedError` because the :code:`**`
        operator should be used for constructing power nodes instead.
        See :meth:`__pow__`.
        """
        raise NotImplementedError(
            "Use ** for exponentiation, not '^', which means xor "
            "in Python, and has the wrong precedence")

    @abstractmethod
    def accept(self, visitor: ASTVisitor[α]) -> α:
        """Accept an AST visitor."""
        ...

    def as_latex(self) -> str:
        """Return a LaTeX representation of this AST node.

        >>> z = Var('z')
        >>> (z + 2 * I).as_latex()
        'z + 2 i'
        """
        return self.accept(LatexFormatter())

    def eval(self) -> tuple[mpq, mpq]:
        """Evaluate this AST node as a complex number and return the real and
        imaginary part. Raise a :class:`ValueError` if the AST node is not
        constant.

        >>> (2 * I).eval()
        (mpq(0,1), mpq(2,1))
        >>> z = Var('z')
        >>> (z + 1).eval()
        Traceback (most recent call last):
          ...
        ValueError: Cannot evaluate variable z
        """
        return self.accept(ConstantEvaluator())

    def factors(self) -> list[AST]:
        """Return a list of factors of this AST node, where each factor is a
        AST node that is not a multiplication.

        >>> z = Var('z')
        >>> (2 * z * I).factors()
        [Rat(mpq(2,1)), Var('z'), _I()]
        >>> (z + 1).factors()
        [Add(Var('z'), Rat(mpq(1,1)))]
        """
        if isinstance(self, Mul):
            return list(self.args)
        else:
            return [self]

    @staticmethod
    def from_real_imag(real: mpq, imag: mpq) -> AST:
        """Construct an AST node from given real and imaginary parts.

        >>> AST.from_real_imag(mpq(2), mpq(-1))
        Add(Rat(mpq(2,1)), Neg(_I()))
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
    def from_number(value: Number) -> AST:  # TODO: how to handle wrong types in general?
        """Construct an AST node from a given :data:`Number <.types.Number>`.
        Raise a :class:`ValueError` if the given value is not a number.

        >>> AST.from_number(3.5)
        Rat(mpq(7,2))
        >>> AST.from_number(2 + 3j)
        Add(Rat(mpq(2,1)), Mul(Rat(mpq(3,1)), _I()))
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

        >>> z = Var('z')
        >>> (z + 2).is_constant()
        False
        >>> (2 * I).is_constant()
        True
        """
        try:
            self.eval()
            return True
        except ValueError:
            return False

    def is_rational(self) -> bool:
        """Return :obj:`True` if this AST node is build from rational numbers.

        >>> Rat(1).is_rational()
        True
        >>> Neg(Rat(1)).is_rational()
        True
        >>> (1 + 2 * I).is_rational()
        False
        """
        def func(ast, result):
            if isinstance(ast, Rat):
                return True
            if isinstance(ast, (_I, Var)):
                return False
            if isinstance(ast, (Add, Mul)):
                return all(result)
            if isinstance(ast, (Pow, Neg, Conj, Re, Im)):
                return result[0]
            assert False, f"Unexpected AST node type: {type(ast)}"
        return self.accept(TraversalASTVisitor(func))

    def is_variable(self) -> bool:
        """Return :obj:`True` if this AST node is a variable.

        >>> z = Var('z')
        >>> (z + 2).is_variable()
        False
        >>> z.is_variable()
        True
        >>> I.is_variable()
        False
        """
        return isinstance(self, Var)

    def is_zero(self) -> bool:
        """Return :obj:`True` if this AST node is the rational number zero.

        >>> z = Var('z')
        >>> (z + 2).is_zero()
        False
        >>> Rat(0).is_zero()
        True
        """
        return isinstance(self, Rat) and self.value == mpq(0)

    def _lc(self) -> AST:
        """Return the left-most constant coefficient of this AST node.

        >>> z = Var('z')
        >>> (2 * z + 3)._lc()
        Mul(Rat(mpq(2,1)), Rat(mpq(1,1)))
        >>> (z + 2 * I)._lc()
        Rat(mpq(1,1))
        """
        if self.is_constant():
            return self
        elif isinstance(self, Add):
            return self.args[0]._lc()
        elif isinstance(self, Neg):
            return -self.arg._lc()
        elif isinstance(self, Mul) and self.args[0].is_constant():
            return self.args[0] * Mul(*self.args[1:])._lc()
        elif isinstance(self, Mul) and isinstance(self.args[0], Neg):
            return -Mul(*self.args[1:])._lc()
        else:
            return Rat(1)

    def _repr_latex_(self) -> str:
        """LaTeX representation for Jupyter notebooks.

        >>> z = Var('z')
        >>> (z + 2 * I)._repr_latex_()
        '$\\\\displaystyle z + 2 i$'
        """
        result = f'$\\displaystyle {self.as_latex()}$'
        if len(result) > 5000:
            raise ValueError('Latex output too long')
        return result

    def sort_key(self) -> SortKey[Self]:
        """A sort key suitable for comparing AST nodes.

        >>> z = Var('z')
        >>> z.sort_key() < (z + 1).sort_key()
        True
        """
        return SortKey(self)

    def subs(self, sigma: Mapping[Var, Number | AST]) -> AST:
        """Formal substitution of variables in this AST node according to the
        given mapping.

        >>> a, b = Var('a'), Var('b')
        >>> (a + 2).subs({a: I})
        Add(_I(), Rat(mpq(2,1)))
        >>> (a + b).subs({a: 3, b: a})
        Add(Rat(mpq(3,1)), Var('a'))
        """
        return self.accept(VariableSubstitutor(sigma))


class Rat(AST):
    """Non-negative rational number node. Negative rational numbers are
    automatically represented by a :class:`.Neg` node. Implements the abstract
    class :class:`.AST`.

    >>> Rat(2)
    Rat(mpq(2,1))
    >>> Rat(1.5)
    Rat(mpq(3,2))
    >>> Rat(-1)
    Neg(Rat(mpq(1,1)))
    """

    value: mpq
    """The value of this node.
    """

    @property
    def args(self) -> tuple[mpq]:
        """Tuple containing the value of this node.

        >>> Rat(2).args
        (mpq(2,1),)
        """
        return (self.value,)

    def __init__(self, value: RationalNumber) -> None:
        """Initialize this node with the given value. The
        value must be non-negative, otherwise this node is represented
        by a negation via :meth:`__new__`.

        >>> Rat(2)
        Rat(mpq(2,1))
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
        """Create a new instance of :class:`Rat` from the given value.
        If the value is negative, create an instance of :class:`.Neg` instead.

        >>> Rat(2)
        Rat(mpq(2,1))
        >>> Rat(-2)
        Neg(Rat(mpq(2,1)))
        """
        if value < 0:
            return Neg(Rat(-value))
        else:
            return super().__new__(cls)

    def accept(self, visitor: ASTVisitor[α]) -> α:
        """Implements the abstract method :meth:`.AST.accept`.
        """
        return visitor.visit_rat(self)


class _I(AST):
    """Imaginary unit node. This is a singleton class and the only instance
    is :obj:`I`. Implements the abstract class :class:`.AST`.

    >>> I
    _I()
    """

    _instance: Optional[_I] = None
    """The singleton instance of this class.
    """

    @property
    def args(self) -> tuple[()]:
        """The imaginary unit has no arguments.

        >>> I.args
        ()
        """
        return ()

    def __init__(self) -> None:
        """This class is a singleton, so the constructor is private and should
        not be called directly.
        """
        pass

    def accept(self, visitor: ASTVisitor[α]) -> α:
        """Implements the abstract method :meth:`.AST.accept`.
        """
        return visitor.visit_i(self)

    def __new__(cls):
        """Create a new instance of :class:`_I`. This is a singleton
        class, so this method always returns the same instance.

        >>> _I()
        _I()
        >>> _I() is _I()
        True
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


I: Final[_I] = _I()
"""The singleton instance of the imaginary unit.
"""


class Var(AST):
    """Variable node. Implements the abstract class :class:`.AST`.

    >>> z = Var('z')
    >>> z
    Var('z')
    """

    name: str
    """The name of this variable.
    """

    @property
    def args(self) -> tuple[str]:
        """A tuple containing the name of this variable.

        >>> Var('z').args
        ('z',)
        """
        return (self.name,)

    def __init__(self, name: str) -> None:
        """Initialize this variable with the given name.

        >>> z = Var('z')
        >>> z
        Var('z')
        """
        self.name = name

    def accept(self, visitor: ASTVisitor[α]) -> α:
        """Implements the abstract method :meth:`.AST.accept`.
        """
        return visitor.visit_var(self)


class MonoidalOperation(AST):
    """Abstract base class for monoidal operations, i.e. associative operations
    with identity element. Implements parts of the abstract class :class:`.AST`
    for the subclasses :class:`.Add` and :class:`.Mul`.
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

        >>> Add(Rat(1), Rat(2)).args
        (Rat(mpq(1,1)), Rat(mpq(2,1)))
        """
        return self._args

    @abstractmethod
    def __init__(self, *args: Number | AST) -> None:
        """Initialize this monoidal operation with the given arguments.
        If any of the arguments is itself a monoidal operation of the same
        type, then the argument is flattened. This abstract class is not
        supposed to have instances itself.
        """
        args_flat = []
        for arg in args:
            if not isinstance(arg, AST):
                arg = AST.from_number(arg)
            if isinstance(arg, self.__class__):
                args_flat.extend(list(arg.args))
            else:
                args_flat.append(arg)
        self._args = tuple(args_flat)

    def __new__(cls, *args: Number | AST):
        """Create a new instance of this monoidal operation with the
        given arguments. If no arguments are given, return the identity
        element. If only one argument is given, return that argument.

        >>> x = Var('x')
        >>> Add(x)
        Var('x')
        >>> Add()
        Rat(mpq(0,1))
        """
        if not args:
            return cls.identity
        if len(args) == 1:
            if isinstance(args[0], AST):
                return args[0]
            else:
                return AST.from_number(args[0])
        return super().__new__(cls)


class Add(MonoidalOperation):  # TODO: overwrite args just for docstring?
    """Addition node. Implements the abstract class
    :class:`.MonoidalOperation`.

    >>> z = Var('z')
    >>> z + 1 + I
    Add(Var('z'), Rat(mpq(1,1)), _I())
    """

    identity: ClassVar[Rat] = Rat(0)
    """The identity element of addition, which is the rational number :math:`0`.
    """

    def __init__(self, *args: Number | AST) -> None:
        """Initialize this addition node with the given arguments. If any of
        the arguments is itself an addition node, then the argument is
        flattened. If zero or one argument is given, the identity element or the
        argument itself is returned by :meth:`.MonoidalOperation.__new__`.

        >>> z = Var('z')
        >>> Add(z, Rat(1), I)
        Add(Var('z'), Rat(mpq(1,1)), _I())
        >>> Add(z, Add(Rat(1), I))
        Add(Var('z'), Rat(mpq(1,1)), _I())
        >>> Add(z)
        Var('z')
        >>> Add()
        Rat(mpq(0,1))
        """
        super().__init__(*args)

    def accept(self, visitor: ASTVisitor[α]) -> α:
        """Implements the abstract method :meth:`.AST.accept`.
        """
        return visitor.visit_add(self)


class Mul(MonoidalOperation):
    """Multiplication node. Implements the abstract class
    :class:`.MonoidalOperation`.

    >>> z = Var('z')
    >>> z * I * 2
    Mul(Var('z'), _I(), Rat(mpq(2,1)))
    """

    identity: ClassVar[Rat] = Rat(1)
    """The identity element of multiplication, which is the rational number $1$.
    """

    def __init__(self, *args: Number | AST) -> None:
        """Initialize this multiplication node with the given arguments. If any
        of the arguments is itself a multiplication node, then the argument is
        flattened. If zero or one argument is given, the identity element or the
        argument itself is returned by :meth:`.MonoidalOperation.__new__`.

        >>> z = Var('z')
        >>> Mul(z, Rat(2), I)
        Mul(Var('z'), Rat(mpq(2,1)), _I())
        >>> Mul(z, Mul(Rat(2), I))
        Mul(Var('z'), Rat(mpq(2,1)), _I())
        >>> Mul(z)
        Var('z')
        >>> Mul()
        Rat(mpq(1,1))
        """
        super().__init__(*args)

    def accept(self, visitor: ASTVisitor[α]) -> α:
        """Implements the abstract method :meth:`.AST.accept`.
        """
        return visitor.visit_mul(self)


class Pow(AST):
    """Power node. Implements the abstract class :class:`.AST`.

    >>> z = Var('z')
    >>> z ** 2
    Pow(Var('z'), 2)
    """

    base: AST
    """The base of this power node.
    """

    exponent: int
    """The exponent of this power node. Must be a non-negative integer.
    """

    @property
    def args(self) -> tuple[AST, int]:
        """Tuple containing the base and the exponent of this power node.

        >>> (I ** 2).args
        (_I(), 2)
        """
        return (self.base, self.exponent)

    def __init__(self, base: Number | AST, exponent: int) -> None:
        """Initialize this power node with the given base and exponent.
        Raise a :class:`TypeError` if the exponent is negative.

        >>> z = Var('z')
        >>> Pow(z, 2)
        Pow(Var('z'), 2)
        >>> Pow(z, -1)
        Traceback (most recent call last):
          ...
        TypeError: Exponent must be a non-negative integer
        """
        if not isinstance(exponent, int) or exponent < 0:
            raise TypeError('Exponent must be a non-negative integer')
        if not isinstance(base, AST):
            base = AST.from_number(base)
        self.base = base
        self.exponent = exponent

    def accept(self, visitor: ASTVisitor[α]) -> α:
        """Implements the abstract method :meth:`.AST.accept`.
        """
        return visitor.visit_pow(self)


class UnaryOperation(AST):
    """Abstract base class for unary operations, i.e. operations with only one
    argument. Implements parts of the abstract class :class:`.AST` for the
    subclasses :class:`.Neg`, :class:`.Conj`, :class:`.Re` and :class:`.Im`.
    """

    arg: AST
    """The single argument of this AST node.
    """

    @property
    def args(self) -> tuple[AST]:
        """Tuple containing the single argument of this AST node.
        """
        return (self.arg,)

    @abstractmethod
    def __init__(self, arg: Number | AST) -> None:
        """Initialize this unary operation with the given argument. This
        abstract class is not supposed to have instances itself.
        """
        if not isinstance(arg, AST):
            arg = AST.from_number(arg)
        self.arg = arg


class Neg(UnaryOperation):
    """Negation node. Implements the abstract class :class:`.UnaryOperation`.

    >>> z = Var('z')
    >>> -z
    Neg(Var('z'))
    """

    def __init__(self, arg: Number | AST) -> None:
        """Initialize this negation node with the given argument.

        >>> z = Var('z')
        >>> Neg(z)
        Neg(Var('z'))
        """
        super().__init__(arg)

    def accept(self, visitor: ASTVisitor[α]) -> α:
        """Implements the abstract method :meth:`.AST.accept`.
        """
        return visitor.visit_neg(self)


class Conj(UnaryOperation):
    """Complex conjugation node. Implements the abstract class
    :class:`.UnaryOperation`.

    >>> z = Var('z')
    >>> ~z
    Conj(Var('z'))
    """

    def __init__(self, arg: Number | AST) -> None:
        """Initialize this conjugation node with the given argument.

        >>> z = Var('z')
        >>> Conj(z)
        Conj(Var('z'))
        """
        super().__init__(arg)

    def accept(self, visitor: ASTVisitor[α]) -> α:
        """Implements the abstract method :meth:`.AST.accept`.
        """
        return visitor.visit_conj(self)


class Re(UnaryOperation):
    """Real part node. Implements the abstract class :class:`.UnaryOperation`.

    >>> z = Var('z')
    >>> Re(z)
    Re(Var('z'))
    """

    def __init__(self, arg: Number | AST) -> None:
        """Initialize this real part node with the given argument.

        >>> z = Var('z')
        >>> Re(z)
        Re(Var('z'))
        """
        super().__init__(arg)

    def accept(self, visitor: ASTVisitor[α]) -> α:
        """Implements the abstract method :meth:`.AST.accept`.
        """
        return visitor.visit_re(self)


class Im(UnaryOperation):
    """Imaginary part node. Implements the abstract
    class :class:`.UnaryOperation`.

    >>> z = Var('z')
    >>> Im(z)
    Im(Var('z'))
    """

    def __init__(self, arg: Number | AST) -> None:
        """Initialize this imaginary part node with the given argument.

        >>> z = Var('z')
        >>> Im(z)
        Im(Var('z'))
        """
        super().__init__(arg)

    def accept(self, visitor: ASTVisitor[α]) -> α:
        """Implements the abstract method :meth:`.AST.accept`.
        """
        return visitor.visit_im(self)


@dataclass
@total_ordering
class SortKey(Generic[η]):
    """Default sort key for comparing AST nodes.

    >>> z = Var('z')
    >>> SortKey(z) < SortKey(z + 1)
    True

    .. seealso:: :meth:`.AST.sort_key`
    """

    ORDER: ClassVar[tuple[type[AST], ...]] = (Rat, _I, Var, Conj, Re, Im, Pow, Neg, Mul, Add)
    """The order of AST node types for sorting.
    """

    ast: η
    """The AST node for which this is a sort key.
    """

    @property
    def op(self) -> type[AST]:
        """The operation of the underlying AST node.

        >>> z = Var('z')
        >>> SortKey(z).op
        <class 'logic1.theories.Complex.ast.Var'>
        """
        return self.ast.op

    @property
    def args(self) -> tuple[object, ...]:
        """The arguments of the underlying AST node, where each argument
        that is itself an AST node is replaced by its sort key.

        >>> z = Var('z')
        >>> SortKey(z + 1).args
        (SortKey(Var('z')), SortKey(Rat(mpq(1,1))))
        """
        return tuple(SortKey(arg) if isinstance(arg, AST) else arg for arg in self.ast.args)

    def __eq__(self, other: object) -> bool:
        """Return :obj:`True` if the underlying AST nodes are equal, i.e.
        have the same operation and the same arguments.

        >>> z = Var('z')
        >>> SortKey(z) == SortKey(z)
        True
        >>> SortKey(z) == SortKey(z + 1)
        False
        """
        if not isinstance(other, SortKey):
            return False
        if self.ast is other.ast:
            return True
        return self.op == other.op and self.args == other.args

    def __hash__(self) -> int:
        """Return the hash value of the underlying AST node.

        >>> z = Var('z')
        >>> hash(SortKey(z)) == hash(z)
        True
        """
        return hash(self.ast)

    def __le__(self, other: SortKey) -> bool:
        """Comparison of the underlying AST nodes first by their operation
        according to :attr:`ORDER`, then recursively by their arguments.
        The remaining comparison operators are derived from this using
        :func:`functools.total_ordering`.

        >>> z = Var('z')
        >>> SortKey(z) <= SortKey(z + 1)
        True
        """
        assert self.op in self.ORDER and other.op in self.ORDER
        if self.op == other.op:
            return self.args <= other.args
        else:
            return self.ORDER.index(self.op) < self.ORDER.index(other.op)

    def __repr__(self) -> str:
        """Return a string representation of this sort key that can be
        evaluated to reconstruct the sort key.

        >>> z = Var('z')
        >>> repr(SortKey(z))
        "SortKey(Var('z'))"
        """
        return f'{self.__class__.__name__}({repr(self.ast)})'


class ASTVisitor(ABC, Generic[α]):
    """Abstract visitor for AST nodes used to implement various
    operations on AST nodes.

    .. seealso::
        * :class:`.IdentityASTVisitor`,
          :class:`.VariableSubstitutor`,
        * :class:`.normalize.ArithmeticEvaluator`,
          :class:`.normalize.ConstantEvaluator`,
          :class:`.normalize.WeakNormalizer`,
          :class:`.normalize.Normalizer`,
          :class:`.normalize.ConjugateNormalizer`
        * :class:`.format.BaseReprFormatter`,
          :class:`.format.TermReprFormatter`,
          :class:`.format.StrFormatter`,
          :class:`.format.LatexFormatter`
        * :class:`.qe.RCF_Evaluator`
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

    >>> z = Var('z')
    >>> (z + 1).accept(IdentityASTVisitor())
    Add(Var('z'), Rat(mpq(1,1)))
    """

    def visit_rat(self, num: Rat) -> AST:
        """Return the same rational number. Implements the abstract
        method :meth:`.ASTVisitor.visit_rat`.

        >>> IdentityASTVisitor().visit_rat(Rat(mpq(2,1)))
        Rat(mpq(2,1))
        """
        return num

    def visit_i(self, i: _I) -> AST:
        """Return the imaginary unit. Implements the abstract method
        :meth:`.ASTVisitor.visit_i`.

        >>> IdentityASTVisitor().visit_i(I)
        _I()
        """
        return i

    def visit_var(self, var: Var) -> AST:
        """Return the same variable. Implements the abstract method
        :meth:`.ASTVisitor.visit_var`.

        >>> IdentityASTVisitor().visit_var(Var('x'))
        Var('x')
        """
        return var

    def visit_add(self, add: Add) -> AST:
        """Return the same addition node, but with all arguments
        visited. Implements the abstract method :meth:`.ASTVisitor.visit_add`.

        >>> x = Var('x')
        >>> IdentityASTVisitor().visit_add(x + 2)
        Add(Var('x'), Rat(mpq(2,1)))
        """
        return Add(*[arg.accept(self) for arg in add.args])

    def visit_mul(self, mul: Mul) -> AST:
        """Return the same multiplication node, but with all arguments
        visited. Implements the abstract method :meth:`.ASTVisitor.visit_mul`.

        >>> x = Var('x')
        >>> IdentityASTVisitor().visit_mul(x * 2)
        Mul(Var('x'), Rat(mpq(2,1)))
        """
        return Mul(*[arg.accept(self) for arg in mul.args])

    def visit_pow(self, pow: Pow) -> AST:
        """Return the same power node, but with the base visited.
        Implements the abstract method :meth:`.ASTVisitor.visit_pow`.

        >>> x = Var('x')
        >>> IdentityASTVisitor().visit_pow(x ** 2)
        Pow(Var('x'), 2)
        """
        return Pow(pow.base.accept(self), pow.exponent)

    def visit_neg(self, neg: Neg) -> AST:
        """Return the same negation node, but with the argument
        visited. Implements the abstract method :meth:`.ASTVisitor.visit_neg`.

        >>> x = Var('x')
        >>> IdentityASTVisitor().visit_neg(Neg(x))
        Neg(Var('x'))
        """
        return Neg(neg.arg.accept(self))

    def visit_conj(self, conj: Conj) -> AST:
        """Return the same conjugation node, but with the argument
        visited. Implements the abstract method :meth:`.ASTVisitor.visit_conj`.

        >>> x = Var('x')
        >>> IdentityASTVisitor().visit_conj(Conj(x))
        Conj(Var('x'))
        """
        return Conj(conj.arg.accept(self))

    def visit_re(self, re: Re) -> AST:
        """Return the same real part node, but with the argument
        visited. Implements the abstract method :meth:`.ASTVisitor.visit_re`.

        >>> x = Var('x')
        >>> IdentityASTVisitor().visit_re(Re(x))
        Re(Var('x'))
        """
        return Re(re.arg.accept(self))

    def visit_im(self, im: Im) -> AST:
        """Return the same imaginary part node, but with the argument
        visited. Implements the abstract method :meth:`.ASTVisitor.visit_im`.

        >>> x = Var('x')
        >>> IdentityASTVisitor().visit_im(Im(x))
        Im(Var('x'))
        """
        return Im(im.arg.accept(self))


class TraversalASTVisitor(ASTVisitor[α], Generic[α]):
    """Visitor that traverses the AST and calls a given function for each node.
    The function is called with the current AST node and a list of results from
    visiting the children of the node. Implements the abstract class
    :class:`.ASTVisitor`.

    >>> f = lambda _, results: 1 + sum(results)
    >>> size_visitor = TraversalASTVisitor(f)
    >>> ast = Var('x') + 2 * I
    >>> ast.accept(size_visitor)
    5
    """

    def __init__(self, func: Callable[[AST, list[α]], α]) -> None:
        """Initialize the visitor with a function that is called for each AST
        node during the traversal.
        """
        self.func = func

    def visit_rat(self, num: Rat) -> α:
        """Implements the abstract method :meth:`.ASTVisitor.visit_rat`.
        """
        return self.func(num, [])

    def visit_i(self, i: _I) -> α:
        """Implements the abstract method :meth:`.ASTVisitor.visit_i`.
        """
        return self.func(i, [])

    def visit_var(self, var: Var) -> α:
        """Implements the abstract method :meth:`.ASTVisitor.visit_var`.
        """
        return self.func(var, [])

    def visit_add(self, add: Add) -> α:
        """Implements the abstract method :meth:`.ASTVisitor.visit_add`.
        """
        return self.func(add, [arg.accept(self) for arg in add.args])

    def visit_mul(self, mul: Mul) -> α:
        """Implements the abstract method :meth:`.ASTVisitor.visit_mul`.
        """
        return self.func(mul, [arg.accept(self) for arg in mul.args])

    def visit_pow(self, pow: Pow) -> α:
        """Implements the abstract method :meth:`.ASTVisitor.visit_pow`.
        """
        return self.func(pow, [pow.base.accept(self)])

    def visit_neg(self, neg: Neg) -> α:
        """Implements the abstract method :meth:`.ASTVisitor.visit_neg`.
        """
        return self.func(neg, [neg.arg.accept(self)])

    def visit_conj(self, conj: Conj) -> α:
        """Implements the abstract method :meth:`.ASTVisitor.visit_conj`.
        """
        return self.func(conj, [conj.arg.accept(self)])

    def visit_re(self, re: Re) -> α:
        """Implements the abstract method :meth:`.ASTVisitor.visit_re`.
        """
        return self.func(re, [re.arg.accept(self)])

    def visit_im(self, im: Im) -> α:
        """Implements the abstract method :meth:`.ASTVisitor.visit_im`.
        """
        return self.func(im, [im.arg.accept(self)])


class VariableSubstitutor(IdentityASTVisitor):
    """Visitor that substitutes variables according to a given mapping. See
    also :meth:`.AST.subs`.

    >>> x = Var('x')
    >>> (x + 2).accept(VariableSubstitutor({x: I}))
    Add(_I(), Rat(mpq(2,1)))
    """

    mapping: dict[Var, Number | AST]

    def __init__(self, mapping: Mapping[Var, Number | AST]) -> None:
        """Initialize the substitutor with a given mapping.
        """
        self.mapping = dict(mapping)

    def visit_var(self, var: Var) -> AST:
        """Return the substituted AST node for the given variable, or the
        variable itself if not found in the mapping.

        >>> x = Var('x')
        >>> visitor = VariableSubstitutor({x: I})
        >>> visitor.visit_var(x)
        _I()
        >>> y = Var('y')
        >>> visitor.visit_var(y)
        Var('y')
        """
        value = self.mapping.get(var, var)
        if isinstance(value, AST):
            return value
        else:
            return AST.from_number(value)

from logic1.theories.Complex.format import LatexFormatter, StrFormatter
from logic1.theories.Complex.normalize import ConstantEvaluator
