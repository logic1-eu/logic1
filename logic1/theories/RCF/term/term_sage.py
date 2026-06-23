from __future__ import annotations

from collections.abc import Container
from dataclasses import dataclass
from enum import auto, Enum
from fractions import Fraction
from functools import lru_cache
from typing import (ClassVar, Final, Generic, Iterable, Iterator, Mapping, Optional, Self, TypeVar)

from gmpy2 import mpq
from sage.all import QQ
# Importing QQ from sage.rings.rational_fields causes problems. Notably, a
# fresh instance of RationalField is assigned to QQ in sage.all.
from sage.rings.fraction_field import FractionField  # type: ignore[import-untyped]
from sage.misc.latex import latex as sage_latex
from sage.rings.integer import Integer
from sage.rings.polynomial.multi_polynomial_libsingular import (
    MPolynomial_libsingular as MPolynomial,
    MPolynomialRing_libsingular as MPolynomialRing)
from sage.rings.polynomial.polynomial_ring_constructor import (
    PolynomialRing as sage_PolynomialRing)
from sage.rings.polynomial.polynomial_element import (
    Polynomial_generic_dense as UPolynomial)
from sage.rings.polynomial.term_order import TermOrder
from sage.rings.rational import Rational

from logic1 import firstorder
from logic1.theories.RCF.atomic import Eq, Ge, Gt, Le, Lt, Ne

from logic1.support.tracing import trace  # noqa


POLYLIB: Final = "SAGE"


τ = TypeVar('τ', bound='Term')
"""A type variable denoting a type of terms with upper bound
:class:`logic1.theories.RCF.Term`.
"""

CACHE_SIZE: Final[Optional[int]] = 2**16


def _caches():
    from logic1.theories.RCF.node.xopt import XoNode
    from logic1.theories.RCF.simplify import Simplify
    from logic1.theories.RCF.substitution import _SubstValue
    return [Term.factor, _SubstValue.as_term, Simplify._simpl_at, XoNode.subs_into_formula]

def cache_clear():
    for cache in _caches():
        cache.cache_clear()

def cache_info():
    return {cache.__wrapped__: cache.cache_info() for cache in _caches()}


def init_env(ring_vars: list[str]) -> None:
    # We pass the ring variables to the workers. The workers reconstruct the ring.
    polynomial_ring.add_vars(ring_vars)

def init_env_arg() -> list[str]:
    return [str(v) for v in polynomial_ring.get_vars()]


class _PolynomialRing:

    sage_ring: MPolynomialRing
    stack: list[MPolynomialRing]

    def __call__(self, obj):
        return self.sage_ring(obj)

    def __init__(self, term_order='deglex'):
        self.sage_ring = self.MPolynomialRing_factory('unused_', order=term_order)
        self.stack = []

    def __repr__(self):
        return str(self.sage_ring)

    def add_var(self, var: str) -> None:
        new_vars = [str(g) for g in self.sage_ring.gens()]
        assert var not in new_vars
        new_vars.append(var)
        new_vars.sort()
        self.sage_ring = self.MPolynomialRing_factory(new_vars, order=self.sage_ring.term_order())

    def add_vars(self, vars_: Iterable[str]) -> None:

        def sort_key(s: str) -> tuple[str, int]:
            base = s.rstrip('0123456789')
            index = s[len(base):]
            n = int(index) if index else -1
            return base, n

        new_vars = []
        for g in self.sage_ring.gens():
            new_vars.append(str(g))
        have_appended = False
        for v in vars_:
            if v not in new_vars:
                new_vars.append(v)
                have_appended = True
        if have_appended:
            new_vars.sort(key=sort_key)
            self.sage_ring = self.MPolynomialRing_factory(
                new_vars, order=self.sage_ring.term_order())

    def get_vars(self) -> tuple[MPolynomial[Integer], ...]:
        gens = (g for g in self.sage_ring.gens() if str(g) != 'unused_')
        return tuple(gens)

    @staticmethod
    def MPolynomialRing_factory(names: str | Iterable[str], order: TermOrder) -> MPolynomialRing:
        return sage_PolynomialRing(QQ, names, order=order, implementation='singular')

    def pop(self) -> None:
        self.sage_ring = self.stack.pop()

    def push(self) -> None:
        self.stack.append(self.sage_ring)
        self.sage_ring = self.MPolynomialRing_factory('unused_', order=self.sage_ring.term_order())


polynomial_ring = _PolynomialRing()


class VariableSet(firstorder.VariableSet['Variable']):
    """The infinite set of all variables belonging to the theory of Real Closed
    Fields. Variables are uniquely identified by their name, which is a
    :external:class:`.str`. This class is a singleton, whose single instance is
    assigned to :data:`.VV`.

    .. seealso::
        Final methods inherited from parent class:

        * :meth:`.firstorder.atomic.VariableSet.get`
            -- obtain several variables simultaneously
        * :meth:`.firstorder.atomic.VariableSet.imp`
            -- import variables into global namespace
    """

    polynomial_ring: ClassVar[_PolynomialRing] = polynomial_ring

    @property
    def stack(self) -> list[MPolynomialRing]:
        """Implements abstract property
        :attr:`.firstorder.atomic.VariableSet.stack`.
        """
        return self.polynomial_ring.stack

    def __getitem__(self, index: str) -> Variable:
        """Implements abstract method
        :meth:`.firstorder.atomic.VariableSet.__getitem__`.
        """
        match index:
            case str():
                self.polynomial_ring.add_vars((index,))
                return Variable(self.polynomial_ring(index))
            case _:
                raise ValueError(f'expecting string as index; {index} is {type(index)}')

    def __repr__(self) -> str:
        vars_ = self.polynomial_ring.get_vars()
        s = ', '.join(str(g) for g in (*vars_, '...'))
        return f'{{{s}}}'

    def fresh(self, suffix: str = '') -> Variable:
        """Return a fresh variable, by default from the sequence G0001, G0002,
        ..., G9999, G10000, ... This naming convention is inspired by Lisp's
        gensym(). If the optional argument :data:`suffix` is specified, the
        sequence G0001<suffix>, G0002<suffix>, ... is used instead.
        """
        vars_ = set(str(g) for g in self.polynomial_ring.get_vars())
        i = 1
        v = f'G{i:04d}{suffix}'
        while v in vars_:
            i += 1
            v = f'G{i:04d}{suffix}'
        self.polynomial_ring.add_var(v)
        return Variable(self.polynomial_ring(v))

    def pop(self) -> None:
        from . import cache_clear
        self.polynomial_ring.pop()
        cache_clear()

    def push(self) -> None:
        from . import cache_clear
        self.polynomial_ring.push()
        cache_clear()


VV = VariableSet()
"""
The unique instance of :class:`.VariableSet`.
"""


class DEFINITE(Enum):
    """Information whether a certain term has positive or negative definiteness
    properties; typically as a result of a heuristic test as in
    :meth:`.Term.is_definite`.
    """

    # This is an ordered Enum, the order of the following properties should not
    # be changed.
    UNKNOWN = auto()
    """It has not been derived that any the other cases holds.
    """

    ZERO = auto()
    """The polynomial is the zero polynomial.
    """

    POSITIVE = auto()
    """The polynomial positive definite, i.e., positive for all real choices of
    variables.
    """

    POSITIVE_SEMI = auto()
    """The polynomial positive semi-definite, i.e., non-negative for all real
    choices of variables.
    """

    NEGATIVE = auto()
    """The polynomial negative definite, i.e., negative for all real choices of
    variables.
    """

    NEGATIVE_SEMI = auto()
    """The polynomial negative semi-definite, i.e., non-positive for all real
    choices of variables.
    """

    # The following is an implementation of OrderedEnum as described in
    # https://docs.python.org/3/howto/enum.html#orderedenum
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    @staticmethod
    def add(x: DEFINITE, y: DEFINITE) -> DEFINITE:
        """Compute DEFINITE of a sum from DEFINITE of the summands.

        >>> l = list(DEFINITE)

        >>> for x in l:
        ...     for y in l:
        ...             print(f'{x.name} + {y.name} = {DEFINITE.add(x,y).name}')
        ...
        UNKNOWN + UNKNOWN = UNKNOWN
        UNKNOWN + ZERO = UNKNOWN
        UNKNOWN + POSITIVE = UNKNOWN
        UNKNOWN + POSITIVE_SEMI = UNKNOWN
        UNKNOWN + NEGATIVE = UNKNOWN
        UNKNOWN + NEGATIVE_SEMI = UNKNOWN
        ZERO + UNKNOWN = UNKNOWN
        ZERO + ZERO = ZERO
        ZERO + POSITIVE = POSITIVE
        ZERO + POSITIVE_SEMI = POSITIVE_SEMI
        ZERO + NEGATIVE = NEGATIVE
        ZERO + NEGATIVE_SEMI = NEGATIVE_SEMI
        POSITIVE + UNKNOWN = UNKNOWN
        POSITIVE + ZERO = POSITIVE
        POSITIVE + POSITIVE = POSITIVE
        POSITIVE + POSITIVE_SEMI = POSITIVE
        POSITIVE + NEGATIVE = UNKNOWN
        POSITIVE + NEGATIVE_SEMI = UNKNOWN
        POSITIVE_SEMI + UNKNOWN = UNKNOWN
        POSITIVE_SEMI + ZERO = POSITIVE_SEMI
        POSITIVE_SEMI + POSITIVE = POSITIVE
        POSITIVE_SEMI + POSITIVE_SEMI = POSITIVE_SEMI
        POSITIVE_SEMI + NEGATIVE = UNKNOWN
        POSITIVE_SEMI + NEGATIVE_SEMI = UNKNOWN
        NEGATIVE + UNKNOWN = UNKNOWN
        NEGATIVE + ZERO = NEGATIVE
        NEGATIVE + POSITIVE = UNKNOWN
        NEGATIVE + POSITIVE_SEMI = UNKNOWN
        NEGATIVE + NEGATIVE = NEGATIVE
        NEGATIVE + NEGATIVE_SEMI = NEGATIVE
        NEGATIVE_SEMI + UNKNOWN = UNKNOWN
        NEGATIVE_SEMI + ZERO = NEGATIVE_SEMI
        NEGATIVE_SEMI + POSITIVE = UNKNOWN
        NEGATIVE_SEMI + POSITIVE_SEMI = UNKNOWN
        NEGATIVE_SEMI + NEGATIVE = NEGATIVE
        NEGATIVE_SEMI + NEGATIVE_SEMI = NEGATIVE_SEMI

        This addition is commutative:
        >>> all(DEFINITE.add(x, y) is DEFINITE.add(y, x) for x in l for y in l)
        True

        DEFINITE.zero is a (unique) neutral element:
        >>> all(DEFINITE.add(x, DEFINITE.ZERO) is x for x in l)
        True
        """
        x, y = sorted([x, y])
        if x is DEFINITE.UNKNOWN:
            return DEFINITE.UNKNOWN
        if x is DEFINITE.ZERO:
            return y
        if x is DEFINITE.POSITIVE:
            if y is DEFINITE.POSITIVE or y is DEFINITE.POSITIVE_SEMI:
                return DEFINITE.POSITIVE
            assert y is DEFINITE.NEGATIVE or y is DEFINITE.NEGATIVE_SEMI, (x, y)
            return DEFINITE.UNKNOWN
        if x is DEFINITE.POSITIVE_SEMI:
            if y is DEFINITE.POSITIVE_SEMI:
                return DEFINITE.POSITIVE_SEMI
            assert y is DEFINITE.NEGATIVE or y is DEFINITE.NEGATIVE_SEMI, (x, y)
            return DEFINITE.UNKNOWN
        if x is DEFINITE.NEGATIVE:
            assert y is DEFINITE.NEGATIVE or y is DEFINITE.NEGATIVE_SEMI, (x, y)
            return DEFINITE.NEGATIVE
        assert x is DEFINITE.NEGATIVE_SEMI, (x, y)
        assert y is DEFINITE.NEGATIVE_SEMI, (x, y)
        return DEFINITE.NEGATIVE_SEMI

    @staticmethod
    def from_constant(q: int | mpq | Rational) -> DEFINITE:
        """Compute DEFINITE of a number.

        >>> print(DEFINITE.from_constant(mpq(42)))
        DEFINITE.POSITIVE

        >>> print(DEFINITE.from_constant(mpq(-4711)))
        DEFINITE.NEGATIVE

        >>> print(DEFINITE.from_constant(mpq(0)))
        DEFINITE.ZERO
        """
        assert isinstance(q, (int, mpq, Rational)), q
        if q > 0:
            return DEFINITE.POSITIVE
        if q < 0:
            return DEFINITE.NEGATIVE
        assert q == 0, q
        return DEFINITE.ZERO

    @staticmethod
    def mul(x: DEFINITE, y: DEFINITE) -> DEFINITE:
        """Compute DEFINITE of a product from DEFINITE of the factors.

        >>> l = list(DEFINITE)

        The multiplication table:
        >>> for x in l:
        ...     for y in l:
        ...             print(f'{x.name} * {y.name} = {DEFINITE.mul(x,y).name}')
        ...
        UNKNOWN * UNKNOWN = UNKNOWN
        UNKNOWN * ZERO = ZERO
        UNKNOWN * POSITIVE = UNKNOWN
        UNKNOWN * POSITIVE_SEMI = UNKNOWN
        UNKNOWN * NEGATIVE = UNKNOWN
        UNKNOWN * NEGATIVE_SEMI = UNKNOWN
        ZERO * UNKNOWN = ZERO
        ZERO * ZERO = ZERO
        ZERO * POSITIVE = ZERO
        ZERO * POSITIVE_SEMI = ZERO
        ZERO * NEGATIVE = ZERO
        ZERO * NEGATIVE_SEMI = ZERO
        POSITIVE * UNKNOWN = UNKNOWN
        POSITIVE * ZERO = ZERO
        POSITIVE * POSITIVE = POSITIVE
        POSITIVE * POSITIVE_SEMI = POSITIVE_SEMI
        POSITIVE * NEGATIVE = NEGATIVE
        POSITIVE * NEGATIVE_SEMI = NEGATIVE_SEMI
        POSITIVE_SEMI * UNKNOWN = UNKNOWN
        POSITIVE_SEMI * ZERO = ZERO
        POSITIVE_SEMI * POSITIVE = POSITIVE_SEMI
        POSITIVE_SEMI * POSITIVE_SEMI = POSITIVE_SEMI
        POSITIVE_SEMI * NEGATIVE = NEGATIVE_SEMI
        POSITIVE_SEMI * NEGATIVE_SEMI = NEGATIVE_SEMI
        NEGATIVE * UNKNOWN = UNKNOWN
        NEGATIVE * ZERO = ZERO
        NEGATIVE * POSITIVE = NEGATIVE
        NEGATIVE * POSITIVE_SEMI = NEGATIVE_SEMI
        NEGATIVE * NEGATIVE = POSITIVE
        NEGATIVE * NEGATIVE_SEMI = POSITIVE_SEMI
        NEGATIVE_SEMI * UNKNOWN = UNKNOWN
        NEGATIVE_SEMI * ZERO = ZERO
        NEGATIVE_SEMI * POSITIVE = NEGATIVE_SEMI
        NEGATIVE_SEMI * POSITIVE_SEMI = NEGATIVE_SEMI
        NEGATIVE_SEMI * NEGATIVE = POSITIVE_SEMI
        NEGATIVE_SEMI * NEGATIVE_SEMI = POSITIVE_SEMI

        This multiplication is commutative:
        >>> all(DEFINITE.mul(x, y) is DEFINITE.mul(y, x) for x in l for y in l)
        True

        DEFINITE.POSITIVE is a (unique) neutral element:
        >>> all(DEFINITE.mul(x, DEFINITE.POSITIVE) is x for x in l)
        True
        """
        x, y = sorted([x, y])
        if x is DEFINITE.UNKNOWN:
            if y is DEFINITE.ZERO:
                return DEFINITE.ZERO
            return DEFINITE.UNKNOWN
        if x is DEFINITE.ZERO:
            return DEFINITE.ZERO
        if x is DEFINITE.POSITIVE:
            return y
        if x is DEFINITE.POSITIVE_SEMI:
            if y is DEFINITE.POSITIVE_SEMI:
                return DEFINITE.POSITIVE_SEMI
            assert y is DEFINITE.NEGATIVE or y is DEFINITE.NEGATIVE_SEMI, (x, y)
            return DEFINITE.NEGATIVE_SEMI
        if x is DEFINITE.NEGATIVE:
            if y is DEFINITE.NEGATIVE:
                return DEFINITE.POSITIVE
            assert y is DEFINITE.NEGATIVE_SEMI, (x, y)
            return DEFINITE.POSITIVE_SEMI
        assert x is DEFINITE.NEGATIVE_SEMI, (x, y)
        assert y is DEFINITE.NEGATIVE_SEMI, (x, y)
        return DEFINITE.POSITIVE_SEMI

    @staticmethod
    def square(x: DEFINITE) -> DEFINITE:
        if x is DEFINITE.UNKNOWN:
            return DEFINITE.POSITIVE_SEMI
        return DEFINITE.mul(x, x)


@dataclass
class SortKey(Generic[τ]):

    term: τ

    def __eq__(self, other: Self) -> bool:  # type: ignore[override]
        if hash(self.term) != hash(other.term):
            return False
        return self.term._poly == other.term._poly

    def __ge__(self, other: Self) -> bool:
        return self.term._poly >= other.term._poly

    def __gt__(self, other: Self) -> bool:
        return self.term._poly > other.term._poly

    def __hash__(self) -> int:
        return hash(self.term)

    def __le__(self, other: Self) -> bool:
        return self.term._poly <= other.term._poly

    def __lt__(self, other: Self) -> bool:
        return self.term._poly < other.term._poly

    def __ne__(self, other: Self) -> bool:  # type: ignore[override]
        if hash(self.term) != hash(other.term):
            return True
        return self.term._poly != other.term._poly


class Term(firstorder.Term['Term', 'Variable', int, SortKey['Term']]):

    polynomial_ring: ClassVar[_PolynomialRing] = polynomial_ring

    _hash: Optional[int]
    _poly: MPolynomial[Rational]

    # The property should be private. We might want a method to_sage()
    @property
    def poly(self) -> MPolynomial[Rational]:
        """
        An instance of :class:`MPolynomial_libsingular
        <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular>`,
        which is wrapped by ``self``.
        """
        parent = self._poly.parent()
        if parent is not self.polynomial_ring.sage_ring:
            poly_gens = parent.gens()
            # Make sure that the manager process in parallel qe knows all
            # variables. Otherwise the following line could be replaced with an
            # assertion.
            self.polynomial_ring.add_vars(map(str, poly_gens))
            # We currently coerce manually in: reduce, subs, derivative,
            # pseudo_quo_rem. The following line might cleaner:
            #
            # TEMPORARY HACK. There is an issue with the derivative of x**2 + 1
            self._poly = self.polynomial_ring(self._poly)
        return self._poly

    def __add__(self, other: object) -> Term:
        if isinstance(other, Term):
            return Term(self.poly + other.poly)
        if isinstance(other, (mpq, float)):
            return Term(self.poly + Rational(other))
        return Term(self.poly + other)

    def __eq__(self, other: Term | int) -> Eq:  # type: ignore[override]
        # MyPy requires "other: object". However, with our use a a constructor,
        # it makes no sense to compare terms with general objects. We have
        # Eq.__bool__, which supports some comparisons in boolean contexts.
        # Same for __ne__.
        lhs = self - other
        # Use poly.lc() in order to support @lru_cache on Term.lc().
        if lhs.poly.lc() < 0:
            lhs = -lhs
        return Eq(lhs, 0)

    def __ge__(self, other: Term | int) -> Ge | Le:
        lhs = self - other
        if lhs.lc() < 0:
            return Le(-lhs, 0)
        return Ge(lhs, 0)

    def __gt__(self, other: Term | int) -> Gt | Lt:
        lhs = self - other
        if lhs.lc() < 0:
            return Lt(-lhs, 0)
        return Gt(lhs, 0)

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(self.poly)
        return self._hash

    def __getstate__(self):
        d = {"_poly": self._poly}
        return d

    def __setstate__(self, state):
        self._poly = state["_poly"]
        self._hash = None

    def __init__(self, arg: float | Fraction | int | Integer | MPolynomial[Rational]
                 | mpq | Rational | UPolynomial) -> None:
        if isinstance(arg, MPolynomial):
            self._poly = arg
        elif isinstance(arg, (float | Fraction, int, Integer, mpq, Rational, UPolynomial)):
            self._poly = self.polynomial_ring(arg)
        else:
            raise ValueError(f'expected polynomial, integer, or rational; {arg} is {type(arg)}')
        self._hash = None

    def __iter__(self) -> Iterator[tuple[mpq, Term]]:
        """Iterate over the polynomial representation of the term, yielding
        pairs of coefficients and power products.

        >>> from logic1.theories.RCF import VV
        >>> x, y = VV.get('x', 'y')
        >>> t = (x - y + 2) ** 2
        >>> [(abs(coef), power_product) for coef, power_product in t]
        [(mpq(1,1), x**2), (mpq(2,1), x*y), (mpq(1,1), y**2), (mpq(4,1), x),
         (mpq(4,1), y), (mpq(4,1), 1)]
        """
        for coefficient, power_product in self.poly:
            yield mpq(coefficient), Term(power_product)

    def __le__(self, other: Term | int | mpq) -> Ge | Le:
        lhs = self - other
        if lhs.lc() < 0:
            return Ge(-lhs, 0)
        return Le(lhs, 0)

    def __lt__(self, other: Term | int | mpq) -> Gt | Lt:
        lhs = self - other
        if lhs.lc() < 0:
            return Gt(-lhs, 0)
        return Lt(lhs, 0)

    def __mul__(self, other: object) -> Term:
        if isinstance(other, Term):
            return Term(self.poly * other.poly)
        if isinstance(other, (mpq, float)):
            return Term(self.poly * Rational(other))
        return Term(self.poly * other)

    def __ne__(  # type: ignore[override]
            self, other: Term | int | mpq) -> Ne:
        lhs = self - other
        if lhs.lc() < 0:
            lhs = -lhs
        return Ne(lhs, Term(0))

    def __neg__(self) -> Term:
        return Term(-self.poly)

    def __pow__(self, other: object) -> Term:
        return Term(self.poly ** other)

    def __radd__(self, other: object) -> Term:
        assert not isinstance(object, Term)
        if isinstance(other, (mpq, float)):
            return Term(Rational(other) + self.poly)
        return Term(other + self.poly)

    def __repr__(self) -> str:
        return repr(self.poly).replace('^', '**')

    def __rmul__(self, other: object) -> Term:
        assert not isinstance(object, Term)
        if isinstance(other, (mpq, float)):
            return Term(Rational(other) * self.poly)
        return Term(other * self.poly)

    def __rsub__(self, other: object) -> Term:
        assert not isinstance(object, Term)
        if isinstance(other, (mpq, float)):
            return Term(Rational(other) - self.poly)
        return Term(other - self.poly)

    def __str__(self):
        return str(self.poly)

    def __sub__(self, other: object) -> Term:
        if isinstance(other, Term):
            return Term(self.poly - other.poly)
        if isinstance(other, (mpq, float)):
            return Term(self.poly - Rational(other))
        return Term(self.poly - other)

    def __truediv__(self, other: object) -> Term:
        if isinstance(other, (mpq, float)):
            return Term(self.poly / Rational(other))
        if isinstance(other, Term):
            return Term(self.poly / other.poly)
        # x*y / x would yield y as a Sage rational function and raise an
        # exception.
        return Term(self.poly / other)

    def __xor__(self, other: object) -> Term:
        raise NotImplementedError(
            "Use ** for exponentiation, not '^', which means xor "
            "in Python, and has the wrong precedence")

    def as_constant(self) -> mpq:
        assert self.is_constant()
        return self.constant_coefficient()

    def as_latex(self) -> str:
        """LaTeX representation as a string. Implements the abstract method
        :meth:`.firstorder.atomic.Term.as_latex`.

        >>> from logic1.theories.RCF import VV
        >>> x, y = VV.get('x', 'y')
        >>> t = (x - y + 2) ** 2
        >>> t.as_latex()
        'x^{2} - 2 x y + y^{2} + 4 x - 4 y + 4'
        """
        return str(sage_latex(self.poly))

    def as_variable(self) -> Variable:
        if not self.is_variable():
            raise ValueError(f'{self} is not a variable')
        return Variable(self.poly)

    def coefficient(self, degrees: dict[Variable, int]) -> Term:
        """Return the coefficient of the variables with the degrees specified
        in the python dictionary `degrees`.

        >>> from logic1.theories.RCF import VV
        >>> x, y = VV.get('x', 'y')
        >>> t = (x - y + 2) ** 2
        >>> t.coefficient({x: 1, y: 1})
        -2
        >>> t.coefficient({x: 1})
        -2*y + 4

        .. seealso::
            :external:meth:`MPolynomial_libsingular.coefficient()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.coefficient>`
        """
        d_poly = {key.poly: value for key, value in degrees.items()}
        return Term(self.poly.coefficient(d_poly))

    @lru_cache(maxsize=CACHE_SIZE)
    def constant_coefficient(self) -> mpq:
        """Return the constant coefficient of this term.

        >>> from logic1.theories.RCF import VV
        >>> x, y = VV.get('x', 'y')
        >>> t = (x - y + 2) ** 2
        >>> t.constant_coefficient()
        mpq(4,1)

        .. seealso::
            :external:meth:`MPolynomial_libsingular.constant_coefficient()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.constant_coefficient>`
        """
        return mpq(self.poly.constant_coefficient())

    @lru_cache(maxsize=CACHE_SIZE)
    def content(self) -> mpq:
        """Return the content of this term, which is defined as the gcd of its
        integer coefficients.

        >>> from logic1.theories.RCF import VV
        >>> x, y = VV.get('x', 'y')
        >>> t = (x - y + 2) ** 2 - (x**2 + y**2)
        >>> t.content()
        mpq(2,1)

        .. seealso::
            :external:meth:`MPolynomial.content()
            <sage.rings.polynomial.multi_polynomial.MPolynomial.content>`
        """
        content = self.poly.content()
        assert content > 0 or (content == 0 and self == 0)
        return mpq(content)

    def degree(self, x: Variable) -> int:
        """Return the degree in `x` of this term.

        >>> from logic1.theories.RCF import VV
        >>> x, y = VV.get('x', 'y')
        >>> t = (x - y + 2) ** 2
        >>> t.degree(y)
        2

        .. seealso::
            :external:meth:`MPolynomial_libsingular.degree()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.degree>`
        """
        return self.poly.degree(x.poly)

    def derivative(self, x: Variable, n: int = 1) -> Term:
        """The `n`-th derivative of this term, with respect to `x`.

        >>> from logic1.theories.RCF import VV
        >>> x, y = VV.get('x', 'y')
        >>> t = (x - y + 2) ** 2
        >>> t.derivative(x)
        2*x - 2*y + 4

        .. seealso::
            :external:meth:`MPolynomial.derivative()
            <sage.rings.polynomial.multi_polynomial.MPolynomial.derivative>`
        """
        return Term(self.poly.derivative(self.polynomial_ring(x.poly), n))

    @lru_cache(maxsize=CACHE_SIZE)
    def factor(self) -> tuple[mpq, dict[Term, int]]:
        """A polynomial factorization of this term.

        :returns: A pair `(unit, D)`, where `unit` is a rational number, the
          keys of `D` are irreducible factors, and the corresponding values are
          their multiplicities. All irreducible factors are monic. Note that
          the return value is uniquely determined by this specification.

        >>> x, y = VV.get('x', 'y')
        >>> t = -x**2 + y**2
        >>> t.factor() == (mpq(-1,1), {x - y: 1, x + y: 1})
        True

        It is noteworthy that Sage factorization over QQ does not always yield
        monic factors.

        >>> a, b = VV.get('a', 'b')
        >>> t = 2*a**2 + 4*a*b + 2*b**2 - 1
        >>> t.factor() == (mpq(2,1), {a**2 + 2*a*b + b**2 - 1/2: 1})
        True
        >>> sage_factorization = t.poly.factor()
        >>> sage_factorization.unit(), list(sage_factorization)
        (1, [(2*a^2 + 4*a*b + 2*b^2 - 1, 1)])

        .. seealso::
            :external:meth:`MPolynomial_libsingular.factor()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.factor>`
        """
        F = self.poly.factor()
        assert F.unit().is_constant()
        unit = mpq(F.unit().constant_coefficient())
        D = dict()
        for poly, multiplicity in F:
            assert not poly.is_constant()
            lc = poly.lc()
            poly /= lc
            unit *= mpq(lc) ** multiplicity
            D[Term(poly)] = multiplicity
        return unit, D

    def is_constant(self) -> bool:
        """Return :obj:`True` if this term is constant.

        .. seealso::
            :external:meth:`MPolynomial_libsingular.is_constant()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.is_constant>`
        """
        return self.poly.is_constant()

    def is_definite(self, assume: Mapping[Variable, DEFINITE] = {}) -> DEFINITE:
        """A fast heuristic test for definitetess properties of this term. This
        is based on *trivial square sum* properties of coefficient signs and
        exponents.

        >>> x, y = VV.get('x', 'y')
        >>> print(Term(0).is_definite())
        DEFINITE.ZERO
        >>> f = x**2 + y**2
        >>> print(f.is_definite())
        DEFINITE.POSITIVE_SEMI
        >>> g = -x**2 - y**2 - 1
        >>> print(g.is_definite())
        DEFINITE.NEGATIVE
        >>> h = (x - y) ** 2
        >>> print(h.is_definite())
        DEFINITE.UNKNOWN
        >>> print(h.is_definite(assume={x: DEFINITE.POSITIVE, y: DEFINITE.NEGATIVE}))
        DEFINITE.POSITIVE
        >>> print(h.is_definite(assume={x: DEFINITE.NEGATIVE_SEMI, y: DEFINITE.POSITIVE_SEMI}))
        DEFINITE.POSITIVE_SEMI
        """
        # Start with the neutral element of DEFINITE.add().
        poly_result = DEFINITE.ZERO
        gens = self.poly.parent().gens()
        for exponent, coefficient in self.poly.dict().items():
            # Start with either POSITIVE or NEGATIVE, depending on the coefficient.
            term_result = DEFINITE.from_constant(coefficient)
            for g, e in zip(gens, exponent):
                if e == 0:
                    # In contrast to a variable with even degree, an absent
                    # variable yields the neutral element of DEFINITE.mul().
                    ge_result = DEFINITE.POSITIVE
                else:
                    ge_result = assume.get(Variable(g), DEFINITE.UNKNOWN)
                    if e % 2 == 0:
                        ge_result = DEFINITE.square(ge_result)
                term_result = DEFINITE.mul(term_result, ge_result)
            poly_result = DEFINITE.add(poly_result, term_result)
            if poly_result is DEFINITE.UNKNOWN:
                return DEFINITE.UNKNOWN
        return poly_result

    def is_monomial(self) -> bool:
        """Return :obj:`True` if this term is a monomial.
        """
        return self.poly.is_monomial()

    def is_variable(self) -> bool:
        """Return :obj:`True` if this term is a variable.
        """
        try:
            return self.poly.is_gen()
        except AttributeError:
            return self.poly.is_generator()

    def is_weakly_parametric_linear(self, X: Container[Variable]) -> bool:
        """Return :obj:`True` if this Term can be written as a_1 x_1 + ... +
        a_n x_n + r such that a_1, ..., a_n in QQ, x_1, ..., x_n in X, and r is
        a polynomial over QQ that does not contain any variable from X.

        >>> a, b, x, y = VV.get('a', 'b', 'x', 'y')
        >>> term = 2 * x - 3 * y + 4 * a**2 + 5 * a * b
        >>> term.is_weakly_parametric_linear({x, y})
        True
        >>> term.is_weakly_parametric_linear({a})
        False
        >>> term.is_weakly_parametric_linear({b})
        False
        """
        for m in self.monomials():
            if m in X:
                continue
            for v in m.vars():
                if v in X:
                    return False
        return True

    def is_zero(self) -> bool:
        """Return :obj:`True` if this term is a zero.

        .. seealso::
            :external:meth:`MPolynomial_libsingular.is_zero()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.is_zero>`
        """
        return self.poly.is_zero()

    @lru_cache(maxsize=CACHE_SIZE)
    def lc(self) -> mpq:
        """Leading coefficient of this term with respect to the degree
        lexicographical term order :mod:`deglex
        <sage.rings.polynomial.term_order>`.

        >>> from logic1.theories.RCF import VV
        >>> x, y = VV.get('x', 'y')
        >>> f = 2*x*y**2 + 3*x**2 + 1
        >>> f.lc()
        mpq(2,1)

        .. seealso::
            :external:meth:`MPolynomial_libsingular.lc()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.lc>`
        """
        return mpq(self.poly.lc())

    def monomial_coefficient(self, mon: Term) -> mpq:
        """Return the coefficient in the base ring of the monomial mon in self,
        where mon must have the same parent as self.

        .. seealso::
            :external:meth:`MPolynomial_libsingular.monomial_coefficient()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.monomial_coefficient>`
        """
        if not mon.is_monomial():
            raise ValueError(f'{mon} is not a monomial')
        return mpq(self.poly.monomial_coefficient(mon.poly))

    def monomials(self) -> list[Term]:
        """List of monomials of this term. A monomial is defined here as a
        summand of a polynomial *without* the coefficient.

        >>> from logic1.theories.RCF import VV
        >>> x, y = VV.get('x', 'y')
        >>> t = (x - y + 2) ** 2
        >>> t.monomials()
        [x**2, x*y, y**2, x, y, 1]

        .. seealso::
            :external:meth:`MPolynomial_libsingular.monomials()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.monomials>`
        """
        return [Term(monomial) for monomial in self.poly.monomials()]

    @lru_cache(maxsize=CACHE_SIZE)
    def normalize(self) -> Term:
        return Term(self.poly / self.poly.lc())

    @lru_cache(maxsize=CACHE_SIZE)
    def primitive_part(self, positive: bool = False) -> Term:
        """Return the primitive part over ``Z``. This is ``self`` divided by its
        (positive) content, so that ``self.content() * self.primitive_part() ==
        self``. If ``positive`` is ``True``, the result is normalized to have a
        positive leading coefficient.
        """
        pp = self / self.content()
        if positive and pp.lc() < 0:
            pp = -pp
        return pp

    def pseudo_quo_rem(self, other: Term, x: Variable) -> tuple[Term, Term]:
        """Pseudo quotient and remainder of this term and other, both as
        univariate polynomials in `x` with polynomial coefficients in all other
        variables.

        >>> a, b, c, x = VV.get('a', 'b', 'c', 'x')
        >>> f = a * x**2 + b*x + c
        >>> g = c * x + b
        >>> q, r = f.pseudo_quo_rem(g, x); q, r
        (a*c*x - a*b + b*c, a*b**2 - b**2*c + c**3)
        >>> assert c**(2 - 1 + 1) * f == q * g + r

        .. seealso::
            :meth:`Polynomial.pseudo_quo_rem()
            <sage.rings.polynomial.polynomial_element.Polynomial.pseudo_quo_rem>`
        """
        self1 = self.poly.polynomial(self.polynomial_ring(x.poly))
        other1 = other.poly.polynomial(self.polynomial_ring(x.poly))
        quotient, remainder = self1.pseudo_quo_rem(other1)
        return Term(quotient), Term(remainder)

    def quo_rem(self, other: Term) -> tuple[Term, Term]:
        """Quotient and remainder of this term and `other`.

        >>> from logic1.theories.RCF import VV
        >>> x, y = VV.get('x', 'y')
        >>> f = 2*y*x**2 + x + 1
        >>> f.quo_rem(x)
        (2*x*y + 1, 1)
        >>> f.quo_rem(y)
        (2*x**2, x + 1)
        >>> f.quo_rem(3*x)  # would yield (0, 2*x**2*y + x + 1) over ZZ
        (2/3*x*y + 1/3, 1)

        .. seealso::
            :external:meth:`MPolynomial_libsingular.quo_rem()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.quo_rem>`
        """
        quo, rem = self.poly.quo_rem(other.poly)
        return Term(quo), Term(rem)

    def reduce(self, G: Iterable[Term]) -> Term:
        """Reduce self modulo G.
        """
        # Sage requires that g.poly can be coerced to self.poly.parent().
        poly = self.polynomial_ring(self.poly).reduce([g.poly for g in G])
        return Term(poly)

    def sort_key(self) -> SortKey[Self]:
        """A sort key suitable for ordering instances of this class. Implements
        the abstract method :meth:`.firstorder.atomic.Term.sort_key`.
        """
        return SortKey(self)

    def subs(self, d: Mapping[Variable, Term | int | mpq]) -> Term:
        """Simultaneous substitution of terms for variables.

        >>> from logic1.theories.RCF import VV
        >>> x, y, z = VV.get('x', 'y', 'z')
        >>> f = 2*y*x**2 + x + 1
        >>> f.subs({x: y, y: 2*z})
        4*y**2*z + y + 1

        .. seealso::
            :external:meth:`MPolynomial_libsingular.subs()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.subs>`
        """
        sage_keywords: dict[str, MPolynomial[Rational] | int | mpq] = dict()
        for variable, substitute in d.items():
            match substitute:
                case Term():
                    sage_keywords[str(variable.poly)] = substitute.poly
                case int() | mpq():
                    sage_keywords[str(variable.poly)] = substitute
                case _:
                    assert False, (self, d)
        return Term(self.polynomial_ring(self.poly).subs(**sage_keywords))

    @lru_cache(maxsize=CACHE_SIZE)
    def subs_linear_solution(self, x: Variable, minimal_polynomial: Term) -> Term:
        """Substitute the solution of the weakly parametric linear
        polynomial ``minimal_polynomial`` this weakly parametric linear
        polynomial.
        """
        # self = a * x + b
        a = self.monomial_coefficient(x)
        b = self - a * x
        assert x not in b.vars()
        # minimal_polynomial = c * x + d
        c = minimal_polynomial.monomial_coefficient(x)
        d = minimal_polynomial - c * x
        assert x not in d.vars()
        result = a * (-d / c) + b
        return result

    def summands(self) -> Iterator[tuple[dict[Variable, int], mpq]]:
        """Iterate over the summands of self yielding pairs of dictionaries
        representing monomials, and coefficients.
        """
        gens = self.polynomial_ring.sage_ring.gens()
        for etuple, coefficient in self.poly.iterator_exp_coeff(as_ETuples=True):
            result = dict()
            for i, exponent in enumerate(etuple):
                if exponent:
                    result[Variable(gens[i])] = int(exponent)
            yield result, mpq(coefficient)

    def vars(self) -> Iterator[Variable]:
        """An iterator that yields each variable of this term once. Implements
        the abstract method :meth:`.firstorder.atomic.Term.vars`.

        .. seealso::
            :external:meth:`MPolynomial_libsingular.variables()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.variables>`
        """
        for g in self.poly.variables():
            yield Variable(g)

# discuss: Variable inherits __init__, and we can create Variable(3), Variable(term.poly), etc.
class Variable(Term, firstorder.Variable['Variable', int, SortKey['Variable']]):

    VV: ClassVar[VariableSet] = VV

    def fresh(self) -> Variable:
        """Returns a variable that has not been used so far. Implements
        abstract method :meth:`.firstorder.atomic.Variable.fresh`.
        """
        return self.VV.fresh(suffix=f'_{str(self)}')