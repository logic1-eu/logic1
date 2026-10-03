from __future__ import annotations

from collections.abc import Container
from dataclasses import dataclass
from functools import lru_cache
from typing import (ClassVar, Final, Generic, Iterable, Iterator, Mapping, Optional, Self, TypeVar)
from typing_extensions import final

from gmpy2 import mpq
from sage.all import QQ
# Importing QQ from sage.rings.rational_fields causes problems. Notably, a
# fresh instance of RationalField is assigned to QQ in sage.all.
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
from logic1.theories.RCF.term import abc as RCF_term_abc
from logic1.theories.RCF.term.definite import Definite
from logic1.theories.RCF.types import Number, _NUMBER_TYPES
from logic1.theories.RCF.atomic import Eq, Ge, Gt, Le, Lt, Ne

from logic1.support.tracing import trace  # noqa


POLYLIB: Final = "SAGE"


τ = TypeVar('τ', bound='Term')
"""A type variable denoting a type of terms with upper bound :class:`Term`.
"""

CACHE_SIZE: Final[Optional[int]] = 2**16


def _caches():
    from logic1.theories.RCF.simplify import Simplify
    from logic1.theories.RCF.substitution import _SubstValue
    return [Term.factor, _SubstValue.as_term, Simplify._simpl_at]

def cache_clear():
    for cache in _caches():
        cache.cache_clear()

def cache_info():
    return {cache.__wrapped__: cache.cache_info() for cache in _caches()}


def init_env(ring_vars: list[str]) -> None:
    VV._used.update(ring_vars)

def init_env_arg() -> list[str]:
    return list(VV._used)


@final
class _PolynomialRing(RCF_term_abc.Ring['MPolynomial[Rational]', MPolynomialRing]):
    """A wrapper around a Sage Singular polynomial ring.
    """

    _ring: MPolynomialRing
    """Underlying Sage polynomial ring. The variable names are sorted according
    to the sort key :meth:`sort_key`
    """

    def __init__(self, vars_: Iterable[str] = (), term_order: str = 'deglex') -> None:
        """Construct a polynomial ring with the given variables and term order.

        >>> R = _PolynomialRing(['x', 'y'], term_order='lex')
        >>> R
        _PolynomialRing(['x', 'y'], order='lex')
        """
        vars_ = set(vars_)
        if 'unused_' in vars_:
            raise ValueError("Variable name 'unused_' is reserved and cannot be used.")
        vars_.add('unused_')
        new_vars = sorted(vars_, key=_PolynomialRing.sort_key)
        self._ring = self.MPolynomialRing_factory(new_vars, order=TermOrder(term_order))

    def __repr__(self) -> str:
        """Return a string representation of the polynomial ring.

        >>> R = _PolynomialRing()
        >>> R
        _PolynomialRing([], order='deglex')
        """
        names = [str(g) for g in self.get_gens()]
        order = self._ring.term_order().name()
        return f'_PolynomialRing({names}, order=\'{order}\')'

    def __str__(self) -> str:
        """Return a string representation of the underlying Sage polynomial
        ring.
        """
        return str(self._ring)

    def coerce_poly(self, poly: MPolynomial[Rational]) -> MPolynomial[Rational]:
        """Coerce poly to the :attr:`_ring` of ``self``.
        """
        return self._ring(poly)

    @classmethod
    def from_raw(cls, ring: MPolynomialRing) -> Self:
        self = cls.__new__(cls)
        self._ring = ring
        return self

    def get_names(self) -> tuple[str, ...]:
        """Return the names of the variables in the polynomial ring.

        >>> R = _PolynomialRing(('a', 'b', 'c'))
        >>> R.get_names()
        ('a', 'b', 'c')
        """
        return tuple(str(var) for var in self.get_gens())

    def get_gens(self) -> tuple[MPolynomial[Rational], ...]:
        """Return the variables of the polynomial ring.

        >>> R = _PolynomialRing(('a', 'b', 'c'))
        >>> R.get_gens()
        (a, b, c)
        """
        return tuple(g for g in self._ring.gens() if str(g) != 'unused_')

    @staticmethod
    def MPolynomialRing_factory(names: str | Iterable[str], order: TermOrder) -> MPolynomialRing:
        """Construct a Sage Singular polynomial ring with the given variable names and term order.
        """
        if not isinstance(names, str):
            names = tuple(names)
            if len(names) > 2**15:
                # https://github.com/Singular/Singular/issues/1383
                # https://github.com/sagemath/sage/issues/42712
                raise OverflowError(f'cannot construct a polynomial ring with {len(names)} variables')
        return sage_PolynomialRing(QQ, names, order=order, implementation='singular')

    @staticmethod
    def sort_key(s: str) -> tuple[str, int]:
        """Sort key for variable names. The sort order is lexicographic, except
        that variables with the same name are ordered by their numeric suffix.

        >>> _PolynomialRing.sort_key('x')
        ('x', -1)
        >>> _PolynomialRing.sort_key('x1')
        ('x', 1)
        """
        base = s.rstrip('0123456789')
        index = s[len(base):]
        n = int(index) if index else -1
        return base, n


class VariableSet(firstorder.VariableSet['Variable']):
    """The infinite set of all variables belonging to the theory of Real Closed
    Fields. Variables are uniquely identified by their name, which is a
    :external:class:`.str`. This class is a singleton, whose single instance is
    assigned to :data:`.VV`.

    The use of :data:`.VV` for the construction of terms, atoms, and formulas is
    described in the introduction of the section :ref:`Real Closed Fields <api-RCF>`.

    .. seealso::
        Final methods inherited from the parent class:

        * :meth:`.firstorder.term.VariableSet.get`
            -- obtain several variables simultaneously
        * :meth:`.firstorder.term.VariableSet.imp`
            -- import variables into global namespace
    """

    _stack: list[set[str]]

    @property
    def stack(self) -> list[MPolynomialRing]:
        """Implements the abstract property :attr:`.firstorder.term.VariableSet.stack`.
        """
        return self.stack

    @property
    def _used(self) -> set[str]:
        return self._stack[-1]

    def __getitem__(self, name: str) -> Variable:
        """Implements the abstract method :meth:`.firstorder.term.VariableSet.__getitem__`.

        >>> from logic1.theories.RCF import VV
        >>> isinstance(VV, VariableSet)
        True
        >>> x = VV['x']
        >>> isinstance(x, Variable)
        True
        """
        if not isinstance(name, str):
            raise TypeError(f'expecting string as index; {name} is {type(name)}')
        self._used.add(name)
        ring = _PolynomialRing((name,))
        return Variable._from_gen(ring.get_gens()[0])

    def __init__(self) -> None:
        self._stack = [set()]

    def __repr__(self) -> str:
        names = sorted(self._used, key=_PolynomialRing.sort_key)
        s = ', '.join(name for name in (*names, '...'))
        return f'{{{s}}}'

    def fresh(self, suffix: str = '') -> Variable:
        """Return a fresh variable, by default from the sequence G0001, G0002,
        ..., G9999, G10000, ... This naming convention is inspired by Lisp's
        gensym(). If the optional argument :data:`suffix` is specified, the
        sequence G0001<suffix>, G0002<suffix>, ... is used instead.

        >>> from logic1.theories.RCF import VV
        >>> VV.fresh('_demo')
        G0001_demo
        >>> VV.fresh('_demo')
        G0002_demo
        """
        i = 1
        v = f'G{i:04d}{suffix}'
        while v in self._used:
            i += 1
            v = f'G{i:04d}{suffix}'
        return self[v]

    def pop(self) -> None:
        from . import cache_clear
        self._stack.pop()
        cache_clear()

    def push(self) -> None:
        from . import cache_clear
        self._stack.append(set())
        cache_clear()


VV: Final = VariableSet()
"""
The unique instance of :class:`.VariableSet`.
"""


@dataclass
class SortKey(Generic[τ]):
    """
    Sort key for comparing terms.

    >>> x = VV['x']
    >>> SortKey(x) < SortKey(x + 1)
    True
    """

    term: τ

    def __eq__(self, other: Self) -> bool:  # type: ignore[override]
        ring = self.term._parent | other.term._parent
        self_poly = ring.coerce_poly(self.term._poly)
        other_poly = ring.coerce_poly(other.term._poly)
        return self_poly == other_poly

    def __ge__(self, other: Self) -> bool:
        ring = self.term._parent | other.term._parent
        self_poly = ring.coerce_poly(self.term._poly)
        other_poly = ring.coerce_poly(other.term._poly)
        return self_poly >= other_poly

    def __gt__(self, other: Self) -> bool:
        ring = self.term._parent | other.term._parent
        self_poly = ring.coerce_poly(self.term._poly)
        other_poly = ring.coerce_poly(other.term._poly)
        return self_poly > other_poly

    def __hash__(self) -> int:
        return hash(self.term)

    def __le__(self, other: Self) -> bool:
        ring = self.term._parent | other.term._parent
        self_poly = ring.coerce_poly(self.term._poly)
        other_poly = ring.coerce_poly(other.term._poly)
        return self_poly <= other_poly

    def __lt__(self, other: Self) -> bool:
        ring = self.term._parent | other.term._parent
        self_poly = ring.coerce_poly(self.term._poly)
        other_poly = ring.coerce_poly(other.term._poly)
        return self_poly < other_poly

    def __ne__(self, other: Self) -> bool:  # type: ignore[override]
        ring = self.term._parent | other.term._parent
        self_poly = ring.coerce_poly(self.term._poly)
        other_poly = ring.coerce_poly(other.term._poly)
        return self_poly != other_poly


class Term(RCF_term_abc.Term['Term', 'Variable', SortKey['Term'], 'MPolynomial[Rational]', MPolynomialRing]):

    _hash: Optional[int]
    _poly: MPolynomial[Rational]

    @property
    def _parent(self) -> _PolynomialRing:
        return _PolynomialRing.from_raw(self._poly.parent())

    def __eq__(self, other: Number | Term) -> Eq:  # type: ignore[override]
        # MyPy requires "other: object". However, with our use a a constructor,
        # it makes no sense to compare terms with general objects. We have
        # Eq.__bool__, which supports some comparisons in Boolean contexts.
        # Same for __ne__.
        lhs = self - other
        # Use poly.lc() in order to support @lru_cache on Term.lc().
        if lhs._poly.lc() < 0:
            lhs = -lhs
        return Eq(lhs, 0)

    def __ge__(self, other: Number | Term) -> Ge | Le:
        lhs = self - other
        if lhs.lc() < 0:
            return Le(-lhs, 0)
        else:
            return Ge(lhs, 0)

    def __gt__(self, other: Number | Term) -> Gt | Lt:
        lhs = self - other
        if lhs.lc() < 0:
            return Lt(-lhs, 0)
        else:
            return Gt(lhs, 0)

    def __hash__(self) -> int:
        return super().__hash__()

    def __init__(self, arg: Number) -> None:
        """Construct a :class:`Term` from a number.

        >>> from fractions import Fraction
        >>> from logic1.theories.RCF import Term
        >>> Term(2)
        2
        >>> Term(0.5)
        1/2
        >>> Term(Fraction(1, 2))
        1/2
        >>> Term(mpq(1, 2))
        1/2

        .. attention::
            Python division of integers yields a float, which can cause
            precision issues:

            >>> Term(1/10 + 2/10)
            415716888680356/1385722962267853

            In contrast:

            >>> Term(mpq(1, 10) + mpq(2, 10))
            3/10
            >>> Term(Fraction(1, 10) + Fraction(2, 10))
            3/10

        """
        if not isinstance(arg, _NUMBER_TYPES):
            raise ValueError(f'expected a number type; {arg} is {type(arg)}')
        ring = _PolynomialRing()
        self._poly = ring._ring(arg)
        self._hash = None

    def __iter__(self) -> Iterator[tuple[mpq, Term]]:
        """Iterate over the polynomial representation of the term, yielding
        pairs of coefficients and power products.

        >>> x, y = VV.get('x', 'y')
        >>> t = (x - y + 2) ** 2
        >>> [(abs(coef), power_product) for coef, power_product in t]
        [(mpq(1,1), x**2), (mpq(2,1), x*y), (mpq(1,1), y**2), (mpq(4,1), x),
         (mpq(4,1), y), (mpq(4,1), 1)]
        """
        for coefficient, power_product in self._poly:
            yield mpq(coefficient), Term._from_raw(power_product)

    def __le__(self, other: Term | int | mpq) -> Ge | Le:
        lhs = self - other
        if lhs.lc() < 0:
            return Ge(-lhs, 0)
        else:
            return Le(lhs, 0)

    def __lt__(self, other: Term | int | mpq) -> Gt | Lt:
        lhs = self - other
        if lhs.lc() < 0:
            return Gt(-lhs, 0)
        else:
            return Lt(lhs, 0)

    def __ne__(self, other: Number | Term) -> Ne:  # type: ignore[override]
        lhs = self - other
        if lhs.lc() < 0:
            lhs = -lhs
        return Ne(lhs, 0)

    def __repr__(self) -> str:
        return repr(self._poly).replace('^', '**')

    def __truediv__(self, other: Number | Term) -> Term:
        """True division. ``self`` must be divisible by ``other``.
        """
        if not isinstance(other, Term):
            other = Term(other)
        ring = self._parent | other._parent
        quotient = ring.coerce_poly(self._poly) / ring.coerce_poly(other._poly)
        # For x*y / y we obtain an element of a fraction field. We must ensure
        # that the denominator is 1 and then convert to an MPolynomial.
        return Term._from_raw(quotient)

    def as_constant(self) -> mpq:
        """Return this term as an :class:`mpq <.gmpy2.mpq>`.
        Raise :class:`ValueError` if this term is not constant.

        >>> from logic1.theories.RCF import VV
        >>> x = VV['x']
        >>> t = x + mpq(1, 2) - x
        >>> t
        1/2
        >>> isinstance(t, Term)
        True
        >>> isinstance(t, mpq)
        False
        >>> c = t.as_constant()
        >>> c
        mpq(1,2)
        >>> isinstance(c, mpq)
        True

        .. seealso::
            :meth:`.Term.is_constant`
        """
        if not self.is_constant():
            raise ValueError(f'{self} is not constant')
        return self.constant_coefficient()

    def as_latex(self) -> str:
        """LaTeX representation as a string. Implements the abstract method
        :meth:`.firstorder.term.Term.as_latex`.

        >>> from logic1.theories.RCF import VV
        >>> x, y = VV.get('x', 'y')
        >>> t = (x - y + 2) ** 2
        >>> t.as_latex()
        'x^{2} - 2 x y + y^{2} + 4 x - 4 y + 4'
        """
        return str(sage_latex(self._poly))

    def as_variable(self) -> Variable:
        """Return this term as an instance of the subclass :class:`.Variable`.
        Raise :class:`ValueError` if this term is not a variable.

        >>> from logic1.theories.RCF import VV
        >>> x = VV['x']
        >>> t = x + 1 - 1
        >>> t
        x
        >>> isinstance(t, Term)
        True
        >>> isinstance(t, Variable)
        False
        >>> v = t.as_variable()
        >>> v
        x
        >>> isinstance(v, Variable)
        True

        .. seealso::
            :meth:`.Term.is_variable`
        """
        if not self.is_variable():
            raise ValueError(f'{self} is not a variable')
        return Variable._from_gen(self._poly)

    def coefficient(self, degrees: dict[Variable, int]) -> Term:
        """Return the coefficient of the variables with the degrees specified in
        the ``degrees``. Mathematically, this is the coefficient in the base
        ring adjoined by the variables of this ring that are not listed in
        ``degrees``.

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
        ring = self._parent
        for variable in degrees:
            ring |= variable._parent
        self_poly = ring.coerce_poly(self._poly)
        degrees_poly = {ring.coerce_poly(key._poly): value for key, value in degrees.items()}
        return Term._from_raw(self_poly.coefficient(degrees_poly))

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
        return mpq(self._poly.constant_coefficient())

    @lru_cache(maxsize=CACHE_SIZE)
    def content(self) -> mpq:
        """Return the content of this term, which is defined as the positive gcd
        of its rational coefficients.

        >>> x, y = VV.get('x', 'y')
        >>> (mpq(2, 3) * x + mpq(4, 9) * y + mpq(8, 15)).content()
        mpq(2,45)

        .. seealso::
            :external:meth:`MPolynomial.content()
            <sage.rings.polynomial.multi_polynomial.MPolynomial.content>`
        """
        content = self._poly.content()
        assert content > 0 or (content == 0 and self == 0)
        return mpq(content)

    def degree(self, x: Variable) -> int:
        """Return the degree of the Term in ``x``.

        >>> x, y = VV.get('x', 'y')
        >>> (2*y*x**2 + x + 1).degree(x)
        2

        .. seealso::
            :external:meth:`MPolynomial_libsingular.degree()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.degree>`
        """
        ring = self._parent | x._parent
        return ring.coerce_poly(self._poly).degree(ring.coerce_poly(x._poly))

    def derivative(self, x: Variable, n: int = 1) -> Term:
        """The ``n``-th derivative of this term, with respect to ``x``.

        >>> x, y, z = VV.get('x', 'y', 'z')
        >>> (x ** 2 * y + x * z ** 2 + y ** 2 * z).derivative(x)
        2*x*y + z**2

        .. seealso::
            :external:meth:`MPolynomial.derivative()
            <sage.rings.polynomial.multi_polynomial.MPolynomial.derivative>`
        """
        if n < 0:
            raise ValueError("Derivative order must be non-negative")
        ring = self._parent | x._parent
        df = ring.coerce_poly(self._poly).derivative(ring.coerce_poly(x._poly), n)
        return Term._from_raw(df)

    @lru_cache(maxsize=CACHE_SIZE)
    def factor(self) -> tuple[mpq, dict[Term, int]]:
        """A polynomial factorization of this term.

        Returns a pair ``(unit, D)``, where ``unit`` is a rational number, the
        keys of ``D`` are irreducible factors, and the corresponding values are
        their multiplicities. All irreducible factors are monic. Note that the
        return value is uniquely determined by this specification.

        >>> x, y = VV.get('x', 'y')
        >>> t = -x**2 + y**2
        >>> t.factor() == (mpq(-1,1), {x - y: 1, x + y: 1})
        True

        It is noteworthy that Sage factorization over :external:class:`QQ
        <sage.rings.rational_field.RationalField>` does not always yield monic
        factors.

        >>> a, b = VV.get('a', 'b')
        >>> t = 2*a**2 + 4*a*b + 2*b**2 - 1
        >>> t.factor() == (mpq(2,1), {a**2 + 2*a*b + b**2 - 1/2: 1})
        True
        >>> sage_factorization = t._poly.factor()
        >>> sage_factorization.unit(), list(sage_factorization)
        (1, [(2*a^2 + 4*a*b + 2*b^2 - 1, 1)])

        .. seealso::
            :external:meth:`MPolynomial_libsingular.factor()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.factor>`
        """
        if self.is_constant():
            return self.constant_coefficient(), {}
        F = self._poly.factor()
        assert F.unit().is_constant()
        unit = mpq(F.unit().constant_coefficient())
        D = dict()
        for poly, multiplicity in F:
            assert not poly.is_constant()
            lc = poly.lc()
            poly /= lc
            unit *= mpq(lc) ** multiplicity
            D[Term._from_raw(poly)] = multiplicity
        return unit, D

    def all_variable_names(p):
        R = p.parent()
        names = []
        while hasattr(R, "variable_names"):
            names.extend(R.variable_names())
            R = R.base_ring()
        return tuple(names)

    @classmethod
    def _from_number(cls, num: Number) -> Term:
        return Term(num)

    @classmethod
    def _from_raw(cls, value: Integer | Rational | MPolynomial[Rational] | UPolynomial) -> Term:
        """Construct a :class:`Term` from a Sage object. Its argument types are
        private and should not be used outside of this module.
        """
        if isinstance(value, MPolynomial):
            poly = value
        elif isinstance(value, (Integer, Rational)):
            ring = _PolynomialRing()
            poly = ring._ring(value)
        elif isinstance(value, UPolynomial):
            R = value.parent()
            names = []
            while R != QQ:
                assert hasattr(R, 'variable_names')
                names.extend(R.variable_names())
                assert hasattr(R, 'base_ring')
                R = R.base_ring()
            names.remove('unused_')
            ring = _PolynomialRing(names)
            poly = ring._ring(value)
        else:
            raise TypeError(f"Unsupported type for Term: {type(value)}")
        term = Term.__new__(Term)
        term._poly = poly
        term._hash = None
        return term

    def is_constant(self) -> bool:
        """Return :obj:`True` if this term is constant.

        >>> from logic1.theories.RCF import VV
        >>> x = VV['x']
        >>> t = x + mpq(1, 2) - x
        >>> t
        1/2
        >>> isinstance(t, mpq)
        False
        >>> t.is_constant()
        True

        .. seealso::
            - :meth:`.as_constant`
            - :external:meth:`MPolynomial_libsingular.is_constant()
              <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.is_constant>`
        """
        return self._poly.is_constant()

    def is_definite(self, assume: Mapping[Variable, Definite] = {}) -> Definite:
        """A fast heuristic test for definitetess properties of this term. This
        is based on *trivial square sum* properties of coefficient signs and
        exponents.

        >>> x, y = VV.get('x', 'y')
        >>> print(Term(0).is_definite())
        Definite.ZERO
        >>> f = x**2 + y**2
        >>> print(f.is_definite())
        Definite.POSITIVE_SEMI
        >>> g = -x**2 - y**2 - 1
        >>> print(g.is_definite())
        Definite.NEGATIVE
        >>> h = (x - y) ** 2
        >>> print(h.is_definite())
        Definite.UNKNOWN
        >>> print(h.is_definite(assume={x: Definite.POSITIVE, y: Definite.NEGATIVE}))
        Definite.POSITIVE
        >>> print(h.is_definite(assume={x: Definite.NEGATIVE_SEMI, y: Definite.POSITIVE_SEMI}))
        Definite.POSITIVE_SEMI
        """
        # Start with the neutral element of Definite.add().
        poly_result = Definite.ZERO
        gens = self._poly.parent().gens()
        for exponent, coefficient in self._poly.dict().items():
            # Start with either POSITIVE or NEGATIVE, depending on the coefficient.
            term_result = Definite.from_constant(mpq(coefficient))
            for g, e in zip(gens, exponent):
                if e == 0:
                    # In contrast to a variable with even degree, an absent
                    # variable yields the neutral element of Definite.mul().
                    ge_result = Definite.POSITIVE
                else:
                    ge_result = assume.get(Variable._from_gen(g), Definite.UNKNOWN)
                    if e % 2 == 0:
                        ge_result = Definite.square(ge_result)
                term_result = Definite.mul(term_result, ge_result)
            poly_result = Definite.add(poly_result, term_result)
            if poly_result is Definite.UNKNOWN:
                return Definite.UNKNOWN
        return poly_result

    def is_monomial(self) -> bool:
        """Check if this term is a monomial. A monomial is a summand without its
        coefficient, i.e., there is a bijection between monomials and exponent
        vectors. In particular, 1 is the only constant monomial.
        """
        return self._poly.is_monomial()

    def is_variable(self) -> bool:
        """Return :obj:`True` if this term is a variable.

        >>> x = VV['x']
        >>> t = x + 1 - 1
        >>> isinstance(t, Term)
        True
        >>> isinstance(t, Variable)
        False
        >>> t.is_variable()
        True

        .. seealso::
            :meth:`.as_variable`
        """
        return self._poly.is_gen()

    def is_weakly_parametric_linear(self, X: Container[Variable]) -> bool:
        r"""Return :obj:`True` if this Term can be written as
        :math:`a_1 x_1 + ... + a_n x_n + r` such that :math:`a_1, ..., a_n \in
        \mathbb{Q}`, :math:`x_1, ..., x_n \in X`, and :math:`r` is a polynomial
        over :math:`\mathbb{Q}` that does not contain any variable from
        :math:`X`.

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
        """Return :obj:`True` if this term equals zero.

        >>> from logic1.theories.RCF import VV
        >>> x = VV['x']
        >>> t = x - x
        >>> t
        0
        >>> isinstance(t, Term)
        True
        >>> isinstance(t, int)
        False
        >>> t.is_zero()
        True

        .. seealso::
            :external:meth:`MPolynomial_libsingular.is_zero()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.is_zero>`
        """
        return self._poly.is_zero()

    @lru_cache(maxsize=CACHE_SIZE)
    def lc(self) -> mpq:
        """Return the leading coefficient of this term with respect to the
        degree lexicographical term order :mod:`deglex <sage.rings.polynomial.term_order>`.

        >>> from logic1.theories.RCF import VV
        >>> x, y = VV.get('x', 'y')
        >>> f = 2*x*y**2 + 3*x**2 + 1
        >>> f.lc()
        mpq(2,1)

        .. seealso::
            :external:meth:`MPolynomial_libsingular.lc()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.lc>`
        """
        return mpq(self._poly.lc())

    def monomial_coefficient(self, mon: Term) -> mpq:
        """Return the coefficient in the base ring of the monomial ``mon`` in
        ``self``, where ``mon`` must have the same parent as ``self``. Raise
        :class:`ValueError` if ``mon`` is not a monomial.

        .. seealso::
            :external:meth:`MPolynomial_libsingular.monomial_coefficient()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.monomial_coefficient>`
        """
        if not mon.is_monomial():
            raise ValueError(f'{mon} is not a monomial')
        ring = self._parent | mon._parent
        self_poly = ring.coerce_poly(self._poly)
        mon_poly = ring.coerce_poly(mon._poly)
        return mpq(self_poly.monomial_coefficient(mon_poly))

    def monomials(self) -> list[Term]:
        """Return a list of all monomials of this term. A monomial is defined
        here as a summand of a polynomial *without* the coefficient.

        >>> from logic1.theories.RCF import VV
        >>> x, y = VV.get('x', 'y')
        >>> t = (x - y + 2) ** 2
        >>> t.monomials()
        [x**2, x*y, y**2, x, y, 1]

        .. seealso::
            :external:meth:`MPolynomial_libsingular.monomials()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.monomials>`
        """
        return [Term._from_raw(monomial) for monomial in self._poly.monomials()]

    @lru_cache(maxsize=CACHE_SIZE)
    def normalize(self) -> Term:
        """Divide this term by its leading coefficient, so that the result is monic.
        """
        return Term._from_raw(self._poly / self._poly.lc())

    @lru_cache(maxsize=CACHE_SIZE)
    def primitive_part(self, positive: bool = False) -> Term:
        """Return the primitive part of this term. This is ``self`` divided by
        its (positive) content, so that ``self.content() * self.primitive_part()
        == self``. If ``positive`` is ``True``, the result is normalized to have
        a positive leading coefficient.
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
        # self, other, quotient are of type UPolynomial
        ring = self._parent | other._parent | x._parent
        x_poly = ring.coerce_poly(x._poly)
        self1 = ring.coerce_poly(self._poly).polynomial(x_poly)
        other1 = ring.coerce_poly(other._poly).polynomial(x_poly)
        quotient, remainder = self1.pseudo_quo_rem(other1)
        return Term._from_raw(quotient), Term._from_raw(remainder)

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
        ring = self._parent | other._parent
        self_poly = ring.coerce_poly(self._poly)
        other_poly = ring.coerce_poly(other._poly)
        quo, rem = self_poly.quo_rem(other_poly)
        return Term._from_raw(quo), Term._from_raw(rem)

    def reduce(self, G: Iterable[Term]) -> Term:
        """Reduce self modulo G. The output is a polynomial ``r`` such that
        ``self - r`` is in the ideal generated by ``G``, and no monomial of
        ``r`` is divisible by the leading monomial of any polynomial in ``G``.
        The result is canonical if ``G`` is a Gröbner basis.

        The elements of G must be coercible to the parent of self. Otherwise,
        a :class:`TypeError` is raised.

        .. seealso::
            :external:meth:`MPolynomial_libsingular.reduce()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.reduce>`
        """
        G = tuple(G)
        ring = self._parent
        for g in G:
            ring |= g._parent
        self_poly = ring.coerce_poly(self._poly)
        poly = self_poly.reduce([ring.coerce_poly(g._poly) for g in G])
        return Term._from_raw(poly)

    def sort_key(self) -> SortKey[Self]:
        """A sort key suitable for ordering instances of this class. Implements
        the abstract method :meth:`.firstorder.term.Term.sort_key`.
        """
        return SortKey(self)

    def subs(self, d: Mapping[Variable, Number | Term]) -> Term:
        """Simultaneous substitution of terms for variables.

        >>> from logic1.theories.RCF import VV
        >>> x, y, z = VV.get('x', 'y', 'z')
        >>> (x + y).subs({x: mpq(1,2)})
        y + 1/2
        >>> (2*y*x**2 + x + 1).subs({x: y, y: 2*z})
        4*y**2*z + y + 1

        .. seealso::
            :external:meth:`MPolynomial_libsingular.subs()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.subs>`
        """
        ring = self._parent
        sage_keywords: dict[str, MPolynomial[Rational]] = dict()
        for variable, substitute in d.items():
            if not isinstance(substitute, Term):
                substitute = Term(substitute)
            sage_keywords[str(variable)] = substitute._poly
            ring = ring | variable._parent | substitute._parent
        for string in sage_keywords:
            sage_keywords[string] = ring.coerce_poly(sage_keywords[string])
        self_poly = ring.coerce_poly(self._poly)
        return Term._from_raw(self_poly.subs(**sage_keywords))

    @lru_cache(maxsize=CACHE_SIZE)
    def subs_linear_solution(self, x: Variable, minimal_polynomial: Term) -> Term:
        """Substitute the solution of the weakly parametric linear polynomial
        ``minimal_polynomial`` into this weakly parametric linear polynomial.

        >>> from logic1.theories.RCF import VV
        >>> a, b, x = VV.get('a', 'b', 'x')
        >>> (2 * x + a).subs_linear_solution(x, 5 * x + b)
        a - 2/5*b

        Both ``self`` and ``minimal_polynomial`` must be weakly parametric
        linear in ``x``.
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
        """Iterate over the summands of this term yielding pairs of monomials
        and coefficients, where the monimials are represented as dictionaries.

        >>> from logic1.theories.RCF import VV
        >>> a, b, c = VV.get('a', 'b', 'c')
        >>> f = a*c**3 + a**2*b + 2*b**4
        >>> list(f.summands())
        [({a: 1, c: 3}, mpq(1,1)), ({b: 4}, mpq(2,1)), ({a: 2, b: 1}, mpq(1,1))]

        .. seealso::
            :external:meth:`MPolynomial_libsingular.iterator_exp_coeff()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.iterator_exp_coeff>`
        """
        gens = self._parent._ring.gens()
        for etuple, coefficient in self._poly.iterator_exp_coeff(as_ETuples=True):
            result = dict()
            for i, exponent in enumerate(etuple):
                if exponent:
                    result[Variable._from_gen(gens[i])] = int(exponent)
            yield result, mpq(coefficient)

    def vars(self) -> Iterator[Variable]:
        """An iterator that yields each variable of this term once. Implements
        the abstract method :meth:`.firstorder.term.Term.vars`.

        .. seealso::
            :external:meth:`MPolynomial_libsingular.variables()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.variables>`
        """
        for g in self._poly.variables():
            yield Variable._from_gen(g)


class Variable(Term, firstorder.Variable['Variable', int, SortKey['Variable']]):

    VV: ClassVar[VariableSet] = VV

    def __init__(self, arg: object) -> None:
        raise NotImplementedError("Use the global variable set VV to create variables.")

    def fresh(self) -> Variable:
        """Returns a variable that has not been used so far. Implements
        abstract method :meth:`.firstorder.term.Variable.fresh`.
        """
        return self.VV.fresh(suffix=f'_{str(self)}')

    @classmethod
    def _from_gen(cls, gen: MPolynomial[Rational]) -> Variable:
        """Construct a :class:`Variable` from an ``MPolynomial``. The argument
        types are private and should not be used outside of this module.
        """
        if not gen.is_gen():
            raise ValueError(f"{gen} is not a generator")
        variable = cls.__new__(cls)
        variable._poly = gen
        variable._hash = None
        return variable
