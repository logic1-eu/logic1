from __future__ import annotations

from enum import auto, Enum
from fractions import Fraction
from typing import Any, ClassVar, Final, Iterable, Iterator, Mapping, Self

from gmpy2 import mpq
from sage.all import QQ
# Importing QQ from sage.rings.rational_fields does not work. In fact, a fresh
# instance of RationalField is assigned to QQ in sage.all.
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

from ... import firstorder
from ...firstorder import _T, _F
from ...support.excepthook import NoTraceException

from ...support.tracing import trace  # noqa


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

    def ensure_vars(self, vars_: Iterable[str]) -> None:

        def sort_key(s: str) -> tuple[str, int]:
            base = s.rstrip('0123456789')
            index = s[len(base):]
            n = int(index) if index else -1
            return base, n

        new_vars = [str(g) for g in self.sage_ring.gens()]
        have_appended = False
        for v in vars_:
            if v not in new_vars:
                have_appended = True
                new_vars.append(v)
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


class VariableSet(firstorder.atomic.VariableSet['Variable']):
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
                self.polynomial_ring.ensure_vars((index,))
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
        self.polynomial_ring.pop()

    def push(self) -> None:
        self.polynomial_ring.push()


VV = VariableSet()
"""
The unique instance of :class:`.VariableSet`.
"""


class TSQ(Enum):
    """Information whether a certain term is has a *trivial square sum*
    property.

    .. seealso:: :meth:`.Term.is_definite`
    """
    NONE = auto()
    """None of the other cases holds..
    """

    STRICT = auto()
    """All other integer coefficients, including the absolute summand, are
    strictly positive. All exponents are even. This is sufficient for a term to
    be positive definite, i.e., positive for all real choices of variables.
    """

    WEAK = auto()
    """The absolute summand is zero. All other integer coefficients are
    strictly positive. All exponents are even. This is sufficient for a term to
    be positive semi-definite, i.e., non-negative for all real choices of
    variables.
    """


class Term(firstorder.Term['Term', 'Variable', int]):

    polynomial_ring: ClassVar[_PolynomialRing] = polynomial_ring

    _poly: MPolynomial[Rational]

    # The property should be private. We might want a method to_sage()
    @property
    def poly(self) -> MPolynomial[Rational]:
        """
        An instance of :class:`MPolynomial_libsingular
        <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular>`,
        which is wrapped by ``self``.
        """
        if self.polynomial_ring.sage_ring is not self._poly.parent():
            self.polynomial_ring.ensure_vars(str(g) for g in self._poly.parent().gens())
            self._poly = self.polynomial_ring(self._poly)
        return self._poly

    @poly.setter
    def poly(self, value: MPolynomial[Rational]):
        # self._poly = value  # old
        self._poly = self.polynomial_ring(value)  # new

    def __add__(self, other: object) -> Term:
        if isinstance(other, Term):
            return Term(self.poly + other.poly)
        return Term(self.poly + other)

    def __eq__(self, other: Term | int) -> Eq:  # type: ignore[override]
        # MyPy requires "other: object". However, with our use a a constructor,
        # it makes no sense to compare terms with general objects. We have
        # Eq.__bool__, which supports some comparisons in boolean contexts.
        # Same for __ne__.
        lhs = self - (other if isinstance(other, Term) else Term(other))
        if lhs.lc() < 0:
            lhs = - lhs
        return Eq(lhs, Term(0))

    def __ge__(self, other: Term | int) -> Ge | Le:
        lhs = self - (other if isinstance(other, Term) else Term(other))
        if lhs.lc() < 0:
            return Le(-lhs, 0)
        return Ge(lhs, 0)

    def __gt__(self, other: Term | int) -> Gt | Lt:
        lhs = self - (other if isinstance(other, Term) else Term(other))
        if lhs.lc() < 0:
            return Lt(- lhs, 0)
        return Gt(lhs, 0)

    def __hash__(self) -> int:
        return hash(self.poly)

    def __init__(self, arg: Fraction | Integer | int | MPolynomial[Rational]
                 | mpq | Rational | UPolynomial) -> None:
        if isinstance(arg, MPolynomial):
            self.poly = arg
        elif isinstance(arg, (Fraction, Integer, int, mpq, Rational, UPolynomial)):
            self.poly = self.polynomial_ring(arg)
        else:
            raise ValueError(
                f'arguments must be polynomial, integer, or rational; {arg} is {type(arg)}')

    def __iter__(self) -> Iterator[tuple[mpq, Term]]:
        """Iterate over the polynomial representation of the term, yielding
        pairs of coefficients and power products.

        >>> from logic1.theories.RCF import VV
        >>> x, y = VV.get('x', 'y')
        >>> t = (x - y + 2) ** 2
        >>> [(abs(coef), power_product) for coef, power_product in t]
        [(mpq(1,1), x^2), (mpq(2,1), x*y), (mpq(1,1), y^2), (mpq(4,1), x),
         (mpq(4,1), y), (mpq(4,1), 1)]
        """
        for coefficient, power_product in self.poly:
            yield mpq(coefficient), Term(power_product)

    def __le__(self, other: Term | int | mpq) -> Ge | Le:
        lhs = self - (other if isinstance(other, Term) else Term(other))
        if lhs.lc() < 0:
            return Ge(-lhs, 0)
        return Le(lhs, 0)

    def __lt__(self, other: Term | int | mpq) -> Gt | Lt:
        lhs = self - (other if isinstance(other, Term) else Term(other))
        if lhs.lc() < 0:
            return Gt(-lhs, 0)
        return Lt(lhs, 0)

    def __mul__(self, other: object) -> Term:
        if isinstance(other, Term):
            return Term(self.poly * other.poly)
        return Term(self.poly * other)

    def __ne__(  # type: ignore[override]
            self, other: Term | int | mpq) -> Ne:
        lhs = self - (other if isinstance(other, Term) else Term(other))
        if lhs.lc() < 0:
            lhs = -lhs
        return Ne(lhs, Term(0))

    def __neg__(self) -> Term:
        return Term(-self.poly)

    def __pow__(self, other: object) -> Term:
        return Term(self.poly ** other)

    def __repr__(self) -> str:
        return str(self.poly)

    def __radd__(self, other: object) -> Term:
        # We know that other is not a :class:`Term`, see :meth:`__add__`.
        return Term(other + self.poly)

    def __rmul__(self, other: object) -> Term:
        # We know that other is not a :class:`Term`, see :meth:`__mul__`.
        return Term(other * self.poly)

    def __rsub__(self, other: object) -> Term:
        # We know that other is not a :class:`Term`, see :meth:`__sub__`.
        return Term(other - self.poly)

    def __sub__(self, other: object) -> Term:
        if isinstance(other, Term):
            return self + (-other)
        return Term(self.poly - other)

    def __truediv__(self, other: object) -> Term:
        # x*y / x yields y as a Sage rational function and throws here.
        if isinstance(other, Term):
            return Term(self.poly / other.poly)
        return Term(self.poly / other)

    def __xor__(self, other: object) -> Term:
        raise NotImplementedError(
            "Use ** for exponentiation, not '^', which means xor "
            "in Python, and has the wrong precedence")

    def as_fraction(self) -> mpq:
        if not self.is_constant():
            raise ValueError(f'{self} is not constant')
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
        return Term(self.poly.derivative(x.poly, n))

    # discuss bug:
    # >>> (2*x).factor()
    def factor(self) -> tuple[Term, Term, dict[Term, int]]:
        """Return the factorization of this term.

        >>> from logic1.theories.RCF import VV
        >>> x, y = VV.get('x', 'y')
        >>> t = x**2 - y**2
        >>> t.factor()
        (1, 1, {x - y: 1, x + y: 1})

        .. seealso::
            :external:meth:`MPolynomial_libsingular.factor()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.factor>`
        """
        F = self.poly.factor()
        unit = Term(F.unit())
        content = Term(1)
        assert unit in (-1, 1), (self, F, unit)
        D = dict()
        for poly, multiplicity in F:
            term = Term(poly)
            if term.lc() < 0:
                term = -term
                unit = -unit
            if term.is_constant():
                content *= term
            else:
                D[term] = multiplicity
        return unit, content, D

    def is_constant(self) -> bool:
        """Return :obj:`True` if this term is constant.

        .. seealso::
            :external:meth:`MPolynomial_libsingular.is_constant()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.is_constant>`
        """
        return self.poly.is_constant()

    def is_definite(self) -> TSQ:
        """A fast heuristic test whether this term is positive or non-negative
        for all real choices of variables.

        >>> from logic1.theories.RCF import VV
        >>> x, y = VV.get('x', 'y')
        >>> f = x**2 + y**2
        >>> f.is_definite()
        <TSQ.WEAK: 3>
        >>> g = x**2 + y**2 + 1
        >>> g.is_definite()
        <TSQ.STRICT: 2>
        >>> h = (x + y) ** 2
        >>> h.is_definite()
        <TSQ.NONE: 1>
        """
        for exponent, coefficient in self.poly.dict().items():
            if coefficient < 0:
                return TSQ.NONE
            for e in exponent:
                if e % 2 == 1:
                    return TSQ.NONE
        if self.poly.constant_coefficient() == 0:
            return TSQ.WEAK
        return TSQ.STRICT

    def is_monomial(self) -> bool:
        """Return :obj:`True` if this term is a monomial.
        """
        return self.poly.is_monomial()

    def is_variable(self) -> bool:
        """Return :obj:`True` if this term is a variable.
        """
        return self.poly.is_generator()

    def is_zero(self) -> bool:
        """Return :obj:`True` if this term is a zero.

        .. seealso::
            :external:meth:`MPolynomial_libsingular.is_zero()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.is_zero>`
        """
        return self.poly.is_zero()

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
        [x^2, x*y, y^2, x, y, 1]

        .. seealso::
            :external:meth:`MPolynomial_libsingular.monomials()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.monomials>`
        """
        return [Term(monomial) for monomial in self.poly.monomials()]

    def pseudo_quo_rem(self, other: Term, x: Variable) -> tuple[Term, Term]:
        """Pseudo quotient and remainder of this term and other, both as
        univariate polynomials in `x` with polynomial coefficients in all other
        variables.

        >>> a, b, c, x = VV.get('a', 'b', 'c', 'x')
        >>> f = a * x**2 + b*x + c
        >>> g = c * x + b
        >>> q, r = f.pseudo_quo_rem(g, x); q, r
        (a*c*x - a*b + b*c, a*b^2 - b^2*c + c^3)
        >>> assert c**(2 - 1 + 1) * f == q * g + r

        .. seealso::
            :meth:`Polynomial.pseudo_quo_rem()
            <sage.rings.polynomial.polynomial_element.Polynomial.pseudo_quo_rem>`
        """
        self1 = self.poly.polynomial(x.poly)
        other1 = other.poly.polynomial(x.poly)
        quotient, remainder = self1.pseudo_quo_rem(other1)
        return Term(quotient), Term(remainder)

    def reduce(self, G: Iterable[Term]) -> Term:
        """Reduce self modulo G. The result (c, p) describes a polynomial q = c
        * p over Q. c >= 0 is the content of of q, and p is a polynomial over
        Z. If c == 0, then p == 0.
        """
        poly = self.poly.reduce([g.poly for g in G])
        return Term(poly)

    def quo_rem(self, other: Term) -> tuple[Term, Term]:
        """Quotient and remainder of this term and `other`.

        >>> from logic1.theories.RCF import VV
        >>> x, y = VV.get('x', 'y')
        >>> f = 2*y*x**2 + x + 1
        >>> f.quo_rem(x)
        (2*x*y + 1, 1)
        >>> f.quo_rem(y)
        (2*x^2, x + 1)
        >>> f.quo_rem(3*x)  # would yield (0, 2*x^2*y + x + 1) over ZZ
        (2/3*x*y + 1/3, 1)

        .. seealso::
            :external:meth:`MPolynomial_libsingular.quo_rem()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.quo_rem>`
        """
        quo, rem = self.poly.quo_rem(other.poly)
        return Term(quo), Term(rem)

    @staticmethod
    def sort_key(term: Term) -> MPolynomial[Rational]:
        """A sort key suitable for ordering instances of this class. ImplementTerm(remainder)s
        the abstract method :meth:`.firstorder.atomic.Term.sort_key`.
        """
        return term.poly

    def subs(self, d: Mapping[Variable, Term | int | mpq]) -> Term:
        """Simultaneous substitution of terms for variables.

        >>> from logic1.theories.RCF import VV
        >>> x, y, z = VV.get('x', 'y', 'z')
        >>> f = 2*y*x**2 + x + 1
        >>> f.subs({x: y, y: 2*z})
        4*y^2*z + y + 1

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
        return Term(self.poly.subs(**sage_keywords))

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
class Variable(Term, firstorder.Variable['Variable', int]):

    VV: ClassVar[VariableSet] = VV

    def fresh(self) -> Variable:
        """Returns a variable that has not been used so far. Implements
        abstract method :meth:`.firstorder.atomic.Variable.fresh`.
        """
        return self.VV.fresh(suffix=f'_{str(self)}')


class AtomicFormula(firstorder.AtomicFormula['AtomicFormula', 'Term', 'Variable', int]):

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
        """In boolean contexts atomic formulas are evaluated via corresponding
        comparisons with respect to the degree lexicographical term order
        :mod:`deglex <sage.rings.polynomial.term_order>`. In particular,
        comparisons between terms representing integers follow the natural
        order.
        """
        match self:
            case Eq():
                return self.lhs.poly == self.rhs.poly
            case Ne():
                return self.lhs.poly != self.rhs.poly
            case Ge():
                return self.lhs.poly >= self.rhs.poly
            case Gt():
                return self.lhs.poly > self.rhs.poly
            case Le():
                return self.lhs.poly <= self.rhs.poly
            case Lt():
                return self.lhs.poly < self.rhs.poly
            case _:
                assert False, self

    def __init__(self, lhs: Term | int, rhs: Term | int):
        if not isinstance(self, (Eq, Ne, Ge, Gt, Le, Lt)):
            raise NoTraceException('Instantiate one of Eq, Ne, Ge, Gt, Le, Lt instead')
        if not isinstance(lhs, Term):
            lhs = Term(lhs)
        if not isinstance(rhs, Term):
            rhs = Term(rhs)
        self.args = (lhs, rhs)

    def __le__(self, other: Formula) -> bool:
        """Returns `True` if this atomic formula should be sorted before or is
        equal to other. Implements abstract method
        :meth:`.firstorder.atomic.AtomicFormula.__le__`.
        """
        match other:
            case AtomicFormula():
                if self.lhs != other.lhs:
                    return Term.sort_key(self.lhs) <= Term.sort_key(other.lhs)
                if self.rhs != other.rhs:
                    return Term.sort_key(self.rhs) <= Term.sort_key(other.rhs)
                L = [Eq, Ne, Le, Lt, Ge, Gt]
                return L.index(self.op) <= L.index(other.op)
            case _:
                return True

    def __repr__(self) -> str:
        if self.lhs.poly.is_constant() and self.rhs.poly.is_constant():
            # Return Eq(1, 2) instead of 1 == 2, because the latter is not
            # suitable as input.
            return super().__repr__()
        return str(self)

    def __str__(self) -> str:
        """String representation of this atomic formula. Implements the
        abstract method :meth:`.firstorder.atomic.AtomicFormula.__str__`.
        """
        SYMBOL: Final = {Eq: '==', Ne: '!=', Ge: '>=', Le: '<=', Gt: '>', Lt: '<'}
        SPACING: Final = ' '
        return f'{self.lhs.poly}{SPACING}{SYMBOL[self.op]}{SPACING}{self.rhs.poly}'

    def as_latex(self) -> str:
        """Latex representation as a string. Implements the abstract method
        :meth:`.firstorder.atomic.AtomicFormula.as_latex`.
        """
        SYMBOL: Final = {
            Eq: '=', Ne: '\\neq', Ge: '\\geq', Le: '\\leq', Gt: '>', Lt: '<'}
        SPACING: Final = ' '
        return f'{self.lhs.as_latex()}{SPACING}{SYMBOL[self.op]}{SPACING}{self.rhs.as_latex()}'

    def as_redlog(self) -> str:
        """Latex representation as a string. Implements the abstract method
        :meth:`.firstorder.atomic.AtomicFormula.as_latex`.
        """
        SYMBOL: Final = {
            Eq: '=', Ne: '<>', Ge: '>=', Le: '<=', Gt: '>', Lt: '<'}
        return f'({self.lhs!r} {SYMBOL[self.op]} {self.rhs!r})'

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
        D: Any = {Eq: Ne, Ne: Eq, Le: Gt, Lt: Ge, Ge: Lt, Gt: Le}
        return D[cls]

    @classmethod
    def converse(cls) -> type[AtomicFormula]:
        """Converse relation.
        """
        D: Any = {Eq: Eq, Ne: Ne, Le: Ge, Lt: Gt, Ge: Le, Gt: Lt}
        return D[cls]

    @classmethod
    def dual(cls) -> type[AtomicFormula]:
        """Dual relation.
        """
        return cls.complement().converse()

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
        lhs = self.lhs - self.rhs
        if lhs.is_constant():
            return _T() if self.op(lhs, 0) else _F()
        if lhs.lc() < 0:
            return self.op.converse()(-lhs, 0)
        return self.op(lhs, 0)

    @classmethod
    def strict_part(cls) -> type[Gt | Lt]:
        """The strict part of a binary relation is the relation without the
        diagonal. Raises :exc:`NotImplementedError` for :class:`Eq` and
        :class:`Ne`.
        """
        if cls in (Eq, Ne):
            raise NotImplementedError()
        D: Any = {Le: Lt, Lt: Lt, Ge: Gt, Gt: Gt}
        return D[cls]

    def subs(self, sigma: Mapping[Variable, Term | int | mpq]) -> Self:
        """Formal simultaneous term substitution into the two argument terms of
        the atomic formula. Implements the abstract method
        :meth:`.firstorder.atomic.AtomicFormula.subs`.
        """
        return self.op(self.lhs.subs(sigma), self.rhs.subs(sigma))


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
