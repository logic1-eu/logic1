from __future__ import annotations

from enum import auto, Enum
import operator
import sage.all as sage  # type: ignore[import-untyped]
from sage.rings.polynomial.multi_polynomial_libsingular import (  # type: ignore[import-untyped]
    MPolynomial_libsingular as Polynomial)
from sage.rings.polynomial.polynomial_element import (  # type: ignore[import-untyped]
    Polynomial_generic_dense as UnivariatePolynomial)
from types import BuiltinFunctionType
from typing import Any, ClassVar, Final, Iterable, Iterator, Self

from ... import firstorder
from ...firstorder import _T, _F

from ...support.tracing import trace  # noqa


class PolynomialRing:

    sage_ring: sage.PolynomialRing

    def __call__(self, obj):
        return self.sage_ring(obj)

    def __init__(self, term_order='deglex'):
        self.sage_ring = sage.PolynomialRing(
            sage.ZZ, 'unused_', implementation='singular', order=term_order)
        self.stack = []

    def __repr__(self):
        return str(self.sage_ring)

    def add_var(self, var: str) -> None:
        new_vars = [str(g) for g in self.sage_ring.gens()]
        assert var not in new_vars
        new_vars.append(var)
        new_vars.sort()
        self.sage_ring = sage.PolynomialRing(
            sage.ZZ, new_vars, implementation='singular', order=self.sage_ring.term_order())

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
            self.sage_ring = sage.PolynomialRing(
                sage.ZZ, new_vars, implementation='singular', order=self.sage_ring.term_order())

    def get_vars(self) -> tuple[Polynomial, ...]:
        gens = self.sage_ring.gens()
        gens = (g for g in gens if str(g) != 'unused_')
        return tuple(gens)

    def pop(self) -> None:
        self.sage_ring = self.stack.pop()

    def push(self) -> None:
        self.stack.append(self.sage_ring)
        self.sage_ring = sage.PolynomialRing(
            sage.ZZ, 'unused_', implementation='singular', order=self.sage_ring.term_order())


polynomial_ring = PolynomialRing()


class VariableSet(firstorder.atomic.VariableSet['Variable']):

    polynomial_ring: ClassVar[PolynomialRing] = polynomial_ring

    @property
    def stack(self) -> list[sage.PolynomialRing]:
        return self.polynomial_ring.stack

    def __getitem__(self, index: str) -> Variable:
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


class TSQ(Enum):
    NONE = auto()
    STRICT = auto()
    WEAK = auto()


class Term(firstorder.Term['Term', 'Variable']):

    polynomial_ring: ClassVar[PolynomialRing] = polynomial_ring

    _poly: Polynomial

    @property
    def poly(self) -> Polynomial:
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
    def poly(self, value: Polynomial):
        self._poly = value

    def __add__(self, other: object) -> Term:
        if isinstance(other, Term):
            return Term(self.poly + other.poly)
        return Term(self.poly + other)

    def __eq__(  # type: ignore[override]
            self, other: Term | Polynomial | sage.Integer | int) -> Eq:
        # MyPy requires "other: object". However, with our use a a constructor,
        # it makes no sense to compare terms with general objects. We have
        # Eq.__bool__, which supports some comparisons in boolean contexts.
        # Same for __ne__.
        lhs = self - (other if isinstance(other, Term) else Term(other))
        if lhs.lc() < 0:
            lhs = - lhs
        return Eq(lhs, Term(0))

    def __ge__(self, other: Term | Polynomial | sage.Integer | int) -> Ge | Le:
        lhs = self - (other if isinstance(other, Term) else Term(other))
        if lhs.lc() < 0:
            return Le(- lhs, 0)
        return Ge(lhs, 0)

    def __gt__(self, other: Term | Polynomial | sage.Integer | int) -> Gt | Lt:
        lhs = self - (other if isinstance(other, Term) else Term(other))
        if lhs.lc() < 0:
            return Lt(- lhs, 0)
        return Gt(lhs, 0)

    def __hash__(self) -> int:
        return hash((tuple(str(cls) for cls in self.__class__.mro()), self.poly))

    def __init__(self, arg: Polynomial | sage.Integer | int) -> None:
        match arg:
            case Polynomial():
                self.poly = arg
            case sage.Integer() | int() | UnivariatePolynomial():
                self.poly = self.polynomial_ring(arg)
            case _:
                raise ValueError(
                    f'arguments must be polynomial or integer; {arg} is {type(arg)}')

    def __iter__(self) -> Iterator[tuple[int, Term]]:
        for coefficient, power_product in self.poly:
            yield int(coefficient), Term(power_product)

    def __le__(self, other: Term | Polynomial | sage.Integer | int) -> Le | Ge:
        lhs = self - (other if isinstance(other, Term) else Term(other))
        if lhs.lc() < 0:
            return Ge(- lhs, 0)
        return Le(lhs, 0)

    def __lt__(self, other: Term | Polynomial | sage.Integer | int) -> Lt | Gt:
        lhs = self - (other if isinstance(other, Term) else Term(other))
        if lhs.lc() < 0:
            return Gt(- lhs, 0)
        return Lt(lhs, 0)

    def __mul__(self, other: object) -> Term:
        if isinstance(other, Term):
            return Term(self.poly * other.poly)
        return Term(self.poly * other)

    def __ne__(  # type: ignore[override]
            self, other: Term | Polynomial | sage.Integer | int) -> Ne:
        lhs = self - (other if isinstance(other, Term) else Term(other))
        if lhs.lc() < 0:
            lhs = - lhs
        return Ne(lhs, Term(0))

    def __neg__(self) -> Term:
        return Term(- self.poly)

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
            return self + (- other)
        return Term(self.poly - other)

    def __truediv__(self, other: object) -> Term:
        return Term(self.poly / other)

    def __xor__(self, other: object) -> Term:
        raise NotImplementedError(
            "Use ** for exponentiation, not '^', which means xor "
            "in Python, and has the wrong precedence")

    def as_latex(self) -> str:
        """Convert `self` to LaTeX.

        Implements the abstract method
        :meth:`.firstorder.atomic.Term.as_Latex`.
        """
        return str(sage.latex(self.poly))

    def coefficient(self, d: dict[Variable, int]) -> Term:
        """
        .. seealso::
            :external:meth:`MPolynomial_libsingular.coefficient()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.coefficient>`
        """
        d_poly = {key.poly: value for key, value in d.items()}
        return Term(self.poly.coefficient(d_poly))

    def constant_coefficient(self) -> int:
        return int(self.poly.constant_coefficient())

    def content(self) -> int:
        """
        .. seealso::
            :external:meth:`MPolynomial.content()
            <sage.rings.polynomial.multi_polynomial.MPolynomial.content>`
        """
        return int(self.poly.content())

    def degree(self, x: Variable) -> int:
        """
        .. seealso::
            :external:meth:`MPolynomial_libsingular.degree()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.degree>`
        """
        return self.poly.degree(x.poly)

    def derivative(self, x: Variable, n: int = 1) -> Term:
        """
        .. seealso::
            :external:meth:`MPolynomial.derivative()
            <sage.rings.polynomial.multi_polynomial.MPolynomial.derivative>`
        """
        return Term(self.poly.derivative(x.poly, n))

    def factor(self) -> tuple[Term, dict[Term, int]]:
        """
        .. seealso::
            :external:meth:`MPolynomial_libsingular.factor()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.factor>`
        """
        F = self.poly.factor()
        unit = Term(F.unit())
        assert unit in (-1, 1), (self, F, unit)
        D = dict()
        for poly, multiplicity in F:
            if poly.lc() < 0:
                poly = - poly
                unit = - unit
            D[Term(poly)] = multiplicity
        return unit, D

    def is_constant(self) -> bool:
        """
        .. seealso::
            :external:meth:`MPolynomial_libsingular.is_constant()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.is_constant>`
        """
        return self.poly.is_constant()

    def is_definite(self) -> TSQ:
        for exponent, coefficient in self.poly.dict().items():
            if coefficient < 0:
                return TSQ.NONE
            for e in exponent:
                if e % 2 == 1:
                    return TSQ.NONE
        if self.poly.constant_coefficient() == 0:
            return TSQ.WEAK
        return TSQ.STRICT

    def is_variable(self) -> bool:
        return self.poly in self.poly.parent().gens()

    def is_zero(self) -> bool:
        """
        .. seealso::
            :external:meth:`MPolynomial_libsingular.is_zero()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.is_zero>`
        """
        return self.poly.is_zero()

    def lc(self) -> int:
        """
        .. seealso::
            :external:meth:`MPolynomial_libsingular.lc()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.lc>`
        """
        return int(self.poly.lc())

    def monomials(self) -> list[Term]:
        """
        .. seealso::
            :external:meth:`MPolynomial_libsingular.monomials()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.monomials>`
        """
        return [Term(monomial) for monomial in self.poly.monomials()]

    def quo_rem(self, other: Term) -> tuple[Term, Term]:
        """
        .. seealso::
            :external:meth:`MPolynomial_libsingular.quo_rem()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.quo_rem>`
        """
        quo, rem = self.poly.quo_rem(other.poly)
        return Term(quo), Term(rem)

    def pseudo_quo_rem(self, other: Term, x: Variable):
        self1 = self.poly.polynomial(x.poly)
        other1 = other.poly.polynomial(x.poly)
        quotient, remainder = self1.pseudo_quo_rem(other1)
        return Term(quotient), Term(remainder)

    @staticmethod
    def sort_key(term: Term) -> Polynomial:
        return term.poly

    def subs(self, d: dict[Variable, Term]) -> Term:
        """
        .. seealso::
            :external:meth:`MPolynomial_libsingular.subs()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.subs>`
        """
        sage_keywords = {str(v.poly): t.poly for v, t in d.items()}
        return Term(self.poly.subs(**sage_keywords))

    def vars(self) -> Iterator[Variable]:
        """
        .. seealso::
            :external:meth:`MPolynomial_libsingular.variables()
            <sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular.variables>`
        """
        for g in self.poly.variables():
            yield Variable(g)


class Variable(Term, firstorder.Variable['Variable']):

    VV: ClassVar[VariableSet] = VV

    def fresh(self) -> Variable:
        """
        .. seealso::
            :meth:`logic1.firstorder.atomic.Term.fresh`
        """
        return self.VV.fresh(suffix=f'_{str(self)}')


class AtomicFormula(firstorder.AtomicFormula['AtomicFormula', 'Term', 'Variable']):
    """
    +--------------------+-------------+-------------+-------------+-------------+-------------+-------------+
    | :data:`self`       | :class:`Eq` | :class:`Ne` | :class:`Le` | :class:`Ge` | :class:`Lt` | :class:`Gt` |
    +--------------------+-------------+-------------+-------------+-------------+-------------+-------------+
    | :meth:`complement` | :class:`Ne` | :class:`Eq` | :class:`Gt` | :class:`Lt` | :class:`Ge` | :class:`Le` |
    +--------------------+-------------+-------------+-------------+-------------+-------------+-------------+
    | :meth:`converse`   | :class:`Eq` | :class:`Ne` | :class:`Ge` | :class:`Le` | :class:`Gt` | :class:`Lt` |
    +--------------------+-------------+-------------+-------------+-------------+-------------+-------------+
    | :meth:`dual`       | :class:`Ne` | :class:`Eq` | :class:`Lt` | :class:`Gt` | :class:`Le` | :class:`Ge` |
    +--------------------+-------------+-------------+-------------+-------------+-------------+-------------+
    """  # noqa
    @classmethod
    def complement(cls) -> type[AtomicFormula]:
        """Complement relation.
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

    @classmethod
    def python_operator(cls) -> BuiltinFunctionType:
        """The operator correponding to `cls` for evaluation of constant
        relations.
        """
        D: Any = {Eq: operator.eq, Ne: operator.ne,
                  Le: operator.le, Lt: operator.lt,
                  Ge: operator.ge, Gt: operator.gt}
        return D[cls]

    @classmethod
    def strict_part(cls) -> type[Formula]:
        """The strict part is the binary relation without the diagonal.
        """
        if cls in (Eq, Ne):
            raise NotImplementedError()
        D: Any = {Le: Lt, Lt: Lt, Ge: Gt, Gt: Gt}
        return D[cls]

    @property
    def lhs(self) -> Term:
        return self.args[0]

    @property
    def rhs(self) -> Term:
        return self.args[1]

    def __init__(self, lhs: Term | Polynomial | sage.Integer | int,
                 rhs: Term | Polynomial | sage.Integer | int):
        # sage.Integer is a candidate for removal
        assert not isinstance(lhs, sage.Integer)
        assert not isinstance(rhs, sage.Integer)
        if not isinstance(lhs, Term):
            lhs = Term(lhs)
        if not isinstance(rhs, Term):
            rhs = Term(rhs)
        self.args = (lhs, rhs)

    def __le__(self, other: Formula) -> bool:
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
        SYMBOL: Final = {Eq: '==', Ne: '!=', Ge: '>=', Le: '<=', Gt: '>', Lt: '<'}
        SPACING: Final = ' '
        return f'{self.lhs.poly}{SPACING}{SYMBOL[self.op]}{SPACING}{self.rhs.poly}'

    def as_latex(self) -> str:
        SYMBOL: Final = {
            Eq: '=', Ne: '\\neq', Ge: '\\geq', Le: '\\leq', Gt: '>', Lt: '<'}
        SPACING: Final = ' '
        return f'{self.lhs.as_latex()}{SPACING}{SYMBOL[self.op]}{SPACING}{self.rhs.as_latex()}'

    def bvars(self, quantified: frozenset[Variable] = frozenset()) -> Iterator[Variable]:
        for v in self.lhs.vars():
            if v in quantified:
                yield v
        for v in self.rhs.vars():
            if v in quantified:
                yield v

    def fvars(self, quantified: frozenset[Variable] = frozenset()) -> Iterator[Variable]:
        for v in self.lhs.vars():
            if v not in quantified:
                yield v
        for v in self.rhs.vars():
            if v not in quantified:
                yield v

    def subs(self, d: dict[Variable, Term]) -> Self:
        """Implements the abstract method :meth:`.firstorder.atomic.AtomicFormula.subs`.
        """
        return self.op(self.lhs.subs(d), self.rhs.subs(d))


class Eq(AtomicFormula):

    def __bool__(self) -> bool:
        return self.lhs.poly == self.rhs.poly

    def simplify(self) -> Formula:
        lhs = self.lhs.poly - self.rhs.poly
        if lhs.is_zero():
            return _T()
        if lhs.is_constant():
            return _F()
        return Eq(Term(lhs), Term(0))


class Ne(AtomicFormula):

    def __bool__(self) -> bool:
        return self.lhs.poly != self.rhs.poly

    def simplify(self) -> Formula:
        lhs = self.lhs.poly - self.rhs.poly
        if lhs.is_zero():
            return _F()
        if lhs.is_constant():
            return _T()
        return Ne(Term(lhs), Term(0))


class Ge(AtomicFormula):

    def __bool__(self) -> bool:
        return self.lhs.poly >= self.rhs.poly

    def simplify(self) -> Formula:
        lhs = self.lhs.poly - self.rhs.poly
        if lhs.is_constant():
            return _T() if lhs >= 0 else _F()
        return Ge(Term(lhs), Term(0))


class Le(AtomicFormula):

    def __bool__(self) -> bool:
        return self.lhs.poly <= self.rhs.poly

    def simplify(self) -> Formula:
        lhs = self.lhs.poly - self.rhs.poly
        if lhs.is_constant():
            return _T() if lhs <= 0 else _F()
        return Le(Term(lhs), Term(0))


class Gt(AtomicFormula):

    def __bool__(self) -> bool:
        return self.lhs.poly > self.rhs.poly

    def simplify(self) -> Formula:
        lhs = self.lhs.poly - self.rhs.poly
        if lhs.is_constant():
            return _T() if lhs > 0 else _F()
        return Gt(Term(lhs), Term(0))


class Lt(AtomicFormula):

    def __bool__(self) -> bool:
        return self.lhs.poly < self.rhs.poly

    def simplify(self) -> Formula:
        lhs = self.lhs.poly - self.rhs.poly
        if lhs.is_constant():
            return _T() if lhs < 0 else _F()
        return Lt(Term(lhs), Term(0))


from .typing import Formula


# from typing import reveal_type, TYPE_CHECKING

# if TYPE_CHECKING:

#     def test() -> None:
#         from .. import Sets
#         x, y = VV.get('x', 'y')
#         z1, z2 = Sets.VV.get('z1', 'z2')
#         reveal_type(x == 0)
#         f = firstorder.And(x == 0, z1 == z2)
#         reveal_type(f)
#         at = next(f.atoms())
#         reveal_type(at)
