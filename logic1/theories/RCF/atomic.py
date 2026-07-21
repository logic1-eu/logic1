from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Iterator, Mapping, Self

from gmpy2 import mpq

import logic1
from logic1.firstorder import _T, _F
from logic1.support.excepthook import NoTraceException

if TYPE_CHECKING:
    from logic1.theories.RCF.types import Formula


class AtomicFormula(logic1.firstorder.AtomicFormula['logic1.theories.RCF.atomic.AtomicFormula',
                                                    'logic1.theories.RCF.term.Term',
                                                    'logic1.theories.RCF.term.Variable',
                                                    int]):

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
        comparisons with respect to the degree lexicographical term order. In
        particular, comparisons between terms representing integers follow the
        natural order.
        """
        match self:
            case Eq():
                return self.lhs.sort_key() == self.rhs.sort_key()
            case Ne():
                return self.lhs.sort_key() != self.rhs.sort_key()
            case Ge():
                return self.lhs.sort_key() >= self.rhs.sort_key()
            case Gt():
                return self.lhs.sort_key() > self.rhs.sort_key()
            case Le():
                return self.lhs.sort_key() <= self.rhs.sort_key()
            case Lt():
                return self.lhs.sort_key() < self.rhs.sort_key()
            case _:
                assert False, self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AtomicFormula):
            return False
        if self.op != other.op:
            return False
        if self.lhs.sort_key() != other.lhs.sort_key():
            return False
        if self.rhs.sort_key() != other.rhs.sort_key():
            return False
        return True

    def __hash__(self) -> int:
        return super().__hash__()

    def __init__(self, lhs: Term | int, rhs: Term | int):
        super().__init__()
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
        if not isinstance(other, AtomicFormula):
            return True
        self_sort_key = self.lhs.sort_key()
        other_sort_key = other.lhs.sort_key()
        if self_sort_key != other_sort_key:
            return self_sort_key <= other_sort_key
        self_sort_key = self.rhs.sort_key()
        other_sort_key = other.rhs.sort_key()
        if self_sort_key != other_sort_key:
            return self_sort_key <= other_sort_key
        L = [Eq, Ne, Le, Lt, Ge, Gt]
        return L.index(self.op) <= L.index(other.op)

    def __repr__(self) -> str:
        if self.lhs.is_constant() and self.rhs.is_constant():
            # Return Eq(1, 2) instead of 1 == 2, because the latter is not
            # suitable as input.
            return super().__repr__()
        SYMBOL: Final = {Eq: '==', Ne: '!=', Ge: '>=', Le: '<=', Gt: '>', Lt: '<'}
        SPACING: Final = ' '
        return f'{self.lhs!r}{SPACING}{SYMBOL[self.op]}{SPACING}{self.rhs!r}'


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

    # def subsq(self, sigma: Mapping[Variable, tuple[Term | int | mpq, Term | int | mpq]],
    #           is_positive: bool = False) -> Self:

    #     def cast(x: Term | int | mpq) -> MPolynomial:
    #         if isinstance(x, Term):
    #             return x.poly
    #         else:
    #             return ring(x)

    #     def subs1(p: MPolynomial, d: dict) -> Any:
    #         return p.subs(**d)  # type: ignore

    #     ring = polynomial_ring.sage_ring
    #     FF = FractionField(ring)
    #     d = {str(x): FF(cast(num), cast(den)) for x, (num, den) in sigma.items()}
    #     lhq = subs1(ring(self.lhs.poly), d)
    #     rhq = subs1(ring(self.rhs.poly), d)
    #     if is_positive or isinstance(self, (Eq, Ne)):
    #         lhp = lhq.numerator()
    #         rhp = rhq.numerator()
    #     else:
    #         assert isinstance(self, (Le, Lt, Ge, Gt))
    #         lhp = (lhq * lhq.denominator() ** 2).numerator()
    #         rhp = (rhq * rhq.denominator() ** 2).numerator()
    #     assert lhp.parent() in (ring, QQ)
    #     assert rhp.parent() in (ring, QQ)
    #     return self.op(Term(lhp), Term(rhp))

    def subsq(self, sigma: Mapping[Variable, tuple[Term | int | mpq, Term | int | mpq]], is_positive: bool = False) -> Self:
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


from logic1.theories.RCF.term import Term, Variable