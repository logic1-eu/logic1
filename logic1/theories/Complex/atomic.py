from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Self

import operator

from logic1 import firstorder
from logic1.firstorder.boolean import And, _F, Or, _T
from logic1.theories.Complex.types import Formula, Number
from logic1.theories.Complex.term import Im, Re, Term, Variable

from gmpy2 import mpq


class AtomicFormula(
        firstorder.AtomicFormula['AtomicFormula', Term, Variable, Number]):

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

    def __bool__(self) -> bool:  # TODO: discuss
        """Compares the sort keys of the two sides of the atomic formula
        using the operator of the formula.
        """
        ops = {Eq: operator.eq, Ne: operator.ne, Le: operator.le,
               Lt: operator.lt, Ge: operator.ge, Gt: operator.gt}
        return ops[self.op](self.lhs.sort_key(), self.rhs.sort_key())

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

    def __init__(self, lhs: Number | Term, rhs: Number | Term) -> None:
        super().__init__(self, lhs, rhs)
        self.args = (
            lhs if isinstance(lhs, Term) else Term(lhs),
            rhs if isinstance(rhs, Term) else Term(rhs)
        )

    def __le__(self, other: Formula) -> bool:
        """Returns `True` if this atomic formula should be sorted before or is
        equal to other. Implements abstract method
        :meth:`.firstorder.atomic.AtomicFormula.__le__`.
        """
        if not isinstance(other, AtomicFormula):
            return True
        self_key = self.lhs.sort_key()
        other_key = other.lhs.sort_key()
        if self_key != other_key:
            return self_key <= other_key
        self_key = self.rhs.sort_key()
        other_key = other.rhs.sort_key()
        if self_key != other_key:
            return self_key <= other_key
        ORDER = [Eq, Ne, Le, Lt, Ge, Gt]
        return ORDER.index(self.op) <= ORDER.index(other.op)

    def __repr__(self) -> str:
        symbols = {Eq: '=', Ne: '!=', Le: '<=', Lt: '<', Ge: '>=', Gt: '>'}
        return f'{repr(self.lhs)} {symbols[self.op]} {repr(self.rhs)}'

    def __str__(self) -> str:
        """String representation of this atomic formula. Implements the
        abstract method :meth:`.firstorder.atomic.AtomicFormula.__str__`.
        """
        symbols = {Eq: '=', Ne: '!=', Le: '<=', Lt: '<', Ge: '>=', Gt: '>'}
        return f'{str(self.lhs)} {symbols[self.op]} {str(self.rhs)}'

    def as_latex(self) -> str:
        """Latex representation as a string. Implements the abstract method
        :meth:`.firstorder.atomic.AtomicFormula.as_latex`.
        """
        symbols = {
            Eq: '=', Ne: '\\neq', Le: '\\leq', Lt: '<', Ge: '\\geq', Gt: '>'
        }
        return f'{self.lhs.as_latex()} {symbols[self.op]} {self.rhs.as_latex()}'

    def as_real_formula(self) -> Formula:
        """Returns an equivalent formula where all terms are real.
        """
        if isinstance(self, Eq):
            return And(Re(self.lhs) == Re(self.rhs), Im(self.lhs) == Im(self.rhs))
        if isinstance(self, Ne):
            return Or(Re(self.lhs) != Re(self.rhs), Im(self.lhs) != Im(self.rhs))
        if isinstance(self, Ge):
            return And(Re(self.lhs) >= Re(self.rhs), Im(self.lhs) == 0, Im(self.rhs) == 0)
        if isinstance(self, Le):
            return And(Re(self.lhs) <= Re(self.rhs), Im(self.lhs) == 0, Im(self.rhs) == 0)
        if isinstance(self, Gt):
            return And(Re(self.lhs) > Re(self.rhs), Im(self.lhs) == 0, Im(self.rhs) == 0)
        if isinstance(self, Lt):
            return And(Re(self.lhs) < Re(self.rhs), Im(self.lhs) == 0, Im(self.rhs) == 0)
        assert False, type(self)

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
            Inherited method
            :meth:`.firstorder.atomic.AtomicFormula.to_complement`
        """
        return {Eq: Ne, Ne: Eq, Le: Gt, Lt: Ge, Ge: Lt, Gt: Le}[cls]

    @classmethod
    def converse(cls) -> type[AtomicFormula]:
        """Converse relation.
        """
        return {Eq: Eq, Ne: Ne, Le: Ge, Lt: Gt, Ge: Le, Gt: Lt}[cls]

    def eval(self) -> bool:
        """Evaluates an atomic formula where both sides are constants.
        Returns `True` if the formula is true, `False` if the formula is false,
        and raises `ValueError` if the formula contains variables.

        >>> from logic1.theories.Complex import *
        >>> (2 * I == 0).eval()
        False
        >>> (I**2 < 0).eval()
        True
        >>> x = VV['x']
        >>> (x == 0).eval()
        Traceback (most recent call last):
        ...
        ValueError: Cannot evaluate variable x
        """
        (lhs_re, lhs_im) = self.lhs.eval()
        (rhs_re, rhs_im) = self.rhs.eval()
        if isinstance(self, Eq):
            return lhs_re == rhs_re and lhs_im == rhs_im
        if isinstance(self, Ne):
            return lhs_re != rhs_re or lhs_im != rhs_im
        if isinstance(self, Ge):
            return lhs_re >= rhs_re and lhs_im == 0 and rhs_im == 0
        if isinstance(self, Le):
            return lhs_re <= rhs_re and lhs_im == 0 and rhs_im == 0
        if isinstance(self, Gt):
            return lhs_re > rhs_re and lhs_im == 0 and rhs_im == 0
        if isinstance(self, Lt):
            return lhs_re < rhs_re and lhs_im == 0 and rhs_im == 0
        assert False, type(self)

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

    def is_imaginary(self) -> bool:
        """Returns `True` if both sides of this atomic formula are imaginary.
        """
        return self.lhs.is_imaginary() and self.rhs.is_imaginary()

    def is_real(self) -> bool:
        """Returns `True` if both sides of this atomic formula are real."""
        return self.lhs.is_real() and self.rhs.is_real()

    def simplify(self) -> AtomicFormula | _T | _F:
        """Fast basic simplification. The result is equivalent to self.
        Implements the abstract method
        :meth:`.firstorder.atomic.AtomicFormula.simplify`.
        """
        try:
            return firstorder._T() if self.eval() else firstorder._F()
        except ValueError:
            pass
        lhs = self.lhs - self.rhs
        a, b = lhs.lc().eval()
        if isinstance(self, (Eq, Ne)):
            if (a, b) != (mpq(0), mpq(0)):
                lhs = (lhs / Term.from_real_imag(a, b))
            return self.op(lhs, 0)
        if isinstance(self, (Le, Lt, Ge, Gt)):
            if a == mpq(0) and b != mpq(0):
                c = b
            elif a != mpq(0):
                c = a
            else:
                return self.op(lhs, 0)
            lhs = (lhs / Term.from_real_imag(c, 0))
            if c < mpq(0):
                return self.op.converse()(lhs, 0)
            else:
                return self.op(lhs, 0)
        assert False, type(self)

    def subs(self, sigma: Mapping[Variable, Number | Term]) -> Self:
        """Formal simultaneous term substitution into the two argument terms of
        the atomic formula. Implements the abstract method
        :meth:`.firstorder.atomic.AtomicFormula.subs`.
        """
        return self.op(self.lhs.subs(sigma), self.rhs.subs(sigma))



class Eq(AtomicFormula):
    pass


class Ne(AtomicFormula):
    pass


class RealAtomicFormula(AtomicFormula):

    def __init__(self, lhs: Number | Term, rhs: Number | Term):
        super().__init__(lhs, rhs)
        if not self.is_real():
            raise ValueError(f'Cannot create atomic formula {self} because it is not real')


class Ge(RealAtomicFormula):
    pass


class Le(RealAtomicFormula):
    pass


class Gt(RealAtomicFormula):
    pass


class Lt(RealAtomicFormula):
    pass


from logic1.theories.Complex.normalize import ComplexNormalizer, ConstantEvaluator, Normalizer, WeakNormalizer
from logic1.theories.Complex.format import LatexFormatter, ReprFormatter, StrFormatter