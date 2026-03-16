from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Final, Optional, Self

import operator

from logic1 import firstorder
from logic1.firstorder.boolean import And, Or
from logic1.theories.Complex.types import Formula, Number
from logic1.theories.Complex.term import Im, Rational, Re, Term, Variable
from gmpy2 import mpq

class AtomicFormula(firstorder.AtomicFormula['AtomicFormula', Term, Variable, Number]):

    _hash: Optional[int] = None  # TODO: understand why we need this

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
        OPS = {
            Eq: operator.eq,
            Ne: operator.ne,
            Ge: operator.ge,
            Gt: operator.gt,
            Le: operator.le,
            Lt: operator.lt
        }
        return OPS[self.op](self.lhs.sort_key(), self.rhs.sort_key())


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
    
    def as_real_formula(self) -> Formula:
        """Returns an equivalent formula where all terms are real.
        """
        a = Re(self.lhs - self.rhs).normalize()
        b = Im(self.lhs - self.rhs).normalize()
        if isinstance(self, Eq):
            return And(a == 0, b == 0)
        if isinstance(self, Ne):
            return Or(a != 0, b != 0)
        if isinstance(self, Ge):
            return And(a >= 0, b == 0)
        if isinstance(self, Le):
            return And(a <= 0, b == 0)
        if isinstance(self, Gt):
            return And(a > 0, b == 0)
        if isinstance(self, Lt):
            return And(a < 0, b == 0)
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
          Inherited method :meth:`.firstorder.atomic.AtomicFormula.to_complement`
          """
        return {Eq: Ne, Ne: Eq, Le: Gt, Lt: Ge, Ge: Lt, Gt: Le}[cls]
    
    def _dump(self) -> str:
        return f'{self.op.__name__}({self.lhs._dump()}, {self.rhs._dump()})'

    def _eval_constant(self) -> bool:
        a, b = (self.lhs - self.rhs).eval_constant()
        if isinstance(self, Eq):
            return a == 0 and b == 0
        if isinstance(self, Ne):
            return a != 0 or b != 0
        if isinstance(self, Ge):
            return a >= 0 and b == 0
        if isinstance(self, Le):
            return a <= 0 and b == 0
        if isinstance(self, Gt):
            return a > 0 and b == 0
        if isinstance(self, Lt):
            return a < 0 and b == 0
        return False

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

    def normalize_weak(self) -> Self:
        lhs = (self.lhs - self.rhs).normalize_weak()
        if isinstance(self, (Eq, Ne)):
            lhs = (lhs / lhs.lc()).normalize_weak()
        return self.op(lhs, 0)
    
    def normalize(self) -> Self:
        lhs = (self.lhs - self.rhs).normalize()
        if isinstance(self, (Eq, Ne)):
            lhs = (lhs / lhs.lc()).normalize()
        return self.op(lhs, 0)
    
    def normalize_complex(self) -> Self:
        lhs = (self.lhs - self.rhs).normalize_complex()
        a, b = lhs.lc().eval_constant()
        if a != mpq(0) or b != mpq(0):
            if isinstance(self, (Eq, Ne)):
                lhs = (lhs / lhs.lc()).normalize_complex()
            elif a == mpq(0):
                lhs = (lhs / Rational(b)).normalize_complex()
            else:
                lhs = (lhs / Rational(a)).normalize_complex()
        else:
            lhs = Rational(0)
        c, d = lhs.lc().eval_constant()
        # assert c == mpq(1), ((self.lhs - self.rhs).normalize_complex(), (self.lhs - self.rhs).normalize_complex()._dump(), (self.lhs - self.rhs).normalize_complex().lc(), (a, b, c, d), lhs, self._dump(), lhs._dump())
        return self.op(lhs, 0)
    
    def is_imaginary(self) -> bool:
        return self.lhs.is_imaginary() and self.rhs.is_imaginary()
    
    def is_real(self) -> bool:
        return self.lhs.is_real() and self.rhs.is_real()

    def simplify(self) -> Formula:
        """Fast basic simplification. The result is equivalent to self.
        Implements the abstract method :meth:`.firstorder.atomic.AtomicFormula.simplify`.
        """
        result = self.normalize_weak()
        try:
            return firstorder._T() if result._eval_constant() else firstorder._F()
        except ValueError:
            return result
        
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


class Ge(AtomicFormula):
    pass


class Le(AtomicFormula):
    pass


class Gt(AtomicFormula):
    pass


class Lt(AtomicFormula):
    pass