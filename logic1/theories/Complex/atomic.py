from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator, Mapping
from typing import Final, Generic, Optional, Self, TypeVar

import operator

from logic1 import firstorder
from logic1.firstorder.boolean import And, Or
from logic1.theories.Complex.types import Formula, Number
from logic1.theories.Complex.term import Im, Re, Term, Variable
from gmpy2 import mpq

α = TypeVar('α')


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
    
    def __bool__(self) -> bool:  # TODO: discuss
        """Compares the sort keys of the two sides of the atomic formula
        using the operator of the formula.
        """
        ops = {Eq: operator.eq, Ne: operator.ne, Le: operator.le, Lt: operator.lt, Ge: operator.ge, Gt: operator.gt}
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
    
    def __hash__(self) -> int:  # TODO: understand why we need this
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
        return self.accept(ReprFormatter())

    def __str__(self) -> str:
        """String representation of this atomic formula. Implements the
        abstract method :meth:`.firstorder.atomic.AtomicFormula.__str__`.
        """
        return self.accept(StrFormatter())
    
    @abstractmethod
    def accept(self, visitor: AtomicFormulaVisitor[α]) -> α:
        ...

    def as_latex(self) -> str:
        """Latex representation as a string. Implements the abstract method
        :meth:`.firstorder.atomic.AtomicFormula.as_latex`.
        """
        return self.accept(LatexFormatter())
    
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
    
    def _dump(self) -> str:
        return f'{self.op.__name__}({self.lhs._dump()}, {self.rhs._dump()})'

    def eval(self, variables: dict[Variable, Number] = dict()) -> bool:
        """Evaluates the atomic formula under the given variable assignment.
        Raises a ValueError if this atomic formula contains any variables
        that are not in the given assignment.
        """
        return self.accept(ConstantEvaluator(variables))

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

    def normalize_weak(self) -> AtomicFormula:
        return self.accept(WeakNormalizer())
    
    def normalize(self) -> AtomicFormula:
        return self.accept(Normalizer())
    
    def normalize_complex(self) -> AtomicFormula:
        return self.accept(ComplexNormalizer())
    
    def is_imaginary(self) -> bool:
        return self.lhs.is_imaginary() and self.rhs.is_imaginary()
    
    def is_real(self) -> bool:
        return self.lhs.is_real() and self.rhs.is_real()

    def simplify(self) -> Formula:
        """Fast basic simplification. The result is equivalent to self.
        Implements the abstract method
        :meth:`.firstorder.atomic.AtomicFormula.simplify`.
        """
        result = self.normalize_weak()
        try:
            return firstorder._T() if result.eval() else firstorder._F()
        except ValueError:
            return result
        
    def subs(self, sigma: Mapping[Variable, Number | Term]) -> Self:
        """Formal simultaneous term substitution into the two argument terms of
        the atomic formula. Implements the abstract method
        :meth:`.firstorder.atomic.AtomicFormula.subs`.
        """
        return self.op(self.lhs.subs(sigma), self.rhs.subs(sigma))



class Eq(AtomicFormula):
    
    def accept(self, visitor: AtomicFormulaVisitor[α]) -> α:
        return visitor.visit_eq(self)


class Ne(AtomicFormula):
    
    def accept(self, visitor: AtomicFormulaVisitor[α]) -> α:
        return visitor.visit_ne(self)


class Ge(AtomicFormula):
    
    def accept(self, visitor: AtomicFormulaVisitor[α]) -> α:
        return visitor.visit_ge(self)


class Le(AtomicFormula):
    
    def accept(self, visitor: AtomicFormulaVisitor[α]) -> α:
        return visitor.visit_le(self)


class Gt(AtomicFormula):
    
    def accept(self, visitor: AtomicFormulaVisitor[α]) -> α:
        return visitor.visit_gt(self)


class Lt(AtomicFormula):
    
    def accept(self, visitor: AtomicFormulaVisitor[α]) -> α:
        return visitor.visit_lt(self)


class AtomicFormulaVisitor(Generic[α]):

    @abstractmethod
    def visit_eq(self, eq: Eq) -> α:
        ...

    @abstractmethod
    def visit_ne(self, ne: Ne) -> α:
        ...

    @abstractmethod
    def visit_ge(self, ge: Ge) -> α:
        ...

    @abstractmethod
    def visit_le(self, le: Le) -> α:
        ...

    @abstractmethod
    def visit_gt(self, gt: Gt) -> α:
        ...

    @abstractmethod
    def visit_lt(self, lt: Lt) -> α:
        ...


from logic1.theories.Complex.normalize import ComplexNormalizer, ConstantEvaluator, Normalizer, WeakNormalizer
from logic1.theories.Complex.format import LatexFormatter, ReprFormatter, StrFormatter