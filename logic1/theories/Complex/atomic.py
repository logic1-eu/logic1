"""Atomic formulas in the theory :mod:`Complex <logic1.theories.Complex>`.
"""

from __future__ import annotations

from abc import abstractmethod
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
    """Abstract base class for atomic formulas in the theory of complex
    numbers. Implements the abstract class
    :class:`.firstorder.atomic.AtomicFormula`.

    .. seealso::
        :class:`.Eq`, :class:`.Ne`, :class:`.Ge`, :class:`.Gt`,
        :class:`.Le`, :class:`.Lt`
    """

    @property
    def lhs(self) -> Term:
        """The left-hand side term of this atomic formula.

        >>> from logic1.theories.Complex import *
        >>> z = VV['z']
        >>> (z == 0).lhs
        z
        """
        return self.args[0]

    @property
    def rhs(self) -> Term:
        """The right-hand side term of this atomic formula.

        >>> from logic1.theories.Complex import *
        >>> z = VV['z']
        >>> (z == 0).rhs
        0
        """
        return self.args[1]

    def __bool__(self) -> bool:  # TODO: maybe dont allow inequalities?
        """Compare the sort keys of both sides of this atomic formula
        using the corresponding operator of this formula. This is used to
        evaluate atomic formulas in boolean contexts. For evaluating constant
        atomic formulas with respect to their usual semantics, use :meth:`.eval`
        instead.

        >>> from logic1.theories.Complex import *
        >>> z = VV['z']
        >>> bool(z == 0)
        False
        >>> bool(z != 0)
        True
        >>> bool(z * ~z >= 1)  # not the usual semantics!
        True
        """
        ops = {Eq: operator.eq, Ne: operator.ne, Le: operator.le,
               Lt: operator.lt, Ge: operator.ge, Gt: operator.gt}
        return ops[self.op](self.lhs.sort_key(), self.rhs.sort_key())

    def __eq__(self, other: object) -> bool:
        """Return :obj:`True` if two formulas have the same relation and sides.

        >>> from logic1.theories.Complex import *
        >>> z = VV['z']
        >>> (z == 0) == (z == 0)
        True
        >>> (z == 0) == (z != 0)
        False
        """
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
        """Return the hash value of this atomic formula. We need to explicitly
        implement this method because we override :meth:`__eq__`.
        """
        return super().__hash__()

    @abstractmethod
    def __init__(self, lhs: Number | Term, rhs: Number | Term) -> None:
        """Initialize an atomic formula. This abstract base class is not
        supposed to have instances itself.
        """
        super().__init__(self, lhs, rhs)
        self.args = (
            lhs if isinstance(lhs, Term) else Term(lhs),
            rhs if isinstance(rhs, Term) else Term(rhs)
        )

    def __le__(self, other: Formula) -> bool:
        """Compare this atomic formula with another formula.
        Implements the abstract method
        :meth:`.firstorder.atomic.AtomicFormula.__le__`.

        >>> from logic1.theories.Complex import *
        >>> z = VV['z']
        >>> (z == 0) <= (z != 0)
        True
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
        """Return a string representation of this atomic formula that is
        valid Python code and can be evaluated to reconstruct the original
        atomic formula. Implements the abstract method
        :meth:`.firstorder.atomic.AtomicFormula.__repr__`.

        >>> from logic1.theories.Complex import *
        >>> z = VV['z']
        >>> repr(z == 0)
        'z == 0'
        """
        symbols = {Eq: '==', Ne: '!=', Le: '<=', Lt: '<', Ge: '>=', Gt: '>'}
        return f'{repr(self.lhs)} {symbols[self.op]} {repr(self.rhs)}'

    def __str__(self) -> str:
        """Return a string representation of this atomic formula. Implements the
        abstract method :meth:`.firstorder.atomic.AtomicFormula.__str__`.

        >>> from logic1.theories.Complex import *
        >>> z = VV['z']
        >>> str(z == 0)
        'z = 0'
        """
        symbols = {Eq: '=', Ne: '!=', Le: '<=', Lt: '<', Ge: '>=', Gt: '>'}
        return f'{str(self.lhs)} {symbols[self.op]} {str(self.rhs)}'

    def as_latex(self) -> str:
        """Return a LaTeX representation of this atomic formula. Implements the
        abstract method :meth:`.firstorder.atomic.AtomicFormula.as_latex`.

        >>> from logic1.theories.Complex import *
        >>> z = VV['z']
        >>> (z * ~z >= 0).as_latex()
        'z \\\\overline{z} \\\\geq 0'
        """
        symbols = {
            Eq: '=', Ne: '\\neq', Le: '\\leq', Lt: '<', Ge: '\\geq', Gt: '>'
        }
        return f'{self.lhs.as_latex()} {symbols[self.op]} {self.rhs.as_latex()}'

    def bvars(self, quantified: frozenset[Variable] = frozenset()) -> Iterator[Variable]:
        """Return an iterator over occurrences of variables that are elements of
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
        """Return the complement relation. Implements the abstract method
        :meth:`.firstorder.atomic.AtomicFormula.complement`.

        .. seealso::
            Inherited method
            :meth:`.firstorder.atomic.AtomicFormula.to_complement`

        >>> Eq.complement()
        <class 'logic1.theories.Complex.atomic.Ne'>
        >>> Lt.complement()
        <class 'logic1.theories.Complex.atomic.Ge'>
        """
        return {Eq: Ne, Ne: Eq, Le: Gt, Lt: Ge, Ge: Lt, Gt: Le}[cls]

    @classmethod
    def converse(cls) -> type[AtomicFormula]:
        """Return the converse relation.

        >>> Le.converse()
        <class 'logic1.theories.Complex.atomic.Ge'>
        >>> Lt.converse()
        <class 'logic1.theories.Complex.atomic.Gt'>
        """
        return {Eq: Eq, Ne: Ne, Le: Ge, Lt: Gt, Ge: Le, Gt: Lt}[cls]

    def eval(self) -> bool:
        """Evaluate an atomic formula where both sides are constants.
        Return :obj:`True` if the formula equivalent to :obj:`T`.
        Raises :class:`ValueError` if the formula contains variables.

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
        """Return an iterator over occurrences of variables that are *not*
        elements of :code:`quantified`. Yield each such variable once for each
        term that it occurs in. Implements the abstract method
        :meth:`.firstorder.atomic.AtomicFormula.fvars`.
        """
        for v in self.lhs.vars():
            if v not in quantified:
                yield v
        for v in self.rhs.vars():
            if v not in quantified:
                yield v

    def is_imaginary(self) -> bool:
        """Return :obj:`True` if both sides of this atomic formula are
        imaginary.

        >>> from logic1.theories.Complex import *
        >>> z = VV['z']
        >>> (Re(z) == 0).is_imaginary()
        False
        >>> (I * Re(z) == 0).is_imaginary()
        True
        """
        return self.lhs.is_imaginary() and self.rhs.is_imaginary()

    def is_real(self) -> bool:
        """Return :obj:`True` if both sides of this atomic formula are real.

        >>> from logic1.theories.Complex import *
        >>> z = VV['z']
        >>> (z == 0).is_real()
        False
        >>> (z * ~z == 0).is_real()
        True
        """
        return self.lhs.is_real() and self.rhs.is_real()

    def real_normal_form(self) -> Formula:
        """Return an equivalent formula in real normal form.

        >>> from logic1.theories.Complex import *
        >>> z = VV['z']
        >>> (z == 0).real_normal_form()
        And(1/2 * z + 1/2 * ~z == 0, -1/2 * I * z + 1/2 * I * ~z == 0)
        >>> (z != 0).real_normal_form()
        Or(1/2 * z + 1/2 * ~z != 0, -1/2 * I * z + 1/2 * I * ~z != 0)
        """
        lhs = self.lhs - self.rhs
        if isinstance(self, Eq):
            return And(Re(lhs) == 0, Im(lhs) == 0)
        if isinstance(self, Ne):
            return Or(Re(lhs) != 0, Im(lhs) != 0)
        if isinstance(self, Ge):
            return And(Re(lhs) >= 0, Im(lhs) == 0)
        if isinstance(self, Le):
            return And(Re(lhs) <= 0, Im(lhs) == 0)
        if isinstance(self, Gt):
            return And(Re(lhs) > 0, Im(lhs) == 0)
        if isinstance(self, Lt):
            return And(Re(lhs) < 0, Im(lhs) == 0)
        assert False, type(self)

    def simplify(self) -> AtomicFormula | _T | _F:
        """Return an equivalent simplified version of this atomic formula.
        Implements the abstract method
        :meth:`.firstorder.atomic.AtomicFormula.simplify`.

        >>> from logic1.theories.Complex import *
        >>> z = VV['z']
        >>> (z * ~z == Re(z)**2 + Im(z)**2).simplify()
        T
        >>> (Re(z) == 0).simplify()
        z + ~z == 0
        >>> (-Re(z) > z * ~z).simplify()
        z * ~z + 1/2 * z + 1/2 * ~z < 0
        """
        lhs = self.lhs - self.rhs
        try:
            return firstorder._T() if self.op(lhs, 0).eval() else firstorder._F()
        except ValueError:
            pass
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
        """Formal simultaneous term substitution into both sides of the atomic
        formula. Implements the abstract method
        :meth:`.firstorder.atomic.AtomicFormula.subs`.

        >>> from logic1.theories.Complex import *
        >>> a, b = VV.get('a', 'b')
        >>> (a + b == 0).subs({a: 1, b: a})
        a + 1 == 0
        """
        return self.op(self.lhs.subs(sigma), self.rhs.subs(sigma))


class Eq(AtomicFormula):
    """Equality relation in the theory of complex numbers.

    >>> from logic1.theories.Complex import *
    >>> z = VV['z']
    >>> z == 0
    z == 0
    """

    def __init__(self, lhs: Number | Term, rhs: Number | Term) -> None:
        """Initialize the equality relation.

        >>> from logic1.theories.Complex import *
        >>> z = VV['z']
        >>> Eq(z, 0)
        z == 0
        """
        super().__init__(lhs, rhs)


class Ne(AtomicFormula):
    """Inequality relation in the theory of complex numbers.

    >>> from logic1.theories.Complex import *
    >>> z = VV['z']
    >>> z != 0
    z != 0
    """

    def __init__(self, lhs: Number | Term, rhs: Number | Term) -> None:
        """Initialize the inequality relation.

        >>> from logic1.theories.Complex import *
        >>> z = VV['z']
        >>> Ne(z, 0)
        z != 0
        """
        super().__init__(lhs, rhs)


class RealAtomicFormula(AtomicFormula):
    """An abstract base class for atomic formulas where both sides are real.
    Raise a :class:`ValueError` when trying to create an instance where
    either side is not real.

    .. seealso::
        :class:`.Ge`, :class:`.Le`, :class:`.Gt`, :class:`.Lt`
    """

    @abstractmethod
    def __init__(self, lhs: Number | Term, rhs: Number | Term):
        """Initialize the real atomic formula. Raise a :class:`ValueError`
        if either side is not real. This abstract base class is not supposed to
        have instances itself.
        """
        super().__init__(lhs, rhs)
        if not self.is_real():
            raise ValueError(f'Cannot create atomic formula {self} because it is not real')


class Ge(RealAtomicFormula):
    """Greater than or equal relation in the theory of complex numbers.

    >>> from logic1.theories.Complex import *
    >>> z = VV['z']
    >>> z * ~z >= 0
    z * ~z >= 0
    >>> z >= 0
    Traceback (most recent call last):
    ...
    ValueError: Cannot create atomic formula z >= 0 because it is not real
    """

    def __init__(self, lhs: Number | Term, rhs: Number | Term) -> None:
        """Initialize the greater than or equal relation. Raise a
        :class:`ValueError` if either side is not real.

        >>> from logic1.theories.Complex import *
        >>> z = VV['z']
        >>> Ge(z * ~z, 0)
        z * ~z >= 0
        >>> Ge(z, 0)
        Traceback (most recent call last):
        ...
        ValueError: Cannot create atomic formula z >= 0 because it is not real
        """
        super().__init__(lhs, rhs)


class Le(RealAtomicFormula):
    """Less than or equal relation in the theory of complex numbers.

    >>> from logic1.theories.Complex import *
    >>> z = VV['z']
    >>> z * ~z <= 0
    z * ~z <= 0
    >>> z <= 0
    Traceback (most recent call last):
    ...
    ValueError: Cannot create atomic formula z <= 0 because it is not real
    """

    def __init__(self, lhs: Number | Term, rhs: Number | Term) -> None:
        """Initialize the less than or equal relation.

        >>> from logic1.theories.Complex import *
        >>> z = VV['z']
        >>> Le(z * ~z, 0)
        z * ~z <= 0
        >>> Le(z, 0)
        Traceback (most recent call last):
        ...
        ValueError: Cannot create atomic formula z <= 0 because it is not real
        """
        super().__init__(lhs, rhs)


class Gt(RealAtomicFormula):
    """Greater than relation in the theory of complex numbers.

    >>> from logic1.theories.Complex import *
    >>> z = VV['z']
    >>> z * ~z > 0
    z * ~z > 0
    >>> z > 0
    Traceback (most recent call last):
    ...
    ValueError: Cannot create atomic formula z > 0 because it is not real
    """

    def __init__(self, lhs: Number | Term, rhs: Number | Term) -> None:
        """Initialize the greater than relation.

        >>> from logic1.theories.Complex import *
        >>> z = VV['z']
        >>> Gt(z * ~z, 0)
        z * ~z > 0
        >>> Gt(z, 0)
        Traceback (most recent call last):
        ...
        ValueError: Cannot create atomic formula z > 0 because it is not real
        """
        super().__init__(lhs, rhs)


class Lt(RealAtomicFormula):
    """Less than relation in the theory of complex numbers.

    >>> from logic1.theories.Complex import *
    >>> z = VV['z']
    >>> z * ~z < 0
    z * ~z < 0
    >>> z < 0
    Traceback (most recent call last):
    ...
    ValueError: Cannot create atomic formula z < 0 because it is not real
    """

    def __init__(self, lhs: Number | Term, rhs: Number | Term) -> None:
        """Initialize the less than relation. Raise a :class:`ValueError`
        if either side is not real.

        >>> from logic1.theories.Complex import *
        >>> z = VV['z']
        >>> Lt(z * ~z, 0)
        z * ~z < 0
        >>> Lt(z, 0)
        Traceback (most recent call last):
        ...
        ValueError: Cannot create atomic formula z < 0 because it is not real
        """
        super().__init__(lhs, rhs)
