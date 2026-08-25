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
    """Base class for atomic formulas over real closed fields. The class is the
    common parent of :class:`Eq <.RCF.atomic.Eq>`, :class:`Ne <.RCF.atomic.Ne>`,
    :class:`Le <.RCF.atomic.Le>`, :class:`Ge <.RCF.atomic.Ge>`, :class:`Lt
    <.RCF.atomic.Lt>`, and :class:`Gt <.RCF.atomic.Gt>`. It is not intended to
    be instantiated directly. Use one of the concrete relation classes instead.
    """

    @property
    def lhs(self) -> Term:
        """The left hand side term of this atomic formula.
        """
        return self.args[0]

    @property
    def rhs(self) -> Term:
        """The right hand side term of this atomic formula.
        """
        return self.args[1]

    def __bool__(self) -> bool:
        """Evaluation of this atomic formula in a Boolean context.

        In a Boolean context, atomic formulas are evaluated by comparing the
        left and right hand side terms using degree lexicographical term order.
        In particular, comparisons between terms representing integers follow
        the natural order.
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
        """Return whether this atomic formula is equal to ``other``.

        Two atomic formulas are equal if they have the same relation and the
        sort keys of their corresponding left- and right-hand side terms are
        equal.
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
        return super().__hash__()

    def __init__(self, lhs: Term | int, rhs: Term | int):
        super().__init__()
        if not isinstance(self, (Eq, Ne, Ge, Gt, Le, Lt)):
            raise NoTraceException('Instantiate one of Eq, Ne, Ge, Gt, Le, Lt instead')
        if not isinstance(lhs, Term):
            lhs = Term(lhs)
        if not isinstance(rhs, Term):
            rhs = Term(rhs)
        self._args = (lhs, rhs)

    def __le__(self, other: Formula) -> bool:
        """Return whether this atomic formula precedes or equals ``other`` in
        order.

        Implements the abstract method :meth:`.firstorder.atomic.AtomicFormula.__le__`.

        Atomic formulas are ordered lexicographically by the sort keys of
        their left-hand and right-hand terms, followed by the ordering
        :class:`Eq` < :class:`Ne` < :class:`Le` < :class:`Lt` < :class:`Ge` <
        :class:`Gt`. Non-atomic formulas are considered greater than atomic
        formulas.
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
        r"""Return the mathematical string representation of this atomic formula.

        Implements the abstract method :meth:`.firstorder.atomic.AtomicFormula.__str__`.

        The relation symbol is written infix, using the usual ASCII operators
        ``'=='``, ``'!='``, ``'<='``, ``'>='``, ``'<'``, and ``'>'``. The
        representation of the left and right hand side terms is delegated to
        :meth:`Term.__str__ <.RCF.term.Term.__str__>`.

        >>> from logic1.theories.RCF import VV
        >>> x, y = VV.get('x', 'y')
        >>> atom = (x - y + 2) ** 2 >= 0
        >>> str(atom)
        'x^2 - 2*x*y + y^2 + 4*x - 4*y + 4 >= 0'
        """
        SYMBOL: Final = {Eq: '==', Ne: '!=', Ge: '>=', Le: '<=', Gt: '>', Lt: '<'}
        SPACING: Final = ' '
        return f'{self.lhs}{SPACING}{SYMBOL[self.op]}{SPACING}{self.rhs}'

    def as_latex(self) -> str:
        r"""Return the LaTeX representation of this atomic formula.

        Implements the abstract method :meth:`.firstorder.atomic.AtomicFormula.as_latex`.

        The relation symbol is rendered infix as ``'='``, ``'\neq'``,
        ``'\leq'``, ``'\geq'``, ``'<'``, and ``'>'``. The representation of the
        left and right hand side terms is delegated to :meth:`Term.as_latex
        <.RCF.term.Term.as_latex>`.

        >>> from logic1.theories.RCF import VV
        >>> x, y = VV.get('x', 'y')
        >>> atom = (x - y + 2) ** 2 >= 0
        >>> atom.as_latex()
        'x^{2} - 2 x y + y^{2} + 4 x - 4 y + 4 \\geq 0'

        .. seealso::

            :meth:`.firstorder.formula.Formula.as_latex`
                LaTeX representation of first-order formulas
        """
        SYMBOL: Final = {
            Eq: '=', Ne: '\\neq', Ge: '\\geq', Le: '\\leq', Gt: '>', Lt: '<'}
        SPACING: Final = ' '
        return f'{self.lhs.as_latex()}{SPACING}{SYMBOL[self.op]}{SPACING}{self.rhs.as_latex()}'

    def as_redlog(self) -> str:
        r"""Return the Redlog representation of this atomic formula.

        Overloads the method :meth:`.firstorder.atomic.AtomicFormula.as_redlog`,
        which raises :exc:`NotImplementedError`.

        Returns the Redlog representation of the atomic formula in parentheses.

        >>> from logic1.theories.RCF import VV
        >>> x, y = VV.get('x', 'y')
        >>> atom = (x - y + 2) ** 2 != 0
        >>> atom.as_redlog()
        '(x**2 - 2*x*y + y**2 + 4*x - 4*y + 4 <> 0)'

        .. seealso::

            :meth:`.firstorder.formula.Formula.as_redlog`
                Redlog representation of first-order formulas
        """
        SYMBOL: Final = {
            Eq: '=', Ne: '<>', Ge: '>=', Le: '<=', Gt: '>', Lt: '<'}
        return f'({self.lhs!r} {SYMBOL[self.op]} {self.rhs!r})'

    def bvars(self, quantified: frozenset[Variable] = frozenset()) -> Iterator[Variable]:
        """The bound variables of this atomic formula.

        Implements the abstract method :meth:`.firstorder.atomic.AtomicFormula.bvars`.

        For each variable occurring in either term, yield it once for each
        term in which it occurs, provided that the variable belongs to
        ``quantified``.

        .. seealso::

            :meth:`.fvars`
                The free variables of this atomic formula.
            :meth:`.firstorder.formula.Formula.bvars`
                An iterator over all bound occurrences of variables in a
                first-order formula
        """
        for v in self.lhs.vars():
            if v in quantified:
                yield v
        for v in self.rhs.vars():
            if v in quantified:
                yield v

    @classmethod
    def complement(cls) -> type[AtomicFormula]:
        """Return the complement relation of ``cls``.

        Implements the abstract method :meth:`.firstorder.atomic.AtomicFormula.complement`.

        .. seealso::

          The inherited method :meth:`.firstorder.atomic.AtomicFormula.to_complement`
        """
        D: Any = {Eq: Ne, Ne: Eq, Le: Gt, Lt: Ge, Ge: Lt, Gt: Le}
        return D[cls]

    @classmethod
    def converse(cls) -> type[AtomicFormula]:
        """Return the converse relation of ``cls``.
        """
        D: Any = {Eq: Eq, Ne: Ne, Le: Ge, Lt: Gt, Ge: Le, Gt: Lt}
        return D[cls]

    @classmethod
    def dual(cls) -> type[AtomicFormula]:
        """Return the dual relation of ``cls``.
        """
        return cls.complement().converse()

    def fvars(self, quantified: frozenset[Variable] = frozenset()) -> Iterator[Variable]:
        """The free variables of this atomic formula.

        Implements the abstract method :meth:`.firstorder.atomic.AtomicFormula.fvars`.

        For each variable occurring in either term, yield it once for each
        term in which it occurs, provided that the variable does not belong to
        ``quantified``.

        .. seealso::

            :meth:`.bvars`
                The bound variables of this atomic formula.
            :meth:`.firstorder.formula.Formula.fvars`
                An iterator over all free occurrences of variables in a
                first-order formula
        """
        for v in self.lhs.vars():
            if v not in quantified:
                yield v
        for v in self.rhs.vars():
            if v not in quantified:
                yield v

    def simplify(self) -> Formula:
        """Return a simplified equivalent of the atomic formula, using basic
        simplification rules.

        Implements the abstract method :meth:`.firstorder.atomic.AtomicFormula.simplify`.

        If the difference between the left and right hand side terms of the
        input is constant, :obj:`.T` or :obj:`.F` is returned. Otherwise an
        :class:`AtomicFormula` is returned which has the following properties:

        1. The right hand side term is 0.
        2. The left hand side term is not constant and its leading coefficient
           is non-negative.

        .. seealso::

            :meth:`.firstorder.formula.Formula.simplify`
                Basic simplification of formulas, which uses this method for atomic formulas.
            :func:`.RCF.simplify.simplify`
                More powerful simplification of formulas.
        """
        lhs = self.lhs - self.rhs
        if lhs.is_constant():
            return _T() if self.op(lhs, 0) else _F()
        if lhs.lc() < 0:
            return self.op.converse()(-lhs, 0)
        return self.op(lhs, 0)

    @classmethod
    def strict_part(cls) -> type[Gt | Lt]:
        """Return the strict part of this subclass of :class:`AtomicFormula
        <.RCF.atomic.AtomicFormula>`.

        Raise :class:`.NotImplementedError` if this class is not an inequality.

        Otherwise, the strict part is defined as the relation without the
        diagonal:

        +---------------------+-------------+-------------+-------------+-------------+
        |                     | :class:`Le` | :class:`Ge` | :class:`Lt` | :class:`Gt` |
        +=====================+=============+=============+=============+=============+
        | :meth:`strict_part` | :class:`Lt` | :class:`Gt` | :class:`Lt` | :class:`Gt` |
        +---------------------+-------------+-------------+-------------+-------------+
        """
        if cls in (Eq, Ne):
            raise NotImplementedError()
        D: Any = {Le: Lt, Lt: Lt, Ge: Gt, Gt: Gt}
        return D[cls]

    def subs(self, sigma: Mapping[Variable, Term | int | mpq]) -> Self:
        """Return the atomic formula obtained from this atomic formula by
        simultaneous term substitution.

        Implements the abstract method :meth:`.firstorder.atomic.AtomicFormula.subs`.

        The substitution ``sigma`` is applied independently and simultaneously
        to the left- and right-hand side terms.

        .. seealso::

            :meth:`.firstorder.formula.Formula.subs`
                Simultaneous substitution of terms for variables in first-order
                formulas
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