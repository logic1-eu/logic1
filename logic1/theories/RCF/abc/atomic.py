from abc import abstractmethod
from collections.abc import Iterator, Mapping
from typing import cast, Self

from logic1 import firstorder
from logic1.firstorder import _T, _F
from logic1.theories.RCF.abc.term import Term, Variable
from logic1.theories.RCF.abc.types import Number, _NUMBER_TYPES

class AtomicFormula[α: AtomicFormula,
                    τ: Term,
                    χ: Variable,
                    T_Eq: AtomicFormula,
                    T_Ge: AtomicFormula,
                    T_Gt: AtomicFormula,
                    T_Le: AtomicFormula,
                    T_Lt: AtomicFormula,
                    T_Ne: AtomicFormula,
    ](firstorder.AtomicFormula[α, τ, χ, Number]):
    """Abstract base class for atomic formulas over real closed fields. This
    class is the parent of a backend-specific implementation of atomic formulas.

    .. seealso::

        :class:`.theories.RCF.flint.atomic.AtomicFormula`
            The Flint-specific base class for all atomic formulas.

        :class:`.theories.RCF.sage.atomic.AtomicFormula`
            The Sage-specific base class for all atomic formulas.
    """

    @property
    def lhs(self) -> τ:
        """The left hand side term of this atomic formula.
        """
        return self.args[0]

    @property
    def rhs(self) -> τ:
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
        if isinstance(self, self.__class__._eq_type()):
            return self.lhs.sort_key() == self.rhs.sort_key()
        if isinstance(self, self.__class__._ne_type()):
            return self.lhs.sort_key() != self.rhs.sort_key()
        if isinstance(self, self.__class__._ge_type()):
            return self.lhs.sort_key() >= self.rhs.sort_key()
        if isinstance(self, self.__class__._gt_type()):
            return self.lhs.sort_key() > self.rhs.sort_key()
        if isinstance(self, self.__class__._le_type()):
            return self.lhs.sort_key() <= self.rhs.sort_key()
        if isinstance(self, self.__class__._lt_type()):
            return self.lhs.sort_key() < self.rhs.sort_key()
        assert False, self

    def __eq__(self, other: object) -> bool:
        """Return whether this atomic formula is equal to ``other``.

        Two atomic formulas are equal if they have the same relation and the
        sort keys of their corresponding left- and right-hand side terms are
        equal.
        """
        if not isinstance(other, self.__class__):
            # Covers both general objects and the wrong relation.
            return False
        if self.lhs.sort_key() != other.lhs.sort_key():
            return False
        if self.rhs.sort_key() != other.rhs.sort_key():
            return False
        return True

    def __hash__(self) -> int:
        return super().__hash__()

    @abstractmethod
    def __init__(self, lhs: Number | τ, rhs: Number | τ) -> None:
        super().__init__(lhs, rhs)
        if isinstance(lhs, _NUMBER_TYPES):
            lhs = self.__class__._term_type()(lhs)
        if isinstance(rhs, _NUMBER_TYPES):
            rhs = self.__class__._term_type()(rhs)
        self._args = (lhs, rhs)

    def __le__(self, other: firstorder.Formula) -> bool:
        """Return whether this atomic formula precedes or equals ``other`` in
        order.

        Implements the abstract method :meth:`.firstorder.atomic.AtomicFormula.__le__`.

        Atomic formulas are ordered lexicographically by the sort keys of
        their left-hand and right-hand terms, followed by the ordering
        :class:`Eq` < :class:`Ne` < :class:`Le` < :class:`Lt` < :class:`Ge` <
        :class:`Gt`. Non-atomic formulas are considered greater than atomic
        formulas.
        """
        L = (self.__class__._eq_type(),
             self.__class__._ne_type(),
             self.__class__._le_type(),
             self.__class__._lt_type(),
             self.__class__._ge_type(),
             self.__class__._gt_type())
        if not isinstance(other, L):
            return True
        self_sort_key = self.lhs.sort_key()
        other_sort_key = other.lhs.sort_key()
        if self_sort_key != other_sort_key:
            return self_sort_key <= other_sort_key
        self_sort_key = self.rhs.sort_key()
        other_sort_key = other.rhs.sort_key()
        if self_sort_key != other_sort_key:
            return self_sort_key <= other_sort_key
        return L.index(self.op) <= L.index(other.op)

    def __repr__(self) -> str:
        if self.lhs.is_constant() and self.rhs.is_constant():
            # Return Eq(1, 2) instead of 1 == 2, because the latter is not
            # suitable as input.
            return super().__repr__()
        SYMBOL = {self.__class__._eq_type(): "==",
                  self.__class__._ne_type(): "!=",
                  self.__class__._ge_type(): ">=",
                  self.__class__._le_type(): "<=",
                  self.__class__._gt_type(): ">",
                  self.__class__._lt_type(): "<"}
        SPACING = " "
        return f"{self.lhs!r}{SPACING}{SYMBOL[self.op]}{SPACING}{self.rhs!r}"

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
        SYMBOL = {self.__class__._eq_type(): "==",
                  self.__class__._ne_type(): "!=",
                  self.__class__._ge_type(): ">=",
                  self.__class__._le_type(): "<=",
                  self.__class__._gt_type(): ">",
                  self.__class__._lt_type(): "<"}
        SPACING = " "
        return f"{self.lhs}{SPACING}{SYMBOL[self.op]}{SPACING}{self.rhs}"

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
        SYMBOL = {self.__class__._eq_type(): "=",
                  self.__class__._ne_type(): "\\neq",
                  self.__class__._ge_type(): "\\geq",
                  self.__class__._le_type(): "\\leq",
                  self.__class__._gt_type(): ">",
                  self.__class__._lt_type(): "<"}
        SPACING = " "
        return f"{self.lhs.as_latex()}{SPACING}{SYMBOL[self.op]}{SPACING}{self.rhs.as_latex()}"

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
        SYMBOL = {self.__class__._eq_type(): "=",
                  self.__class__._ne_type(): "<>",
                  self.__class__._ge_type(): ">=",
                  self.__class__._le_type(): "<=",
                  self.__class__._gt_type(): ">",
                  self.__class__._lt_type(): "<"}
        SPACING = " "
        return f"({self.lhs!r}{SPACING}{SYMBOL[self.op]}{SPACING}{self.rhs!r})"

    def bvars(self, quantified: frozenset[χ] = frozenset()) -> Iterator[χ]:
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
    def complement(cls) -> type[α]:
        """Return the complement relation of ``cls``.

        Implements the abstract method :meth:`.firstorder.atomic.AtomicFormula.complement`.

        .. seealso::

          The inherited method :meth:`.firstorder.atomic.AtomicFormula.to_complement`
        """
        D = {cls._eq_type(): cls._ne_type(),
             cls._ne_type(): cls._eq_type(),
             cls._le_type(): cls._gt_type(),
             cls._lt_type(): cls._ge_type(),
             cls._ge_type(): cls._lt_type(),
             cls._gt_type(): cls._le_type()}
        return cast(type[α], D[cls])

    @classmethod
    def converse(cls) -> type[α]:
        """Return the converse relation of ``cls``.
        """
        D = {cls._eq_type(): cls._eq_type(),
             cls._ne_type(): cls._ne_type(),
             cls._le_type(): cls._ge_type(),
             cls._lt_type(): cls._gt_type(),
             cls._ge_type(): cls._le_type(),
             cls._gt_type(): cls._lt_type()}
        return cast(type[α], D[cls])

    @classmethod
    def dual(cls) -> type[α]:
        """Return the dual relation of ``cls``.
        """
        return cls.complement().converse()

    @staticmethod
    @abstractmethod
    def _eq_type() -> type[T_Eq]:
        ...

    def fvars(self, quantified: frozenset[χ] = frozenset()) -> Iterator[χ]:
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

    @staticmethod
    @abstractmethod
    def _ge_type() -> type[T_Ge]:
        ...

    @staticmethod
    @abstractmethod
    def _gt_type() -> type[T_Gt]:
        ...

    @staticmethod
    @abstractmethod
    def _le_type() -> type[T_Le]:
        ...

    @staticmethod
    @abstractmethod
    def _lt_type() -> type[T_Lt]:
        ...

    @staticmethod
    @abstractmethod
    def _ne_type() -> type[T_Ne]:
        ...

    def simplify(self) -> firstorder.Formula[α, τ, χ, Number]:
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
    def strict_part(cls) -> type[T_Gt | T_Lt]:
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
        if cls in (cls._eq_type(), cls._ne_type()):
            raise NotImplementedError()
        D: dict[object, type[T_Lt | T_Gt]] = {
             cls._le_type(): cls._lt_type(),
             cls._lt_type(): cls._lt_type(),
             cls._ge_type(): cls._gt_type(),
             cls._gt_type(): cls._gt_type()}
        return D[cls]

    def subs(self, sigma: Mapping[χ, Number | τ]) -> α:
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
        return cast(α, self.op(self.lhs.subs(sigma), self.rhs.subs(sigma)))

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

    def subsq(self, sigma: Mapping[χ, tuple[Number | τ, Number | τ]], is_positive: bool = False) -> α:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def _term_type() -> type[τ]:
        ...