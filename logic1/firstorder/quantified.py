r"""Provides subclasses of :class:`Formula <.formula.Formula>` that implement
quanitfied formulas in the sense that their toplevel operator is a one of the
quantifiers :math:`\exists` or :math:`\forall`.
"""
from __future__ import annotations

from typing import Any, Callable, Iterator, TypeAlias, TYPE_CHECKING

from .formula import Formula
from ..support.containers import GetVars
from ..support.decorators import classproperty

from ..support.tracing import trace  # noqa

if TYPE_CHECKING:
    from .atomic import AtomicFormula


QuantifierBlock: TypeAlias = tuple[Any, list]


class QuantifiedFormula(Formula):
    r"""A class whose instances are quanitfied formulas in the sense that their
    toplevel operator is a one of the quantifiers :math:`\exists` or
    :math:`\forall`.

    Note that members of :class:`QuantifiedFormula` may have subformulas with
    other logical operators deeper in the expression tree.
    """

    # The following would be abstract class variables, which are not available
    # at the moment.
    func: type[QuantifiedFormula]  #: :meta private:
    dual_func: type[QuantifiedFormula]  #: :meta private:

    # Similarly the following would be an abstract instance variable:
    args: tuple[Any, Formula]  #: :meta private:

    @property
    def var(self) -> Any:
        """The variable of the quantifier.

        >>> from logic1.theories.Sets import Eq
        >>> from sympy.abc import x, y
        >>>
        >>> e1 = All(x, Ex(y, Eq(x, y)))
        >>> e1.var
        x
        """
        return self.args[0]

    @var.setter
    def var(self, value: Any) -> None:
        self.args = (value, *self.args[1:])

    @property
    def arg(self) -> Formula:
        """The subformula in the scope of the :class:`QuantifiedFormula`.

        >>> from logic1.theories.Sets import Eq
        >>> from sympy.abc import x, y
        >>>
        >>> e1 = All(x, Ex(y, Eq(x, y)))
        >>> e1.arg
        Ex(y, Eq(x, y))
        """
        return self.args[1]

    def __init__(self, variable: Any, arg: Formula) -> None:
        self.args = (variable, arg)

    def _count_alternations(self) -> tuple[int, set]:
        count, quantifiers = self.arg._count_alternations()
        if self.dual_func in quantifiers:
            return (count + 1, {self.func})
        return (count, quantifiers)

    def get_qvars(self) -> set:
        """Implements the abstract method :meth:`Formula.get_qvars`.
        """
        return self.arg.get_qvars() | {self.var}

    def get_vars(self, assume_quantified: set = set()) -> GetVars:
        """Implements the abstract method :meth:`Formula.get_vars`.
        """
        quantified = assume_quantified | {self.var}
        return self.arg.get_vars(assume_quantified=quantified)

    def matrix(self) -> tuple[Formula, list[QuantifierBlock]]:
        blocks = []
        vars_ = []
        mat: Formula = self
        while isinstance(mat, QuantifiedFormula):
            Q = type(mat)
            while isinstance(mat, Q):
                vars_.append(mat.var)
                mat = mat.arg
            blocks.append((Q, vars_))
            vars_ = []
        return mat, blocks

    def simplify(self) -> Formula:
        """Compare the parent method :meth:`Formula.simplify`.

        >>> from logic1.theories.Sets import Eq
        >>> from sympy.abc import x, y
        >>>
        >>> All(x, Ex(y, Eq(x, y))).simplify()
        All(x, Ex(y, Eq(x, y)))
        """
        return self.func(self.var, self.arg.simplify())

    def subs(self, substitution: dict) -> QuantifiedFormula:
        """Implements the abstract method :meth:`Formula.subs`.
        """
        atom = next(self.atoms(), None)
        if atom is None:
            return self
        # A copy of the mutable could be avoided by keeping track of the
        # changes and undoing them at the end.
        substitution = substitution.copy()
        # (1) Remove substitution for the quantified variable. In principle,
        # this is covered by (2) below, but deleting here preserves the name.
        if self.var in substitution:
            del substitution[self.var]
        # Collect all variables on the right hand sides of substitutions:
        substituted_vars = set()
        for term in substitution.values():
            substituted_vars |= atom.term_get_vars(term)
        # (2) Make sure the quantified variable is not a key and does not occur
        # in a value of substitution:
        if self.var in substituted_vars or self.var in substitution:
            var = atom.rename_var(self.var)
            # We now know the following:
            #   (i) var is not a key,
            #  (ii) var does not occur in the values,
            # (iii) self.var is not a key.
            # We do *not* know whether self.var occurs in the values.
            substitution[self.var] = var
            # All free occurrences of self.var in self.arg will be renamed to
            # var. In case of (iv) above, substitution will introduce new free
            # occurrences of self.var, which do not clash with the new
            # quantified variable var:
            return self.func(var, self.arg.subs(substitution))
        return self.func(self.var, self.arg.subs(substitution))

    def to_nnf(self, to_positive: bool = True,
               _implicit_not: bool = False) -> Formula:
        """Implements the abstract method :meth:`Formula.to_nnf`.
        """
        func_nnf = self.dual_func if _implicit_not else self.func
        arg_nnf = self.arg.to_nnf(to_positive=to_positive,
                                  _implicit_not=_implicit_not)
        return func_nnf(self.var, arg_nnf)

    def transform_atoms(self, transformation: Callable) -> QuantifiedFormula:
        """Implements the abstract method :meth:`Formula.transform_atoms`.
        """
        return self.func(self.var, self.arg.transform_atoms(transformation))


class Ex(QuantifiedFormula):
    r"""A class whose instances are existentially quanitfied formulas in the
    sense that their toplevel operator represents the quantifier symbol
    :math:`\exists`.

    >>> from logic1.theories.Sets import Eq
    >>> from sympy.abc import x, y
    >>>
    >>> Ex(x, Eq(x, y))
    Ex(x, Eq(x, y))
    """
    @classproperty
    def func(cls):
        """A class property yielding the class :class:`Ex` itself.
        """
        return cls

    @classproperty
    def dual_func(cls):
        """A class property yielding the dual class :class:`All` of class:`Ex`.
        """
        return All


class All(QuantifiedFormula):
    r"""A class whose instances are universally quanitfied formulas in the
    sense that their toplevel operator represents the quantifier symbol
    :math:`\forall`.

    >>> from logic1.theories.RCF import Eq, ring
    >>> x, y = ring.set_vars('x', 'y')
    >>>
    >>> All(x, All(y, Eq((x + y)**2 + 1, x**2 + 2*x*y + y**2)))
    All(x, All(y, Eq(x^2 + 2*x*y + y^2 + 1, x^2 + 2*x*y + y^2)))
    """
    @classproperty
    def func(cls):
        """A class property yielding the class :class:`Ex` itself.
        """
        return cls

    @classproperty
    def dual_func(cls):
        """A class property yielding the dual class :class:`Ex` of class:`All`.
        """
        return Ex
