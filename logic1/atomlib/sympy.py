from __future__ import annotations

import sympy

from typing import Any, TypeAlias, Union

from .. import firstorder
from ..support.containers import GetVars
from ..support.renaming import rename
from . import generic

# Type alias
Card = Union[int, sympy.core.numbers.Infinity]

Term: TypeAlias = sympy.Expr
Variable: TypeAlias = sympy.Symbol

oo = sympy.oo


class TermMixin():

    @staticmethod
    def term_get_vars(term: Term) -> set[Variable]:
        """Implements the abstract method
        :meth:`.firstorder.AtomicFormula.term_get_vars`.
        """
        return sympy.S(term).atoms(Variable)

    @staticmethod
    def term_to_latex(term: Term) -> str:
        """Implements the abstract method
        :meth:`.firstorder.AtomicFormula.term_to_latex`.
        """
        return sympy.latex(term)

    @staticmethod
    def variable_type() -> type[Variable]:
        """Implements the abstract method
        :meth:`.firstorder.AtomicFormula.variable_type`.
        """
        return Variable

    @staticmethod
    def rename_var(variable: Variable) -> Variable:
        """Implements the abstract method
        :meth:`.firstorder.AtomicFormula.rename_var`.
        """
        return rename(variable)


class AtomicFormula(TermMixin, firstorder.AtomicFormula):
    """Atomic Formula with Sympy Terms. All terms are :class:`sympy.Expr`.
    """

    # Instance methods
    def get_vars(self, assume_quantified: set = set()) -> GetVars:
        """Implements the abstract method :meth:`.firstorder.Formula.get_vars`.
        """
        all_vars = set()
        for term in self.args:
            all_vars |= term.atoms(sympy.Symbol)
        return GetVars(free=all_vars - assume_quantified,
                       bound=all_vars & assume_quantified)

    def subs(self, substitution: dict) -> AtomicFormula:
        """Implements the abstract method :meth:`.firstorder.Formula.subs`.
        """
        args = (arg.subs(substitution, simultaneous=True) for arg in self.args)
        return self.func(*args)


class BinaryAtomicFormula(generic.BinaryAtomicFormulaMixin, AtomicFormula):
    """A class whose instances are binary formulas in the sense that both
    their m-arity and their p-arity is 2.
    """

    def __init__(self, *args):
        if len(args) != 2:
            raise ValueError(f'bad number of arguments for binary relation')
        args_ = []
        for arg in args:
            arg_ = sympy.Integer(arg) if isinstance(arg, int) else arg
            if not isinstance(arg_, Term):
                raise ValueError(f"{arg!r} is not a Term")
            args_.append(arg_)
        super().__init__(*args_)

    def _sprint(self, mode: str) -> str:
        if mode == 'latex':
            symbol = self.__class__.latex_symbol
            lhs = sympy.latex(self.lhs)
            rhs = sympy.latex(self.rhs)
            spacing = self.__class__.latex_symbol_spacing
        else:
            assert mode == 'text'
            symbol = self.__class__.text_symbol
            lhs = self.lhs.__str__()
            rhs = self.rhs.__str__()
            spacing = self.__class__.text_symbol_spacing
        return f'{lhs}{spacing}{symbol}{spacing}{rhs}'


class IndexedConstantAtomicFormula(AtomicFormula):
    r"""A class whose instances form a family of atomic formulas with m-arity
    0. Their p-arity is 1, where the one argument of the constructor is the
    index.
    """
    @property
    def index(self) -> Any:
        """The index of the :class:`IndexedConstantAtomicFormula`.
        """
        return self.args[0]

    def get_vars(self, assume_quantified: set = set()) -> GetVars:
        """Implements the abstract method :meth:`.firstorder.Formula.get_vars`.
        """
        return GetVars()


# Removed the following docstring from BinaryAtomicFormula above. Not sure
# where it belongs yet.

r"""Let `R` be a subclass of :class:`BinaryAtomicFormula` implementing atomic
    formulas with a binary relation symbol :math:`r`. For instance, if `R` is
    :class:`Eq`, then :math:`r` stands for the equality relation :math:`=` in
    the following discussion:

    >>> Eq(0, 0).func
    <class 'logic1.atomlib.sympy.Eq'>
    >>> Eq.func
    <class 'logic1.atomlib.sympy.Eq'>

    Assume that :math:`r` is defined on a domain :math:`D`. Then the
    *complement relation* of :math:`r` is defined as :math:`\overline{r} = D^2
    \setminus r`. It is avaialable as a class property `R.complement_func`,
    e.g.:

    >>> rels = (Eq, Ne, Ge, Le, Gt, Lt)
    >>> tuple(r.complement_func for r in rels)
    (<class 'logic1.atomlib.sympy.Ne'>,
     <class 'logic1.atomlib.sympy.Eq'>,
     <class 'logic1.atomlib.sympy.Lt'>,
     <class 'logic1.atomlib.sympy.Gt'>,
     <class 'logic1.atomlib.sympy.Le'>,
     <class 'logic1.atomlib.sympy.Ge'>)

    Since :math:`\overline{r}(s, t)` is equivalent to :math:`\neg r(s, t)`, the
    availability of complement relations is relevant for the computation of
    positive negation normal forms; compare :meth:`.firstorder.Formula.to_nnf`
    with keyword argument `to_positive=True`.

    The *converse relation* of :math:`r` is defined as
    :math:`r^{-1} = \{ (x, y) \in D : (y, x) \in r \}`.
    It is avaialable as a class property `R.converse_func`, e.g.:

    >>> tuple(r.converse_func for r in rels)
    (<class 'logic1.atomlib.sympy.Eq'>,
     <class 'logic1.atomlib.sympy.Ne'>,
     <class 'logic1.atomlib.sympy.Le'>,
     <class 'logic1.atomlib.sympy.Ge'>,
     <class 'logic1.atomlib.sympy.Lt'>,
     <class 'logic1.atomlib.sympy.Gt'>)

    The converse relation is the inverse with respect to composition.

    Finally, the *dual relation* of :math:`r` is defined as
    :math:`\overline{r}^{-1}`. It is available as a class property
    `R.dual_func`. Generally, :math:`\overline{r}^{-1} = \overline{r^{-1}}`,
    e.g.:

    >>> tuple(r.dual_func for r in rels)
    (<class 'logic1.atomlib.sympy.Ne'>,
     <class 'logic1.atomlib.sympy.Eq'>,
     <class 'logic1.atomlib.sympy.Gt'>,
     <class 'logic1.atomlib.sympy.Lt'>,
     <class 'logic1.atomlib.sympy.Ge'>,
     <class 'logic1.atomlib.sympy.Le'>)
    >>> all(r.dual_func == r.complement_func.converse_func for r in rels)
    True
    >>> all(r.dual_func == r.converse_func.complement_func for r in rels)
    True

    In the context of orderings, dualization turns strict inequalities into
    weak inequalities, and vice versa. Note that we also have duality and
    corresponding properties with Boolean functions, which is defined
    differently.

    All those operators on relations are involutive:

    >>> all(r.complement_func.complement_func == r for r in rels)
    True
    >>> all(r.converse_func.converse_func == r for r in rels)
    True
    >>> all(r.dual_func.dual_func == r for r in rels)
    True
"""
