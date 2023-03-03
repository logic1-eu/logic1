from __future__ import annotations

from typing import Any, Callable, Optional, TYPE_CHECKING

from .formula import Formula
from ..support.containers import GetVars
from ..support.decorators import classproperty
from ..support.renaming import rename

# from ..support.tracing import trace

if TYPE_CHECKING:
    from .atomic import AtomicFormula


class QuantifiedFormula(Formula):

    # Class variables
    latex_symbol_spacing = ' \\, '
    text_symbol_spacing = ' '
    print_precedence = 99

    latex_symbol: str
    text_symbol: str

    func: type[QuantifiedFormula]
    dual_func: type[QuantifiedFormula]

    # Instance variables
    args: tuple[Any, Formula]

    @property
    def var(self) -> Any:
        """The variable of the quantifier.

        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import x, y
        >>> e1 = All(x, Ex(y, EQ(x, y)))
        >>> e1.var
        x
        """
        return self.args[0]

    @var.setter
    def var(self, value: Any) -> None:
        self.args = (value, *self.args[1:])

    @property
    def arg(self) -> Formula:
        """The subformula in the scope of the QuantifiedFormula.

        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import x, y
        >>> e1 = All(x, Ex(y, EQ(x, y)))
        >>> e1.arg
        Ex(y, Eq(x, y))
        """
        return self.args[1]

    # Class methods
    def __new__(cls, variable: Any, arg: Formula):
        """A type-checking constructor.

        >>> from logic1 import Ex
        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import x
        >>> Ex(x, EQ(x, x))
        Ex(x, Eq(x, x))

        >>> Ex('x', 'y')
        Traceback (most recent call last):
        ...
        ValueError: 'y' is not a Formula

        >>> Ex('x', EQ(x, x))
        Traceback (most recent call last):
        ...
        ValueError: 'x' is not a Variable
        """
        if not isinstance(arg, Formula):
            raise ValueError(f'{arg!r} is not a Formula')
        atom = arg.get_any_atom()
        # If atom is None, then arg does not contain any atomic formula.
        # Therefore we cannot know what are valid variables, and we will accept
        # anything. Otherwise atom has a static method providing the type of
        # variables. This assumes that there is only one class of atomic
        # formulas used within a formula.
        if atom is not None and not isinstance(variable, atom.variable_type()):
            raise ValueError(f'{variable!r} is not a Variable')
        return super().__new__(cls)

    # Instance methods
    def _count_alternations(self) -> tuple[int, set]:
        count, quantifiers = self.arg._count_alternations()
        if self.dual_func in quantifiers:
            return (count + 1, {self.func})
        return (count, quantifiers)

    def get_any_atom(self) -> Optional[AtomicFormula]:
        return self.arg.get_any_atom()

    def get_qvars(self) -> set:
        return self.arg.get_qvars() | {self.var}

    def get_vars(self, assume_quantified: set = set()) -> GetVars:
        quantified = assume_quantified | {self.var}
        return self.arg.get_vars(assume_quantified=quantified)

    def simplify(self, Theta=None) -> Formula:
        """Simplification.

        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import x, y
        >>> All(x, Ex(y, EQ(x, y))).simplify()
        All(x, Ex(y, Eq(x, y)))
        """
        return self.func(self.var, self.arg.simplify())

    def _sprint(self, mode: str) -> str:
        def arg_in_parens(inner):
            inner_sprint = inner._sprint(mode)
            if inner.func not in (Ex, All, Not):
                inner_sprint = '(' + inner_sprint + ')'
            return inner_sprint

        if mode == 'latex':
            atom = self.get_any_atom()
            symbol = self.__class__.latex_symbol
            var = atom.term_to_latex(self.var) if atom else self.var
            spacing = self.__class__.latex_symbol_spacing
        else:
            assert mode == 'text'
            symbol = self.__class__.text_symbol
            var = self.var.__str__()
            spacing = self.__class__.text_symbol_spacing
        return f'{symbol} {var}{spacing}{arg_in_parens(self.arg)}'

    def subs(self, substitution: dict) -> QuantifiedFormula:
        """Substitution.
        """
        atom = self.get_any_atom()
        if not atom:
            return self
        # A copy of the mutual could be avoided by keeping track of the changes
        # and undoing them at the end.
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

    def _to_distinct_vars(self, badlist: set) -> QuantifiedFormula:
        arg = self.arg._to_distinct_vars(badlist)
        if self.var in badlist:
            var = rename(self.var)
            arg = arg.subs({self.var: var})
            badlist |= {var}  # mutable
            return self.func(var, arg)
        badlist |= {self.var}
        return self.func(self.var, arg)

    def to_nnf(self, implicit_not: bool = False,
               to_positive: bool = True) -> Formula:
        func_nnf = self.dual_func if implicit_not else self.func
        arg_nnf = self.arg.to_nnf(implicit_not=implicit_not,
                                  to_positive=to_positive)
        return func_nnf(self.var, arg_nnf)

    def _to_pnf(self) -> dict:
        """Prenex normal form. self must be in negation normal form.
        """
        pnf = self.func(self.var, self.arg._to_pnf()[self.func])
        return {Ex: pnf, All: pnf}

    def to_sympy(self, *args, **kwargs):
        raise NotImplementedError(f'sympy does not know {type(self)}')

    def transform_atoms(self, transformation: Callable) -> QuantifiedFormula:
        return self.func(self.var, self.arg.transform_atoms(transformation))


class Ex(QuantifiedFormula):
    """Existentially quantified formula factory.

    >>> from logic1.atomlib.sympy import EQ
    >>> from sympy.abc import x
    >>> Ex(x, EQ(x, 1))
    Ex(x, Eq(x, 1))
    """

    # Class variables
    latex_symbol = '\\exists'
    text_symbol = 'Ex'

    @classproperty
    def func(cls):
        return cls

    @classproperty
    def dual_func(cls):
        return All

    # Instance methods
    def __init__(self, variable: Any, arg: Formula) -> None:
        self.args = (variable, arg)


class All(QuantifiedFormula):
    """Universally quantified formula factory.

    >>> from logic1.atomlib.sympy import EQ
    >>> from sympy.abc import x, y
    >>> All(x, All(y, EQ((x + y)**2 + 1, x**2 + 2*x*y + y**2)))
    All(x, All(y, Eq((x + y)**2 + 1, x**2 + 2*x*y + y**2)))
    """

    # Class variables
    latex_symbol = '\\forall'
    text_symbol = 'All'

    @classproperty
    def func(cls):
        return cls

    @classproperty
    def dual_func(cls):
        return Ex

    # Instance methods
    def __init__(self, variable: Any, arg: Formula) -> None:
        self.args = (variable, arg)


# The following import is intentionally late to avoid circularity.
from .boolean import Not
