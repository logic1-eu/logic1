from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, TYPE_CHECKING, Union

from .formula import Formula
from ..support.containers import Variables
from ..support.renaming import rename

# from ..support.tracing import trace

if TYPE_CHECKING:
    from .atomic import AtomicFormula


class QuantifiedFormula(Formula):

    print_precedence = 99
    text_symbol_spacing = ' '
    latex_symbol_spacing = ' \\, '

    func: type[QuantifiedFormula]

    @property
    def var(self):
        """The variable of the quantifier.

        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import x, y
        >>> e1 = All(x, Ex(y, EQ(x, y)))
        >>> e1.var
        x
        """
        return self.args[0]

    @var.setter
    def var(self, value: Any):
        self.args = (value, *self.args[1:])

    @property
    def arg(self):
        """The subformula in the scope of the QuantifiedFormula.

        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import x, y
        >>> e1 = All(x, Ex(y, EQ(x, y)))
        >>> e1.arg
        Ex(y, Eq(x, y))
        """
        return self.args[1]

    @staticmethod
    @abstractmethod
    def to_dual(conditional: bool = True):
        ...

    @classmethod
    def interactive_new(cls, variable, arg):
        """A type-checking convenience wrapper for the constructor.

        This is intended for inteactive use.

        >>> from logic1 import EX
        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import x
        >>> EX(x, EQ(x, x))
        Ex(x, Eq(x, x))

        For efficiency reasons, the constructors of subclasses of Formula do
        not check argument types. Trouble following later on can be hard to
        diagnose:

        >>> f = Ex('x', 'y')
        >>> f
        Ex('x', 'y')
        >>> f.simplify()
        Traceback (most recent call last):
        ...
        AttributeError: 'str' object has no attribute 'simplify'

        EX checks and raises an exception immediately:

        >>> EX('x', EQ(x, x))
        Traceback (most recent call last):
        ...
        TypeError: 'x' is not a Variable
        """
        if not isinstance(arg, Formula):
            raise TypeError(f'{repr(arg)} is not a Formula')
        atom = arg.get_any_atomic_formula()
        # If atom is None, then arg does not contain any atomic formula.
        # Therefore we cannot know what are valid variables, and we will accept
        # anything. Otherwise atom has a static method providing the type of
        # variables. This assumes that there is only one class of atomic
        # formulas used within a formula.
        if atom and not isinstance(variable, atom.variable_type()):
            raise TypeError(f'{repr(variable)} is not a Variable')
        return cls(variable, arg)

    def get_any_atomic_formula(self) -> Union[AtomicFormula, None]:
        return self.arg.get_any_atomic_formula()

    def _count_alternations(self) -> tuple:
        count, quantifiers = self.arg._count_alternations()
        if self.func.to_dual() in quantifiers:
            return (count + 1, {self.func})
        return (count, quantifiers)

    def qvars(self) -> set:
        return self.arg.qvars() | {self.var}

    def simplify(self, Theta=None):
        """Simplification.

        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import x, y
        >>> ALL(x, EX(y, EQ(x, y))).simplify()
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
            atom = self.get_any_atomic_formula()
            symbol = self.__class__.latex_symbol
            var = atom.term_to_latex(self.var) if atom else self.var
            spacing = self.__class__.latex_symbol_spacing
        else:
            assert mode == 'text'
            symbol = self.__class__.text_symbol
            var = self.var.__str__()
            spacing = self.__class__.text_symbol_spacing
        return f'{symbol} {var}{spacing}{arg_in_parens(self.arg)}'

    def sympy(self, *args, **kwargs):
        raise NotImplementedError(f'sympy does not know {type(self)}')

    def _to_distinct_vars(self, badlist: set) -> Self:
        arg = self.arg._to_distinct_vars(badlist)
        if self.var in badlist:
            var = rename(self.var)
            badlist |= {var}  # mutable
            arg = arg.subs({self.var: var})
            return self.func(var, arg)
        return self.func(self.var, arg)

    def to_nnf(self, implicit_not: bool = False,
               to_positive: bool = True) -> Formula:
        func_nnf = self.func.to_dual(conditional=implicit_not)
        arg_nnf = self.arg.to_nnf(implicit_not=implicit_not,
                                  to_positive=to_positive)
        return func_nnf(self.var, arg_nnf)

    def _to_pnf(self) -> dict:
        """Prenex normal form. self must be in negation normal form.
        """
        pnf = self.func(self.var, self.arg._to_pnf()[self.func])
        return {Ex: pnf, All: pnf}

    def subs(self, substitution: dict) -> Self:
        """Substitution.
        """
        atom = self.get_any_atomic_formula()
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
            substituted_vars |= atom.get_variables_from_term(term)
        # (2) Make sure the quantified variable is not a key and does not occur
        # in a value of substitution:
        if self.var in substituted_vars or self.var in substitution:
            var = atom.rename_variable(self.var)
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

    def transform_atoms(self, transformation: Callable) -> Self:
        return self.func(self.var,
                         self.arg.transform_atoms(transformation))

    def vars(self, assume_quantified: set = set()) -> Variables:
        quantified = assume_quantified | {self.var}
        return self.arg.vars(assume_quantified=quantified)


class Ex(QuantifiedFormula):
    """Existentially quantified formula factory.

    >>> from logic1.atomlib.sympy import EQ
    >>> from sympy.abc import x
    >>> Ex(x, EQ(x, 1))
    Ex(x, Eq(x, 1))
    """
    text_symbol = 'Ex'
    latex_symbol = '\\exists'

    @staticmethod
    def to_dual(conditional: bool = True):
        if conditional:
            return All
        return Ex

    def __init__(self, variable, arg):
        self.func = Ex
        self.args = (variable, arg)


EX = Ex.interactive_new


class All(QuantifiedFormula):
    """Universally quantified formula factory.

    >>> from logic1.atomlib.sympy import EQ
    >>> from sympy.abc import x, y
    >>> All(x, All(y, EQ((x + y)**2 + 1, x**2 + 2*x*y + y**2)))
    All(x, All(y, Eq((x + y)**2 + 1, x**2 + 2*x*y + y**2)))
    """
    text_symbol = 'All'
    latex_symbol = '\\forall'

    @staticmethod
    def to_dual(conditional: bool = True):
        if conditional:
            return Ex
        return All

    def __init__(self, variable, arg):
        self.func = All
        self.args = (variable, arg)


ALL = All.interactive_new


# The following import is intentionally late to avoid circularity.
from .boolean import Not
