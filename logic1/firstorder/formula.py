"""Module for first-order formulas.

There are classes Ex, All, Equivaelent, Implies, And, Or, Not, _T, _F which
provide constructors for corresponding (sub)formulas. Furthermore, there are
interactive constructors EX, ALL, EQUIV, IMPL (>>), AND (&), OR (|), NOT (~),
T, F, which do some additional error checking.

There is no language L in the sense of model theory specified. Application
modules implement L as follows:

1. Choose sets R of admissible relation symbols and F of admissible function
   symbols, which includes constants.

2. For each relation in R derive a class Rel from the Abstract Base Class
   firstorder.AtomicFormula and implement at least the abstract methods
   specified by firstorder.AtomicFormula.

3. In combination with the first-order constructors above, the constructors Rel
   allow the construction of first-order L-formulas. Similarly to EX, ALL etc.
   above, respective interactive constructors REL can provide error checking.

The firstorder module imposes no restrictions on the representation of atomic
formulas and terms. It is quite natural to represent relations similarly to the
logical operators in the subclasses of firstorder.Formula.

For the argument terms, sympy.Expr are certainly an option. This will support a
super set of the function symbols F in L, in general. The interacitve
constructors REL can check that only valid L-terms are used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, ClassVar, final, Tuple, TYPE_CHECKING, Union

import pyeda.inter  # type: ignore
from pyeda.inter import expr, exprvar
import sympy

from ..support.containers import Variables

# from ..support.tracing import trace

if TYPE_CHECKING:
    from .atomic import AtomicFormula


class Formula(ABC):
    """An abstract base class for first-order formulas.
    """

    latex_symbol_spacing: ClassVar[str]
    latex_symbol: ClassVar[str]
    print_style: ClassVar[str]
    print_precedence: ClassVar[int]
    text_symbol: ClassVar[str]

    sympy_func: ClassVar[type[sympy.core.basic.Basic]]

    func: type[Formula]
    args: Tuple

    @final
    def __and__(self, other: Self) -> Self:
        """Override the ``&`` operator to apply logical AND.

        Note that ``&`` delegates to the convenience wrapper AND in contrast to
        the constructor And.

        >>> from logic1.atomlib.sympy import EQ
        >>> EQ(0, 0) & EQ(1 + 1, 2) & EQ(1 + 1 + 1, 3)
        And(Eq(0, 0), Eq(2, 2), Eq(3, 3))
        """
        return And.interactive_new(self, other)

    @final
    def __invert__(self) -> Self:
        """Override the ``~`` operator to apply logical NOT.

        Note that ``~`` delegates to the convenience wrapper NOT in contrast to
        the constructor Not.

        >>> from logic1.atomlib.sympy import EQ
        >>> ~ EQ(1,0)
        Not(Eq(1, 0))
        """
        return Not.interactive_new(self)

    @final
    def __lshift__(self, other: Self) -> Self:
        """Override ``>>`` operator to apply logical IMPL.

        Note that ``>>`` delegates to the convenience wrapper IMPL in contrast
        to the constructor Implies.

        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import x, y, z
        >>> EQ(x + z, y + z) << EQ(x, y)
        Implies(Eq(x, y), Eq(x + z, y + z))
        """
        return Implies.interactive_new(other, self)

    @final
    def __or__(self, other: Self) -> Self:
        """Override the ``|`` operator to apply logical OR.

        Note that ``|`` delegates to the convenience wrapper OR in contrast to
        the constructor Or.

        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import x, y, z
        >>> EQ(x, 0) | EQ(x, y) | EQ(x, z)
        Or(Eq(x, 0), Eq(x, y), Eq(x, z))
        """
        return Or.interactive_new(self, other)

    @final
    def __rshift__(self, other: Self) -> Self:
        """Override the ``<<`` operator to apply logical IMPL with reversed
        sides.

        Note that ``<<`` uses the convenience wrapper IMPL in contrast to the
        constructor implies.

        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import x, y, z
        >>> EQ(x, y) >> EQ(x + z, y + z)
        Implies(Eq(x, y), Eq(x + z, y + z))
        """
        return Implies.interactive_new(self, other)

    def __eq__(self, other: Self) -> bool:
        """Recursive equality of the formulas self and other.

        This is *not* logical ``equal.``

        >>> from logic1.atomlib.sympy import NE
        >>> e1 = NE(1, 0)
        >>> e2 = NE(1, 0)
        >>> e1 == e2
        True
        >>> e1 is e2
        False
        """
        return self.func == other.func and self.args == other.args

    def __hash__(self) -> int:
        return hash((self.func, self.args))

    @abstractmethod
    def __init__(self, *args) -> None:
        ...

    @final
    def __repr__(self) -> str:
        """Representation of the Formula suitable for use as an input.
        """
        r = self.func.__name__
        r += '('
        if self.args:
            r += self.args[0].__repr__()
            for a in self.args[1:]:
                r += ', ' + a.__repr__()
        r += ')'
        return r

    @final
    def __str__(self) -> str:
        """Representation of the Formula used in printing.
        """
        return self._sprint(mode='text')

    @final
    def _repr_latex_(self) -> str:
        r"""A LaTeX representation of the formula as it is used within jupyter
        notebooks

        >>> F._repr_latex_()
        '$\\displaystyle \\bot$'

        Subclasses have latex() methods yielding plain LaTeX without the
        surrounding $\\displaystyle ... $.
        """
        return '$\\displaystyle ' + self.latex() + '$'

    @abstractmethod
    def get_any_atomic_formula(self) -> Union[AtomicFormula, None]:
        """Return any atomic formula contained in self, None if there is none.

        A typical use cass is getting access to methods of classes derived from
        AtomicFormula elsewhere.
        """
        ...

    @final
    def count_alternations(self) -> int:
        """Count number of quantifier alternations.

        Returns the maximal number of quantifier alternations along a path in
        the expression tree. Occurrence of quantified variables is not checked.
        >>> from logic1 import EX, ALL, T
        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import x, y, z
        >>> EX(x, EQ(x, y) & ALL(x, EX(y, EX(z, T)))).count_alternations()
        2
        """
        return self._count_alternations()[0]

    @abstractmethod
    def _count_alternations(self) -> tuple:
        ...

    @final
    def latex(self) -> str:
        """Convert to LaTeX representation.
        """
        return self._sprint(mode='latex')

    @abstractmethod
    def qvars(self) -> set:
        """The set of all variables that are quantified in self.

        This should not be confused with bound ocurrences of variables. Compare
        the Formula.vars() method.

        >>> from logic1 import EX, ALL
        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import a, b, c, x, y, z
        >>> ALL(y, EX(x, EQ(a, y)) & EX(z, EQ(a, y))).qvars() == {x, y, z}
        True
        """
        ...

    def simplify(self, Theta=None) -> Self:
        """Identity as a default implemenation of a simplifier for formulas.

        This should be overridden in the majority of the classes that
        are finally instantiated.
        """
        return self

    @abstractmethod
    def _sprint(self, mode: str) -> str:
        """Print to string.
        """
        ...

    @abstractmethod
    def subs(self, substitution: dict) -> Self:
        """Substitution.

        >>> from logic1 import push, pop, EX
        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import a, b, c, x, y, z
        >>> push()

        >>> EX(x, EQ(x, a)).subs({x: a})
        Ex(x, Eq(x, a))

        >>> f = EX(x, EQ(x, a)).subs({a: x})
        >>> g = Ex(x, f & EQ(b, 0))
        >>> g
        Ex(x, And(Ex(x_R1, Eq(x_R1, x)), Eq(b, 0)))

        >>> g.subs({b: x})
        Ex(x_R2, And(Ex(x_R1, Eq(x_R1, x_R2)), Eq(x, 0)))

        >>> pop()
        """
        ...

    def sympy(self, **kwargs) -> sympy.Basic:
        """Provide a sympy representation of the Formula if possible.

        Subclasses that have no match in sympy can raise NotImplementedError.

        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import x, y
        >>> e1 = Equivalent(EQ(x, y), EQ(x + 1, y + 1))
        >>> e1
        Equivalent(Eq(x, y), Eq(x + 1, y + 1))
        >>> type(e1)
        <class 'logic1.firstorder.formula.Equivalent'>
        >>> e1.sympy()
        Equivalent(Eq(x, y), Eq(x + 1, y + 1))
        >>> type(e1.sympy())
        Equivalent

        >>> e2 = Equivalent(EQ(x, y), EQ(y, x))
        >>> e2
        Equivalent(Eq(x, y), Eq(y, x))
        >>> e2.sympy()
        True

        >>> e3 = T
        >>> e3.sympy()
        Traceback (most recent call last):
        ...
        NotImplementedError:
            sympy does not know <class 'logic1.firstorder.truth._T'>

        >>> e4 = All(x, Ex(y, EQ(x, y)))
        >>> e4.sympy()
        Traceback (most recent call last):
        ...
        NotImplementedError:
            sympy does not know <class 'logic1.firstorder.quantified.All'>
        """
        args_sympy = (arg.sympy(**kwargs) for arg in self.args)
        return self.__class__.sympy_func(*args_sympy)

    @final
    def to_distinct_vars(self) -> Self:
        """Convert to equivalent formulas with distinct variables.

        Bound variables are renamed such that that set of all bound
        variables is disjoint from the set of all free variables.
        Furthermore, each bound variables occurs with one and only one
        quantifier.

        >>> from logic1 import push, pop, EX, ALL, T
        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import x, y, z
        >>> push()
        >>> f0 = ALL(z, EX(y, EQ(x, y) & EQ(y, z) & EX(x, T)))
        >>> f = EQ(x, y) & EX(x, EQ(x, y) & f0)
        >>> f.to_distinct_vars()
        And(Eq(x, y), Ex(x_R3, And(Eq(x_R3, y),
            All(z, Ex(y_R2, And(Eq(x_R3, y_R2), Eq(y_R2, z), Ex(x_R1, T)))))))
        >>> pop()
        """
        # Recursion starts with a badlist (technically a set) of all free
        # variables.
        return self._to_distinct_vars(self.vars().free)

    @abstractmethod
    def _to_distinct_vars(self, badlist: set) -> Self:
        # Traverses self. If a badlisted variable is encountered as a
        # quantified variable, it will be replaced with a fresh name in the
        # respective QuantifiedFormula, and the fresh name will be badlisted
        # for the future. Note that this can includes variables that do not
        # *occur* in a mathematical sense.
        ...

    @abstractmethod
    def to_nnf(self, implicit_not: bool = False,
               to_positive: bool = True) -> Self:
        """Convert to Negation Normal Form.

        An NNF is a formula where logical Not is only applied to atomic
        formulas and thruth values. The only other allowed Boolean operators
        are And and Or. Besides those Boolean operators, we also admit
        quantifiers Ex and All. If the input is quanitfier-free, to_nnf will
        not introduce any quanitfiers.

        >>> from logic1 import EX, EQUIV, NOT, T
        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import a, y
        >>> f = EQUIV(EQ(a, 0) & T, EX(y, ~ EQ(y, a)))
        >>> f.to_nnf()
        And(Or(Ne(a, 0), F, Ex(y, Ne(y, a))),
            Or(All(y, Eq(y, a)), And(Eq(a, 0), T)))
        """
        ...

    @final
    def to_pnf(self, prefer_universal: bool = False,
               is_admissible: bool = False) -> Self:
        """Convert to Prenex Normal Form.

        A Prenex Normal Form (PNF) is a Negation Normal Form (NNF) where all
        quantifiers Ex and All stand at the beginning of the formula. The
        method minimizes the number of quantifier alternations in the prenex
        block. Results starting with an existential quantifier are prefered.
        This can be changed by passing prefer_universal=True. The argument
        is_nnf can be used as a hint that self is already in NNF.

        Burhenne p.88:

        >>> from logic1 import push, pop, EX, ALL, T, F
        >>> from logic1.atomlib.sympy import EQ
        >>> push()
        >>> x = sympy.symbols('x:8')
        >>> f1 = EX(x[1], ALL(x[2], ALL(x[3], T)))
        >>> f2 = ALL(x[4], EX(x[5], ALL(x[6], F)))
        >>> f3 = EX(x[7], EQ(x[0], 0))
        >>> (f1 & f2 & f3).to_pnf()
        All(x4, Ex(x1, Ex(x5, Ex(x7, All(x2, All(x3, All(x6,
            And(T, F, Eq(x0, 0)))))))))
        >>> pop()

        Derived from redlog.tst:

        >>> push()
        >>> from logic1 import EQUIV, AND, OR
        >>> from sympy.abc import a, b, y
        >>> f1 = EQ(a, 0) & EQ(b, 0) & EQ(y, 0)
        >>> f2 = EX(y, EQ(y, a) | EQ(a, 0))
        >>> EQUIV(f1, f2).to_pnf()
        Ex(y_R1, All(y_R2,
            And(Or(Ne(a, 0), Ne(b, 0), Ne(y, 0), Eq(y_R1, a), Eq(a, 0)),
                Or(And(Ne(y_R2, a), Ne(a, 0)),
                   And(Eq(a, 0), Eq(b, 0), Eq(y, 0))))))
        >>> pop()

        >>> push()
        >>> EQUIV(f1, f2).to_pnf(prefer_universal=True)
        All(y_R2, Ex(y_R1,
            And(Or(Ne(a, 0), Ne(b, 0), Ne(y, 0), Eq(y_R1, a), Eq(a, 0)),
                Or(And(Ne(y_R2, a), Ne(a, 0)),
                   And(Eq(a, 0), Eq(b, 0), Eq(y, 0))))))
        >>> pop()
        """
        if is_admissible:
            phi = self
        else:
            phi = self.to_nnf().to_distinct_vars()
        return phi._to_pnf()[All if prefer_universal else Ex]

    # abstract - see docstring
    def _to_pnf(self) -> dict:
        """Private convert to Prenex Normal Form.

        self must be in NNF. All NNF operators (QuantifiedFormula, AndOr,
        TruthValue, AtomicFormula) must override this method. The result is a
        dict d with len(d) == 2. The keys are Ex and All, the values are both
        prenex equivalents of self with the same minimized number of quantifier
        alternations. Either d[Ex] starts with an existential quantifier and
        d[All] starts with a universal quantifier, or d[Ex] is d[All], i.e.,
        identity is guaranteed.
        """
        raise NotImplementedError(f'{self.func} is not an NNF operator')

    @abstractmethod
    def transform_atoms(self, transformation: Callable) -> Self:
        ...

    @abstractmethod
    def vars(self, assume_quantified: set = set()) -> Variables:
        """Get variables.

        >>> from logic1 import EX, ALL
        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import x, y, z
        >>> f = EQ(3 * x, 0) \
                >> ALL(z, ALL(x, (~ EQ(x, 0) >> EX(y, EQ(x * y, 1)))))
        >>> f.vars().free == {x}
        True
        >>> f.vars().bound == {x, y}
        True
        >>> z not in f.vars().all
        True
        """
        ...


latex = Formula.latex


class BooleanFormula(Formula):
    """Boolean Formulas have a Boolean operator at the top level.

    An operator of a Formula is either a quantifier Ex, All or a Boolean
    operator And, Or, Not, Implies, Equivaelent, T, F. Note that members of
    BooleanFormula start, in the sense of prefix notation, with a Boolean
    operator but may have quantified subformulas deeper in the expression tree.
    """
    text_symbol_spacing = ' '
    latex_symbol_spacing = ' \\, '

    @staticmethod
    def _from_pyeda(f: expr, d: dict[exprvar, AtomicFormula]) -> Formula:
        from_dict = {'Implies': Implies, 'Or': Or, 'And': And, 'Not': Not}
        if isinstance(f, pyeda.boolalg.expr.Variable):
            return d[f]
        if isinstance(f, pyeda.boolalg.expr.Complement):
            variable = pyeda.boolalg.expr.Not(f, simplify=True)
            return d[variable].to_complement()
        assert isinstance(f, pyeda.boolalg.expr.Operator)
        func = from_dict[f.NAME]
        args = (BooleanFormula._from_pyeda(arg, d) for arg in f.xs)
        return func(*args)

    def get_any_atomic_formula(self) -> Union[AtomicFormula, None]:
        for arg in self.args:
            atom = arg.get_any_atomic_formula()
            if atom:
                return atom
        return None

    def _count_alternations(self) -> tuple:
        best_count = -1
        best_quantifiers = {Ex, All}
        for arg in self.args:
            count, quantifiers = arg._count_alternations()
            if count > best_count:
                best_count = count
                best_quantifiers = quantifiers
            elif count == best_count:
                best_quantifiers |= quantifiers
        return (best_count, best_quantifiers)

    def qvars(self) -> set:
        qvars = set()
        for arg in self.args:
            qvars |= arg.qvars()
        return qvars

    def _sprint(self, mode: str) -> str:
        def not_arg(outer, inner) -> str:
            inner_sprint = inner._sprint(mode)
            if inner.func not in (Ex, All, Not):
                inner_sprint = '(' + inner_sprint + ')'
            return inner_sprint

        def infix_arg(outer, inner) -> str:
            inner_sprint = inner._sprint(mode)
            if (outer.__class__.print_precedence
                    >= inner.__class__.print_precedence):
                inner_sprint = '(' + inner_sprint + ')'
            return inner_sprint

        if mode == 'latex':
            symbol = self.__class__.latex_symbol
            spacing = self.__class__.latex_symbol_spacing
        else:
            assert mode == 'text'
            symbol = self.__class__.text_symbol
            spacing = self.__class__.text_symbol_spacing
        if self.__class__.print_style == 'constant':
            return symbol
        if self.__class__.print_style == 'not':
            return f'{symbol}{spacing}{not_arg(self, self.args[0])}'
        if self.__class__.print_style == 'infix':
            s = infix_arg(self, self.args[0])
            for a in self.args[1:]:
                s = f'{s}{spacing}{symbol}{spacing}{infix_arg(self, a)}'
            return s
        assert False

    def subs(self, substitution: dict) -> Self:
        """Substitution.
        """
        return self.func(*(arg.subs(substitution) for arg in self.args))

    def _to_distinct_vars(self, badlist: set) -> Self:
        return self.func(*(arg._to_distinct_vars(badlist)
                           for arg in self.args))

    def to_cnf(self) -> Self:
        """ Convert to Conjunctive Normal Form.

        >>> from logic1.atomlib.sympy import EQ, NE
        >>> from sympy.abc import a
        >>> ((EQ(a, 0) & NE(a, 1) | ~EQ(a, 0) | NE(a, 2)) >> NE(a, 1)).to_cnf()
        And(Or(Ne(a, 1), Eq(a, 2)), Or(Eq(a, 0), Ne(a, 1)))
        """
        return Not(self).to_dnf().to_nnf(implicit_not=True)

    def to_dnf(self) -> Self:
        """ Convert to Disjunctive Normal Form.

        >>> from logic1.atomlib.sympy import EQ, NE
        >>> from sympy.abc import a
        >>> ((EQ(a, 0) & NE(a, 1) | ~EQ(a, 0) | NE(a, 2)) >> NE(a, 1)).to_dnf()
        Or(Ne(a, 1), And(Eq(a, 0), Eq(a, 2)))
        """
        d: dict[AtomicFormula, exprvar] = {}
        self_pyeda = self._to_pyeda(d)
        # _to_pyeda() has populated the mutable dictionary d
        d_rev: dict[exprvar, AtomicFormula] = dict(map(reversed, d.items()))
        dnf = self_pyeda.to_dnf()
        dnf, = pyeda.boolalg.minimization.espresso_exprs(dnf)
        self_dnf = BooleanFormula._from_pyeda(dnf, d_rev)
        return self_dnf

    def _to_pyeda(self, d: dict[AtomicFormula, exprvar], c: list = [0]) \
            -> pyeda.boolalg.expr:
        to_dict = {Equivalent: pyeda.boolalg.expr.Equal,
                   Implies: pyeda.boolalg.expr.Implies,
                   And: pyeda.boolalg.expr.And,
                   Or: pyeda.boolalg.expr.Or,
                   Not: pyeda.boolalg.expr.Not}
        NAME = to_dict[self.func]
        xs = (arg._to_pyeda(d, c) for arg in self.args)
        return NAME(*xs, simplify=False)

    def transform_atoms(self, transformation: Callable) -> Self:
        return self.func(*(arg.transform_atoms(transformation)
                           for arg in self.args))

    def vars(self, assume_quantified: set = set()) -> Variables:
        vars = Variables()
        for arg in self.args:
            vars |= arg.vars(assume_quantified=assume_quantified)
        return vars


class Equivalent(BooleanFormula):

    print_style = 'infix'
    print_precedence = 10
    text_symbol = '<-->'
    latex_symbol = '\\longleftrightarrow'

    sympy_func = sympy.Equivalent

    @property
    def lhs(self):
        """The left-hand side of the Equivalence."""
        return self.args[0]

    @property
    def rhs(self):
        """The right-hand side of the Equivalence."""
        return self.args[1]

    @classmethod
    def interactive_new(cls, lhs, rhs):
        if not isinstance(lhs, Formula):
            raise TypeError(f'{lhs} is not a Formula')
        if not isinstance(rhs, Formula):
            raise TypeError(f'{rhs} is not a Formula')
        return cls(lhs, rhs)

    def __init__(self, lhs, rhs):
        self.func = Equivalent
        self.args = (lhs, rhs)

    def to_nnf(self, implicit_not: bool = False,
               to_positive: bool = True) -> Formula:
        tmp = And(Implies(self.lhs, self.rhs), Implies(self.rhs, self.lhs))
        return tmp.to_nnf(implicit_not=implicit_not, to_positive=to_positive)

    def simplify(self, Theta=None):
        """Recursively simplify the Equivalence.

        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import x, y
        >>> e1 = EQUIV(~ EQ(x, y), F)
        >>> e1.simplify()
        Eq(x, y)
        """
        lhs = self.lhs.simplify(Theta=Theta)
        rhs = self.rhs.simplify(Theta=Theta)
        if lhs is T:
            return rhs
        if rhs is T:
            return lhs
        if lhs is F:
            if rhs.func is Not:
                return rhs.arg
            return Not(rhs)
        if rhs is F:
            if lhs.func is Not:
                return lhs.arg
            return Not(lhs)
        if lhs == rhs:
            return True
        return Equivalent(lhs, rhs)


EQUIV = Equivalent.interactive_new


class Implies(BooleanFormula):

    print_style = 'infix'
    print_precedence = 10
    text_symbol = '-->'
    latex_symbol = '\\longrightarrow'

    sympy_func = sympy.Implies

    @property
    def lhs(self):
        """The left-hand side of the Implies."""
        return self.args[0]

    @property
    def rhs(self):
        """The right-hand side of the Implies."""
        return self.args[1]

    @classmethod
    def interactive_new(cls, lhs, rhs):
        if not isinstance(lhs, Formula):
            raise TypeError(f'{lhs} is not a Formula')
        if not isinstance(rhs, Formula):
            raise TypeError(f'{rhs} is not a Formula')
        return cls(lhs, rhs)

    def __init__(self, lhs, rhs):
        self.func = Implies
        self.args = (lhs, rhs)

    def simplify(self, Theta=None):
        if self.rhs is T:
            return self.lhs
        lhs_simplify = self.lhs.simplify(Theta=Theta)
        if lhs_simplify is F:
            return T
        rhs_simplify = self.rhs.simplify(Theta=Theta)
        if rhs_simplify is T:
            return T
        if lhs_simplify is T:
            return rhs_simplify
        if rhs_simplify is F:
            return involutive_not(lhs_simplify)
        assert {lhs_simplify, rhs_simplify}.isdisjoint({T, F})
        if lhs_simplify == rhs_simplify:
            return T
        return Implies(lhs_simplify, rhs_simplify)

    def to_nnf(self, implicit_not: bool = False,
               to_positive: bool = True) -> Formula:
        if self.rhs.func is Or:
            tmp = Or(Not(self.lhs), *self.rhs.args)
        else:
            tmp = Or(Not(self.lhs), self.rhs)
        return tmp.to_nnf(implicit_not=implicit_not, to_positive=to_positive)


IMPL = Implies.interactive_new


class AndOr(BooleanFormula):

    print_style = 'infix'
    print_precedence = 50

    func: type[AndOr]

    @staticmethod
    @abstractmethod
    def to_dual(conditional: bool = True):
        ...

    @classmethod
    def interactive_new(cls, *args):
        for arg in args:
            if not isinstance(arg, Formula):
                raise TypeError(f'{arg} is not a Formula')
        args_flat = []
        for arg in args:
            if arg.func is cls:
                args_flat.extend(list(arg.args))
            else:
                args_flat.append(arg)
        return cls(*args_flat)

    def simplify(self, Theta=None):
        """Simplification.

        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import x, y, z
        >>> And(EQ(x, y), T, EQ(x, y), And(EQ(x, z), EQ(x, x + z))).simplify()
        And(Eq(x, y), Eq(x, z), Eq(x, x + z))
        >>> f = Or(EQ(x, 0), Or(EQ(x, 1), EQ(x, 2)), And(EQ(x, y), EQ(x, z)))
        >>> f.simplify()
        Or(Eq(x, 0), Eq(x, 1), Eq(x, 2), And(Eq(x, y), Eq(x, z)))
        """
        gAnd = And.to_dual(conditional=self.func is Or)
        gT = _T.to_dual(conditional=self.func is Or)()
        gF = _F.to_dual(conditional=self.func is Or)()
        # gAnd is an AndOr func, gT and gF are complete TruthValue singletons
        simplified_args = []
        for arg in self.args:
            arg_simplify = arg.simplify()
            if arg_simplify is gF:
                return gF
            if arg_simplify is gT:
                continue
            if arg_simplify in simplified_args:
                continue
            if arg_simplify.func is gAnd:
                simplified_args.extend(arg_simplify.args)
            else:
                simplified_args.append(arg_simplify)
        if not simplified_args:
            return gT
        return gAnd(*simplified_args)

    def to_nnf(self, implicit_not: bool = False,
               to_positive: bool = True) -> Self:
        """Convert to Negation Normal Form.
        """
        func_nnf = self.func.to_dual(conditional=implicit_not)
        args_nnf = []
        for arg in self.args:
            arg_nnf = arg.to_nnf(implicit_not=implicit_not,
                                 to_positive=to_positive)
            if arg_nnf.func is func_nnf:
                args_nnf += arg_nnf.args
            else:
                args_nnf += [arg_nnf]
        return func_nnf(*args_nnf)

    def _to_pnf(self) -> dict:
        """Convert to Prenex Normal Form. self must be in NNF.
        """

        def interchange(self: AndOr, q: Union[type[Ex], type[All]]) -> Formula:
            quantifiers = []
            quantifier_positions = set()
            args = list(self.args)
            while True:
                found_quantifier = False
                for i, arg_i in enumerate(args):
                    while arg_i.func is q:
                        quantifiers += [(q, arg_i.var)]
                        arg_i = arg_i.arg
                        quantifier_positions |= {i}
                        found_quantifier = True
                    args[i] = arg_i
                if not found_quantifier:
                    break
                q = q.to_dual()
            # The lifting of quantifiers above can introduce direct nested
            # ocurrences of self.func, which is one of And, Or. We
            # flatten those now, but not any others.
            args_pnf = []
            for i, arg in enumerate(args):
                if i in quantifier_positions and arg.func is self.func:
                    args_pnf += arg.args
                else:
                    args_pnf += [arg]
            pnf: Formula = self.func(*args_pnf)
            for q, v in reversed(quantifiers):
                pnf = q(v, pnf)
            return pnf

        L1 = []
        L2 = []
        for arg in self.args:
            d = arg._to_pnf()
            L1.append(d[Ex])
            L2.append(d[All])
        phi1 = interchange(self.func(*L1), Ex)
        phi2 = interchange(self.func(*L2), All)
        if phi1.func is not Ex and phi2.func is not All:
            # self is quantifier-free
            return {Ex: self, All: self}
        phi1_alternations = phi1.count_alternations()
        phi2_alternations = phi2.count_alternations()
        d = {}
        if phi1_alternations == phi2_alternations:
            d[Ex] = phi1 if phi1.func is Ex else phi2
            d[All] = phi2 if phi2.func is All else phi1
            return d
        if phi1_alternations < phi2_alternations:
            d[Ex] = d[All] = phi1
            return d
        d[Ex] = d[All] = phi2
        return d


class And(AndOr):
    """Constructor for conjunctions of Formulas.

    >>> from logic1.atomlib.sympy import EQ
    >>> from sympy.abc import x, y, z
    >>> And()
    T
    >>> And(EQ(0, 0))
    Eq(0, 0)
    >>> And(EQ(x, 0), EQ(x, y), EQ(y, z))
    And(Eq(x, 0), Eq(x, y), Eq(y, z))
    """
    text_symbol = '&'
    latex_symbol = '\\wedge'

    sympy_func = sympy.And

    @staticmethod
    def to_dual(conditional: bool = True):
        if conditional:
            return Or
        return And

    def __new__(cls, *args):
        if not args:
            return T
        if len(args) == 1:
            return args[0]
        return super().__new__(cls)

    def __init__(self, *args):
        self.func = And
        self.args = args


AND = And.interactive_new


class Or(AndOr):
    """Constructor for disjunctions of Formulas.

    >>> from logic1.atomlib.sympy import EQ
    >>> Or()
    F
    >>> Or(EQ(1, 0))
    Eq(1, 0)
    >>> Or(EQ(1, 0), EQ(2, 0), EQ(3, 0))
    Or(Eq(1, 0), Eq(2, 0), Eq(3, 0))
    """
    text_symbol = '|'
    latex_symbol = '\\vee'

    sympy_func = sympy.Or

    @staticmethod
    def to_dual(conditional: bool = True):
        if conditional:
            return And
        return Or

    def __new__(cls, *args):
        if not args:
            return F
        if len(args) == 1:
            return args[0]
        return super().__new__(cls)

    def __init__(self, *args):
        self.func = Or
        self.args = args


OR = Or.interactive_new


class Not(BooleanFormula):

    print_precedence = 99
    print_style = 'not'
    text_symbol = '~'
    latex_symbol = '\\neg'

    sympy_func = sympy.Not

    @staticmethod
    def to_dual(conditional: bool = True):
        return Not

    @property
    def arg(self):
        """The one argument of the Not."""
        return self.args[0]

    @classmethod
    def interactive_new(cls, arg):
        if not isinstance(arg, Formula):
            raise TypeError(f'{repr(arg)} is not a Formula')
        return cls(arg)

    def __init__(self, arg):
        self.func = Not
        self.args = (arg, )

    def simplify(self, Theta=None):
        """Simplification.

        >>> from logic1 import EX, ALL
        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import x, y, z
        >>> f = And(EQ(x, y), T, EQ(x, y), And(EQ(x, z), EQ(y, x)))
        >>> ~ ALL(x, EX(y, f)).simplify()
        Not(All(x, Ex(y, And(Eq(x, y), Eq(x, z), Eq(y, x)))))
        """
        arg_simplify = self.arg.simplify(Theta=Theta)
        if arg_simplify is T:
            return F
        if arg_simplify is F:
            return T
        return involutive_not(arg_simplify)

    def to_nnf(self, implicit_not: bool = False,
               to_positive: bool = True) -> Self:
        """Negation normal form.

        >>> from logic1 import EX, ALL
        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import x, y, z
        >>> f = ALL(x, EX(y, And(EQ(x, y), T, EQ(x, y), EQ(x, z) & EQ(y, x))))
        >>> (~f).to_nnf()
        Ex(x, All(y, Or(Ne(x, y), F, Ne(x, y), Ne(x, z), Ne(y, x))))
        """
        return self.arg.to_nnf(implicit_not=not implicit_not,
                               to_positive=to_positive)

    def _to_pnf(self) -> dict:
        """Convert to Prenex Normal Form. self must be in NNF.
        """
        return {Ex: self, All: self}


NOT = Not.interactive_new


def involutive_not(arg: Formula):
    """Construct a formula equivalent Not(arg) using the involutive law if
    applicable.

    >>> from logic1.atomlib.sympy import EQ
    >>> involutive_not(EQ(0, 0))
    Not(Eq(0, 0))
    >>> involutive_not(~EQ(1, 0))
    Eq(1, 0)
    >>> involutive_not(T)
    Not(T)
    """
    if isinstance(arg, Not):
        return arg.arg
    return Not(arg)


# The following imports are intentionally late to avoid circularity.
from .quantified import Ex, All
from .truth import _T, _F, T, F
