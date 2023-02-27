from __future__ import annotations

from typing import Callable, Optional

import pyeda.inter  # type: ignore
from pyeda.inter import expr, exprvar
import sympy

from .formula import Formula
from ..support.containers import GetVars
from ..support.decorators import classproperty

# from ..support.tracing import trace


class BooleanFormula(Formula):
    """Boolean Formulas have a Boolean operator at the top level.

    An operator of a Formula is either a quantifier Ex, All or a Boolean
    operator And, Or, Not, Implies, Equivaelent, T, F. Note that members of
    BooleanFormula start, in the sense of prefix notation, with a Boolean
    operator but may have quantified subformulas deeper in the expression tree.
    """

    # Class variables
    latex_symbol: str
    latex_symbol_spacing = ' \\, '
    print_style: str
    text_symbol: str
    text_symbol_spacing = ' '

    func: type[BooleanFormula]

    # Instance variables
    args: tuple[Formula, ...]

    # Instance methods
    def _count_alternations(self) -> tuple[int, set]:
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

    def get_any_atom(self) -> Optional[AtomicFormula]:
        for arg in self.args:
            atom = arg.get_any_atom()
            if atom:
                return atom
        return None

    def get_qvars(self) -> set:
        qvars = set()
        for arg in self.args:
            qvars |= arg.get_qvars()
        return qvars

    def get_vars(self, assume_quantified: set = set()) -> GetVars:
        vars = GetVars()
        for arg in self.args:
            vars |= arg.get_vars(assume_quantified=assume_quantified)
        return vars

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

    def subs(self, substitution: dict) -> BooleanFormula:
        """Substitution.
        """
        return self.func(*(arg.subs(substitution) for arg in self.args))

    def to_cnf(self) -> Formula:
        """ Convert to Conjunctive Normal Form.

        >>> from logic1.atomlib.sympy import EQ, NE
        >>> from sympy.abc import a
        >>> ((EQ(a, 0) & NE(a, 1) | ~EQ(a, 0) | NE(a, 2)) >> NE(a, 1)).to_cnf()
        And(Or(Ne(a, 1), Eq(a, 2)), Or(Eq(a, 0), Ne(a, 1)))
        """
        return Not(self).to_dnf().to_nnf(implicit_not=True)

    def _to_distinct_vars(self, badlist: set) -> BooleanFormula:
        return self.func(*(arg._to_distinct_vars(badlist)
                           for arg in self.args))

    def to_dnf(self) -> BooleanFormula | AtomicFormula:
        """ Convert to Disjunctive Normal Form.

        >>> from logic1.atomlib.sympy import EQ, NE
        >>> from sympy.abc import a
        >>> ((EQ(a, 0) & NE(a, 1) | ~EQ(a, 0) | NE(a, 2)) >> NE(a, 1)).to_dnf()
        Or(Ne(a, 1), And(Eq(a, 0), Eq(a, 2)))
        """
        d: dict[AtomicFormula, exprvar] = {}
        self_pyeda = self._to_pyeda(d)
        # _to_pyeda() has populated the mutable dictionary d
        d_rev: dict[exprvar, AtomicFormula]
        d_rev = dict(map(reversed, d.items()))  # type: ignore
        dnf = self_pyeda.to_dnf()
        if not isinstance(dnf, pyeda.boolalg.expr.Constant):
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
        name = to_dict[self.func]
        xs = []
        for arg in self.args:
            assert isinstance(arg, (BooleanFormula, AtomicFormula))
            xs.append(arg._to_pyeda(d, c))
        return name(*xs, simplify=False)

    def transform_atoms(self, transformation: Callable) -> BooleanFormula:
        return self.func(*(arg.transform_atoms(transformation)
                           for arg in self.args))

    # Static methods
    @staticmethod
    def _from_pyeda(f: expr, d: dict[exprvar, AtomicFormula]) \
            -> BooleanFormula | AtomicFormula:
        if isinstance(f, pyeda.boolalg.expr._Zero):
            return F
        if isinstance(f, pyeda.boolalg.expr._One):
            return T
        if isinstance(f, pyeda.boolalg.expr.Variable):
            return d[f]
        if isinstance(f, pyeda.boolalg.expr.Complement):
            variable = pyeda.boolalg.expr.Not(f, simplify=True)
            return d[variable].to_complement()
        assert isinstance(f, pyeda.boolalg.expr.Operator)
        from_dict = {'Implies': Implies, 'Or': Or, 'And': And, 'Not': Not}
        func = from_dict[f.NAME]
        args = (BooleanFormula._from_pyeda(arg, d) for arg in f.xs)
        return func(*args)


class Equivalent(BooleanFormula):

    # Class variables
    latex_symbol = '\\longleftrightarrow'
    print_precedence = 10
    print_style = 'infix'
    sympy_func = sympy.Equivalent
    text_symbol = '<-->'

    @classproperty
    def func(cls):
        return cls

    # Instance variables
    args: tuple[Formula, Formula]

    @property
    def lhs(self) -> Formula:
        """The left-hand side of the Equivalence."""
        return self.args[0]

    @property
    def rhs(self) -> Formula:
        """The right-hand side of the Equivalence."""
        return self.args[1]

    # Class methods
    @classmethod
    def interactive_new(cls, lhs: Formula, rhs: Formula):
        if not isinstance(lhs, Formula):
            raise TypeError(f'{lhs} is not a Formula')
        if not isinstance(rhs, Formula):
            raise TypeError(f'{rhs} is not a Formula')
        return cls(lhs, rhs)

    # Instance methods
    def __init__(self, lhs: Formula, rhs: Formula) -> None:
        self.args = (lhs, rhs)

    def simplify(self, Theta=None) -> Formula:
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
            if isinstance(rhs, Not):
                return rhs.arg
            return Not(rhs)
        if rhs is F:
            if isinstance(lhs, Not):
                return lhs.arg
            return Not(lhs)
        if lhs == rhs:
            return T
        return Equivalent(lhs, rhs)

    def to_nnf(self, implicit_not: bool = False, to_positive: bool = True) \
            -> BooleanFormula | AtomicFormula:
        tmp = And(Implies(self.lhs, self.rhs), Implies(self.rhs, self.lhs))
        return tmp.to_nnf(implicit_not=implicit_not, to_positive=to_positive)


EQUIV = Equivalent.interactive_new


class Implies(BooleanFormula):

    # Class variables
    latex_symbol = '\\longrightarrow'
    print_precedence = 10
    print_style = 'infix'
    sympy_func = sympy.Implies
    text_symbol = '-->'

    @classproperty
    def func(cls):
        return cls

    # Instance variables
    args: tuple[Formula, Formula]

    @property
    def lhs(self) -> Formula:
        """The left-hand side of the Implies."""
        return self.args[0]

    @property
    def rhs(self) -> Formula:
        """The right-hand side of the Implies."""
        return self.args[1]

    # Class methods
    @classmethod
    def interactive_new(cls, lhs: Formula, rhs: Formula):
        if not isinstance(lhs, Formula):
            raise TypeError(f'{lhs} is not a Formula')
        if not isinstance(rhs, Formula):
            raise TypeError(f'{rhs} is not a Formula')
        return cls(lhs, rhs)

    # Instance methods
    def __init__(self, lhs: Formula, rhs: Formula) -> None:
        self.args = (lhs, rhs)

    def simplify(self, Theta=None) -> Formula:
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

    def to_nnf(self, implicit_not: bool = False, to_positive: bool = True) \
            -> BooleanFormula | AtomicFormula:
        if isinstance(self.rhs, Or):
            tmp = Or(Not(self.lhs), *self.rhs.args)
        else:
            tmp = Or(Not(self.lhs), self.rhs)
        return tmp.to_nnf(implicit_not=implicit_not, to_positive=to_positive)


IMPL = Implies.interactive_new


class AndOr(BooleanFormula):

    # Class variables
    print_precedence = 50
    print_style = 'infix'

    func: type[AndOr]
    dual_func: type[AndOr]

    # Instance variables
    args: tuple[Formula, ...]

    # Class methods
    @classmethod
    def interactive_new(cls, *args):
        for arg in args:
            if not isinstance(arg, Formula):
                raise TypeError(f'{arg} is not a Formula')
        args_flat = []
        for arg in args:
            if isinstance(arg, cls):
                args_flat.extend(list(arg.args))
            else:
                args_flat.append(arg)
        return cls(*args_flat)

    # Instance methods
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
        gAnd = And if self.func is And else Or
        gT = T if self.func is And else F
        gF = F if self.func is And else T
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
               to_positive: bool = True) -> AndOr:
        """Convert to Negation Normal Form.
        """
        func_nnf = self.dual_func if implicit_not else self.func
        args_nnf: list[Formula] = []
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

        def interchange(self: AndOr, q: type[Ex] | type[All]) -> Formula:
            quantifiers = []
            quantifier_positions = set()
            args = list(self.args)
            while True:
                found_quantifier = False
                for i, arg_i in enumerate(args):
                    while isinstance(arg_i, q):
                        # I think it follows from the type hints that arg_i is
                        # an instance of Ex or All, but mypy 1.0.1 cannot see
                        # that.
                        quantifiers += [(q, arg_i.var)]  # type: ignore
                        arg_i = arg_i.arg  # type: ignore
                        quantifier_positions |= {i}
                        found_quantifier = True
                    args[i] = arg_i
                if not found_quantifier:
                    break
                q = q.dual_func
            # The lifting of quantifiers above can introduce direct nested
            # ocurrences of self.func, which is one of And, Or. We
            # flatten those now, but not any other nestings.
            args_pnf: list[Formula] = []
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

    # Class variables
    latex_symbol = '\\wedge'
    sympy_func = sympy.And
    text_symbol = '&'

    @classproperty
    def func(cls):
        return cls

    @classproperty
    def dual_func(cls):
        return Or

    # Instance variables
    args: tuple[Formula, ...]

    # Class methods
    def __new__(cls, *args):
        if not args:
            return T
        if len(args) == 1:
            return args[0]
        return super().__new__(cls)

    # Instance methods
    def __init__(self, *args) -> None:
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

    # Class variables
    latex_symbol = '\\vee'
    sympy_func = sympy.Or
    text_symbol = '|'

    @classproperty
    def func(cls):
        return cls

    @classproperty
    def dual_func(cls):
        return And

    # Instance variables
    args: tuple[Formula, ...]

    # Class methods
    def __new__(cls, *args):
        if not args:
            return F
        if len(args) == 1:
            return args[0]
        return super().__new__(cls)

    # Instance methods
    def __init__(self, *args) -> None:
        self.args = args


OR = Or.interactive_new


class Not(BooleanFormula):

    # Class variables
    latex_symbol = '\\neg'
    print_precedence = 99
    print_style = 'not'
    sympy_func = sympy.Not
    text_symbol = '~'

    @classproperty
    def func(cls):
        return cls

    # Instance variables
    @property
    def arg(self) -> Formula:
        """The one argument of the Not."""
        return self.args[0]

    # Class methods
    @classmethod
    def interactive_new(cls, arg: Formula):
        if not isinstance(arg, Formula):
            raise TypeError(f'{repr(arg)} is not a Formula')
        return cls(arg)

    # Instance methods
    def __init__(self, arg: Formula) -> None:
        self.args = (arg, )

    def simplify(self, Theta=None) -> Formula:
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

    def to_nnf(self, implicit_not: bool = False, to_positive: bool = True) \
            -> Formula:
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


def involutive_not(arg: Formula) -> Formula:
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
from .atomic import AtomicFormula
from .quantified import Ex, All
from .truth import T, F
