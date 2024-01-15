from __future__ import annotations

from typing import TYPE_CHECKING

from .formula import Formula
from ..support.containers import GetVars
from ..support.decorators import classproperty

from ..support.tracing import trace  # noqa

if TYPE_CHECKING:
    from .atomic import AtomicFormula
    from .quantified import QuantifierBlock


class BooleanFormula(Formula):
    r"""A class whose instances are Boolean formulas in the sense that their
    toplevel operator is one of the Boolean operators :math:`\lnot`,
    :math:`\wedge`, :math:`\vee`, :math:`\longrightarrow`,
    :math:`\longleftrightarrow`.

    Note that members of :class:`BooleanFormula` may have subformulas with
    other logical operators deeper in the expression tree.
    """
    # The following would be abstract class variables, which are not available
    # at the moment.
    func: type[BooleanFormula]  #: :meta private:
    dual_func: type[BooleanFormula]  #: :meta private:

    # Similarly the following would be an abstract instance variable:
    args: tuple[Formula, ...]  #: :meta private:

    def get_qvars(self) -> set:
        """Implements the abstract method :meth:`Formula.get_qvars`.
        """
        qvars = set()
        for arg in self.args:
            qvars |= arg.get_qvars()
        return qvars

    def get_vars(self, assume_quantified: set = set()) -> GetVars:
        """Implements the abstract method :meth:`Formula.get_vars`.
        """
        vars = GetVars()
        for arg in self.args:
            vars |= arg.get_vars(assume_quantified=assume_quantified)
        return vars

    def matrix(self) -> tuple[Formula, list[QuantifierBlock]]:
        return self, []

    def subs(self, substitution: dict) -> BooleanFormula:
        """Implements the abstract method :meth:`Formula.subs`.
        """
        return self.func(*(arg.subs(substitution) for arg in self.args))


class Equivalent(BooleanFormula):
    r"""A class whose instances are equivalences in the sense that their
    toplevel operator represents the Boolean operator
    :math:`\longleftrightarrow`.
    """
    @classproperty
    def func(cls):
        """A class property yielding the class :class:`Equivalent` itself.
        """
        return cls

    # Instance variables
    args: tuple[Formula, Formula]

    @property
    def lhs(self) -> Formula:
        """The left-hand side of the equivalence."""
        return self.args[0]

    @property
    def rhs(self) -> Formula:
        """The right-hand side of the equivalence."""
        return self.args[1]

    def __init__(self, lhs: Formula, rhs: Formula) -> None:
        self.args = (lhs, rhs)

    def simplify(self) -> Formula:
        """Compare the parent method :meth:`Formula.simplify`.

        >>> from logic1.theories.Sets import Eq
        >>> from sympy.abc import x, y
        >>>
        >>> e1 = Equivalent(~ Eq(x, y), F)
        >>> e1.simplify()
        Eq(x, y)
        """
        lhs = self.lhs.simplify()
        rhs = self.rhs.simplify()
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

    def to_nnf(self, to_positive: bool = True,
               _implicit_not: bool = False) -> BooleanFormula | AtomicFormula:
        """Implements the abstract method :meth:`Formula.to_nnf`.
        """
        tmp = And(Implies(self.lhs, self.rhs), Implies(self.rhs, self.lhs))
        return tmp.to_nnf(to_positive=to_positive, _implicit_not=_implicit_not)


class Implies(BooleanFormula):
    r"""A class whose instances are equivalences in the sense that their
    toplevel operator represents the Boolean operator :math:`\longrightarrow`.
    """
    @classproperty
    def func(cls):
        """A class property yielding the class :class:`Equivalent` itself.
        """
        return cls

    # Instance variables
    args: tuple[Formula, Formula]

    @property
    def lhs(self) -> Formula:
        """The left-hand side of the implication."""
        return self.args[0]

    @property
    def rhs(self) -> Formula:
        """The right-hand side of the implication."""
        return self.args[1]

    def __init__(self, lhs: Formula, rhs: Formula) -> None:
        self.args = (lhs, rhs)

    def simplify(self) -> Formula:
        """Compare the parent method :meth:`Formula.simplify`.
        """
        if self.rhs is T:
            return self.lhs
        lhs_simplify = self.lhs.simplify()
        if lhs_simplify is F:
            return T
        rhs_simplify = self.rhs.simplify()
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

    def to_nnf(self, to_positive: bool = True,
               _implicit_not: bool = False) -> BooleanFormula | AtomicFormula:
        """Implements the abstract method :meth:`Formula.to_nnf`.
        """
        if isinstance(self.rhs, Or):
            tmp = Or(Not(self.lhs), *self.rhs.args)
        else:
            tmp = Or(Not(self.lhs), self.rhs)
        return tmp.to_nnf(to_positive=to_positive, _implicit_not=_implicit_not)


class AndOr(BooleanFormula):
    # The following would be abstract class variables, which are not available
    # at the moment.
    func: type[AndOr]  #: :meta private:
    dual_func: type[AndOr]  #: :meta private:
    definite_func: type[BooleanFormula]  #: :meta private:
    neutral_func: type[BooleanFormula]  #: :meta private:

    # Similarly the following would be an abstract instance variable:
    args: tuple[Formula, ...]  #: :meta private:

    def simplify(self):
        """Compare the parent method :meth:`Formula.simplify`.

        >>> from logic1.theories.RCF import Eq, ring
        >>> x, y, z = ring.set_vars('x', 'y', 'z')
        >>>
        >>> f1 = And(Eq(x, y), T, Eq(x, y), And(Eq(x, z), Eq(x, x + z)))
        >>> f1.simplify()
        And(Eq(x - y, 0), Eq(x - z, 0), Eq(-z, 0))
        >>>
        >>> f2 = Or(Eq(x, 0), Or(Eq(x, 1), Eq(x, 2)), And(Eq(x, y), Eq(x, z)))
        >>> f2.simplify()
        Or(Eq(x, 0), Eq(x - 1, 0), Eq(x - 2, 0), And(Eq(x - y, 0), Eq(x - z, 0)))
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

    def to_nnf(self, to_positive: bool = True,
               _implicit_not: bool = False) -> AndOr:
        """Implements the abstract method :meth:`Formula.to_nnf`.
        """
        func_nnf = self.dual_func if _implicit_not else self.func
        args_nnf: list[Formula] = []
        for arg in self.args:
            arg_nnf = arg.to_nnf(to_positive=to_positive,
                                 _implicit_not=_implicit_not)
            if arg_nnf.func is func_nnf:
                args_nnf += arg_nnf.args
            else:
                args_nnf += [arg_nnf]
        return func_nnf(*args_nnf)


class And(AndOr):
    r"""A class whose instances are conjunctions in the sense that their
    toplevel operator represents the Boolean operator
    :math:`\wedge`.

    >>> from logic1.theories.Sets import Eq
    >>> from sympy.abc import x, y, z, O
    >>>
    >>> And()
    T
    >>>
    >>> And(Eq(O, O))
    Eq(O, O)
    >>>
    >>> And(Eq(x, O), Eq(x, y), Eq(y, z))
    And(Eq(x, O), Eq(x, y), Eq(y, z))
    """
    @classproperty
    def func(cls):
        """A class property yielding the class :class:`And` itself.
        """
        return cls

    @classproperty
    def dual_func(cls):
        r"""A class property yielding the class :class:`Or`, which implements
        the dual operator :math:`\vee` of :math:`\wedge`.
        """
        return Or

    @classproperty
    def definite_func(cls):
        r"""A class property yielding the class :class:`_F`, which implements
        the definite operator :math:`\bot` of :math:`\wedge`. The definite
        operator is the dual of the neutral.

        Note that the return value :class:`_F` is the naked operator, in
        contrast to the formula :data:`F`.
        """
        return _F

    @classproperty
    def neutral_func(cls):
        r"""A class property yielding the class :class:`_T`, which implements
        the neutral operator :math:`\top` of :math:`\wedge`.

        Note that the return value :class:`_T` is the naked operator, in
        contrast to the formula :data:`T`.
        """
        return _T

    # Instance variables
    args: tuple[Formula, ...]

    def __new__(cls, *args: Formula):
        if not args:
            return T
        if len(args) == 1:
            return args[0]
        return super().__new__(cls)

    def __init__(self, *args: Formula) -> None:
        args_flat = []
        for arg in args:
            if isinstance(arg, And):
                args_flat.extend(list(arg.args))
            else:
                args_flat.append(arg)
        self.args = tuple(args_flat)


class Or(AndOr):
    r"""A class whose instances are disjunctions in the sense that their
    toplevel operator represents the Boolean operator
    :math:`\vee`.

    >>> from logic1.theories.RCF import Eq
    >>> Or()
    F
    >>>
    >>> Or(Eq(1, 0))
    Eq(1, 0)
    >>>
    >>> Or(Eq(1, 0), Eq(2, 0), Eq(3, 0))
    Or(Eq(1, 0), Eq(2, 0), Eq(3, 0))
    """
    @classproperty
    def func(cls):
        """A class property yielding the class :class:`Or` itself.
        """
        return cls

    @classproperty
    def dual_func(cls):
        r"""A class property yielding the class :class:`And`, which implements
        the dual operator :math:`\wedge` of :math:`\vee`.
        """
        return And

    @classproperty
    def definite_func(cls):
        r"""A class property yielding the class :class:`_T`, which implements
        the definite operator :math:`\top` of :math:`\vee`. The definite
        operator is the dual of the neutral.

        Note that the return value :class:`_T` is the naked operator, in
        contrast to the formula :data:`T`.
        """
        return _T

    @classproperty
    def neutral_func(cls):
        r"""A class property yielding the class :class:`_F`, which implements
        the neutral operator :math:`\bot` of :math:`\vee`.

        Note that the return value :class:`_F` is the naked operator, in
        contrast to the formula :data:`F`.
        """
        return _F

    # Instance variables
    args: tuple[Formula, ...]

    def __new__(cls, *args):
        if not args:
            return F
        if len(args) == 1:
            return args[0]
        return super().__new__(cls)

    def __init__(self, *args) -> None:
        args_flat = []
        for arg in args:
            if isinstance(arg, Or):
                args_flat.extend(list(arg.args))
            else:
                args_flat.append(arg)
        self.args = tuple(args_flat)


class Not(BooleanFormula):
    r"""A class whose instances are negated formulas in the sense that their
    toplevel operator is the Boolean operator
    :math:`\neg`.
    """
    @classproperty
    def func(cls):
        """A class property yielding the class :class:`Not` itself.
        """
        return cls

    # Instance variables
    args: tuple[Formula]

    @property
    def arg(self) -> Formula:
        r"""The one argument of the operator :math:`\neg`.
        """
        return self.args[0]

    def __init__(self, arg: Formula) -> None:
        self.args = (arg, )

    def simplify(self) -> Formula:
        """Compare the parent method :meth:`Formula.simplify`.

        >>> from logic1 import Ex, All
        >>> from logic1.theories.Sets import Eq
        >>> from sympy.abc import x, y, z
        >>>
        >>> f = And(Eq(x, y), T, Eq(x, y), And(Eq(x, z), Eq(y, x)))
        >>> ~ All(x, Ex(y, f)).simplify()
        Not(All(x, Ex(y, And(Eq(x, y), Eq(x, z)))))
        """
        arg_simplify = self.arg.simplify()
        if arg_simplify is T:
            return F
        if arg_simplify is F:
            return T
        return involutive_not(arg_simplify)

    def to_nnf(self, to_positive: bool = True,
               _implicit_not: bool = False) -> Formula:
        """Implements the abstract method :meth:`Formula.to_nnf`.

        >>> from logic1 import Ex, All
        >>> from logic1.theories.Sets import Eq
        >>> from sympy.abc import x, y, z
        >>>
        >>> f = All(x, Ex(y, And(Eq(x, y), T, Eq(x, y), Eq(x, z) & Eq(y, x))))
        >>> (~f).to_nnf()
        Ex(x, All(y, Or(Ne(x, y), F, Ne(x, y), Ne(x, z), Ne(y, x))))
        """
        return self.arg.to_nnf(to_positive=to_positive,
                               _implicit_not=not _implicit_not)


def involutive_not(arg: Formula) -> Formula:
    """Construct a formula equivalent Not(arg) using the involutive law if
    applicable.

    >>> from logic1.theories.RCF import Eq
    >>> involutive_not(Eq(0, 0))
    Not(Eq(0, 0))
    >>> involutive_not(~Eq(1, 0))
    Eq(1, 0)
    >>> involutive_not(T)
    Not(T)
    """
    if isinstance(arg, Not):
        return arg.arg
    return Not(arg)


# The following imports are intentionally late to avoid circularity.
from .atomic import AtomicFormula  # noqa
from .truth import _T, _F, T, F
