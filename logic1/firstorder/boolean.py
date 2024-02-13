from __future__ import annotations

from .formula import Formula
from ..support.decorators import classproperty

from ..support.tracing import trace  # noqa


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
        # discuss: To what extent does this check for 2 args?
        self.args = (lhs, rhs)


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


class AndOr(BooleanFormula):
    # The following would be abstract class variables, which are not available
    # at the moment.
    func: type[AndOr]  #: :meta private:
    dual_func: type[AndOr]  #: :meta private:
    definite_func: type[BooleanFormula]  #: :meta private:
    neutral_func: type[BooleanFormula]  #: :meta private:

    # Similarly the following would be an abstract instance variable:
    args: tuple[Formula, ...]  #: :meta private:


class And(AndOr):
    r"""A class whose instances are conjunctions in the sense that their
    toplevel operator represents the Boolean operator
    :math:`\wedge`.

    >>> from logic1.theories.Sets import Eq, VV
    >>> x, y, z, O = VV.set_vars('x', 'y', 'z', 'O')
    >>>
    >>> And()
    T
    >>>
    >>> And(Eq(O, O))
    O == O
    >>>
    >>> And(Eq(x, O), Eq(x, y), Eq(y, z))
    And(x == O, x == y, y == z)
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

    >>> from logic1.theories.RCF import VV
    >>> Or()
    F
    >>> x, = VV.get('x')
    >>> Or(x == 0)
    x == 0
    >>>
    >>> Or(x == 1, x == 2, x == 3)
    Or(x == 1, x == 2, x == 3)
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


def involutive_not(arg: Formula) -> Formula:
    """Construct a formula equivalent Not(arg) using the involutive law if
    applicable.

    >>> from logic1.theories.RCF import VV
    >>> x, = VV.get('x')
    >>> involutive_not(x == 0)
    Not(x == 0)
    >>> involutive_not(~ (x == 0))
    x == 0
    >>> involutive_not(T)
    Not(T)
    """
    if isinstance(arg, Not):
        return arg.arg
    return Not(arg)


# The following imports are intentionally late to avoid circularity.
from .truth import _T, _F, T, F
