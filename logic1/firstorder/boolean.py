"""We introduce formulas with Boolean toplevel operators as subclasses of
:class:`.Formula`.
"""
from __future__ import annotations

from typing import final, Optional

from .atomic import AtomicFormula, Term, Variable
from .formula import α, τ, χ, σ, Formula

from ..support.tracing import trace  # noqa


class BooleanFormula(Formula[α, τ, χ, σ]):
    r"""A class whose instances are Boolean formulas in the sense that their
    toplevel operator is one of the Boolean operators :math:`\top`,
    :math:`\bot`, :math:`\lnot`, :math:`\wedge`, :math:`\vee`,
    :math:`\longrightarrow`, :math:`\longleftrightarrow`.
    """
    pass


@final
class Equivalent(BooleanFormula[α, τ, χ, σ]):
    r"""A class whose instances are equivalences in the sense that their
    toplevel operator represents the Boolean operator
    :math:`\longleftrightarrow`.

    >>> from logic1.theories.RCF import *
    >>> x, = VV.get('x')
    >>> Equivalent(x >= 0, Or(x > 0, x == 0))
    Equivalent(x >= 0, Or(x > 0, x == 0))
    """

    def __init__(self, lhs: Formula[α, τ, χ, σ], rhs: Formula[α, τ, χ, σ]) -> None:
        super().__init__()
        self.args = (lhs, rhs)

    @property
    def lhs(self) -> Formula[α, τ, χ, σ]:
        """The left-hand side of the equivalence.

        .. seealso::
            * :attr:`args <.formula.Formula.op>` -- all arguments as a tuple
            * :attr:`op <.formula.Formula.op>` -- operator
        """
        return self.args[0]

    @property
    def rhs(self) -> Formula[α, τ, χ, σ]:
        """The right-hand side of the equivalence.

        .. seealso::
            * :attr:`args <.formula.Formula.op>` -- all arguments as a tuple
            * :attr:`op <.formula.Formula.op>` -- operator
        """
        return self.args[1]


@final
class Implies(BooleanFormula[α, τ, χ, σ]):
    """A class whose instances are equivalences in the sense that their
    toplevel operator represents the Boolean operator :math:`\\longrightarrow`.

    >>> from logic1.theories.RCF import *
    >>> x, = VV.get('x')
    >>> Implies(x == 0, x >= 0)
    Implies(x == 0, x >= 0)

    .. seealso::
        * :meth:`>>, __rshift__() <.formula.Formula.__rshift__>` -- \
            infix notation of :class:`Implies`
        * :meth:`\\<\\<, __lshift__() <.formula.Formula.__lshift__>` -- \
            infix notation of converse :class:`Implies`
    """  # noqa

    def __init__(self, lhs: Formula[α, τ, χ, σ], rhs: Formula[α, τ, χ, σ]) -> None:
        super().__init__()
        self.args = (lhs, rhs)

    @property
    def lhs(self) -> Formula[α, τ, χ, σ]:
        """The left-hand side of the implication.

        .. seealso::
            * :attr:`args <.formula.Formula.op>` -- all arguments as a tuple
            * :attr:`op <.formula.Formula.op>` -- operator
        """
        return self.args[0]

    @property
    def rhs(self) -> Formula[α, τ, χ, σ]:
        """The right-hand side of the implication.

        .. seealso::
            * :attr:`args <.formula.Formula.op>` -- all arguments as a tuple
            * :attr:`op <.formula.Formula.op>` -- operator
        """
        return self.args[1]


@final
class And(BooleanFormula[α, τ, χ, σ]):
    """A class whose instances are conjunctions in the sense that their
    toplevel operator represents the Boolean operator
    :math:`\\wedge`.

    >>> from logic1.theories.RCF import *
    >>> x, y, z = VV.get('x', 'y', 'z')
    >>> And()
    T
    >>> And(x == 0)
    x == 0
    >>> And(x == 1, x == y, y == z)
    And(x - 1 == 0, x - y == 0, y - z == 0)

    .. seealso::
        * :meth:`&, __and__() <.formula.Formula.__and__>` -- \
            infix notation of :class:`And`
        * :attr:`args <.formula.Formula.op>` -- all arguments as a tuple
        * :attr:`op <.formula.Formula.op>` -- operator
    """

    def __init__(self, *args: Formula[α, τ, χ, σ]) -> None:
        """
        >>> from logic1.theories.RCF import *
        >>> a, = VV.get('a')
        >>> And(a >= 0, a != 0)
        And(a >= 0, a != 0)
        """
        super().__init__()
        args_flat = []
        for arg in args:
            if isinstance(arg, And):
                args_flat.extend(list(arg.args))
            else:
                args_flat.append(arg)
        self.args = tuple(args_flat)

    def __new__(cls, *args: Formula[α, τ, χ, σ]):
        if not args:
            return T
        if len(args) == 1:
            return args[0]
        return super().__new__(cls)

    @classmethod
    def dual(cls) -> type[Or[α, τ, χ, σ]]:
        r"""A class method yielding the class :class:`Or`, which implements
        the dual operator :math:`\vee` of :math:`\wedge`.
        """
        return Or

    @classmethod
    def definite(cls) -> type[_F[α, τ, χ, σ]]:
        r"""A class method yielding the class :class:`_F`, which is the
        operator of the constant Formula :data:`F`. The definite is the dual of
        the neutral.
        """
        return _F

    @classmethod
    def definite_element(cls) -> _F[α, τ, χ, σ]:
        r"""A class method yielding the unique instance :data:`F` of the
        :class:`_F`.
        """
        return _F()

    @classmethod
    def neutral(cls) -> type[_T[α, τ, χ, σ]]:
        r"""A class method yielding the class :class:`_T`, which is the
        operator of the constant Formula :data:`T`. The neutral is the dual of
        the definite.
        """
        return _T

    @classmethod
    def neutral_element(cls) -> _T[α, τ, χ, σ]:
        r"""A class method yielding the unique instance :data:`T` of the
        :class:`_T`.
        """
        return _T()


@final
class Or(BooleanFormula[α, τ, χ, σ]):
    """A class whose instances are disjunctions in the sense that their
    toplevel operator represents the Boolean operator
    :math:`\\vee`.

    >>> from logic1.theories.RCF import *
    >>> x, = VV.get('x')
    >>> Or()
    F
    >>> Or(x == 0)
    x == 0
    >>> Or(x == 1, x == 2, x == 3)
    Or(x - 1 == 0, x - 2 == 0, x - 3 == 0)

    .. seealso::
        * :meth:`|, __or__() <.formula.Formula.__or__>` -- \
            infix notation of :class:`Or`
        * :attr:`args <.formula.Formula.op>` -- all arguments as a tuple
        * :attr:`op <.formula.Formula.op>` -- operator
    """

    def __init__(self, *args: Formula[α, τ, χ, σ]) -> None:
        """
        >>> from logic1.theories.RCF import *
        >>> a, = VV.get('a')
        >>> Or(a > 0, a == 0)
        Or(a > 0, a == 0)
        """
        super().__init__()
        args_flat = []
        for arg in args:
            if isinstance(arg, Or):
                args_flat.extend(list(arg.args))
            else:
                args_flat.append(arg)
        self.args = tuple(args_flat)

    def __new__(cls, *args: Formula[α, τ, χ, σ]):
        if not args:
            return F
        if len(args) == 1:
            return args[0]
        return super().__new__(cls)

    @classmethod
    def dual(cls) -> type[And[α, τ, χ, σ]]:
        r"""A class method yielding the class :class:`And`, which implements
        the dual operator :math:`\wedge` of :math:`\vee`.
        """
        return And

    @classmethod
    def definite(cls) -> type[_T[α, τ, χ, σ]]:
        r"""A class method yielding the class :class:`_T`, which is the
        operator of the constant Formula :data:`T`. The definite is the dual of
        the neutral.
        """
        return _T

    @classmethod
    def definite_element(cls) -> _T[α, τ, χ, σ]:
        r"""A class method yielding the unique instance :data:`T` of the
        :class:`_T`.
        """
        return _T()

    @classmethod
    def neutral(cls) -> type[_F[α, τ, χ, σ]]:
        r"""A class method yielding the class :class:`_F`, which is the
        operator of the constant Formula :data:`F`. The neutral is the dual of
        the definite.
        """
        return _F

    @classmethod
    def neutral_element(cls) -> _F[α, τ, χ, σ]:
        r"""A class method yielding the unique instance :data:`F` of the
        :class:`_F`.
        """
        return _F()


@final
class Not(BooleanFormula[α, τ, χ, σ]):
    """A class whose instances are negated formulas in the sense that their
    toplevel operator is the Boolean operator
    :math:`\\neg`.

    >>> from logic1.theories.RCF import *
    >>> a, = VV.get('a')
    >>> Not(a == 0)
    Not(a == 0)

    .. seealso::
        * :meth:`~, __invert__() <.formula.Formula.__invert__>` -- \
            short notation of :class:`Not`
    """

    def __init__(self, arg: Formula[α, τ, χ, σ]) -> None:
        """
        >>> from logic1.theories.RCF import *
        >>> a, = VV.get('a')
        >>> Not(a == 0)
        Not(a == 0)
        """
        super().__init__()
        self.args = (arg, )

    @property
    def arg(self) -> Formula[α, τ, χ, σ]:
        """The one argument of the operator :math:`\\neg`.

        .. seealso::
            * :attr:`args <.formula.Formula.op>` -- all arguments as a tuple
            * :attr:`op <.formula.Formula.op>` -- operator
        """
        return self.args[0]


def involutive_not(arg: Formula[α, τ, χ, σ]) -> Formula[α, τ, χ, σ]:
    """Construct a formula equivalent Not(arg) using the involutive law if
    applicable.

    >>> from logic1.theories.RCF import *
    >>> x, = VV.get('x')
    >>> involutive_not(x == 0)
    Not(x == 0)
    >>> involutive_not(Not(x == 0))
    x == 0
    >>> involutive_not(T)
    Not(T)
    """
    if isinstance(arg, Not):
        return arg.arg
    return Not(arg)


@final
class _T(BooleanFormula[α, τ, χ, σ]):
    """A singleton class whose sole instance represents the constant Formula
    that is always true.

    >>> _T()
    T
    >>> _T() is _T()
    True
    """

    # This is a quite basic implementation of a singleton class. It does not
    # support subclassing. We do not use a module because we need _T to be a
    # subclass itself.

    _instance: Optional[_T[α, τ, χ, σ]] = None

    def __init__(self) -> None:
        super().__init__()
        self.args = ()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return 'T'

    @classmethod
    def dual(cls) -> type[_F[α, τ, χ, σ]]:
        r"""A class method yielding the class :class:`_F`, which implements
        the dual operator :math:`\bot` of :math:`\top`.
        """
        return _F


T: _T['AtomicFormula', 'Term', 'Variable', object] = _T()
"""Support use as a constant without parentheses.

We instantiate type variables with their respective upper bounds, which is the
best we can do at the module level. `T` cannot be generic. Therefore, _T()
should be used within statically typed code instead.

    >>> T is _T()
    True
"""


@final
class _F(BooleanFormula[α, τ, χ, σ]):
    """A singleton class whose sole instance represents the constant Formula
    that is always false.

    >>> _F()
    F
    >>> _F() is _F()
    True
    """

    # This is a quite basic implementation of a singleton class. It does not
    # support subclassing. We do not use a module because we need _F to be a
    # subclass itself.

    _instance: Optional[_F[α, τ, χ, σ]] = None

    def __init__(self) -> None:
        super().__init__()
        self.args = ()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return 'F'

    @classmethod
    def dual(cls) -> type[_T[α, τ, χ, σ]]:
        r"""A class method yielding the class :class:`_T`, which implements
        the dual operator :math:`\top` of :math:`\bot`.
        """
        return _T


F: _F['AtomicFormula', 'Term', 'Variable', object] = _F()
"""Support use as a constant without parentheses.

We instantiate type variables with their respective upper bounds, which is the
best we can do at the module level. `F` cannot be generic. Therefore, _F()
should be used within statically typed code instead.

>>> F is _F()
True
"""
