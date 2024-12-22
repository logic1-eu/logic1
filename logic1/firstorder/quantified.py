r"""We provide subclasses of :class:`Formula <.formula.Formula>` that implement
quanitfied formulas in the sense that their toplevel operator is a one of the
quantifiers :math:`\exists` or :math:`\forall`.
"""
from __future__ import annotations

from collections import deque
from typing import final, Sequence

from .atomic import Variable
from .formula import α, τ, χ, σ, Formula

from ..support.tracing import trace  # noqa


class QuantifiedFormula(Formula[α, τ, χ, σ]):
    r"""A class whose instances are quanitfied formulas in the sense that their
    toplevel operator is a one of the quantifiers :math:`\exists` or
    :math:`\forall`. Note that members of :class:`QuantifiedFormula` may have
    subformulas with other logical operators deeper in the expression tree.
    """
    @property
    def var(self) -> χ:
        """The variable of the quantifier.

        >>> from logic1.theories.RCF import *
        >>> x, y = VV.get('x', 'y')
        >>> f = All(x, Ex(y, x == y))
        >>> f.var
        x

        .. seealso::
            * :attr:`args <.formula.Formula.op>` -- all arguments as a tuple
            * :attr:`op <.formula.Formula.op>` -- operator
        """
        return self.args[0]

    @var.setter
    def var(self, value: χ) -> None:
        self.args = (value, *self.args[1:])

    @property
    def arg(self) -> Formula[α, τ, χ, σ]:
        """The subformula in the scope of the :class:`QuantifiedFormula`.

        >>> from logic1.theories.RCF import *
        >>> x, y = VV.get('x', 'y')
        >>> f = All(x, Ex(y, x == y))
        >>> f.arg
        Ex(y, x - y == 0)

        .. seealso::
            * :attr:`args <.formula.Formula.op>` -- all arguments as a tuple
            * :attr:`op <.formula.Formula.op>` -- operator
        """
        return self.args[1]

    def __init__(self, vars_: χ | Sequence[χ], arg: Formula[α, τ, χ, σ]) -> None:
        """Construct a quantified formula.

        >>> from logic1.theories.RCF import VV
        >>> a, b, x = VV.get('a', 'b', 'x')
        >>> All((a, b), Ex(x, a*x + b >= 0))
        All(a, All(b, Ex(x, a*x + b >= 0)))
        """
        assert self.op in (Ex, All)  # in lack of abstract class properties
        super().__init__()
        if not isinstance(arg, Formula):
            raise ValueError(f'{arg!r} is not a Formula')
        match vars_:
            case Variable():
                assert not isinstance(vars_, Sequence)
                self.args = (vars_, arg)
            case (Variable(), *_):
                f = arg
                for v in reversed(vars_[1:]):
                    f = self.op(v, f)
                self.args = (vars_[0], f)
            case _:
                raise ValueError(f'{vars_!r} is not a Variable')


@final
class Ex(QuantifiedFormula[α, τ, χ, σ]):
    r"""A class whose instances are existentially quanitfied formulas in the
    sense that their toplevel operator represents the quantifier symbol
    :math:`\exists`. Besides variables, the quantifier accepts sequences of
    variables as a shorthand.

    >>> from logic1.firstorder import *
    >>> from logic1.theories.RCF import *
    >>> x, y, z = VV.get('x', 'y', 'z')
    >>> Ex(x, x**2 == y)
    Ex(x, x^2 - y == 0)
    >>> Ex([x, y], And(x > 0, y > 0, z == x - y))
    Ex(x, Ex(y, And(x > 0, y > 0, x - y - z == 0)))
    """
    @classmethod
    def dual(cls) -> type[All[α, τ, χ, σ]]:
        r"""A class method yielding the class :class:`All`, which implements
        the dual operator :math:`\forall` of :math:`\exists`.
        """
        return All


@final
class All(QuantifiedFormula[α, τ, χ, σ]):
    r"""A class whose instances are universally quanitfied formulas in the
    sense that their toplevel operator represents the quantifier symbol
    :math:`\forall`. Besides variables, the quantifier accepts sequences of
    variables as a shorthand.

    >>> from logic1.theories.RCF import *
    >>> x, y = VV.get('x', 'y')
    >>> All(x, x**2 >= 0)
    All(x, x^2 >= 0)
    >>> All([x, y], (x + y)**2 >= 0)
    All(x, All(y, x^2 + 2*x*y + y^2 >= 0))
    """
    @classmethod
    def dual(cls) -> type[Ex[α, τ, χ, σ]]:
        """A class method yielding the dual class :class:`Ex` of class:`All`.
        """
        return Ex


class Prefix(deque[tuple[type[All | Ex], list[χ]]]):
    """Holds a quantifier prefix of a formula.

    >>> from logic1.theories.RCF import *
    >>> x, x0, epsilon, delta = VV.get('x', 'x0', 'epsilon', 'delta')
    >>> Prefix((All, [x0, epsilon]), (Ex, [delta]), (All, [x]))
    Prefix([(<class 'logic1.firstorder.quantified.All'>, [x0, epsilon]),
            (<class 'logic1.firstorder.quantified.Ex'>, [delta]),
            (<class 'logic1.firstorder.quantified.All'>, [x])])
    >>> print(_)
    All [x0, epsilon]  Ex [delta]  All [x]

    .. seealso::
        * :external:class:`collections.deque` -- for mehods inherited from double-ended queues
        * :meth:`matrix <.Formula.matrix>` -- the matrix of a prenex formula
        * :meth:`quantify <.Formula.quantify>` -- add quantifier prefix
    """

    def __init__(self, *blocks: tuple[type[All[α, τ, χ, σ] | Ex[α, τ, χ, σ]], list[χ]]) -> None:
        super().__init__(blocks)

    def __str__(self) -> str:
        return '  '.join(q.__name__ + ' ' + str(vars_) for q, vars_ in self)
