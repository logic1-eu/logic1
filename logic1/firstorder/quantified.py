r"""We provide subclasses of :class:`Formula <.formula.Formula>` that implement
quanitfied formulas in the sense that their toplevel operator is a one of the
quantifiers :math:`\exists` or :math:`\forall`.
"""
from __future__ import annotations

from typing import Any, final, Sequence, TypeAlias

from .formula import Formula
from .atomic import Variable
from ..support.decorators import classproperty
from ..support.tracing import trace  # noqa


class QuantifiedFormula(Formula):
    r"""A class whose instances are quanitfied formulas in the sense that their
    toplevel operator is a one of the quantifiers :math:`\exists` or
    :math:`\forall`.
    """

    # Note that members of :class:`QuantifiedFormula` may have subformulas with
    # other logical operators deeper in the expression tree.

    # The following would be abstract class variables, which are not available
    # at the moment.
    func: Any  #: :meta private:
    dual_func: type[QuantifiedFormula]  #: :meta private:

    @property
    def var(self) -> Any:
        """The variable of the quantifier.

        >>> from logic1.theories.RCF import *
        >>> x, y = VV.get('x', 'y')
        >>> f = All(x, Ex(y, x == y))
        >>> f.var
        x
        """
        return self.args[0]

    @var.setter
    def var(self, value: Any) -> None:
        self.args = (value, *self.args[1:])

    @property
    def arg(self) -> Formula:
        """The subformula in the scope of the :class:`QuantifiedFormula`.

        >>> from logic1.theories.RCF import *
        >>> x, y = VV.get('x', 'y')
        >>> f = All(x, Ex(y, x == y))
        >>> f.arg
        Ex(y, x == y)
        """
        return self.args[1]

    def __init__(self, vars_: Variable | Sequence[Variable], arg: Formula) -> None:
        """Construct a quantified formula.

        >>> from logic1.theories.RCF import VV
        >>> a, b, x = VV.get('a', 'b', 'x')
        >>> All((a, b), Ex(x, a*x + b >= 0))
        All(a, All(b, Ex(x, a*x + b >= 0)))
        """
        assert self.func in (Ex, All)  # in lack of abstract class properties
        if not isinstance(arg, Formula):
            raise ValueError(f'{arg!r} is not a Formula')
        match vars_:
            case Variable():
                assert not isinstance(vars_, Sequence)
                self.args = (vars_, arg)
            case (Variable(), *_):
                f = arg
                for v in reversed(vars_[1:]):
                    f = self.func(v, f)
                self.args = (vars_[0], f)
            case _:
                raise ValueError(f'{vars_!r} is not a Variable')


@final
class Ex(QuantifiedFormula):
    r"""A class whose instances are existentially quanitfied formulas in the
    sense that their toplevel operator represents the quantifier symbol
    :math:`\exists`. Besides variables, the quantifier accepts sequences of
    variables as a shorthand.

    >>> from logic1.firstorder import *
    >>> from logic1.theories.RCF import *
    >>> x, y, z = VV.get('x', 'y', 'z')
    >>> Ex(x, x**2 == y)
    Ex(x, x^2 == y)
    >>> Ex([x, y], And(x > 0, y > 0, z == x - y))
    Ex(x, Ex(y, And(x > 0, y > 0, z == x - y)))
    """
    @classproperty
    def dual_func(cls):
        r"""A class property yielding the class :class:`All`, which implements
        the dual operator :math:`\forall` of :math:`\exists`.
        """
        return All


@final
class All(QuantifiedFormula):
    r"""A class whose instances are universally quanitfied formulas in the
    sense that their toplevel operator represents the quantifier symbol
    :math:`\forall`. Besides variables, the quantifier accepts sequences of
    variables as a shorthand.

    >>> from logic1.theories.RCF import *
    >>> x, y = VV.get('x', 'y')
    >>> All(x, x**2 >= 0)
    All(x, x^2 >= 0)
    >>> All([x, y], (x + y)**2 == x**2 + 2*x*y + y**2)
    All(x, All(y, x^2 + 2*x*y + y^2 == x^2 + 2*x*y + y^2))
    """
    @classproperty
    def dual_func(cls):
        """A class property yielding the dual class :class:`Ex` of class:`All`.
        """
        return Ex


QuantifierBlock: TypeAlias = tuple[type[All | Ex], list]
"""
.. seealso::
    :meth:`matrix <.Formula.matrix>` -- the matrix of a prenex formula
"""
