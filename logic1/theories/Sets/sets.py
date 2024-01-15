from __future__ import annotations

from abc import ABCMeta
import logging
import sympy
from typing import ClassVar, Final, TypeAlias

from ... import atomlib
from ...firstorder import T, F
from ...support.decorators import classproperty


oo: TypeAlias = atomlib.sympy.oo
Term: TypeAlias = atomlib.sympy.Term
Variable: TypeAlias = atomlib.sympy.Variable


logging.basicConfig(
    format='%(levelname)s[%(relativeCreated)0.0f ms]: %(message)s',
    level=logging.CRITICAL)


class BinaryAtomicFormula(atomlib.sympy.BinaryAtomicFormula):

    def __str__(self) -> str:
        SYMBOL: Final = {Eq: '==', Ne: '!='}
        SPACING: Final = ' '
        return f'{self.lhs}{SPACING}{SYMBOL[self.func]}{SPACING}{self.rhs}'

    def as_latex(self) -> str:
        SYMBOL: Final = {Eq: '=', Ne: '\\neq'}
        SPACING: Final = ' '
        lhs = sympy.latex(self.lhs)
        rhs = sympy.latex(self.rhs)
        return f'{lhs}{SPACING}{SYMBOL[self.func]}{SPACING}{rhs}'

    def relations(self) -> list[ABCMeta]:
        return [C, C_, Eq, Ne]


class Eq(atomlib.generic.EqMixin, BinaryAtomicFormula):
    """Equations with only variables as terms.

    This implements that fact that the language of sets has no functions and,
    in particular, no constants.

    >>> from sympy.abc import x, y
    >>> Eq(x, y)
    Eq(x, y)

    >>> Eq(x, 0)
    Traceback (most recent call last):
    ...
    ValueError: 0 is not a variable

    >>> Eq(x + x, y)
    Traceback (most recent call last):
    ...
    ValueError: 2*x is not a variable
    """

    func: type[Eq]

    @classproperty
    def complement_func(cls):
        """The complement relation Ne of Eq.
        """
        return Ne

    @classproperty
    def converse_func(cls):
        """The converse relation Eq of Eq.
        """
        return Eq

    def __init__(self, *args):
        for arg in args:
            if not isinstance(arg, Variable):
                raise ValueError(f"{arg!r} is not a variable")
        super().__init__(*args)

    def simplify(self):
        c = self.lhs.compare(self.rhs)
        if c == 0:
            return T
        if c == 1:
            return Eq(self.rhs, self.lhs)
        assert c == -1
        return self


class Ne(atomlib.generic.NeMixin, BinaryAtomicFormula):
    """Inequations with only variables as terms.

    This implements that fact that the language of sets has no functions and,
    in particular, no constants.

    >>> from sympy.abc import x, y
    >>> Ne(y, x)
    Ne(y, x)

    >>> Ne(x, y + 1)
    Traceback (most recent call last):
    ...
    ValueError: y + 1 is not a variable
    """

    func: type[Ne]

    @classproperty
    def complement_func(cls):
        """The complement relation Eq of Ne.
        """
        return Eq

    @classproperty
    def converse_func(cls):
        """The converse relation Me of Ne.
        """
        return Ne

    def __init__(self, *args):
        for arg in args:
            if not isinstance(arg, Variable):
                raise ValueError(f"{arg!r} is not a variable")
        super().__init__(*args)

    def simplify(self):
        c = self.lhs.compare(self.rhs)
        if c == 0:
            return F
        if c == 1:
            return Ne(self.rhs, self.lhs)
        assert c == -1
        return self


class IndexedConstantAtomicFormula(atomlib.sympy.IndexedConstantAtomicFormula):

    def __str__(self) -> str:
        # Normally __str__ defaults to __repr__. However, we have inherited
        # Formula.__str__, which recursively calls exactly this method here.
        return repr(self)

    def as_latex(self) -> str:
        match self.index:
            case int() | sympy.Integer():  # no int admitted - discuss!
                k = str(self.index)
            case sympy.core.numbers.Infinity():
                k = '\\infty'
            case _:
                assert False
        match self.func:
            case C():
                return f'C_{{{k}}}'
            case C_():
                return f'\\overline{{C_{{{k}}}}}'
            case _:
                assert False

    def relations(self) -> list[ABCMeta]:
        return [C, C_, Eq, Ne]


class C(IndexedConstantAtomicFormula):
    r"""A class whose instances are cardinality constraints in the sense that
    their toplevel operator represents a constant relation symbol :math:`C_n`
    where :math:`n \in \mathbb{N} \cup \{\infty\}`. A typical interpretation in
    a domain :math:`D` is that :math:`C_n` holds iff :math:`|D| \geq n`.

    The class constructor takes one argument, which is the index `n`. It takes
    care that instance with equal indices are identical.

    >>> c_0_1 = C(0)
    >>> c_0_2 = C(0)
    >>> c_oo = C(oo)
    >>> c_0_1 is c_0_2
    True
    >>> c_0_1 == c_oo
    False
    """

    # Class variables
    func: type[C]  #: :meta private:
    """A type annotation for the class property `func` inherited from
    :attr:`.firstorder.AtomicFormula.func`.
    """

    @classproperty
    def complement_func(cls):
        """A class property yielding the complement class :class:`C_` of
        :class:`C`.
        """
        return C_

    _instances: ClassVar[dict] = {}
    """A private class variable, which is a dictionary holding unique instances
    of `C(n)` with key `n`.
    """

    # Instance variables
    args: tuple[atomlib.sympy.Index]  #: :meta private:
    """A type annotation for the property `func` inherited from
    :attr:`.firstorder.AtomicFormula.func`.

    :meta private:
    """

    def __repr__(self):
        return f'C({self.index})'

    def _sprint(self, mode: str) -> str:
        if mode == 'text':
            return repr(self)
        assert mode == 'latex', f'bad print mode {mode!r}'
        k = str(self.index) if isinstance(self.index, int) else '\\infty'
        return f'C_{{{k}}}'


class C_(IndexedConstantAtomicFormula):
    r"""A class whose instances are cardinality constraints in the sense that
    their toplevel operator represents a constant relation symbol
    :math:`\bar{C}_n` where :math:`n \in \mathbb{N} \cup \{\infty\}`. A typical
    interpretation in a domain :math:`D` is that :math:`\bar{C}_n` holds iff
    :math:`|D| < n`.

    The class constructor takes one argument, which is the index `n`. It takes
    care that instance with equal indices are identical.

    >>> c_0_1 = C_(0)
    >>> c_0_2 = C_(0)
    >>> c_oo = C_(oo)
    >>> c_0_1 is c_0_2
    True
    >>> c_0_1 == c_oo
    False
    """

    # Class variables
    func: type[C_]  #: :meta private:
    """A type annotation for the class property `func` inherited from
    :attr:`.firstorder.AtomicFormula.func`.
    """

    @classproperty
    def complement_func(cls):
        """A class property yielding the complement class :class:`C` of
        :class:'C_'.
        """
        return C

    _instances: ClassVar[dict] = {}
    """A private class variable, which is a dictionary holding unique instances
    of `C_(n)` with key `n`.
    """

    # Instance variables
    args: tuple[atomlib.sympy.Index]
    """A type annotation for the property `func` inherited from
    :attr:`.firstorder.AtomicFormula.func`.

    :meta private:
    """

    def __repr__(self) -> str:
        return f'C_({self.index})'

    def _sprint(self, mode: str) -> str:
        if mode == 'text':
            return repr(self)
        assert mode == 'latex', f'bad print mode {mode!r}'
        k = str(self.index) if isinstance(self.index, int) else '\\infty'
        return f'\\overline{{C_{{{k}}}}}'
