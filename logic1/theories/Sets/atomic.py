from __future__ import annotations

import logging
import string
from typing import Any, ClassVar, Final, Iterator, Never, Optional, Self, TypeAlias

from ... import firstorder
from ...firstorder import _F, _T


logging.basicConfig(
    format='%(levelname)s[%(relativeCreated)0.0f ms]: %(message)s',
    level=logging.CRITICAL)


oo = float('Inf')
"""A symbolic name for the float `inf` as an :data:`.Index`
"""

Index: TypeAlias = int | float
"""An index, which is either a positive integer or the float `inf`, which is
represented by :data:`oo`
"""


class VariableSet(firstorder.atomic.VariableSet['Variable']):
    """The infinite set of all variables belonging to the theory of Sets.
    Variables are uniquely identified by their name, which is a
    :external:class:`.str`. This class is a singleton, whose single instance is
    assigned to :data:`.VV`.

    .. seealso::
        Final methods inherited from parent class:

        * :meth:`.firstorder.atomic.VariableSet.get`
            -- obtain several variables simultaneously
        * :meth:`.firstorder.atomic.VariableSet.imp`
            -- import variables into global namespace
    """

    _instance: ClassVar[Optional[VariableSet]] = None

    @property
    def stack(self) -> list[set[str]]:
        return self._stack

    def __getitem__(self, index: str) -> Variable:
        """Implements abstract method
        :meth:`.firstorder.atomic.VariableSet.__getitem__`.
        """
        match index:
            case str():
                self._used.update((index,))
                return Variable(index)
            case _:
                raise ValueError(f'expecting string as index; {index} is {type(index)}')

    def __init__(self) -> None:
        self._stack: list[set[str]] = []
        self._used: set[str] = set()

    def __new__(cls) -> VariableSet:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        s = ', '.join(str(g) for g in (*self._used, '...'))
        return f'{{{s}}}'

    def fresh(self, suffix: str = '') -> Variable:
        """Return a fresh variable, by default from the sequence G0001, G0002,
        ..., G9999, G10000, ... This naming convention is inspired by Lisp's
        gensym(). If the optional argument :data:`suffix` is specified, the
        sequence G0001<suffix>, G0002<suffix>, ... is used instead.
        """
        i = 1
        v_as_str = f'G{i:04d}{suffix}'
        while v_as_str in self._used:
            i += 1
            v_as_str = f'G{i:04d}{suffix}'
        return self[v_as_str]

    def pop(self) -> None:
        self._used = self._stack.pop()

    def push(self) -> None:
        self._stack.append(self._used)
        self._used = set()


VV = VariableSet()


class Variable(firstorder.Variable['Variable', Never, str]):

    wrapped_variable_set: VariableSet = VV

    string: str

    def __eq__(self, other: Variable) -> Eq:  # type: ignore[override]
        if isinstance(other, Variable):
            return Eq(self, other)
        raise ValueError(f'arguments must be terms - {other} is {type(other)}')

    def __hash__(self) -> int:
        return hash((tuple(str(cls) for cls in self.__class__.mro()), self.string))

    def __init__(self, arg: str) -> None:
        if not isinstance(arg, str):
            raise ValueError(f'argument must be a string; {arg} is {type(arg)}')
        self.string = arg

    def __ne__(self, other: Variable) -> Ne:  # type: ignore[override]
        if isinstance(other, Variable):
            return Ne(self, other)
        raise ValueError(f'arguments must be terms; {other} is {type(other)}')

    def __repr__(self) -> str:
        return self.string

    def as_latex(self) -> str:
        """LaTeX representation as a string. Implements the abstract method
        :meth:`.firstorder.atomic.Term.as_latex`.
        """
        base = self.string.rstrip(string.digits)
        index = self.string[len(base):]
        if index:
            return f'{base}_{{{index}}}'
        return base

    def fresh(self) -> Variable:
        """Returns a variable that has not been used so far. Implements
        abstract method :meth:`.firstorder.atomic.Variable.fresh`.
        """
        return self.wrapped_variable_set.fresh(suffix=f'_{str(self)}')

    def sort_key(self) -> str:
        """A sort key suitable for ordering instances of this class. Implements
        the abstract method :meth:`.firstorder.atomic.Term.sort_key`.
        """
        return self.string

    def subs(self, d: dict[Variable, Variable]) -> Variable:
        """Simultaneous substitution of variables for variables.

        >>> from logic1.theories.Sets import VV
        >>> x, y, z = VV.get('x', 'y', 'z')
        >>> f = x
        >>> f.subs({x: y, y: z})
        y
        """
        return d.get(self, self)

    def vars(self) -> Iterator[Variable]:
        """An iterator that yields this variable. Implements the abstract
        method :meth:`.firstorder.atomic.Term.vars`.
        """
        yield self


class AtomicFormula(firstorder.AtomicFormula['AtomicFormula', 'Variable', 'Variable', Never]):

    def __le__(self, other: Formula) -> bool:
        """Returns `True` if this atomic formula should be sorted before or is
        equal to other. Implements abstract method
        :meth:`.firstorder.atomic.AtomicFormula.__le__`.
        """
        L: Final = [C, C_, Eq, Ne]
        match other:
            case AtomicFormula():
                if isinstance(self, (C, C_)) and isinstance(other, (Eq, Ne)):
                    return True
                if isinstance(self, (Eq, Ne)) and isinstance(other, (C, C_)):
                    return False
                match self:
                    case C() | C_():
                        assert isinstance(other, (C, C_))
                        if self.index == other.index:
                            return L.index(self.op) <= L.index(other.op)
                        return self.index <= other.index
                    case Eq() | Ne():
                        assert isinstance(other, (Eq, Ne))
                        if self.op != other.op:
                            return L.index(self.op) <= L.index(other.op)
                        if Variable.sort_key(self.lhs) != Variable.sort_key(other.lhs):
                            return Variable.sort_key(self.lhs) <= Variable.sort_key(other.lhs)
                        return Variable.sort_key(self.rhs) <= Variable.sort_key(other.rhs)
                    case _:
                        assert False, f'{self}: {type(self)}'
            case _:
                return True

    def __repr__(self) -> str:
        SYMBOL: Final = {Eq: '==', Ne: '!=', C: 'C', C_: 'C_'}
        SPACING: Final = ' '
        match self:
            case C() | C_():
                if self.index is oo:
                    return f'{SYMBOL[self.op]}(oo)'
                return f'{SYMBOL[self.op]}({self.index})'
            case Eq() | Ne():
                return f'{self.lhs}{SPACING}{SYMBOL[self.op]}{SPACING}{self.rhs}'
            case _:
                assert False, f'{self}: {type(self)}'

    def __str__(self) -> str:
        """String representation of this atomic formula. Implements the
        abstract method :meth:`.firstorder.atomic.AtomicFormula.__str__`.
        """
        return repr(self)

    def as_latex(self) -> str:
        """Latex representation as a string. Implements the abstract method
        :meth:`.firstorder.atomic.AtomicFormula.as_latex`.
        """
        match self:
            case C(index=index) if index == oo:
                return f'C_\\infty'
            case C(index=index):
                return f'C_{{{index}}}'
            case C_(index=index) if index == oo:
                return f'\\overline{{C_\\infty}}'
            case C_(index=index):
                return f'\\overline{{C_{{{index}}}}}'
            case Eq(lhs=lhs, rhs=rhs):
                return f'{lhs.as_latex()} = {rhs.as_latex()}'
            case Ne(lhs=lhs, rhs=rhs):
                return f'{lhs.as_latex()} \\neq {rhs.as_latex()}'
            case _:
                assert False, f'{self}: {type(self)}'

    def bvars(self, quantified: frozenset[Variable] = frozenset()) -> Iterator[Variable]:
        """Iterate over occurrences of variables that are elements of
        `quantified`. Implements the abstract method
        :meth:`.firstorder.atomic.AtomicFormula.bvars`.
        """
        match self:
            case Eq() | Ne():
                yield from (v for v in (self.lhs, self.rhs) if v in quantified)
            case C() | C_():
                yield from ()
            case _:
                assert False, f'{self}: {type(self)}'

    @classmethod
    def complement(cls) -> type[AtomicFormula]:
        """Complement relation. Implements the abstract method
        :meth:`.firstorder.atomic.AtomicFormula.complement`.

        .. seealso::
          Inherited method :meth:`.firstorder.atomic.AtomicFormula.to_complement`
        """
        D: Any = {C: C_, C_: C, Eq: Ne, Ne: Eq}
        return D[cls]

    def fvars(self, quantified: frozenset[Variable] = frozenset()) -> Iterator[Variable]:
        """Iterate over occurrences of variables that are *not* elements of
        `quantified`. Implements the abstract method
        :meth:`.firstorder.atomic.AtomicFormula.fvars`.
        """
        match self:
            case Eq() | Ne():
                yield from (v for v in (self.lhs, self.rhs) if v not in quantified)
            case C() | C_():
                yield from ()
            case _:
                assert False, f'{self}: {type(self)}'

    def simplify(self) -> Formula:
        """Fast basic simplification. The result is equivalent to self.
        Implements the abstract method :meth:`.firstorder.atomic.AtomicFormula.simplify`.
        """
        match self:
            case Eq(lhs=lhs, rhs=rhs):
                if lhs == rhs:
                    return _T()
                if Variable.sort_key(lhs) > Variable.sort_key(rhs):
                    return rhs == lhs
            case Ne(lhs=lhs, rhs=rhs):
                if lhs == rhs:
                    return _F()
                if Variable.sort_key(lhs) > Variable.sort_key(rhs):
                    return rhs != lhs
            case C():
                if self.index == 1:
                    return _T()
            case C_():
                if self.index == 1:
                    return _F()
            case _:
                assert False, self
        return self

    def subs(self, d: dict[Variable, Variable]) -> Self:
        """Simultaneous substitution of variables for variables. Implements the
        abstract method :meth:`.firstorder.atomic.AtomicFormula.subs`.
        """
        match self:
            case C() | C_():
                return self  # type: ignore[return-value]
            case Eq() | Ne():
                return self.op(self.lhs.subs(d), self.rhs.subs(d))  # type: ignore[return-value]
            case _:
                assert False, f'{self}: {type(self)}'


class Eq(AtomicFormula):

    @property
    def lhs(self) -> Variable:
        return self.args[0]

    @property
    def rhs(self) -> Variable:
        return self.args[1]

    def __bool__(self) -> bool:
        return self.lhs.string == self.rhs.string

    def __init__(self, lhs: Variable, rhs: Variable) -> None:
        super().__init__()
        for arg in (lhs, rhs):
            if not isinstance(arg, Variable):
                raise ValueError(
                    f'arguments must be variables; {arg} is {type(arg)}')
        self.args = (lhs, rhs)


class Ne(AtomicFormula):

    @property
    def lhs(self) -> Variable:
        return self.args[0]

    @property
    def rhs(self) -> Variable:
        return self.args[1]

    def __bool__(self) -> bool:
        return self.lhs.string != self.rhs.string

    def __init__(self, lhs: Variable, rhs: Variable) -> None:
        super().__init__()
        for arg in (lhs, rhs):
            if not isinstance(arg, Variable):
                raise ValueError(
                    f'arguments must be variables - {arg} is {type(arg)}')
        self.args = (lhs, rhs)


class C(AtomicFormula):
    """Cardinality constraints. From a mathematical perspective, the instances
    are constant relation symbols with an index, which is either a positive
    integer or the float `inf`, represented as ``oo``. ``C(n)`` holds iff there
    are at least ``n`` different elements in the universe. This is not a
    statement about the index ``n`` but about a range of models where this
    constant relation holds.

    In the following example, ``f`` states that there should be at least 2
    elements but not 3 elements or more:

    >>> from logic1.firstorder import *
    >>> from logic1.theories.Sets import *
    >>> x, y, z = VV.get('x', 'y', 'z')
    >>> f = Ex([x, y], x != y) & All([x, y, z], Or(x == y, y == z, z == x))
    >>> qe(f)  # quantifier elimination:
    And(C(2), C_(3))

    The class constructor takes care that instances with equal indices are
    identical:

    >>> C(1) is C(1)
    True
    >>> C(1) == C(2)
    False
    """

    _instances: ClassVar[dict[Index, C]] = dict()

    @property
    def index(self) -> Index:
        """The index of the constant relation symbol
        """
        return self.args[0]

    def __init__(self, index: Index) -> None:
        """Implements abstract method
        :meth:`firstorder.formula.Formula.__init__`.
        """
        super().__init__()
        self.args = (index,)

    def __new__(cls, index: Index):
        if not (isinstance(index, int) and index > 0 or index == oo):
            raise ValueError(f'argument must be positive int or oo; '
                             f'{index} is {type(index)}')
        if index not in cls._instances:
            cls._instances[index] = super().__new__(cls)
        return cls._instances[index]


class C_(AtomicFormula):
    """Cardinality constraints. The class :class:`C_` is dual to :class:`C`;
    more precisely, for every index ``n``, we have that ``C_(n)`` is the dual
    relation of ``C(n)``, and vice versa.
    """

    _instances: ClassVar[dict[Index, C_]] = dict()

    @property
    def index(self) -> Index:
        """The index of the constant relation symbol
        """
        return self.args[0]

    def __init__(self, index: Index) -> None:
        """Implements abstract method
        :meth:`firstorder.formula.Formula.__init__`.
        """
        super().__init__()
        self.args = (index,)

    def __new__(cls, index: Index):
        if not (isinstance(index, int) and index > 0 or index == oo):
            raise ValueError(f'argument must be positive int or oo; '
                             f'{index} is {type(index)}')
        if index not in cls._instances:
            cls._instances[index] = super().__new__(cls)
        return cls._instances[index]


from .typing import Formula
