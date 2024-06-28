from __future__ import annotations

import functools
import inspect
import logging
import string
from types import FrameType
from typing import Any, ClassVar, Final, Iterator, Optional, TypeAlias

from ... import firstorder
from ...firstorder import F, Formula, T


logging.basicConfig(
    format='%(levelname)s[%(relativeCreated)0.0f ms]: %(message)s',
    level=logging.CRITICAL)


oo = float('Inf')

Index: TypeAlias = int | float


class _VariableSet:
    """Instances of the singleton VariableSet register, store, and provide
    variables, which are instances of Terms and suitable for building complex
    instances of Terms using operators and methods defined in :class:`Term`.
    """

    _instance: ClassVar[Optional[_VariableSet]] = None
    _stack: list[set[str]]
    _used: set[str]

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

    def get(self, *args) -> tuple[Variable, ...]:
        return tuple(self[name] for name in args)

    def __getitem__(self, index: str) -> Variable:
        match index:
            case str():
                self._used.update((index,))
                return Variable(index)
            case _:
                raise ValueError(f'expecting string as index; {index} is {type(index)}')

    def imp(self, *args) -> None:
        """Import variables into global namespace.
        """
        vars_ = self.get(*args)
        frame = inspect.currentframe()
        assert isinstance(frame, FrameType)
        frame = frame.f_back
        try:
            assert isinstance(frame, FrameType)
            module = frame.f_globals['__name__']
            assert module == '__main__', \
                f'expecting imp to be called from the top level of module __main__; ' \
                f'context is module {module}'
            function = frame.f_code.co_name
            assert function == '<module>', \
                f'expecting imp to be called from the top level of module __main__; ' \
                f'context is function {function} in module {module}'
            for v in vars_:
                frame.f_globals[str(v)] = v
        finally:
            # Compare Note here:
            # https://docs.python.org/3/library/inspect.html#inspect.Traceback
            del frame

    def __init__(self):
        self._stack = []
        self._used = set()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def pop(self) -> set[str]:
        self._used = self._stack.pop()
        return self._used

    def push(self) -> list[set[str]]:
        self._stack.append(self._used)
        self._used = set()
        return self._stack

    def __repr__(self):
        s = ', '.join(str(g) for g in (*self._used, '...'))
        return f'{{{s}}}'


VV = _VariableSet()


class Term(firstorder.Term):

    wrapped_variable_set: _VariableSet = VV

    string: str

    def __eq__(self, other: Term) -> Eq:  # type: ignore[override]
        if isinstance(other, Term):
            return Eq(self, other)
        raise ValueError(f'arguments must be terms - {other} is {type(other)}')

    def __hash__(self) -> int:
        return hash((tuple(str(cls) for cls in self.__class__.mro()), self.string))

    def __init__(self, arg: str) -> None:
        if not isinstance(arg, str):
            raise ValueError(f'argument must be a string; {arg} is {type(arg)}')
        self.string = arg

    def __ne__(self, other: Term) -> Ne:  # type: ignore[override]
        if isinstance(other, Term):
            return Ne(self, other)
        raise ValueError(f'arguments must be terms; {other} is {type(other)}')

    def __repr__(self) -> str:
        return self.string

    def as_latex(self) -> str:
        """Convert `self` to LaTeX.

        Implements the abstract method
        :meth:`logic1.firstorder.atomic.Term.as_Latex`.
        """
        base = self.string.rstrip(string.digits)
        index = self.string[len(base):]
        if index:
            return f'{base}_{{{index}}}'
        return base

    def fresh(self) -> Variable:
        return self.wrapped_variable_set.fresh(suffix=f'_{str(self)}')

    @staticmethod
    def sort_key(term: Term) -> str:
        return term.string

    def subs(self, d: dict[Variable, Term]) -> Term:
        return d.get(self, self)

    def vars(self) -> Iterator[Variable]:
        """Extract the set of variables occurring in `self`.

        Implements the abstract method
        :meth:`logic1.firstorder.atomic.Term.vars`.
        """
        yield self


Variable: TypeAlias = Term


@functools.total_ordering
class AtomicFormula(firstorder.AtomicFormula):

    @classmethod
    def complement(cls) -> type[AtomicFormula]:
        D: Any = {C: C_, C_: C, Eq: Ne, Ne: Eq}
        return D[cls]

    def __le__(self, other: Formula) -> bool:
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
                        if Term.sort_key(self.lhs) != Term.sort_key(other.lhs):
                            return Term.sort_key(self.lhs) <= Term.sort_key(other.lhs)
                        return Term.sort_key(self.rhs) <= Term.sort_key(other.rhs)
                    case _:
                        assert False, f'{self}: {type(self)}'
            case _:
                return True

    def __repr__(self) -> str:
        match self:
            case C() | C_():
                return super().__repr__()
            case Eq() | Ne():
                SYMBOL: Final = {Eq: '==', Ne: '!='}
                SPACING: Final = ' '
                return f'{self.lhs}{SPACING}{SYMBOL[self.op]}{SPACING}{self.rhs}'
            case _:
                assert False, f'{self}: {type(self)}'

    def as_latex(self) -> str:
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

    def _bvars(self, quantified: set) -> Iterator[Variable]:
        match self:
            case Eq() | Ne():
                yield from (v for v in (self.lhs, self.rhs) if v in quantified)
            case C() | C_():
                yield from ()
            case _:
                assert False, f'{self}: {type(self)}'

    def _fvars(self, quantified: set) -> Iterator[Variable]:
        match self:
            case Eq() | Ne():
                yield from (v for v in (self.lhs, self.rhs) if v not in quantified)
            case C() | C_():
                yield from ()
            case _:
                assert False, f'{self}: {type(self)}'

    def subs(self, d: dict[Variable, Term]) -> AtomicFormula:
        """Implements abstract :meth:`.firstorder.atomic.AtomicFormula.subs`.
        """
        match self:
            case C() | C_():
                return self
            case Eq() | Ne():
                return self.op(self.lhs.subs(d), self.rhs.subs(d))
            case _:
                assert False, f'{self}: {type(self)}'


class Eq(AtomicFormula):

    @property
    def lhs(self) -> Term:
        return self.args[0]

    @property
    def rhs(self) -> Term:
        return self.args[1]

    def __bool__(self) -> bool:
        return self.lhs.string == self.rhs.string

    def __init__(self, lhs: Term, rhs: Term) -> None:
        for arg in (lhs, rhs):
            if not isinstance(arg, Term):
                raise ValueError(
                    f'arguments must be variables; {arg} is {type(arg)}')
        super().__init__(lhs, rhs)

    def simplify(self) -> Formula:
        if self.lhs == self.rhs:
            return T
        if Term.sort_key(self.lhs) > Term.sort_key(self.rhs):
            return Eq(self.rhs, self.lhs)
        return self


class Ne(AtomicFormula):

    @property
    def lhs(self) -> Term:
        return self.args[0]

    @property
    def rhs(self) -> Term:
        return self.args[1]

    def __bool__(self) -> bool:
        return self.lhs.string != self.rhs.string

    def __init__(self, lhs: Term, rhs: Term) -> None:
        for arg in (lhs, rhs):
            if not isinstance(arg, Term):
                raise ValueError(
                    f'arguments must be variables - {arg} is {type(arg)}')
        super().__init__(lhs, rhs)

    def simplify(self) -> Formula:
        if self.lhs == self.rhs:
            return F
        if Term.sort_key(self.lhs) > Term.sort_key(self.rhs):
            return Ne(self.rhs, self.lhs)
        return self


class C(AtomicFormula):
    r"""A class whose instances are cardinality constraints in the sense that
    their toplevel operator represents a constant relation symbol :math:`C_n`
    where :math:`n \in \mathbb{N} \cup \{\infty\}`. A typical interpretation in
    a domain :math:`D` is that :math:`C_n` holds iff :math:`|D| \geq n`.

    The class constructor takes one argument, which is the index `n`. It takes
    care that instance with equal indices are identical.

    >>> c_1_1 = C(1)
    >>> c_1_2 = C(1)
    >>> c_oo = C(oo)
    >>> c_1_1 is c_1_2
    True
    >>> c_1_1 == c_oo
    False
    """

    _instances: ClassVar[dict[Index, C]] = dict()

    @property
    def index(self) -> Index:
        return self.args[0]

    def __new__(cls, index: Index):
        if not (isinstance(index, int) and index > 0 or index == oo):
            raise ValueError(f'argument must be positive int or oo; '
                             f'{index} is {type(index)}')
        if index not in cls._instances:
            cls._instances[index] = super().__new__(cls)
        return cls._instances[index]


class C_(AtomicFormula):
    r"""A class whose instances are cardinality constraints in the sense that
    their toplevel operator represents a constant relation symbol
    :math:`\bar{C}_n` where :math:`n \in \mathbb{N} \cup \{\infty\}`. A typical
    interpretation in a domain :math:`D` is that :math:`\bar{C}_n` holds iff
    :math:`|D| < n`.

    The class constructor takes one argument, which is the index `n`. It takes
    care that instance with equal indices are identical.

    >>> c_1_1 = C_(1)
    >>> c_1_2 = C_(1)
    >>> c_oo = C_(oo)
    >>> c_1_1 is c_1_2
    True
    >>> c_1_1 == c_oo
    False
    """

    _instances: ClassVar[dict[Index, C_]] = dict()

    @property
    def index(self) -> Index:
        return self.args[0]

    def __new__(cls, index: Index):
        if not (isinstance(index, int) and index > 0 or index == oo):
            raise ValueError(f'argument must be positive int or oo; '
                             f'{index} is {type(index)}')
        if index not in cls._instances:
            cls._instances[index] = super().__new__(cls)
        return cls._instances[index]
