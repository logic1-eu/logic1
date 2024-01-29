from __future__ import annotations

from dataclasses import dataclass
import functools
import inspect
import logging
import string
import sys
from types import FrameType
from typing import Any, ClassVar, Final, Iterator, Optional, TypeAlias

from ... import firstorder
from ...firstorder import F, Formula, T
from ...support.decorators import classproperty


logging.basicConfig(
    format='%(levelname)s[%(relativeCreated)0.0f ms]: %(message)s',
    level=logging.CRITICAL)


oo = float('Inf')

Index: TypeAlias = int | float


class _VariableRegistry:
    """Instances of the singleton VariableRegsitry register, store, and provide
    variables, which are instances of Terms and suitable for building complex
    insrtances of Terms using operators and methods defined in :class:`Term`.
    """

    _instance: ClassVar[Optional[_VariableRegistry]] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.variables = []
        self.stack = []

    def __repr__(self):
        return f'Term Algebra in {", ".join(self.variables)}'

    def add_var(self, v: str) -> Variable:
        return self.add_vars(v)[0]

    def add_vars(self, *args) -> tuple[Variable, ...]:
        added_as_str = list(args)
        old_as_str = self.variables
        for v in added_as_str:
            if not isinstance(v, str):
                raise ValueError(f'{v} is not a string')
            if v in old_as_str:
                raise ValueError(f'{v} is already a variable')
        new_as_str = sorted(old_as_str + added_as_str)
        self.variables = new_as_str
        added_as_var = (Variable(v) for v in added_as_str)
        return tuple(added_as_var)

    def get_vars(self) -> tuple[Variable, ...]:
        return tuple(Variable(v) for v in self.variables)

    def import_vars(self, force: bool = False):
        critical = []
        variables = self.get_vars()
        frame = inspect.currentframe()
        assert isinstance(frame, FrameType)
        frame = frame.f_back
        try:
            assert isinstance(frame, FrameType)
            name = frame.f_globals['__name__']
            assert name == '__main__', f'import_vars called from {name}'
            for v in variables:
                try:
                    expr = frame.f_globals[str(v)]
                    if expr != v:
                        critical.append(str(v))
                except KeyError:
                    pass
            for v in variables:
                if force or str(v) not in critical:
                    frame.f_globals[str(v)] = v
            if not force:
                if len(critical) == 1:
                    print(f'{critical[0]} has another value already, '
                          f'use force=True to overwrite ',
                          file=sys.stderr)
                elif len(critical) > 1:
                    print(f'{", ".join(critical)} have other values already, '
                          f'use force=True to overwrite ',
                          file=sys.stderr)
        finally:
            # Compare Note here:
            # https://docs.python.org/3/library/inspect.html#inspect.Traceback
            del frame

    def pop(self) -> list[str]:
        self.variables = self.stack.pop()
        return self.variables

    def push(self) -> list[list[str]]:
        self.stack.append(self.variables)
        return self.stack

    def set_vars(self, *args) -> tuple[Variable, ...]:
        self.variables = []
        return self.add_vars(*args)


VV = _VariableRegistry()


@dataclass
class Term(firstorder.Term):

    var: str

    @classmethod
    def fresh_variable(cls, suffix: str = '') -> Variable:
        """Return a fresh variable, by default from the sequence G0001, G0002,
        ..., G9999, G10000, ... This naming convention is inspired by Lisp's
        gensym(). If the optional argument :data:`suffix` is specified, the
        sequence G0001<suffix>, G0002<suffix>, ... is used instead.
        """
        vars_as_str = tuple(str(v) for v in VV.get_vars())
        i = 1
        v_as_str = f'G{i:04d}{suffix}'
        while v_as_str in vars_as_str:
            i += 1
            v_as_str = f'G{i:04d}{suffix}'
        v = VV.add_var(v_as_str)
        return v

    def __eq__(self, other: Term) -> Eq:  # type: ignore[override]
        if isinstance(other, Term):
            return Eq(self, other)
        raise ValueError(f'arguments must be terms - {other} is {type(other)}')

    def __hash__(self) -> int:
        return hash((tuple(str(cls) for cls in self.__class__.mro()), self.var))

    def __init__(self, arg: str) -> None:
        if not isinstance(arg, str):
            raise ValueError(f'argument must be a string; {arg} is {type(arg)}')
        self.var = arg

    def __ne__(self, other: Term) -> Ne:  # type: ignore[override]
        if isinstance(other, Term):
            return Ne(self, other)
        raise ValueError(f'arguments must be terms; {other} is {type(other)}')

    def __repr__(self) -> str:
        return self.var

    def as_latex(self) -> str:
        """Convert `self` to LaTeX.

        Implements the abstract method
        :meth:`logic1.firstorder.atomic.Term.as_Latex`.
        """
        base = self.var.rstrip(string.digits)
        index = self.var[len(base):]
        if index:
            return f'{base}_{{{index}}}'
        return base

    @staticmethod
    def sort_key(term: Term) -> str:
        return term.var

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

    @classproperty
    def complement_func(cls) -> type[AtomicFormula]:
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
                            return L.index(self.func) <= L.index(other.func)
                        return self.index <= other.index
                    case Eq() | Ne():
                        assert isinstance(other, (Eq, Ne))
                        if self.func != other.func:
                            return L.index(self.func) <= L.index(other.func)
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
                return f'{self.lhs}{SPACING}{SYMBOL[self.func]}{SPACING}{self.rhs}'
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
                return self.func(self.lhs.subs(d), self.rhs.subs(d))
            case _:
                assert False, f'{self}: {type(self)}'


class Eq(AtomicFormula):

    func: type[Eq]
    args: tuple[Term, Term]

    @property
    def lhs(self):
        return self.args[0]

    @property
    def rhs(self):
        return self.args[1]

    def __bool__(self) -> bool:
        return self.lhs.var == self.rhs.var

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

    func: type[Ne]
    args: tuple[Term, Term]

    @property
    def lhs(self):
        return self.args[0]

    @property
    def rhs(self):
        return self.args[1]

    def __bool__(self) -> bool:
        return self.lhs.var != self.rhs.var

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

    func: type[C]
    args: tuple[Index]

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

    func: type[C_]
    args: tuple[Index]

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
