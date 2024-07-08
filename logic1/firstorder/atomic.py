from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
from types import FrameType
from typing import Any, final, Generic, Iterator, Self, Sequence, TypeVar

from .formula import Formula

from ..support.tracing import trace  # noqa


V = TypeVar('V', bound='Variable')


class _VariableSet(ABC, Generic[V]):

    @property
    @abstractmethod
    def stack(self) -> Sequence[object]:
        ...

    @abstractmethod
    def __getitem__(self, index: str) -> V:
        ...

    def get(self, *args) -> tuple[V, ...]:
        return tuple(self[name] for name in args)

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

    @abstractmethod
    def pop(self) -> None:
        ...

    @abstractmethod
    def push(self) -> None:
        ...


class Term(ABC, Generic[V]):

    @abstractmethod
    def as_latex(self) -> str:
        """LaTeX representation.

        .. seealso::
            :meth:`.Formula.as_latex` -- LaTeX representation
        """
        ...

    @staticmethod
    @abstractmethod
    def sort_key(term: Any) -> Any:
        """A sort key suitable for ordering terms. Note that ``<``, ``<=`` etc.
        are reserved as constructors for instances of :class:`.AtomicFormula`.
        """
        ...

    @abstractmethod
    def vars(self) -> Iterator[V]:
        """All variables occurring in self.

        .. seealso::
            * :meth:`.Formula.bvars` -- all occurring bound variables
            * :meth:`.Formula.fvars` -- all occurring free variables
            * :meth:`.Formula.qvars` -- all occurring quantified variables
        """
        ...


class Variable(Term, Generic[V]):

    @abstractmethod
    def fresh(self) -> V:
        """Returns a variable that has not been used so far.
        """
        ...


class AtomicFormula(Formula, Generic[V]):

    @classmethod
    @abstractmethod
    def complement(cls):
        """The complement operator of an atomic formula.
        """
        ...

    @abstractmethod
    def __str__(self) -> str:
        # Overloading __str__ here breaks an infinite recursion in the
        # inherited Formula.__str__. Nicer string representations are provided
        # by various theory modules.
        ...

    @abstractmethod
    def as_latex(self) -> str:
        """Latex representation.

        .. seealso::
            :meth:`.Formula.as_latex` -- LaTeX representation
        """
        ...

    @abstractmethod
    def _bvars(self, quantified: set) -> Iterator[V]:
        ...

    @abstractmethod
    def _fvars(self, quantified: set) -> Iterator[V]:
        ...

    @abstractmethod
    def simplify(self) -> Formula:
        ...

    @abstractmethod
    def subs(self, substitution: dict) -> Self:
        """
        .. seealso::
            :meth:`.Formula.subs` -- substitution
        """
        ...

    @final
    def to_complement(self) -> Self:
        """Returns an :class:`AtomicFormula` equivalent to ``Not(self)``.

        .. seealso::
            :attr:`complement` -- complement relation
        """
        return self.complement()(*self.args)
