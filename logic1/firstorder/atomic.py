from __future__ import annotations

from abc import abstractmethod
import inspect
from types import FrameType
from typing import Any, final, Generic, Iterator, Sequence

from .formula import α, τ, χ, Formula

from ..support.tracing import trace  # noqa


class _VariableSet(Generic[χ]):

    @property
    @abstractmethod
    def stack(self) -> Sequence[object]:
        ...

    @abstractmethod
    def __getitem__(self, index: str) -> χ:
        ...

    def get(self, *args: str) -> tuple[χ, ...]:
        return tuple(self[name] for name in args)

    def imp(self, *args: str) -> None:
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


class Term(Generic[τ, χ]):

    @abstractmethod
    def as_latex(self) -> str:
        """LaTeX representation.

        .. seealso::
            :meth:`.Formula.as_latex` -- LaTeX representation
        """
        ...

    @staticmethod
    @abstractmethod
    def sort_key(term: τ) -> Any:
        """A sort key suitable for ordering terms. Note that ``<``, ``<=`` etc.
        are reserved as constructors for instances of :class:`.AtomicFormula`.
        """
        ...

    @abstractmethod
    def vars(self) -> Iterator[χ]:
        """All variables occurring in self.

        .. seealso::
            * :meth:`.Formula.bvars` -- all occurring bound variables
            * :meth:`.Formula.fvars` -- all occurring free variables
            * :meth:`.Formula.qvars` -- all occurring quantified variables
        """
        ...


class Variable(Term[χ, χ]):

    @abstractmethod
    def fresh(self) -> χ:
        """Returns a variable that has not been used so far.
        """
        ...


class AtomicFormula(Formula[α, τ, χ]):

    @classmethod
    @abstractmethod
    def complement(cls) -> type[α]:
        """The complement operator of an atomic formula.
        """
        ...

    @abstractmethod
    def __le__(self, other: Formula[α, τ, χ]) -> bool:
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

    @final
    def atoms(self: α) -> Iterator[α]:
        yield self

    @abstractmethod
    def _bvars(self, quantified: set[χ]) -> Iterator[χ]:
        ...

    @abstractmethod
    def _fvars(self, quantified: set[χ]) -> Iterator[χ]:
        ...

    @abstractmethod
    def simplify(self) -> Formula[α, τ, χ]:
        ...

    @abstractmethod
    def subs(self, substitution: dict[χ, τ]) -> α:
        """
        .. seealso::
            :meth:`.Formula.subs` -- substitution
        """
        ...

    @final
    def to_complement(self) -> α:
        """Returns an :class:`AtomicFormula` equivalent to ``Not(self)``.

        .. seealso::
            :attr:`complement` -- complement relation
        """
        return self.complement()(*self.args)
