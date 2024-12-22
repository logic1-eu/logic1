"""Generic abstract classes specify atomic formulas, terms, and variables at
the first-order level, where the syntax and semantics of the underlying
theories is unknown. The classes primarily act as interfaces specifying methods
that are used as black boxes within :class:`.Formula` methods.
"""

from __future__ import annotations

from abc import abstractmethod
import inspect
from types import FrameType
from typing import Any, final, Generic, Iterator, Sequence, TypeVar

from .formula import α, τ, χ, σ, Formula

from ..support.tracing import trace  # noqa


κ = TypeVar('κ')
"""A type variable denoting a sort key.
"""


class VariableSet(Generic[χ]):
    """The infinite set of all variables of a theory. Variables are uniquely
    identified by their name, which is a :external:class:`str`. Subclasses
    within theories are singletons, and their unique instance is assigned to a
    module variable :data:`VV` there.

    .. seealso::
      Derived classes in various theories and their unique instances:
      :class:`.RCF.atomic.VariableSet`, :data:`.RCF.atomic.VV` for Real Closed
      Fields and :class:`.Sets.atomic.VariableSet`, :data:`.Sets.atomic.VV`
      for Sets.
    """

    @property
    @abstractmethod
    def stack(self) -> Sequence[object]:
        """The class internally keeps track of variables already used. This is
        relevant when creating unused variables via :meth:`.fresh`. The
        :attr:`stack` can hold such internal states.

        .. seealso::
          * :meth:`.push` -- push information to :attr:`.stack` and reset
          * :meth:`.pop` -- restore information from :attr:`.stack`
        """
        ...

    @abstractmethod
    def __getitem__(self, index: str) -> χ:
        """Obtain the unique variable with name `index`.

        >>> from logic1.theories import RCF
        >>> assert isinstance(RCF.VV, RCF.atomic.VariableSet)
        >>> x = RCF.VV['x']; x
        x
        >>> assert isinstance(x, RCF.atomic.Variable)

        .. seealso::
          * :meth:`get` -- obtain several variables simultaneously
          * :meth:`imp` -- import variables into global namespace
        """

    @final
    def get(self, *args: str) -> tuple[χ, ...]:
        """Obtain several variables simultaneously by their names.

        >>> from logic1.theories import RCF
        >>> assert isinstance(RCF.VV, RCF.atomic.VariableSet)
        >>> x, y = RCF.VV.get('x', 'y')
        >>> assert isinstance(x, RCF.atomic.Variable)
        >>> assert isinstance(y, RCF.atomic.Variable)

        .. seealso::
          * :meth:`__getitem__` -- obtain variable by its name
          * :meth:`imp` -- import variables into global namespace
        """
        return tuple(self[name] for name in args)

    @final
    def imp(self, *args: str) -> None:
        """Import variables into global namespace. This works only
        interactively, i.e., ``if __name__ == '__main__'``. Otherwise use
        :meth:`.get`.

        >>> if __name__ == '__main__':  # to prevent doctest failure
        ...     from logic1.theories import RCF
        ...     assert isinstance(RCF.VV, RCF.atomic.VariableSet)
        ...     RCF.VV.imp('x', 'y')
        ...     assert isinstance(x, RCF.atomic.Variable)
        ...     assert isinstance(y, RCF.atomic.Variable)

        .. seealso::
          * :meth:`__getitem__` -- obtain variable by its name
          * :meth:`get` -- obtain several variables simultaneously
        """
        vars_ = self.get(*args)
        frame = inspect.currentframe()
        assert isinstance(frame, FrameType)
        frame = frame.f_back
        try:
            assert isinstance(frame, FrameType)
            module = frame.f_globals['__name__']
            if module != '__main__':
                raise RuntimeError(
                    f'expecting imp to be called from the top level of module __main__; '
                    f'context is module {module}')
            function = frame.f_code.co_name
            if function != '<module>':
                raise RuntimeError(
                    f'expecting imp to be called from the top level of module __main__; '
                    f'context is function {function} in module {module}')
            for v in vars_:
                frame.f_globals[str(v)] = v
        finally:
            # Compare Note here:
            # https://docs.python.org/3/library/inspect.html#inspect.Traceback
            del frame

    @abstractmethod
    def pop(self) -> None:
        """Restore information about used variables from :attr:`stack`.
        """
        ...

    @abstractmethod
    def push(self) -> None:
        """Push information about used variables to :attr:`stack` and reset
        that information.
        """
        ...


class Term(Generic[τ, χ, σ, κ]):
    """This abstract class specifies an interface via the definition of
    abstract methods on terms required by :class:`.Formula`. The methods are
    supposed to be implemented for the various theories. We need a type
    variable <.firstorder.atomic.τ>` for this class itself, because `Self`
    cannot be used in the static method :meth:`.sort_key`.

    .. seealso::
      Derived classes in various theories: :class:`.RCF.atomic.Term` for Real
      Closed Fields.

    .. note::
      The theory :mod:`.logic1.theories.Sets` does not subclass :class:`.Term`.
      Since it has no function symbols, it can use instances of
      :class:`.Sets.atomic.Variable` as terms.
    """

    @abstractmethod
    def as_latex(self) -> str:
        """LaTeX representation as a string. This is required by
        :meth:`.Formula.as_latex` for the representation of quantified
        variables.
        """
        ...

    @abstractmethod
    def sort_key(self) -> κ:
        """A sort key suitable for ordering instances of :data:`τ
        <.firstorder.atomic.τ>`.

        .. note::
          We reserve Python's rich comparisons :external:obj:`__lt__
          <operator.__lt__>`, :external:obj:`__le__ <operator.__le__>` etc. as
          constructors for instances of subclasses of :class:`.AtomicFormula`.
          For example, :obj:`.RCF.atomic.Term.__lt__` constructs an inequality

          >>> from logic1.theories.RCF import *
          >>> a, b = VV.get('a', 'b')
          >>> a < b
          a - b < 0
          >>> type(_)
          <class 'logic1.theories.RCF.atomic.Lt'>

          As a consquence, rich comparisons are not available for defining an
          ordering on terms, and we instead provid a `key`, which can be used,
          e.g., with Python's :external:func:`sorted <sorted>`, or directly as
          follows:

          >>> Term.sort_key(a) < Term.sort_key(b)
          False
          >>> sorted([a, b], key=Term.sort_key)
          [b, a]

          In contrast, atomic formulas and, more generally, formulas support
          rich comparisons.
        """
        ...

    @abstractmethod
    def vars(self) -> Iterator[χ]:
        """An iterator over all occurring variables. Each occurring variable is
        reported once.

        .. seealso::
          * :meth:`.Formula.bvars` -- all occurring bound variables
          * :meth:`.Formula.fvars` -- all occurring free variables
          * :meth:`.Formula.qvars` -- all quantified variables
        """
        ...


class Variable(Term[χ, χ, σ, κ]):
    """This abstract class specifies an interface via the definition of
    abstract methods on variables required by Formula. The methods are supposed
    to be implemented for the various theories.

    .. seealso::
      Derived classes in various theories: :class:`.RCF.atomic.Variable` for
      Real Closed Fields and :class:`.Sets.atomic.Variable` for Sets.
    """

    @abstractmethod
    def fresh(self) -> χ:
        """Returns a variable that has not been used so far.
        """
        ...


class AtomicFormula(Formula[α, τ, χ, σ]):
    """This abstract class primarily specifies an interface via the
    definition of abstract methods on atomic formulas that are required by
    :class:`.Formula`. In addition, it provides some final implementations of
    such methods, where they do not depend on the syntax or sematic of the
    specific theory.

    .. seealso::
      Derived classes in various theories: :class:`.RCF.atomic.AtomicFormula`
      for Real Closed Fields and :class:`.Sets.atomic.AtomicFormula` for Sets.
    """

    @abstractmethod
    def __le__(self, other: Formula[α, τ, χ, σ]) -> bool:
        """Returns :external:obj:`True` if `self` should be sorted before or is
        equal to other. This method is required by the corresponding
        first-order method :meth:`.Formula.__le__`.
        """
        ...

    @abstractmethod
    def __str__(self) -> str:
        """Representation of this atomic formula used in printing. This method
        is required by the corresponding recursive first-order method.
        """
        #  Overloading here breaks an infinite recursion in the inherited
        #  method.
        ...

    @abstractmethod
    def as_latex(self) -> str:
        """Latex representation as a string. This method is required by the
        corresponding recursive first-order method :meth:`.Formula.as_latex`.
        """
        ...

    def as_redlog(self) -> str:
        """Redlog representation as a string. This method is required by the
        corresponding recursive first-order method :meth:`.Formula.as_redlog`.
        """
        raise NotImplementedError()

    @final
    def atoms(self: α) -> Iterator[α]:
        yield self

    @abstractmethod
    def bvars(self, quantified: frozenset[χ] = frozenset()) -> Iterator[χ]:
        """Iterate over occurrences of variables that are elements of
        `quantified`. Yield each such variable once for each term that it
        occurs in. This method is required by the corresponding recursive
        first-order method :meth:`.Formula.bvars`.
        """
        ...

    @classmethod
    @abstractmethod
    def complement(cls) -> type[α]:
        """The complement operator of an atomic formula, i.e.,
        :code:`a.complement(*a.args)` is an atomic formula equivalent to
        :code:`Not(a.op(*a.args))`.

        .. seealso::
          * :meth:`.to_complement` -- \
                generalization from relations to atomic formulas
        """
        ...

    @abstractmethod
    def fvars(self, quantified: frozenset[χ] = frozenset()) -> Iterator[χ]:
        """Iterate over occurrences of variables that are *not* elements of
        `quantified`. Yield each such variable once for each term that it
        occurs in. This method is required by the corresponding recursive
        first-order method :meth:`.Formula.fvars`.
        """
        ...

    @abstractmethod
    def simplify(self) -> Formula[α, τ, χ, σ]:
        """Fast basic simplification. The result is equivalent to self. This
        method is required by the corresponding recursive first-order method
        :meth:`.Formula.simplify`.
        """
        ...

    @abstractmethod
    def subs(self, substitution: dict[χ, τ | σ]) -> α:
        """Simultaneous substitution of terms from `τ` or constants from `σ`
        for variables from `χ`. This method is required by the corresponding
        recursive first-order method :meth:`.Formula.subs`.
        """
        ...

    @final
    def to_complement(self) -> α:
        """Returns an :class:`AtomicFormula` equivalent to ``Not(self)``.

        .. seealso::
            :attr:`complement` -- complement relation
        """
        return self.complement()(*self.args)
