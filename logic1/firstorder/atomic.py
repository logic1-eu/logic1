"""Generic abstract classes specify atomic formulas, terms, and variables at
the first-order level, where the syntax and semantics of the underlying
theories is unknown. The classes primarily act as interfaces specifying methods
that are used as black boxes within :class:`.Formula` methods.
"""

from __future__ import annotations

from abc import abstractmethod
import inspect
from types import FrameType
from typing import Any, final, Generic, Iterator, Sequence

from .formula import α, τ, χ, Formula

from ..support.tracing import trace  # noqa


class _VariableSet(Generic[χ]):
    """The infinite set of all variables of a theory. Variables are uniquely
    identified by their name, which is a :external:class:`str`. Subclasses
    within theories are singletons, and their unique instance is assigned to a
    module variable :data:`VV` there.

    .. seealso::
      * :class:`.RCF.atomic._VariableSet`
      * :class:`.Sets.atomic._VariableSet`
      * :data:`.RCF.atomic.VV`
      * :data:`.Sets.atomic.VV`
    """

    @property
    @abstractmethod
    def stack(self) -> Sequence[object]:
        """The class internally keeps track of variables already used. This is
        relevant when creating unused variables via :meth:`.Variable.fresh`.
        The :attr:`stack` can hold such internal states.

        .. seealso::
          * :meth:`.push` -- push current state to :attr:`.stack`
          * :meth:`.pop` -- pop current state from :attr:`.stack`
        """
        ...

    @abstractmethod
    def __getitem__(self, index: str) -> χ:
        """Obtain the unique variable with name `index`.

        >>> from logic1.theories import RCF
        >>> assert isinstance(RCF.VV, RCF.atomic._VariableSet)
        >>> x = RCF.VV['x']; x
        x
        >>> assert isinstance(x, RCF.atomic.Variable)

        .. seealso::
          * :meth:`get` -- simultaneously obtain several variables
          * :meth:`imp` -- import variables into global namespace
        """

    @final
    def get(self, *args: str) -> tuple[χ, ...]:
        """Simultaneously obtain several variables by their names.

        >>> from logic1.theories import RCF
        >>> assert isinstance(RCF.VV, RCF.atomic._VariableSet)
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
        ...     assert isinstance(RCF.VV, RCF.atomic._VariableSet)
        ...     RCF.VV.imp('x', 'y')
        ...     assert isinstance(x, RCF.atomic.Variable)
        ...     assert isinstance(y, RCF.atomic.Variable)

        .. seealso::
          * :meth:`__getitem__` -- obtain variable by its name
          * :meth:`get` -- simultaneously obtain several variables
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
    """This abstract class specifies an interface via the definition of
    abstract methods on terms required by :class:`.Formula`. The methods are
    supposed to be implemented for the various theories.

    .. seealso::
      * :class:`.RCF.atomic.Term`
      * :class:`.Sets.atomic.Term`
    """

    @abstractmethod
    def as_latex(self) -> str:
        """LaTeX representation as a string, which can be used elsewhere.

        .. seealso::
          :meth:`.Formula.as_latex` -- LaTeX representation
        """
        ...

    @staticmethod
    @abstractmethod
    def sort_key(term: τ) -> Any:
        """A sort key suitable for ordering instances of terms :data:`.τ`.

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
        """An iterator over all variables occurring in self.

        .. seealso::
          * :meth:`.Formula.bvars` -- all occurring bound variables
          * :meth:`.Formula.fvars` -- all occurring free variables
          * :meth:`.Formula.qvars` -- all occurring quantified variables
        """
        ...


class Variable(Term[χ, χ]):
    """
    .. seealso::
      * :class:`.RCF.atomic.Variable`
      * :class:`.Sets.atomic.Variable`
    """

    @abstractmethod
    def fresh(self) -> χ:
        """Returns a variable that has not been used so far.
        """
        ...


class AtomicFormula(Formula[α, τ, χ]):
    """This abstract class primarily specifies an interface via the
    definition of abstract methods on atomic formulas that are required by
    :class:`.Formula`. In addition, it provides some final implementations of
    such methods, where they do not depend on the syntax or sematic of the
    specific theory.

    .. seealso::
        * :class:`.RCF.atomic.AtomicFormula` -- \
            implementation for real closed fields
        * :class:`.Sets.atomic.AtomicFormula` -- \
            implementation for the theory of Sets
    """

    @abstractmethod
    def __le__(self, other: Formula[α, τ, χ]) -> bool:
        """Returns :external:obj:`True` if `self` should be sorted before or is
        equal to other.

        .. seealso::
          * :meth:`.RCF.atomic.AtomicFormula.__le__` --
                implementation for real closed fields
          * :meth:`.Sets.atomic.AtomicFormula.__le__` --
                implementation for the theory of Sets
        """
        ...

    @abstractmethod
    def __str__(self) -> str:
        # Overloading __str__ here breaks an infinite recursion in the
        # inherited Formula.__str__. Nicer string representations are provided
        # by various theory modules.
        ...

    @classmethod
    @abstractmethod
    def complement(cls) -> type[α]:
        """The complement operator of an atomic formula, i.e.,
        :code:`a.complement(*a.args)` is an atomic formula equivalent to
        :code:`Not(a.op(*a.args))`.

        .. seealso::
          * :meth:`.RCF.atomic.AtomicFormula.complement` --
                implementation for real closed fields
          * :meth:`.Sets.atomic.AtomicFormula.complement` --
                implementation for the theory of Sets
          * :meth:`.to_complement` --
                generalization from the class to instances
    """
        ...

    @abstractmethod
    def as_latex(self) -> str:
        """Latex representation.

        .. seealso::
            :meth:`.Formula.as_latex` -- \
                the corresponding recursive first-order method
        """
        ...

    @final
    def atoms(self: α) -> Iterator[α]:
        yield self

    @abstractmethod
    def bvars(self, quantified: frozenset[χ] = frozenset()) -> Iterator[χ]:
        """Iterate over occurrences of variables. Each variable is reported
        once for each term that it occurs in, provided that the variable is in
        the set `quantified`.
        """
        ...

    @abstractmethod
    def _fvars(self, quantified: set[χ]) -> Iterator[χ]:
        ...

    @abstractmethod
    def simplify(self) -> Formula[α, τ, χ]:
        """Fast basic simplification. The result is equivalent to self.

        .. seealso::
            :meth:`.Formula.simplify` -- \
                the corresponding recursive first-order method
        """
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
