"""Generic abstract classes specify atomic formulas at the first-order level,
where the syntax and semantics of the underlying theories is unknown. The
classes primarily act as interfaces specifying methods that are used as black
boxes within :class:`.Formula` methods.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import final, Iterator, TypeVar

from logic1.firstorder.formula import α, τ, χ, σ, Formula

from logic1.support.tracing import trace


κ = TypeVar('κ')
"""A type variable denoting a sort key.
"""

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
