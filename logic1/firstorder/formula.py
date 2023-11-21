"""Provides an abstract base class for first-order formulas."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, final, Iterator, Optional, TYPE_CHECKING
from typing_extensions import Self

from ..support.containers import GetVars

# from ..support.tracing import trace

if TYPE_CHECKING:
    from .atomic import AtomicFormula


class Formula(ABC):
    """An abstract base class for first-order formulas.

    All other classes in the :mod:`.firstorder` package are derived from
    :class:`Formula`.
    """

    # The following would be an abstract class variables, which are not
    # available at the moment.
    func: type[Formula]  #: :meta private:

    # Similarly the following would be an abstract instance variable:
    args: tuple  #: :meta private:

    # Instance methods
    @final
    def __and__(self, other: Formula) -> Formula:
        """Override the :obj:`& <object.__and__>` operator to apply
        :class:`And`.

        >>> from logic1.theories.RCF import Eq
        >>>
        >>> Eq(0, 0) & Eq(1 + 1, 2) & Eq(1 + 1 + 1, 3)
        And(Eq(0, 0), Eq(2, 2), Eq(3, 3))
        """
        return And(self, other)

    @final
    def __invert__(self) -> Formula:
        """Override the :obj:`~ <object.__invert__>` operator to apply
        :class:`Not`.

        >>> from logic1.theories.RCF import Eq
        >>>
        >>> ~ Eq(1,0)
        Not(Eq(1, 0))
        """
        return Not(self)

    @final
    def __lshift__(self, other: Formula) -> Formula:
        r"""Override the :obj:`\<\< <object.__lshift__>` operator to apply
        :class:`Implies` with reversed sides.

        >>> from logic1.theories.RCF import Eq, ring
        >>> x, y, z = ring.set_vars('x', 'y', 'z')
        >>>
        >>> Eq(x + z, y + z) << Eq(x, y)
        Implies(Eq(x, y), Eq(x + z, y + z))
        """
        return Implies(other, self)

    @final
    def __or__(self, other: Formula) -> Formula:
        """Override the :obj:`| <object.__or__>` operator to apply :class:`Or`.

        >>> from logic1.theories.RCF import Eq, ring
        >>> x, y, z = ring.set_vars('x', 'y', 'z')
        >>>
        >>> Eq(x, 0) | Eq(x, y) | Eq(x, z)
        Or(Eq(x, 0), Eq(x, y), Eq(x, z))
        """
        return Or(self, other)

    @final
    def __rshift__(self, other: Formula) -> Formula:
        """Override the :obj:`>> <object.__rshift__>` operator to apply
        :class:`Implies`.

        >>> from logic1.theories.RCF import Eq, ring
        >>> x, y, z = ring.set_vars('x', 'y', 'z')
        >>>
        >>> Eq(x, y) >> Eq(x + z, y + z)
        Implies(Eq(x, y), Eq(x + z, y + z))
        """
        return Implies(self, other)

    def __eq__(self, other: object) -> bool:
        """A recursive test for equality of the `self` and `other`.

        Note that this is is not a logical operator for equality.

        >>> from logic1.theories.RCF import Ne
        >>>
        >>> e1 = Ne(1, 0)
        >>> e2 = Ne(1, 0)
        >>> e1 == e2
        True
        >>> e1 is e2
        False
        """
        if self is other:
            return True
        if not isinstance(other, Formula):
            return NotImplemented
        return self.func == other.func and self.args == other.args

    def __ne__(self, other: object) -> bool:
        """A recursive test for unequality of the `self` and `other`.
        """
        return not self == other

    def __hash__(self) -> int:
        return hash((self.func, self.args))

    @abstractmethod
    def __init__(self, *args) -> None:
        """This abstract base class is not supposed to have instances
        itself.
        """
        ...

    def __repr__(self) -> str:
        """A Representation of the :class:`Formula` `self` that is suitable for
        use as an input.
        """
        r = self.func.__name__
        r += '('
        if self.args:
            r += self.args[0].__repr__()
            for a in self.args[1:]:
                r += ', ' + a.__repr__()
        r += ')'
        return r

    @final
    def __str__(self) -> str:
        """Representation of the Formula used in printing.
        """
        return self._sprint(mode='text')

    @final
    def _repr_latex_(self) -> Optional[str]:
        r"""A LaTeX representation of the :class:`Formula` `self` as it is used
        within jupyter notebooks.

        >>> from logic1 import F
        >>>
        >>> F._repr_latex_()
        '$\\displaystyle \\bot$'

        Subclasses have to_latex() methods yielding plain LaTeX without the
        surrounding $\\displaystyle ... $.
        """
        limit = 5000
        as_latex = self.to_latex()
        if len(as_latex) > limit:
            as_latex = as_latex[:limit]
            opc = 0
            for pos in range(limit):
                match as_latex[pos]:
                    case '{':
                        opc += 1
                    case '}':
                        opc -= 1
            assert opc >= 0
            while opc > 0:
                match as_latex[-1]:
                    case '{':
                        opc -= 1
                    case '}':
                        opc += 1
                as_latex = as_latex[:-1]
            as_latex += '{}\\dots'
        return f'$\\displaystyle {as_latex}$'

    @abstractmethod
    def atoms(self) -> Iterator[AtomicFormula]:
        """
        An iterator over all instances of AtomicFormula occurring in
        :data:`self`.

        >>> from logic1 import Ex, All, T, F
        >>> from logic1.theories.RCF import Eq, ring
        >>> x, y, z = ring.set_vars('x', 'y', 'z')
        >>>
        >>> f = Eq(3 * x, 0) >> All(z, Eq(3 * x, 0) & All(x,
        ...     ~ Eq(x, 0) >> Ex(y, Eq(x * y, 1))))
        >>> type(f.atoms())
        <class 'generator'>
        >>> list(f.atoms())
        [Eq(3*x, 0), Eq(3*x, 0), Eq(x, 0), Eq(x*y, 1)]
        >>> set(f.atoms()) == {Eq(x, 0), Eq(3*x, 0), Eq(x*y, 1)}
        True

        This admits counting using common Python constructions:

        >>> sum(1 for _ in f.atoms())
        4
        >>> from collections import Counter
        >>> Counter(f.atoms())
        Counter({Eq(3*x, 0): 2, Eq(x, 0): 1, Eq(x*y, 1): 1})

        >>> empty = (T & F).atoms()
        >>> next(empty)
        Traceback (most recent call last):
        ...
        StopIteration

        One use case within firstorder is getting access to static methods of
        classes derived from :class:`.atomic.AtomicFormula` elsewhere:

        >>> f = Ex(x, Eq(x, -y) & Eq(y, z ** 2))
        >>> isinstance(f.var, next(f.atoms()).variable_type())
        True
        """
        ...

    @final
    def count_alternations(self) -> int:
        """Count number of quantifier alternations.

        Returns the maximal number of quantifier alternations along a path in
        the expression tree. Occurrence of quantified variables is not checked,
        so that quantifiers with unused variables are counted.

        >>> from logic1 import Ex, All, T
        >>> from logic1.theories.Sets import Eq
        >>> from sympy.abc import x, y, z
        >>>
        >>> Ex(x, Eq(x, y) & All(x, Ex(y, Ex(z, T)))).count_alternations()
        2
        """
        return self._count_alternations()[0]

    @abstractmethod
    def _count_alternations(self) -> tuple[int, set]:
        ...

    @abstractmethod
    def depth(self) -> int:
        ...

    @abstractmethod
    def get_qvars(self) -> set:
        """The set of all variables that are quantified in self.

        >>> from logic1 import Ex, All
        >>> from logic1.theories.Sets import Eq
        >>> from sympy.abc import a, b, c, x, y, z
        >>>
        >>> All(y, Ex(x, Eq(a, y)) & Ex(z, Eq(a, y))).get_qvars() == {x, y, z}
        True

        Note that the mere quantification of a variable does not establish a
        bound ocurrence of that variable. Compare :meth:`get_vars`.
        """
        ...

    @abstractmethod
    def get_vars(self, assume_quantified: set = set()) -> GetVars:
        """Extract all variables occurring in *self*.

        The result is an instance of :class:`GetVars
        <logic1.support.containers.GetVars>`, which extract certain subsects of
        variables as a :class:`set`.

        >>> from logic1 import Ex, All
        >>> from logic1.theories.RCF import Eq, ring
        >>> x, y, z = ring.set_vars('x', 'y', 'z')
        >>>
        >>> # Variables with free occurrences:
        >>> f = Eq(3 * x, 0) >> All(z, All(x,
        ...     ~ Eq(x, 0) >> Ex(y, Eq(x * y, 1))))
        >>> f.get_vars().free == {x}
        True
        >>>
        >>> # Variables with bound occurrences:
        >>> f.get_vars().bound == {x, y}
        True
        >>>
        >>> # All occurring variables:
        >>> z not in f.get_vars().all
        True

        Note that following the common definition in logic, *occurrence* refers
        to the occurrence in a term. Appearances of variables as a quantified
        variables without use in any term are not considered. Compare
        :meth:`get_qvars`.
        """
        ...

    @abstractmethod
    def matrix(self) -> tuple[Formula, list[tuple[Any, list]]]:
        ...

    def simplify(self) -> Formula:
        """Fast simplification. The result is equivalent to `self`.

        Primary simplification goals are the elimination of occurrences of
        :data:`T` and :data:`F` and of occurrences of equal subformulas as
        siblings in the expression tree.
        """
        return self

    @abstractmethod
    def _sprint(self, mode: str) -> str:
        """Print to string.
        """
        ...

    @abstractmethod
    def subs(self, substitution: dict) -> Self:
        """Substitution of terms for variables.

        >>> from logic1 import Ex
        >>> from logic1.theories.RCF import Eq, ring
        >>> a, b, x = ring.set_vars('a', 'b', 'x')
        >>>
        >>> f = Ex(x, Eq(x, a))
        >>> f.subs({x: a})
        Ex(x, Eq(x, a))
        >>>
        >>> f.subs({a: x})
        Ex(x_R1, Eq(x_R1, x))
        >>>
        >>> g = Ex(x, _ & Eq(b, 0))
        >>> g.subs({b: x})
        Ex(x_R2, And(Ex(x_R1, Eq(x_R1, x_R2)), Eq(x, 0)))
        """
        ...

    @final
    def to_latex(self) -> str:
        """Convert to a LaTeX representation.
        """
        return self._sprint(mode='latex')

    @abstractmethod
    def to_nnf(self, to_positive: bool = True,
               _implicit_not: bool = False) -> Formula:
        """Convert to Negation Normal Form.

        A Negation Normal Form (NNF) is an equivalent formula within which the
        application of :class:`Not` is restricted to atomic formulas, i.e.,
        instances of :class:`AtomicFormula`, and truth values :data:`T` and
        :data:`F`. The only other operators admitted are :class:`And`,
        :class:`Or`, :class:`Ex`, and :class:`All`.

        If the input is quanitfier-free, :meth:`to_nnf` will not introduce any
        quanitfiers.

        If `to_positive` is `True`, :class:`Not` is eliminated via replacing
        relation symbols with their complements. The result is then even a
        Positive Normal Form.

        >>> from logic1 import Ex, Equivalent, T
        >>> from logic1.theories.RCF import Eq, ring
        >>> a, y = ring.set_vars('a', 'y')
        >>>
        >>> f = Equivalent(Eq(a, 0) & T, Ex(y, ~ Eq(y, a)))
        >>> f.to_nnf()
        And(Or(Ne(a, 0), F, Ex(y, Ne(y, a))),
            Or(All(y, Eq(y, a)), And(Eq(a, 0), T)))
        """
        ...

    @abstractmethod
    def transform_atoms(self, transformation: Callable) -> Self:
        """Apply `transformation` to all atomic formulas.

        Replaces each atomic subformula of `self` with the :class:`Formula`
        `transformation(self)`.

        >>> from logic1 import And
        >>> from logic1.theories.RCF import Eq, Lt, ring
        >>> x, y, z = ring.set_vars('x', 'y', 'z')
        >>>
        >>> f = Eq(x, y) & Lt(y, z)
        >>> f.transform_atoms(lambda atom: atom.func(atom.lhs - atom.rhs, 0))
        And(Eq(x - y, 0), Lt(y - z, 0))
        """
        ...


# The following imports are intentionally late to avoid circularity.
from .boolean import Implies, And, Or, Not
