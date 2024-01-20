"""Provides an abstract base class for first-order formulas."""

from __future__ import annotations

from abc import ABC, abstractmethod
import functools
from typing import Any, Callable, Final, Iterable, Iterator, TypeAlias
from typing_extensions import Self

from ..support.containers import GetVars

from ..support.tracing import trace  # noqa


QuantifierBlock: TypeAlias = tuple[Any, list]


@functools.total_ordering
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

    def __and__(self, other: Formula) -> Formula:
        """Override the :obj:`& <object.__and__>` operator to apply
        :class:`And`.

        >>> from logic1.theories.RCF import Eq
        >>>
        >>> Eq(0, 0) & Eq(1 + 1, 2) & Eq(1 + 1 + 1, 3)
        And(Eq(0, 0), Eq(2, 2), Eq(3, 3))
        """
        return And(self, other)

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

    def __getnewargs__(self):
        return self.args

    def __hash__(self) -> int:
        """
        Hash function.

        hash() yields deterministic results for a fixed hash seed. Set the
        environment variable PYTHONHASHSEED to a positive integer when
        comparing hashes from various Python sessions, e.g. for debugging.
        Recall from the Python documentation that PYTHONHASHSEED should not be
        fixed in general.
        """
        return hash((tuple(str(cls) for cls in self.func.__mro__), self.args))

    @abstractmethod
    def __init__(self, *args: object) -> None:
        """This abstract base class is not supposed to have instances
        itself.
        """
        ...

    def __invert__(self) -> Formula:
        """Override the :obj:`~ <object.__invert__>` operator to apply
        :class:`Not`.

        >>> from logic1.theories.RCF import Eq
        >>>
        >>> ~ Eq(1,0)
        Not(Eq(1, 0))
        """
        return Not(self)

    def __le__(self, other: Formula) -> bool:
        match other:
            case AtomicFormula():
                return False
            case Formula():
                L = [And, Or, Not, Implies, Equivalent, Ex, All, _T, _F]
                if self.func != other.func:
                    return L.index(self.func) < L.index(other.func)
                return self.args <= other.args

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

    def __ne__(self, other: object) -> bool:
        """A recursive test for unequality of the `self` and `other`.
        """
        return not self == other

    def __or__(self, other: Formula) -> Formula:
        """Override the :obj:`| <object.__or__>` operator to apply :class:`Or`.

        >>> from logic1.theories.RCF import Eq, ring
        >>> x, y, z = ring.set_vars('x', 'y', 'z')
        >>>
        >>> Eq(x, 0) | Eq(x, y) | Eq(x, z)
        Or(Eq(x, 0), Eq(x, y), Eq(x, z))
        """
        return Or(self, other)

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

    def __str__(self) -> str:
        """Representation of the Formula used in printing.
        """
        SYMBOL: Final = {
            All: 'All', Ex: 'Ex', And: '&', Or: 'or', Implies: '>>',
            Equivalent: 'equivalent', Not: '~', _F: 'F', _T: 'T'}
        PRECEDENCE: Final = {
            All: 99, Ex: 99, And: 50, Or: 50, Implies: 10, Equivalent: 10,
            Not: 99, _F: 99, _T: 99}
        SPACING: Final = ' '
        match self:
            case All() | Ex():
                L = []
                arg: Formula = self
                while isinstance(arg, (All, Ex)) and arg.func == self.func:
                    L.append(arg.var)
                    arg = arg.arg
                variables = tuple(L) if len(L) > 1 else L[0]
                return f'{SYMBOL[self.func]}({variables}, {arg})'
            case And() | Or() | Equivalent() | Implies():
                L = []
                for arg in self.args:
                    arg_as_str = str(arg)
                    if PRECEDENCE[self.func] >= PRECEDENCE.get(arg.func, 0):
                        arg_as_str = f'({arg_as_str})'
                    L.append(arg_as_str)
                return f'{SPACING}{SYMBOL[self.func]}{SPACING}'.join(L)
            case Not():
                arg_as_str = str(self.arg)
                if self.arg.func not in (Ex, All, Not):
                    arg_as_str = f'({arg_as_str})'
                return f'{SYMBOL[Not]}{SPACING}{arg_as_str}'
            case _F() | _T():
                return SYMBOL[self.func]
            case _:
                # Atomic formulas must be caught by the implementation of the
                # abstract method AtomicFormula.__str__.
                assert False, repr(self)

    def all(self, ignore: Iterable = set()) -> Formula:
        """Universal closure.

        Universally quantifiy all variables occurring free in self, except the
        ones mentioned in ignore.
        """
        variables = list(self.get_vars().free - set(ignore))
        if variables:
            variables.sort(key=variables[0].sort_key)
        f = self
        for v in reversed(variables):
            f = All(v, f)
        return f

    def as_latex(self) -> str:
        SYMBOL: Final = {
            All: '\\forall', Ex: '\\exists', And: '\\wedge', Or: '\\vee',
            Implies: '\\longrightarrow', Equivalent: '\\longleftrightarrow',
            Not: '\\neg', _F: '\\bot', _T: '\\top'}
        PRECEDENCE: Final = {
            All: 99, Ex: 99, And: 50, Or: 50, Equivalent: 10, Implies: 10,
            Not: 99, _F: 99, _T: 99}
        SPACING: Final = ' \\, '
        match self:
            case All() | Ex():
                var_as_latex = self.var.as_latex()
                arg_as_latex = self.arg.as_latex()
                if self.arg.func not in (Ex, All, Not):
                    arg_as_latex = f'({arg_as_latex})'
                return f'{SYMBOL[self.func]} {var_as_latex}{SPACING}{arg_as_latex}'
            case And() | Or() | Equivalent() | Implies():
                L = []
                for arg in self.args:
                    arg_as_latex = arg.as_latex()
                    if PRECEDENCE[self.func] >= PRECEDENCE.get(arg.func, 99):
                        arg_as_latex = f'({arg_as_latex})'
                    L.append(arg_as_latex)
                return f'{SPACING}{SYMBOL[self.func]}{SPACING}'.join(L)
            case Not():
                arg_as_latex = self.arg.as_latex()
                if self.arg.func not in (Ex, All, Not):
                    arg_as_latex = f'({arg_as_latex})'
                return f'{SYMBOL[Not]}{SPACING}{arg_as_latex}'
            case _F() | _T():
                return SYMBOL[self.func]
            case _:
                # Atomic formulas must be caught by the implementation of the
                # abstract method AtomicFormula.as_latex.
                assert False

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
        """
        match self:
            case All() | Ex():
                yield from self.arg.atoms()
            case And() | Or() | Not() | Implies() | Equivalent() | _F() | _T():
                for arg in self.args:
                    yield from arg.atoms()
            case AtomicFormula():
                yield self
            case _:
                assert False, type(self)

    def count_alternations(self) -> int:
        """Count the number of quantifier alternations.

        Returns the maximal number of quantifier alternations along a path from
        the root to a leaf of the expression tree. Occurrence of quantified
        variables is not checked, so that quantifiers with unused variables are
        counted.

        >>> from logic1 import Ex, All, T
        >>> from logic1.theories.Sets import Eq
        >>> from sympy.abc import x, y, z
        >>>
        >>> Ex(x, Eq(x, y) & All(x, Ex(y, Ex(z, T)))).count_alternations()
        2
        """
        return self._count_alternations()[0]

    def _count_alternations(self) -> tuple[int, set[type[All | Ex]]]:
        match self:
            case All() | Ex():
                count, quantifiers = self.arg._count_alternations()
                if self.dual_func in quantifiers:
                    return (count + 1, {self.func})
                return (count, quantifiers)
            case And() | Or() | Not() | Implies() | Equivalent():
                highest_count = -1
                highest_count_quantifiers: set[type[All | Ex]] = {All, Ex}
                for arg in self.args:
                    count, quantifiers = arg._count_alternations()
                    if count > highest_count:
                        highest_count = count
                        highest_count_quantifiers = quantifiers
                    elif count == highest_count:
                        highest_count_quantifiers.update(quantifiers)
                return (highest_count, highest_count_quantifiers)
            case _F() | _T() | AtomicFormula():
                return (-1, {All, Ex})
            case _:
                assert False, type(self)

    def depth(self) -> int:
        match self:
            case All() | Ex():
                return self.arg.depth() + 1
            case And() | Or() | Not() | Implies() | Equivalent():
                return max(arg.depth() for arg in self.args) + 1
            case _F() | _T() | AtomicFormula():
                return 0
            case _:
                assert False, type(self)

    def ex(self, ignore: Iterable = set()) -> Formula:
        """Existential closure.

        Existentially quantifiy all variables occurring free in self, except
        the ones mentioned in ignore.
        """
        variables = list(self.get_vars().free - set(ignore))
        if variables:
            variables.sort(key=variables[0].sort_key)
        f = self
        for v in reversed(variables):
            f = Ex(v, f)
        return f

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

    def matrix(self) -> tuple[Formula, list[QuantifierBlock]]:
        blocks = []
        block_vars = []
        f: Formula = self
        while isinstance(f, QuantifiedFormula):
            block_quantifier = type(f)
            while isinstance(f, block_quantifier):
                block_vars.append(f.var)
                f = f.arg
            blocks.append((block_quantifier, block_vars))
            block_vars = []
        return f, blocks

    def _repr_latex_(self) -> str:
        """A LaTeX representation of the :class:`Formula` `self` for jupyter
        notebooks. In general, use the method :meth:`to_latex` instead.
        """
        limit = 5000
        as_latex = self.as_latex()
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

    def simplify(self) -> Formula:
        """Fast simplification. The result is equivalent to `self`.

        Primary simplification goals are the elimination of occurrences of
        :data:`T` and :data:`F` and of occurrences of equal subformulas as
        siblings in the expression tree.
        """
        return self

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
        Ex(G0001_x, Eq(G0001_x, x))
        >>>
        >>> g = Ex(x, _ & Eq(b, 0))
        >>> g.subs({b: x})
        Ex(G0002_x, And(Ex(G0001_x, Eq(G0001_x, G0002_x)), Eq(x, 0)))
        """
        ...

    def to_nnf(self, to_positive: bool = True, _not: bool = False) -> Formula:
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
        rewrite: Formula
        match self:
            case All() | Ex():
                nnf_func = self.dual_func if _not else self.func
                nnf_arg = self.arg.to_nnf(to_positive=to_positive, _not=_not)
                return nnf_func(self.var, nnf_arg)
            case Equivalent():
                rewrite = And(Implies(*self.args), Implies(self.rhs, self.lhs))
                return rewrite.to_nnf(to_positive=to_positive, _not=_not)
            case Implies():
                if isinstance(self.rhs, Or):
                    rewrite = Or(Not(self.lhs), *self.rhs.args)
                else:
                    rewrite = Or(Not(self.lhs), self.rhs)
                return rewrite.to_nnf(to_positive=to_positive, _not=_not)
            case And() | Or():
                nnf_func = self.dual_func if _not else self.func
                nnf_args: list[Formula] = []
                for arg in self.args:
                    nnf_arg = arg.to_nnf(to_positive=to_positive, _not=_not)
                    if nnf_arg.func is nnf_func:
                        nnf_args.extend(nnf_arg.args)
                    else:
                        nnf_args.append(nnf_arg)
                return nnf_func(*nnf_args)
            case Not():
                return self.arg.to_nnf(to_positive=to_positive, _not=not _not)
            case _F() | _T():
                if _not:
                    return self.dual_func() if to_positive else Not(self)
                return self
            case AtomicFormula():
                if _not:
                    return self.to_complement() if to_positive else Not(self)
                return self
            case _:
                assert False, type(self)

    def transform_atoms(self, tr: Callable[[Any], Formula]) -> Formula:
        """Apply `tr` to all atomic formulas.

        Replaces each atomic subformula of `self` with the :class:`Formula`
        `tr(self)`.

        >>> from logic1 import And
        >>> from logic1.theories.RCF import Eq, Lt, ring
        >>> x, y, z = ring.set_vars('x', 'y', 'z')
        >>>
        >>> f = Eq(x, y) & Lt(y, z)
        >>> f.transform_atoms(lambda atom: atom.func(atom.lhs - atom.rhs, 0))
        And(Eq(x - y, 0), Lt(y - z, 0))
        """
        # type of tr requieres discussion
        match self:
            case All() | Ex():
                return self.func(self.var, self.arg.transform_atoms(tr))
            case And() | Or() | Not() | Implies() | Equivalent() | _F() | _T():
                args = (arg.transform_atoms(tr) for arg in self.args)
                return self.func(*args)
            case AtomicFormula():
                return tr(self)
            case _:
                assert False, type(self)


# The following imports are intentionally late to avoid circularity.
from .atomic import AtomicFormula
from .boolean import And, Equivalent, Implies, Not, Or
from .quantified import All, Ex, QuantifiedFormula
from .truth import _F, _T
