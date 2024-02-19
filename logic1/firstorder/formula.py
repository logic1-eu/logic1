from __future__ import annotations

from abc import ABC, abstractmethod
import functools
from typing import Any, Callable, Final, Iterable, Iterator
from typing_extensions import Self

from ..support.decorators import classproperty  # noqa
from ..support.tracing import trace  # noqa


@functools.total_ordering
class Formula(ABC):
    r"""This abstract base class implements representations of and methods on
    first-order formulas recursively built using first-order operators:

    1. Boolean operators:

       a. Truth values :math:`\top` and :math:`\bot`

       b. Negation :math:`\lnot`

       c. Conjunction :math:`\land` and discjunction :math:`\lor`

       d. Implication :math:`\longrightarrow`

       e. Bi-implication (syntactic equivalence) :math:`\longleftrightarrow`

    2. Quantifiers :math:`\exists x` and :math:`\forall x`, where :math:`x` is
       a variable.

    As an abstract base class, :class:`Formula` cannot be instantiated.
    Nevertheless, it implements a number of methods on first-order formulas.
    The methods implemented here  are typically syntactic in the sense that
    they do not need to know the semantics of the underlying theories.
    """

    @classproperty
    def func(cls) -> Self:
        """This class property is supposed to be used with instances of
        subclasses of :class:`Formula`. It yields the respective subclass.
        """
        return cls

    @property
    def args(self) -> tuple[Any, ...]:
        """The argument tuple of the formula.
        """
        return self._args

    @args.setter
    def args(self, args: tuple[Any, ...]) -> None:
        self._args = args

    def __and__(self, other: Formula) -> Formula:
        """Override the :obj:`& <object.__and__>` operator to apply
        :class:`.boolean.And`.

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
        return hash((tuple(str(cls) for cls in self.func.mro()), self.args))

    @abstractmethod
    def __init__(self, *args: object) -> None:
        """This abstract base class is not supposed to have instances itself.
        Technically this is enforced via this abstract inializer.
        """
        ...

    def __invert__(self) -> Formula:
        """Override the :obj:`~ <object.__invert__>` operator to apply
        :class:`Not`.

        >>> from logic1.theories.RCF import VV
        >>> x, = VV.get('x')
        >>> ~ (x == 0)
        Not(x == 0)
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

        >>> from logic1.theories.RCF import Eq, VV
        >>> x, y, z = VV.get('x', 'y', 'z')
        >>>
        >>> Eq(x + z, y + z) << Eq(x, y)
        Implies(x == y, x + z == y + z)
        """
        return Implies(other, self)

    def __ne__(self, other: object) -> bool:
        """A recursive test for unequality of the `self` and `other`.
        """
        return not self == other

    def __or__(self, other: Formula) -> Formula:
        """Override the :obj:`| <object.__or__>` operator to apply :class:`Or`.

        >>> from logic1.theories.RCF import Eq, VV
        >>> x, y, z = VV.get('x', 'y', 'z')
        >>>
        >>> Eq(x, 0) | Eq(x, y) | Eq(x, z)
        Or(x == 0, x == y, x == z)
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

        >>> from logic1.theories.RCF import Eq, VV
        >>> x, y, z = VV.get('x', 'y', 'z')
        >>>
        >>> Eq(x, y) >> Eq(x + z, y + z)
        Implies(x == y, x + z == y + z)
        """
        return Implies(self, other)

    def __str__(self) -> str:
        """Representation of the Formula used in printing.
        """
        SYMBOL: Final = {
            All: 'All', Ex: 'Ex', And: 'and', Or: 'or', Implies: '-->',
            Equivalent: '<-->', Not: 'not', _F: 'F', _T: 'T'}
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
                    if PRECEDENCE[self.func] >= PRECEDENCE.get(arg.func, 100):
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
                # Atomic formulas are caught by the implementation of the
                # abstract method AtomicFormula.__str__.
                assert False, repr(self)

    def all(self, ignore: Iterable = set()) -> Formula:
        """Universal closure.

        Universally quantifiy all variables occurring free in self, except the
        ones mentioned in ignore.
        """
        variables = list(set(self.fvars()) - set(ignore))
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
                # Atomic formulas are caught by the implementation of as_latex
                # in AtomicFormula or its subclasses.
                assert False

    def atoms(self) -> Iterator[AtomicFormula]:
        """
        An iterator over all instances of AtomicFormula occurring in
        :data:`self`.

        >>> from logic1 import Ex, All, T, F
        >>> from logic1.theories.RCF import Eq, VV
        >>> x, y, z = VV.get('x', 'y', 'z')
        >>>
        >>> f = Eq(3 * x, 0) >> All(z, Eq(3 * x, 0) & All(x,
        ...     ~ Eq(x, 0) >> Ex(y, Eq(x * y, 1))))
        >>> type(f.atoms())
        <class 'generator'>
        >>> list(f.atoms())
        [3*x == 0, 3*x == 0, x == 0, x*y == 1]
        >>> set(f.atoms()) == {Eq(x, 0), Eq(3*x, 0), Eq(x*y, 1)}
        True

        This admits counting using common Python constructions:

        >>> sum(1 for _ in f.atoms())
        4
        >>> from collections import Counter
        >>> Counter(f.atoms())
        Counter({3*x == 0: 2, x == 0: 1, x*y == 1: 1})

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

    def bvars(self) -> Iterator[Variable]:
        """An iterator over all variables with bound ocurrences in self. Each
        variable is reported once for each term that it occurs in.

        >>> from logic1.theories.RCF import VV
        >>> a, x, y, z = VV.get('a', 'x', 'y', 'z')
        >>>
        >>> list(All(y, Ex(x, a + x == y) & Ex(z, x + y == a + x)).bvars())
        [x, y, y]

        Note that following the common definition in logic, *occurrence* refers
        to the occurrence in a term. Appearances of variables as a quantified
        variables without use in any term are not considered. Compare
        :meth:`qvars`.
        """
        return self._bvars(set())

    def _bvars(self, quantified: set) -> Iterator[Variable]:
        match self:
            case All() | Ex():
                yield from self.arg._bvars(quantified.union({self.var}))
            case And() | Or() | Not() | Implies() | Equivalent() | _F() | _T():
                for arg in self.args:
                    yield from arg._bvars(quantified)
            case AtomicFormula():
                yield from self._bvars(quantified)
            case _:
                assert False, type(self)

    def count_alternations(self) -> int:
        """Count the number of quantifier alternations.

        Returns the maximal number of quantifier alternations along a path from
        the root to a leaf of the expression tree. Occurrence of quantified
        variables is not checked, so that quantifiers with unused variables are
        counted.

        >>> from logic1 import Ex, All, T
        >>> from logic1.theories.Sets import Eq, VV
        >>> x, y, z = VV.set_vars('x', 'y', 'z')
        >>>
        >>> Ex(x, (x == y) & All(x, Ex(y, Ex(z, T)))).count_alternations()
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
        variables = list(set(self.fvars()) - set(ignore))
        if variables:
            variables.sort(key=variables[0].sort_key)
        f = self
        for v in reversed(variables):
            f = Ex(v, f)
        return f

    def fvars(self) -> Iterator[Variable]:
        """An iterator over all variables with free ocurrences in self. Each
        variable is reported once for each term that it occurs in.

        >>> from logic1.theories.RCF import VV
        >>> a, x, y, z = VV.get('a', 'x', 'y', 'z')
        >>>
        >>> list(All(y, Ex(x, a + x == y) & Ex(z, x + y == a + x)).fvars())
        [a, x, a, x]
        """
        return self._fvars(set())

    def _fvars(self, quantified: set) -> Iterator[Variable]:
        match self:
            case All() | Ex():
                yield from self.arg._fvars(quantified.union({self.var}))
            case And() | Or() | Not() | Implies() | Equivalent() | _F() | _T():
                for arg in self.args:
                    yield from arg._fvars(quantified)
            case AtomicFormula():
                yield from self._fvars(quantified)
            case _:
                assert False, type(self)

    def matrix(self) -> tuple[Formula, list[QuantifierBlock]]:
        blocks = []
        block_vars = []
        f: Formula = self
        while isinstance(f, (Ex, All)):
            block_quantifier = type(f)
            while isinstance(f, block_quantifier):
                block_vars.append(f.args[0])
                f = f.args[1]
            blocks.append((block_quantifier, block_vars))
            block_vars = []
        return f, blocks

    def qvars(self) -> Iterator[Variable]:
        """An iterator over all variables that are quantified in self.

        >>> from logic1.theories.Sets import VV
        >>> a, b, c, x, y, z = VV.set_vars('a', 'b', 'c', 'x', 'y', 'z')
        >>>
        >>> list(All(y, Ex(x, a == y) & Ex(z, a == y)).qvars())
        [y, x, z]

        Note that the mere quantification of a variable does not establish a
        bound ocurrence of that variable. Compare :meth:`bvars`, :meth:`fvars`.
        """
        match self:
            case All() | Ex():
                yield self.var
                yield from self.arg.qvars()
            case And() | Or() | Not() | Implies() | Equivalent() | _F() | _T():
                for arg in self.args:
                    yield from arg.qvars()
            case AtomicFormula():
                yield from ()
            case _:
                assert False, type(self)

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
        :data:`T` and :data:`F` and of duplicate siblings in the expression
        tree.
        """
        match self:
            case _F() | _T():
                return self
            case Not():
                arg_simplify = self.arg.simplify()
                if arg_simplify is T:
                    return F
                if arg_simplify is F:
                    return T
                return involutive_not(arg_simplify)
            case And() | Or():
                simplified_args: list[Formula] = []
                for arg in self.args:
                    arg_simplify = arg.simplify()
                    if arg_simplify is self.definite_func():
                        return self.definite_func()
                    if arg_simplify is self.neutral_func():
                        continue
                    if arg_simplify in simplified_args:
                        continue
                    if arg_simplify.func is self.func:
                        simplified_args.extend(arg_simplify.args)
                    else:
                        simplified_args.append(arg_simplify)
                return self.func(*simplified_args)
            case Implies():
                if self.rhs is T:
                    return self.lhs
                lhs_simplify = self.lhs.simplify()
                if lhs_simplify is F:
                    return T
                rhs_simplify = self.rhs.simplify()
                if rhs_simplify is T:
                    return T
                if lhs_simplify is T:
                    return rhs_simplify
                if rhs_simplify is F:
                    return involutive_not(lhs_simplify)
                assert {lhs_simplify, rhs_simplify}.isdisjoint({T, F})
                if lhs_simplify == rhs_simplify:
                    return T
                return Implies(lhs_simplify, rhs_simplify)
            case Equivalent():
                lhs_simplify = self.lhs.simplify()
                rhs_simplify = self.rhs.simplify()
                if lhs_simplify is T:
                    return rhs_simplify
                if rhs_simplify is T:
                    return lhs_simplify
                if lhs_simplify is F:
                    if isinstance(rhs_simplify, Not):
                        return rhs_simplify.arg
                    return Not(rhs_simplify)
                if rhs_simplify is F:
                    if isinstance(lhs_simplify, Not):
                        return lhs_simplify.arg
                    return Not(lhs_simplify)
                if lhs_simplify == rhs_simplify:
                    return T
                return Equivalent(lhs_simplify, rhs_simplify)
            case All() | Ex():
                return self.func(self.var, self.arg.simplify())
            case _:
                # Atomic formulas are caught by the implementation of simplify
                # in AtomicFormula or its subclasses.
                assert False, type(self)

    def subs(self, substitution: dict) -> Self:
        """Substitution of terms for variables.

        >>> from logic1 import Ex
        >>> from logic1.theories.RCF import VV
        >>> a, b, x = VV.get('a', 'b', 'x')
        >>>
        >>> f = Ex(x, x == a)
        >>> f.subs({x: a})
        Ex(x, x == a)
        >>>
        >>> f.subs({a: x})
        Ex(G0001_x, G0001_x == x)
        >>>
        >>> g = Ex(x, _ & (b == 0))
        >>> g.subs({b: x})
        Ex(G0002_x, And(Ex(G0001_x, G0001_x == G0002_x), x == 0))
        """
        match self:
            case All() | Ex():
                # A copy of the mutable could be avoided by keeping track of
                # the changes and undoing them at the end.
                substitution = substitution.copy()
                # (1) Remove substitution for the quantified variable. In
                # principle, this is covered by (2) below, but deleting here
                # preserves the name.
                if self.var in substitution:
                    del substitution[self.var]
                # Collect all variables on the right hand sides of
                # substitutions:
                substituted_vars: set[Variable] = set()
                for term in substitution.values():
                    substituted_vars.update(tuple(term.vars()))
                # (2) Make sure the quantified variable is not a key and does
                # not occur in a value of substitution:
                if self.var in substituted_vars or self.var in substitution:
                    var = self.var.fresh()
                    # We now know the following:
                    #   (i) var is not a key,
                    #  (ii) var does not occur in the values,
                    # (iii) self.var is not a key.
                    # We do *not* know whether self.var occurs in the values.
                    substitution[self.var] = var
                    # All free occurrences of self.var in self.arg will be
                    # renamed to var. In case of (iv) above, substitution will
                    # introduce new free occurrences of self.var, which do not
                    # clash with the new quantified variable var:
                    return self.func(var, self.arg.subs(substitution))
                return self.func(self.var, self.arg.subs(substitution))
            case And() | Or() | Not() | Implies() | Equivalent() | _F() | _T():
                return self.func(*(arg.subs(substitution) for arg in self.args))
            case _:
                # Atomic formulas are caught by the implementation of the
                # abstract method AtomicFormula.subs.
                assert False, type(self)

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
        >>> from logic1.theories.RCF import Eq, VV
        >>> a, y = VV.get('a', 'y')
        >>>
        >>> f = Equivalent(Eq(a, 0) & T, Ex(y, ~ Eq(y, a)))
        >>> f.to_nnf()
        And(Or(a != 0, F, Ex(y, y != a)), Or(All(y, y == a), And(a == 0, T)))
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
        >>> from logic1.theories.RCF import Eq, Lt, VV
        >>> x, y, z = VV.get('x', 'y', 'z')
        >>>
        >>> f = Eq(x, y) & Lt(y, z)
        >>> f.transform_atoms(lambda atom: atom.func(atom.lhs - atom.rhs, 0))
        And(x - y == 0, y - z < 0)
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
from .atomic import AtomicFormula, Variable
from .boolean import And, Equivalent, Implies, involutive_not, Not, Or, _F, F, _T, T
from .quantified import All, Ex, QuantifierBlock
