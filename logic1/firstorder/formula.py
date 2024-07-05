from __future__ import annotations

from abc import ABC, abstractmethod
import functools
from typing import Any, Callable, Final, Iterable, Iterator
from typing_extensions import Self

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

    @property
    def op(self) -> type[Self]:
        """This property is supposed to be used with instances of subclasses of
        :class:`Formula`. It yields the respective subclass.
        """
        return type(self)

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
        return self.op == other.op and self.args == other.args

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
        return hash((tuple(str(cls) for cls in self.op.mro()), self.args))

    @abstractmethod
    def __init__(self, *args: object) -> None:
        """This abstract base class is not supposed to have instances itself.
        Technically this is enforced via this abstract initializer.
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
                if self.op != other.op:
                    return L.index(self.op) < L.index(other.op)
                return self.args <= other.args

    def __lshift__(self, other: Formula) -> Formula:
        r"""Override the :obj:`\<\< <object.__lshift__>` operator to apply
        :class:`Implies` with reversed sides.

        >>> from logic1.theories.RCF import Eq, VV
        >>> x, y, z = VV.get('x', 'y', 'z')
        >>>
        >>> (x + z == y + z) << (x == y)
        Implies(x - y == 0, x - y == 0)
        """
        return Implies(other, self)

    def __ne__(self, other: object) -> bool:
        """A recursive test for unequality of the `self` and `other`.
        """
        return not self == other

    def __or__(self, other: Formula) -> Formula:
        """Override the :obj:`| <object.__or__>` operator to apply :class:`Or`.

        >>> from logic1.theories.RCF import *
        >>> x, y, z = VV.get('x', 'y', 'z')
        >>>
        >>> (x == 0) | (x == y) | (x == z)
        Or(x == 0, x - y == 0, x - z == 0)
        """
        return Or(self, other)

    def __repr__(self) -> str:
        """A Representation of the :class:`Formula` `self` that is suitable for
        use as an input.
        """
        r = self.op.__name__
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

        >>> from logic1.theories.RCF import *
        >>> x, y, z = VV.get('x', 'y', 'z')
        >>>
        >>> (x == y) >> (x + z == y + z)
        Implies(x - y == 0, x - y == 0)
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
                while isinstance(arg, (All, Ex)) and arg.op == self.op:
                    L.append(arg.var)
                    arg = arg.arg
                variables = tuple(L) if len(L) > 1 else L[0]
                return f'{SYMBOL[self.op]}({variables}, {arg})'
            case And() | Or() | Equivalent() | Implies():
                L = []
                for arg in self.args:
                    arg_as_str = str(arg)
                    if PRECEDENCE[self.op] >= PRECEDENCE.get(arg.op, 100):
                        arg_as_str = f'({arg_as_str})'
                    L.append(arg_as_str)
                return f'{SPACING}{SYMBOL[self.op]}{SPACING}'.join(L)
            case Not():
                arg_as_str = str(self.arg)
                if self.arg.op not in (Ex, All, Not):
                    arg_as_str = f'({arg_as_str})'
                return f'{SYMBOL[Not]}{SPACING}{arg_as_str}'
            case _F() | _T():
                return SYMBOL[self.op]
            case _:
                # Atomic formulas are caught by the implementation of the
                # abstract method AtomicFormula.__str__.
                assert False, repr(self)

    def all(self, ignore: Iterable = set()) -> Formula:
        """Universal closure. Universally quantifiy all variables occurring
        free in `self`, except the ones in `ignore`.

        >>> from logic1.theories import RCF
        >>> a, b, x = RCF.VV.get('a', 'b', 'x')
        >>> f = Ex(x, (x >= 0) & (a*x + b == 0))
        >>> f.all()
        All(b, All(a, Ex(x, And(x >= 0, a*x + b == 0))))

        .. seealso:: :meth:`ex` -- existential closure
        """
        variables = list(set(self.fvars()) - set(ignore))
        if variables:
            variables.sort(key=variables[0].sort_key)
        f = self
        for v in reversed(variables):
            f = All(v, f)
        return f

    def as_latex(self) -> str:
        r"""LaTeX representation.

        >>> from logic1.theories import RCF
        >>> x, y = RCF.VV.get('x', 'y')
        >>> f = All(x, (x < 1) | (x - 1 == 0) | (x > 1))
        >>> f.as_latex()
        '\\forall x \\, (x - 1 < 0 \\, \\vee \\, x - 1 = 0 \\, \\vee \\, x - 1 > 0)'

        .. seealso:: :meth:`_repr_latex_` -- LaTeX representation for Jupyter notebooks
        """
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
                if self.arg.op not in (Ex, All, Not):
                    arg_as_latex = f'({arg_as_latex})'
                return f'{SYMBOL[self.op]} {var_as_latex}{SPACING}{arg_as_latex}'
            case And() | Or() | Equivalent() | Implies():
                L = []
                for arg in self.args:
                    arg_as_latex = arg.as_latex()
                    if PRECEDENCE[self.op] >= PRECEDENCE.get(arg.op, 99):
                        arg_as_latex = f'({arg_as_latex})'
                    L.append(arg_as_latex)
                return f'{SPACING}{SYMBOL[self.op]}{SPACING}'.join(L)
            case Not():
                arg_as_latex = self.arg.as_latex()
                if self.arg.op not in (Ex, All, Not):
                    arg_as_latex = f'({arg_as_latex})'
                return f'{SYMBOL[Not]}{SPACING}{arg_as_latex}'
            case _F() | _T():
                return SYMBOL[self.op]
            case _:
                # Atomic formulas are caught by the implementation of as_latex
                # in AtomicFormula or its subclasses.
                assert False

    def atoms(self) -> Iterator[AtomicFormula]:
        """
        An iterator over all instances of :class:`AtomicFormula
        <.firstorder.atomic.AtomicFormula>` occurring in `self`.

        Recall that the truth values :data:`T <.boolean.T>` and :data:`F
        <.boolean.F>` are not atoms:

        >>> from logic1.theories import RCF
        >>> x, y, z = RCF.VV.get('x', 'y', 'z')
        >>> f = ((x == 0) & (y == 0) & T) | ((x == 0) & (y == z) & (z != 0))
        >>> list(f.atoms())
        [x == 0, y == 0, x == 0, y - z == 0, z != 0]

        The overall number of atoms:

        >>> sum(1 for _ in f.atoms())
        5

        Count numbers of occurrences for each occurring atom using a
        :external+python:class:`Counter <collections.Counter>`:

        >>> from collections import Counter
        >>> Counter(f.atoms())
        Counter({x == 0: 2, y == 0: 1, y - z == 0: 1, z != 0: 1})

        Recall the Python builtin :func:`next`:

        >>> iter = (x == 0).atoms()
        >>> next(iter)
        x == 0
        >>> next(iter)
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
        """An iterator over all bound occurrences of variables in `self`. Each
        variable is reported once for each term that it occurs in.

        >>> from logic1.theories import RCF
        >>> a, x, y, z = RCF.VV.get('a', 'x', 'y', 'z')
        >>> f = All(y, Ex(x, a + x == y) & Ex(z, x + y == a + x))
        >>> list(f.bvars())
        [x, y, y]

        Note that following the common definition in logic, *occurrence* refers
        to the occurrence in a term. Appearances of variables as a quantified
        variables without use in any term are not considered.

        .. seealso::
            * :meth:`fvars` -- all occurring free variables
            * :meth:`qvars` -- all occurring quantified variables
            * :meth:`Term.vars() <.firstorder.atomic.Term.vars>` -- all occurring variables
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

        >>> from logic1.theories import RCF
        >>> x, y, z = RCF.VV.get('x', 'y', 'z')
        >>> f = Ex(x, (x == y) & All(x, Ex(y, Ex(z, x == x + 1))))
        >>> f.count_alternations()
        2

        In this example the following path has two alternations, one from
        :class:`Ex <.quantified.Ex>` to :class:`All <.quantified.All>` and
        another one from :class:`All <.quantified.All>` to
        :class:`Ex <.quantified.Ex>`::

            Ex ———— And ———— All ———— Ex ———— Ex ———— x == y + 1
        """
        return self._count_alternations()[0]

    def _count_alternations(self) -> tuple[int, set[type[All | Ex]]]:
        match self:
            case All() | Ex():
                count, quantifiers = self.arg._count_alternations()
                if self.dual() in quantifiers:
                    return (count + 1, {self.op})
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
        """The depth of a formula is the maximal length of a path from the root
        to a truth value or an :class:`AtomicFormula
        <.firstorder.atomic.AtomicFormula>` in the expression tree:

        >>> from logic1.theories import RCF
        >>> x, y, z = RCF.VV.get('x', 'y', 'z')
        >>> f = Ex(x, (x == y) & All(x, Ex(y, Ex(z, x == y + 1))))
        >>> f.depth()
        5

        In this example the the following path has the maximal length 5::

            Ex ———— And ———— All ———— Ex ———— Ex ———— x == y + 1

        Note that for this purpose truth values and :class:`AtomicFormula
        <.firstorder.atomic.AtomicFormula>` are considered to have depth 0.
        """
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
        """Existential closure. Existentially quantifiy all variables occurring
        free in `self`, except the ones in `ignore`.

        >>> from logic1.theories import RCF
        >>> a, b, c, x = RCF.VV.get('a', 'b', 'c', 'x')
        >>> f = All(x, (a < x) & (a + b + c < x))
        >>> f.ex(ignore={c})
        Ex(b, Ex(a, All(x, And(a - x < 0, a + b + c - x < 0))))

        .. seealso:: :meth:`all` -- universal closure
        """
        variables = list(set(self.fvars()) - set(ignore))
        if variables:
            variables.sort(key=variables[0].sort_key)
        f = self
        for v in reversed(variables):
            f = Ex(v, f)
        return f

    def fvars(self) -> Iterator[Variable]:
        """An iterator over all free occurrences of variables in `self`. Each
        variable is reported once for each term that it occurs in.

        >>> from logic1.theories import RCF
        >>> a, x, y, z = RCF.VV.get('a', 'x', 'y', 'z')
        >>> f = All(y, Ex(x, a + x - y == 0) & Ex(z, x + y - a == 0))
        >>> list(f.fvars())
         [a, a, x]

        .. seealso::
            * :meth:`bvars` -- all occurring bound variables
            * :meth:`qvars` -- all occurring quantified variables
            * :meth:`Term.vars() <.firstorder.atomic.Term.vars>` -- all occurring variables
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
        """The matrix of a prenex formula is its quantifier free part. This
        method returns the matrix along with the leading quantifiers.

        >>> from logic1.theories import RCF
        >>> x, y, z = RCF.VV.get('x', 'y', 'z')
        >>> f = All(x, All(y, Ex(z, x - y + z == 0)))
        >>> m, B = f.matrix()
        >>> m
        x - y + z == 0
        >>> B
        [(<class 'logic1.firstorder.quantified.All'>, [x, y]),
         (<class 'logic1.firstorder.quantified.Ex'>, [z])]

        Reconstruct ``f`` from ``m`` and ``B``:

        >>> g = m
        >>> for q, V in reversed(B):
        ...     g= q(V, g)
        >>> g == f
        True

        If `self` is not prenex, then the leading quantifiers are considered
        and the matrix will not be quantifier-free:

        >>> h = All(x, All(y, (x != 0) >> Ex(z, x * z == y)))
        >>> m, B = h.matrix()
        >>> m
        Implies(x != 0, Ex(z, x*z - y == 0))
        >>> B
        [(<class 'logic1.firstorder.quantified.All'>, [x, y])]

        .. seealso::
            * :mod:`.firstorder`
            * :mod:`.firstorder.boolean`
            * :mod:`.firstorder.pnf`
            * :mod:`.RCF`
            * :class:`pnf() <.firstorder.pnf>` -- prenex normal form
            * :data:`QuantifierBlock <.quantified.QuantifierBlock>` \
                -- a type that holds a block of quantifiers
        """
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
        """An iterator over all quantified variables in `self`.

        In the following example, ``z`` is a quantified variable but not a
        bound variable:

        >>> from logic1.theories import RCF
        >>> a, b, c, x, y, z = RCF.VV.get('a', 'b', 'c', 'x', 'y', 'z')
        >>> f = All(y, Ex(x, a == y) & Ex(z, a == y))
        >>> list(f.qvars())
        [y, x, z]

        .. seealso::
            * :meth:`bvars` -- all occurring bound variables
            * :meth:`fvars` -- all occurring free variables
            * :meth:`Term.vars() <.firstorder.atomic.Term.vars>` -- all occurring variables
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
        """A LaTeX representation for Jupyter notebooks. In general, the
        underlying method :meth:`as_latex` should be used instead.

        Due to a current limitation of Jupyter, the LaTeX representration is
        cut off after at most 5000 characters.

        .. seealso:: :meth:`as_latex` -- LaTeX representation
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
        """Fast basic simplification. The result is equivalent to `self`. The
        following first-order simplifications are applied:

        1. Truth values:

           a. Evaluate ``Not(F)`` to ``T``, and evaluate ``Not(T)`` to ``F``.

           b. Evaluate ``And(..., F, ...)`` to ``F`` and ``Or(..., T, ...)`` to
              ``T``.

           c. Evaluate ``Implies(F, arg)`` and ``Implies(arg, T)`` to ``T``.

           d. Remove ``T`` from ``And(..., T, ...)`` and ``F`` from ``Or(...,
              F, ...)``.

           e. Transform ``Implies(T, arg)`` into ``arg``, and transform
              ``Implies(arg, F)`` into ``Not(arg)``.

           f. Transform ``Equivalent(T, arg)`` and ``Equivalent(arg, T)`` into
              ``arg``, and transform ``Equivalent(F, arg)``, ``Equivalent(arg,
              F)`` into ``Not(arg)``.

        2. Nested operators:

           a. Transform ``Not(Not(arg))`` into ``arg``.

           b. Transform ``And(..., And(*args), ...)`` into ``And(..., *args,
              ...)``. The same for ``Or`` instead of ``And``.

        3. Equal arguments:

           a. Transform ``And(..., arg, ..., arg, ...)`` into ``And(..., arg,
              ...)``. The same for ``Or`` instead of ``And``.

           b. Evaluate ``Implies(arg, arg)`` to ``T``. The same for
              ``Equivalent`` instead of ``Implies``.

        4. Sort ``arg_1, ..., arg_n`` within ``And(arg_1, ..., arg_n)`` using a
           canonical order. The same for ``Or`` instead of ``And``.

        Overloading of :class:`AtomicFormula
        <.firstorder.atomic.AtomicFormula>` provides a hook for theories to
        extend :meth:`simplify` to atomic formulas.

        .. seealso::
           `simplify` methods of classes derived from :class:`AtomicFormula
           <.firstorder.atomic.AtomicFormula>` within various theories:

           * :meth:`logic1.theories.RCF.atomic.AtomicFormula.simplify`
           * :meth:`logic1.theories.Sets.atomic.AtomicFormula.simplify`

           More powerful simplifiers provided by various theories:

           * :func:`logic1.theories.RCF.simplify.simplify`
           * :func:`logic1.theories.Sets.simplify.simplify`
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
                    if arg_simplify is self.definite_element():
                        return self.definite_element()
                    if arg_simplify is self.neutral_element():
                        continue
                    if arg_simplify in simplified_args:
                        continue
                    if arg_simplify.op is self.op:
                        simplified_args.extend(arg_simplify.args)
                    else:
                        simplified_args.append(arg_simplify)
                return self.op(*simplified_args)
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
                return self.op(self.var, self.arg.simplify())
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
        Ex(x, a - x == 0)
        >>>
        >>> f.subs({a: x})
        Ex(G0001_x, -G0001_x + x == 0)
        >>>
        >>> g = Ex(x, _ & (b == 0))
        >>> g.subs({b: x})
        Ex(G0002_x, And(Ex(G0001_x, -G0001_x + G0002_x == 0), x == 0))
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
                    return self.op(var, self.arg.subs(substitution))  # type: ignore[return-value]
                return self.op(self.var, self.arg.subs(substitution))  # type: ignore[return-value]
            case And() | Or() | Not() | Implies() | Equivalent() | _F() | _T():
                return_value = self.op(*(arg.subs(substitution) for arg in self.args))
                return return_value  # type: ignore[return-value]
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
        >>> f = Equivalent(And(a == 0, T), Ex(y, Not(y == a)))
        >>> f.to_nnf()
        And(Or(a != 0, F, Ex(y, a - y != 0)), Or(All(y, a - y == 0), And(a == 0, T)))
        """
        nnf_op: type[Formula]
        rewrite: Formula
        match self:
            case All() | Ex():
                nnf_op = self.dual() if _not else self.op
                nnf_arg = self.arg.to_nnf(to_positive=to_positive, _not=_not)
                return nnf_op(self.var, nnf_arg)
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
                nnf_op = self.dual() if _not else self.op
                nnf_args: list[Formula] = []
                for arg in self.args:
                    nnf_arg = arg.to_nnf(to_positive=to_positive, _not=_not)
                    if nnf_arg.op is nnf_op:
                        nnf_args.extend(nnf_arg.args)
                    else:
                        nnf_args.append(nnf_arg)
                return nnf_op(*nnf_args)
            case Not():
                return self.arg.to_nnf(to_positive=to_positive, _not=not _not)
            case _F() | _T():
                if _not:
                    return self.dual()() if to_positive else Not(self)
                return self
            case AtomicFormula():
                if _not:
                    return self.to_complement() if to_positive else Not(self)
                return self
            case _:
                assert False, type(self)

    def to_pnf(self, prefer_universal: bool = False, is_nnf: bool = False):
        """Convert to Prenex Normal Form.

        A Prenex Normal Form (PNF) is a Negation Normal Form (NNF) in which all
        quantifiers :class:`Ex` and :class:`All` stand at the beginning of the
        formula. The method used here minimizes the number of quantifier
        alternations in the prenex block [Burhenne90]_.

        If the minimal number of alternations in the result can be achieved
        with both :class:`Ex` and :class:`All` as the first quantifier in the
        result, then the former is preferred. This preference can be changed
        with a keyword argument `prefer_universal=True`.

        An keyword argument `is_nnf=True` indicates that `self` is already in
        NNF. :meth:`to_pnf` then skips the initial NNF computation, which can
        be useful in time-critical situations.

        >>> from logic1.theories.RCF import *
        >>> a, b, y = VV.get('a', 'b', 'y')
        >>> f = Equivalent(And(a == 0, b == 0, y == 0),
        ...                Ex(y, Or(y == a, a == 0)))
        >>> f.to_pnf()
        Ex(G0001_y, All(G0002_y,
            And(Or(a != 0, b != 0, y != 0, -G0001_y + a == 0, a == 0),
                Or(And(-G0002_y + a != 0, a != 0), And(a == 0, b == 0, y == 0)))))

        .. [Burhenne90]
               Klaus-Dieter Burhenne. Implementierung eines Algorithmus zur
               Quantorenelimination für lineare reelle Probleme.
               Diploma Thesis, University of Passau, Germany, 1990
        """
        from .pnf import pnf
        return pnf(self, prefer_universal, is_nnf)

    def transform_atoms(self, tr: Callable[[Any], Formula]) -> Formula:
        """Apply `tr` to all atomic formulas.

        Replaces each atomic subformula of `self` with the :class:`Formula`
        `tr(self)`.

        >>> from logic1 import And
        >>> from logic1.theories.RCF import Eq, Lt, VV
        >>> x, y, z = VV.get('x', 'y', 'z')
        >>>
        >>> f = And(x == y, y < z)
        >>> f.transform_atoms(lambda atom: atom.op(atom.lhs - atom.rhs, 0))
        And(x - y == 0, y - z < 0)
        """
        # type of tr requieres discussion
        match self:
            case All() | Ex():
                return self.op(self.var, self.arg.transform_atoms(tr))
            case And() | Or() | Not() | Implies() | Equivalent() | _F() | _T():
                args = (arg.transform_atoms(tr) for arg in self.args)
                return self.op(*args)
            case AtomicFormula():
                return tr(self)
            case _:
                assert False, type(self)


# The following imports are intentionally late to avoid circularity.
from .atomic import AtomicFormula, Variable
from .boolean import And, Equivalent, Implies, involutive_not, Not, Or, _F, F, _T, T
from .quantified import All, Ex, QuantifierBlock
