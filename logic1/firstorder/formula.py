"""Provides an abstract base class for first-order formulas."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, final, Optional, TYPE_CHECKING
from typing_extensions import Self

from ..support.containers import GetVars
# from ..support.tracing import trace

if TYPE_CHECKING:
    import sympy  # noqa

    from .atomic import AtomicFormula


class Formula(ABC):
    """An abstract base class for first-order formulas.

    All other classes in the :mod:`.firstorder` package are derived from
    :class:`Formula`.
    """

    # The following would be abstract class variables, which are not available
    # at the moment.
    func: type[Formula]  #: :meta private:
    sympy_func: type[sympy.core.basic.Basic]  #: :meta private:

    # Similaryly the following would be an abstract instance variable:
    args: tuple  #: :meta private:

    # Instance methods
    @final
    def __and__(self, other: Formula) -> Formula:
        """Override the :obj:`&` operator to apply :class:`.boolean.And`.

        >>> from logic1.atomlib.sympy import Eq
        >>>
        >>> Eq(0, 0) & Eq(1 + 1, 2) & Eq(1 + 1 + 1, 3)
        And(Eq(0, 0), Eq(2, 2), Eq(3, 3))
        """
        return And(self, other)

    @final
    def __invert__(self) -> Formula:
        """Override the :obj:`~` operator to apply :class:`.boolean.Not`.

        >>> from logic1.atomlib.sympy import Eq
        >>>
        >>> ~ Eq(1,0)
        Not(Eq(1, 0))
        """
        return Not(self)

    @final
    def __lshift__(self, other: Formula) -> Formula:
        """Override :obj:`<<` operator to apply :class:`.boolean.Implies`
        with reversed sides.

        >>> from logic1.atomlib.sympy import Eq
        >>> from sympy.abc import x, y, z
        >>>
        >>> Eq(x + z, y + z) << Eq(x, y)
        Implies(Eq(x, y), Eq(x + z, y + z))
        """
        return Implies(other, self)

    @final
    def __or__(self, other: Formula) -> Formula:
        """Override the :obj:`|` operator to apply :class:`.boolean.Or`.

        >>> from logic1.atomlib.sympy import Eq
        >>> from sympy.abc import x, y, z
        >>>
        >>> Eq(x, 0) | Eq(x, y) | Eq(x, z)
        Or(Eq(x, 0), Eq(x, y), Eq(x, z))
        """
        return Or(self, other)

    @final
    def __rshift__(self, other: Formula) -> Formula:
        """Override the :obj:`>>` operator to apply :class:`.boolean.Implies`.

        >>> from logic1.atomlib.sympy import Eq
        >>> from sympy.abc import x, y, z
        >>>
        >>> Eq(x, y) >> Eq(x + z, y + z)
        Implies(Eq(x, y), Eq(x + z, y + z))
        """
        return Implies(self, other)

    def __eq__(self, other: object) -> bool:
        """A recursive test for equality of the :class:`Formula` `self`
        and the :class:`object` `other`.

        Note that this is is not a logical operator for equality.

        >>> from logic1.atomlib.sympy import Ne
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
        """A recursive test for unequality of the :class:`Formula` `self`
        and the :class:`object` `other`.
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
    def _repr_latex_(self) -> str:
        r"""A LaTeX representation of the :class:`Formula` `self` as it is used
        within jupyter notebooks.

        >>> from logic1 import F
        >>>
        >>> F._repr_latex_()
        '$\\displaystyle \\bot$'

        Subclasses have to_latex() methods yielding plain LaTeX without the
        surrounding $\\displaystyle ... $.
        """
        return '$\\displaystyle ' + self.to_latex() + '$'

    @final
    def count_alternations(self) -> int:
        """Count number of quantifier alternations.

        Returns the maximal number of quantifier alternations along a path in
        the expression tree. Occurrence of quantified variables is not checked,
        so that quantifiers with unused variables are counted.

        >>> from logic1 import Ex, All, T
        >>> from logic1.atomlib.sympy import Eq
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
    def get_any_atom(self) -> Optional[AtomicFormula]:
        """Return any atomic formula contained in *self*, or None if there is
        none.

        A typical use case is getting access to static methods of classes
        derived from :class:`.atomic.AtomicFormula` elsewhere.

        >>> from logic1 import Ex, And
        >>> from logic1.atomlib.sympy import Eq
        >>> from sympy.abc import x, y, z
        >>>
        >>> f = Ex(x, Eq(x, -y) & Eq(y, z ** 2))
        >>> isinstance(f.var, f.get_any_atom().variable_type())
        True
        """
        ...

    @abstractmethod
    def get_qvars(self) -> set:
        """The set of all variables that are quantified in self.

        >>> from logic1 import Ex, All
        >>> from logic1.atomlib.sympy import Eq
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
        >>> from logic1.atomlib.sympy import Eq
        >>> from sympy.abc import x, y, z
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

    def simplify(self, Theta=None) -> Formula:
        """Fast simplification. The result is equivalent to `self`.

        Primary simplification goals are the elimination of occurrences of
        :data:`.truth.T` and :data:`.truth.F` and of occurrences of equal
        subformulas as siblings in the expression tree.
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
        >>> from logic1.support import renaming
        >>> from logic1.atomlib.sympy import Eq
        >>> from sympy.abc import a, b, c, x, y, z
        >>> renaming.push()  # temporarily create a fresh counter for renaming
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
        >>>
        >>> renaming.pop()  # restore renaming counter
        """
        ...

    @final
    def to_distinct_vars(self) -> Self:
        """Convert to equivalent formulas with distinct variables.

        Bound variables are renamed such that that set of all bound variables
        is disjoint from the set of all free variables. Furthermore, each bound
        variable in the result occurs with one and only one quantifier.

        >>> from logic1 import Ex, All, T
        >>> from logic1.support import renaming
        >>> from logic1.atomlib.sympy import Eq
        >>> from sympy.abc import x, y, z
        >>> renaming.push()  # temporarily create a fresh counter for renaming
        >>>
        >>> f0 = All(z, Ex(y, Eq(x, y) & Eq(y, z) & Ex(x, T)))
        >>> f = Eq(x, y) & Ex(x, Eq(x, y) & f0)
        >>> f
        And(Eq(x, y), Ex(x, And(Eq(x, y),
            All(z, Ex(y, And(Eq(x, y), Eq(y, z), Ex(x, T)))))))
        >>>
        >>> f.to_distinct_vars()
        And(Eq(x, y), Ex(x_R3, And(Eq(x_R3, y),
            All(z, Ex(y_R2, And(Eq(x_R3, y_R2), Eq(y_R2, z), Ex(x_R1, T)))))))
        >>>
        >>> renaming.pop()  # restore renaming counter
        """
        # Recursion starts with a badlist (technically a set) of all free
        # variables.
        return self._to_distinct_vars(self.get_vars().free)

    @abstractmethod
    def _to_distinct_vars(self, badlist: set) -> Self:
        # Traverses self. If a badlisted variable is encountered as a
        # quantified variable, it will be replaced with a fresh name in the
        # respective QuantifiedFormula, and the fresh name will be badlisted
        # for the future. Note that this can includes variables that do not
        # *occur* in a mathematical sense.
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
        application of :class:`.boolean.Not` is restricted to atomic formulas,
        i.e., instances of :class:`.atomic.AtomicFormula`, and truth values
        :data:`.truth.T` and :data:`.truth.F`. The only other operators
        admitted are :class:`.boolean.And`, :class:`.boolean.Or`,
        :class:`.quantified.Ex`, and :class:`.quantified.All`.

        If the input is quanitfier-free, :meth:`to_nnf` will not introduce any
        quanitfiers.

        If `to_positive` is `True`, :class:`.boolean.Not` is eliminated by
        replacing relation symbols with their complements. The result is then
        even a Positive Normal Form.

        >>> from logic1 import Ex, Equivalent, T
        >>> from logic1.atomlib.sympy import Eq
        >>> from sympy.abc import a, y
        >>>
        >>> f = Equivalent(Eq(a, 0) & T, Ex(y, ~ Eq(y, a)))
        >>> f.to_nnf()
        And(Or(Ne(a, 0), F, Ex(y, Ne(y, a))),
            Or(All(y, Eq(y, a)), And(Eq(a, 0), T)))
        """
        ...

    @final
    def to_pnf(self, prefer_universal: bool = False,
               is_nnf: bool = False) -> Formula:
        """Convert to Prenex Normal Form.

        A Prenex Normal Form (PNF) is a Negation Normal Form (NNF) in which all
        quantifiers :class:`.quantified.Ex` and :class:`.quantified.All` stand
        at the beginning of the formula. The method used here minimizes the
        number of quantifier alternations in the prenex block [Burhenne90].

        If the minimal number of alternations in the result can be achieved
        with both :class:`.quantified.Ex` and :class:`.quantified.All` as the
        first quantifier in the result, then the former is preferred. This
        preference can be changed with a keyword argument
        `prefer_universal=True`.

        An keyword argument `is_nnf=True` indicates that `self` is already in
        NNF. :meth:`to_pnf` then skips the initial NNF computation, which can
        be useful in time-critical situations.

        Example from p.88 in [Burhenne90]_:

        >>> from logic1 import Ex, All, T, F
        >>> from logic1.support import renaming
        >>> from logic1.atomlib.sympy import Eq
        >>> import sympy
        >>>
        >>> renaming.push()  # temporarily create a fresh counter for renaming
        >>> x = sympy.symbols('x:8')
        >>> f1 = Ex(x[1], All(x[2], All(x[3], T)))
        >>> f2 = All(x[4], Ex(x[5], All(x[6], F)))
        >>> f3 = Ex(x[7], Eq(x[0], 0))
        >>> (f1 & f2 & f3).to_pnf()
        All(x4, Ex(x1, Ex(x5, Ex(x7, All(x2, All(x3, All(x6,
            And(T, F, Eq(x0, 0)))))))))
        >>> renaming.pop()  # restore renaming counter

        Derived from the `rlpnf` test in `redlog.tst
        <https://sourceforge.net/p/reduce-algebra/code/HEAD/tree/trunk/packages/redlog/rl/redlog.tst>`_:

        >>> from logic1 import Ex, All, Equivalent, And, Or
        >>> from logic1.support import renaming
        >>> from logic1.atomlib.sympy import Eq
        >>> from sympy.abc import a, b, y
        >>>
        >>> renaming.push()
        >>> f1 = Eq(a, 0) & Eq(b, 0) & Eq(y, 0)
        >>> f2 = Ex(y, Eq(y, a) | Eq(a, 0))
        >>> Equivalent(f1, f2).to_pnf()
        Ex(y_R1, All(y_R2,
            And(Or(Ne(a, 0), Ne(b, 0), Ne(y, 0), Eq(y_R1, a), Eq(a, 0)),
                Or(And(Ne(y_R2, a), Ne(a, 0)),
                   And(Eq(a, 0), Eq(b, 0), Eq(y, 0))))))
        >>> renaming.pop()
        >>>
        >>> renaming.push()
        >>> Equivalent(f1, f2).to_pnf(prefer_universal=True)
        All(y_R2, Ex(y_R1,
            And(Or(Ne(a, 0), Ne(b, 0), Ne(y, 0), Eq(y_R1, a), Eq(a, 0)),
                Or(And(Ne(y_R2, a), Ne(a, 0)),
                   And(Eq(a, 0), Eq(b, 0), Eq(y, 0))))))
        >>> renaming.pop()

        .. [Burhenne90]
               Klaus-Dieter Burhenne. Implementierung eines Algorithmus zur
               Quantorenelimination für lineare reelle Probleme.
               Diploma Thesis, University of Passau, Germany, 1990

        """
        if is_nnf:
            phi = self
        else:
            phi = self.to_nnf().to_distinct_vars()
        return phi._to_pnf()[All if prefer_universal else Ex]

    # abstract - see docstring
    def _to_pnf(self) -> dict:
        """Private convert to Prenex Normal Form.

        self must be in NNF. All NNF operators (QuantifiedFormula, AndOr,
        TruthValue, AtomicFormula) must override this method. The result is a
        dict d with len(d) == 2. The keys are Ex and All, the values are both
        prenex equivalents of self with the same minimized number of quantifier
        alternations. Either d[Ex] starts with an existential quantifier and
        d[All] starts with a universal quantifier, or d[Ex] is d[All], i.e.,
        identity is guaranteed.
        """
        raise NotImplementedError(f'{self.func} is not an NNF operator')

    def to_sympy(self, **kwargs) -> sympy.core.basic.Basic:
        """Convert to SymPy representation if possible.

        Subclasses that have no match in Symy raise NotImplementedError. All
        keyword arguments are passed on to the SymPy constructors.

        >>> from logic1 import Equivalent, T
        >>> from logic1.atomlib.sympy import Eq
        >>> from sympy.abc import x, y
        >>>
        >>> e1 = Equivalent(Eq(x, y), Eq(x + 1, y + 1))
        >>> type(e1)
        <class 'logic1.firstorder.boolean.Equivalent'>
        >>>
        >>> e1.to_sympy()
        Equivalent(Eq(x, y), Eq(x + 1, y + 1))
        >>> type(e1.to_sympy())
        Equivalent
        >>>
        >>> e2 = Equivalent(Eq(x, y), Eq(y, x))
        >>> e2.to_sympy()
        True
        >>>
        >>> e3 = T
        >>> e3.to_sympy()
        Traceback (most recent call last):
        ...
        NotImplementedError:
            sympy does not know <class 'logic1.firstorder.truth._T'>
        >>>
        >>> e4 = All(x, Ex(y, Eq(x, y)))
        >>> e4.to_sympy()
        Traceback (most recent call last):
        ...
        NotImplementedError:
            sympy does not know <class 'logic1.firstorder.quantified.All'>
        """
        args_sympy = (arg.to_sympy(**kwargs) for arg in self.args)
        return self.__class__.sympy_func(*args_sympy)

    @abstractmethod
    def transform_atoms(self, transformation: Callable) -> Self:
        """Apply `transformation` to all atomic formulas.

        Replaces each atomic subformula of `self` with the :class:`Formula`
        `transformation(self)`.

        >>> from logic1 import And
        >>> from logic1.atomlib.sympy import Eq, Lt
        >>> from sympy.abc import x, y, z
        >>>
        >>> f = Eq(x, y) & Lt(y, z)
        >>> f.transform_atoms(lambda atom: atom.func(atom.lhs - atom.rhs, 0))
        And(Eq(x - y, 0), Lt(y - z, 0))
        """
        ...


# The following imports are intentionally late to avoid circularity.
from .boolean import Implies, And, Or, Not
from .quantified import Ex, All
