"""Module for first-order formulas.

There are classes Ex, All, Equivalent, Implies (>>), And (&), Or (|), Not (~),
_T (T), and _F (F) which provide constructors for corresponding (sub)formulas.

There is no language L in the sense of model theory specified. Application
modules implement L as follows:

1. Choose sets R of admissible relation symbols and F of admissible function
   symbols, which includes constants.

2. For each relation in R derive a class Rel from the Abstract Base Class
   firstorder.AtomicFormula and implement at least the abstract methods
   specified by firstorder.AtomicFormula.

3. In combination with the first-order constructors above, the constructors Rel
   allow the construction of first-order L-formulas. Similarly to Ex, All etc.
   above, respective interactive constructors REL can provide error checking.

The firstorder module imposes no restrictions on the representation of atomic
formulas and terms. It is quite natural to represent relations similarly to the
logical operators in the subclasses of firstorder.Formula.

For the argument terms, sympy.Expr are certainly an option. This will support a
super set of the function symbols F in L, in general. The interacitve
constructors REL can check that only valid L-terms are used.
"""

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
    """

    # Class variables
    func: type[Formula]
    sympy_func: type[sympy.Basic]

    # Instance variables
    args: tuple

    # Instance methods
    @final
    def __and__(self, other: Formula) -> Formula:
        """Override the ``&`` operator to apply logical AND.

        Note that ``&`` delegates to the convenience wrapper AND in contrast to
        the constructor And.

        >>> from logic1.atomlib.sympy import EQ
        >>> EQ(0, 0) & EQ(1 + 1, 2) & EQ(1 + 1 + 1, 3)
        And(Eq(0, 0), Eq(2, 2), Eq(3, 3))
        """
        return And.interactive_new(self, other)

    @final
    def __invert__(self) -> Formula:
        """Override the ``~`` operator to apply logical NOT.

        Note that ``~`` delegates to the convenience wrapper NOT in contrast to
        the constructor Not.

        >>> from logic1.atomlib.sympy import EQ
        >>> ~ EQ(1,0)
        Not(Eq(1, 0))
        """
        return Not.interactive_new(self)

    @final
    def __lshift__(self, other: Formula) -> Formula:
        """Override ``>>`` operator to apply logical Implies.

        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import x, y, z
        >>> EQ(x + z, y + z) << EQ(x, y)
        Implies(Eq(x, y), Eq(x + z, y + z))
        """
        return Implies(other, self)

    @final
    def __or__(self, other: Formula) -> Formula:
        """Override the ``|`` operator to apply logical OR.

        Note that ``|`` delegates to the convenience wrapper OR in contrast to
        the constructor Or.

        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import x, y, z
        >>> EQ(x, 0) | EQ(x, y) | EQ(x, z)
        Or(Eq(x, 0), Eq(x, y), Eq(x, z))
        """
        return Or.interactive_new(self, other)

    @final
    def __rshift__(self, other: Formula) -> Formula:
        """Override the ``<<`` operator to apply logical Implies with reversed
        sides.

        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import x, y, z
        >>> EQ(x, y) >> EQ(x + z, y + z)
        Implies(Eq(x, y), Eq(x + z, y + z))
        """
        return Implies(self, other)

    def __eq__(self, other: object) -> bool:
        """Recursive equality of the formulas self and other.

        This is *not* logical ``equal.``

        >>> from logic1.atomlib.sympy import NE
        >>> e1 = NE(1, 0)
        >>> e2 = NE(1, 0)
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
        return not self == other

    def __hash__(self) -> int:
        return hash((self.func, self.args))

    @abstractmethod
    def __init__(self, *args) -> None:
        ...

    def __repr__(self) -> str:
        """Representation of the Formula suitable for use as an input.
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
        r"""A LaTeX representation of the formula as it is used within jupyter
        notebooks

        >>> from logic1 import F
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
        the expression tree. Occurrence of quantified variables is not checked.

        >>> from logic1 import Ex, All, T
        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import x, y, z
        >>> Ex(x, EQ(x, y) & All(x, Ex(y, Ex(z, T)))).count_alternations()
        2
        """
        return self._count_alternations()[0]

    @abstractmethod
    def _count_alternations(self) -> tuple[int, set]:
        ...

    @abstractmethod
    def get_any_atom(self) -> Optional[AtomicFormula]:
        """Return any atomic formula contained in self, None if there is none.

        A typical use cass is getting access to methods of classes derived from
        AtomicFormula elsewhere.
        """
        ...

    @abstractmethod
    def get_vars(self, assume_quantified: set = set()) -> GetVars:
        """Get variables.

        >>> from logic1 import Ex, All
        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import x, y, z

        >>> f = EQ(3 * x, 0) >> All(z, All(x,
        ...     ~ EQ(x, 0) >> Ex(y, EQ(x * y, 1))))
        >>> f.get_vars().free == {x}
        True

        >>> f.get_vars().bound == {x, y}
        True

        >>> z not in f.get_vars().all
        True
        """
        ...

    @abstractmethod
    def get_qvars(self) -> set:
        """The set of all variables that are quantified in self.

        This should not be confused with bound ocurrences of variables. Compare
        the Formula.get_vars() method.

        >>> from logic1 import Ex, All
        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import a, b, c, x, y, z
        >>> All(y, Ex(x, EQ(a, y)) & Ex(z, EQ(a, y))).get_qvars() == {x, y, z}
        True
        """
        ...

    def simplify(self, Theta=None) -> Formula:
        """Identity as a default implemenation of a simplifier for formulas.

        This should be overridden in the majority of the classes that
        are finally instantiated.
        """
        return self

    @abstractmethod
    def _sprint(self, mode: str) -> str:
        """Print to string.
        """
        ...

    @abstractmethod
    def subs(self, substitution: dict) -> Self:
        """Substitution.

        >>> from logic1 import push, pop, Ex
        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import a, b, c, x, y, z
        >>> push()

        >>> Ex(x, EQ(x, a)).subs({x: a})
        Ex(x, Eq(x, a))

        >>> f = Ex(x, EQ(x, a)).subs({a: x})
        >>> g = Ex(x, f & EQ(b, 0))
        >>> g
        Ex(x, And(Ex(x_R1, Eq(x_R1, x)), Eq(b, 0)))

        >>> g.subs({b: x})
        Ex(x_R2, And(Ex(x_R1, Eq(x_R1, x_R2)), Eq(x, 0)))

        >>> pop()
        """
        ...

    @final
    def to_distinct_vars(self) -> Self:
        """Convert to equivalent formulas with distinct variables.

        Bound variables are renamed such that that set of all bound
        variables is disjoint from the set of all free variables.
        Furthermore, each bound variables occurs with one and only one
        quantifier.

        >>> from logic1 import push, pop, Ex, All, T
        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import x, y, z
        >>> push()
        >>> f0 = All(z, Ex(y, EQ(x, y) & EQ(y, z) & Ex(x, T)))
        >>> f = EQ(x, y) & Ex(x, EQ(x, y) & f0)
        >>> f.to_distinct_vars()
        And(Eq(x, y), Ex(x_R3, And(Eq(x_R3, y),
            All(z, Ex(y_R2, And(Eq(x_R3, y_R2), Eq(y_R2, z), Ex(x_R1, T)))))))
        >>> pop()
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
        """Convert to LaTeX representation.
        """
        return self._sprint(mode='latex')

    @abstractmethod
    def to_nnf(self, implicit_not: bool = False,
               to_positive: bool = True) -> Formula:
        """Convert to Negation Normal Form.

        An NNF is a formula where logical Not is only applied to atomic
        formulas and thruth values. The only other allowed Boolean operators
        are And and Or. Besides those Boolean operators, we also admit
        quantifiers Ex and All. If the input is quanitfier-free, to_nnf will
        not introduce any quanitfiers.

        >>> from logic1 import Ex, Equivalent, NOT, T
        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import a, y
        >>> f = Equivalent(EQ(a, 0) & T, Ex(y, ~ EQ(y, a)))
        >>> f.to_nnf()
        And(Or(Ne(a, 0), F, Ex(y, Ne(y, a))),
            Or(All(y, Eq(y, a)), And(Eq(a, 0), T)))
        """
        ...

    @final
    def to_pnf(self, prefer_universal: bool = False,
               is_admissible: bool = False) -> Formula:
        """Convert to Prenex Normal Form.

        A Prenex Normal Form (PNF) is a Negation Normal Form (NNF) where all
        quantifiers Ex and All stand at the beginning of the formula. The
        method minimizes the number of quantifier alternations in the prenex
        block. Results starting with an existential quantifier are prefered.
        This can be changed by passing prefer_universal=True. The argument
        is_nnf can be used as a hint that self is already in NNF.

        Burhenne p.88:

        >>> from logic1 import push, pop, Ex, All, T, F
        >>> from logic1.atomlib.sympy import EQ
        >>> import sympy
        >>> push()
        >>> x = sympy.symbols('x:8')
        >>> f1 = Ex(x[1], All(x[2], All(x[3], T)))
        >>> f2 = All(x[4], Ex(x[5], All(x[6], F)))
        >>> f3 = Ex(x[7], EQ(x[0], 0))
        >>> (f1 & f2 & f3).to_pnf()
        All(x4, Ex(x1, Ex(x5, Ex(x7, All(x2, All(x3, All(x6,
            And(T, F, Eq(x0, 0)))))))))
        >>> pop()

        Derived from redlog.tst:

        >>> push()
        >>> from logic1 import Equivalent, AND, OR
        >>> from sympy.abc import a, b, y
        >>> f1 = EQ(a, 0) & EQ(b, 0) & EQ(y, 0)
        >>> f2 = Ex(y, EQ(y, a) | EQ(a, 0))
        >>> Equivalent(f1, f2).to_pnf()
        Ex(y_R1, All(y_R2,
            And(Or(Ne(a, 0), Ne(b, 0), Ne(y, 0), Eq(y_R1, a), Eq(a, 0)),
                Or(And(Ne(y_R2, a), Ne(a, 0)),
                   And(Eq(a, 0), Eq(b, 0), Eq(y, 0))))))
        >>> pop()

        >>> push()
        >>> Equivalent(f1, f2).to_pnf(prefer_universal=True)
        All(y_R2, Ex(y_R1,
            And(Or(Ne(a, 0), Ne(b, 0), Ne(y, 0), Eq(y_R1, a), Eq(a, 0)),
                Or(And(Ne(y_R2, a), Ne(a, 0)),
                   And(Eq(a, 0), Eq(b, 0), Eq(y, 0))))))
        >>> pop()
        """
        if is_admissible:
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

    def to_sympy(self, **kwargs) -> sympy.Basic:
        """Provide a sympy representation of the Formula if possible.

        Subclasses that have no match in sympy can raise NotImplementedError.

        >>> from logic1 import Equivalent, T
        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import x, y
        >>> e1 = Equivalent(EQ(x, y), EQ(x + 1, y + 1))
        >>> e1
        Equivalent(Eq(x, y), Eq(x + 1, y + 1))
        >>> type(e1)
        <class 'logic1.firstorder.boolean.Equivalent'>
        >>> e1.to_sympy()
        Equivalent(Eq(x, y), Eq(x + 1, y + 1))
        >>> type(e1.to_sympy())
        Equivalent

        >>> e2 = Equivalent(EQ(x, y), EQ(y, x))
        >>> e2
        Equivalent(Eq(x, y), Eq(y, x))
        >>> e2.to_sympy()
        True

        >>> e3 = T
        >>> e3.to_sympy()
        Traceback (most recent call last):
        ...
        NotImplementedError:
            sympy does not know <class 'logic1.firstorder.truth._T'>

        >>> e4 = All(x, Ex(y, EQ(x, y)))
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
        ...


latex = Formula.to_latex


# The following imports are intentionally late to avoid circularity.
from .boolean import Implies, And, Or, Not
from .quantified import Ex, All
