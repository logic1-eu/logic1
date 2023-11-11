"""This module provides a library of atomic formulas based on SymPy terms.
"""
from __future__ import annotations

import sympy

from typing import Any, ClassVar, TypeAlias, Union

from .. import firstorder
from ..firstorder import T, F
from ..support.containers import GetVars
from ..support.decorators import classproperty
from ..support.renaming import rename

# Type alias
Card = Union[int, sympy.core.numbers.Infinity]

Term: TypeAlias = sympy.Expr
Variable: TypeAlias = sympy.Symbol

oo = sympy.oo


class TermMixin():

    @staticmethod
    def term_get_vars(term: Term) -> set[Variable]:
        """Implements the abstract method
        :meth:`.firstorder.AtomicFormula.term_get_vars`.
        """
        return sympy.S(term).atoms(Variable)

    @staticmethod
    def term_to_latex(term: Term) -> str:
        """Implements the abstract method
        :meth:`.firstorder.AtomicFormula.term_to_latex`.
        """
        return sympy.latex(term)

    @staticmethod
    def variable_type() -> type[Variable]:
        """Implements the abstract method
        :meth:`.firstorder.AtomicFormula.variable_type`.
        """
        return Variable

    @staticmethod
    def rename_var(variable: Variable) -> Variable:
        """Implements the abstract method
        :meth:`.firstorder.AtomicFormula.rename_var`.
        """
        return rename(variable)


class AtomicFormula(TermMixin, firstorder.AtomicFormula):
    """Atomic Formula with Sympy Terms. All terms are :class:`sympy.Expr`.
    """

    # Instance methods
    def get_vars(self, assume_quantified: set = set()) -> GetVars:
        """Implements the abstract method :meth:`.firstorder.Formula.get_vars`.
        """
        all_vars = set()
        for term in self.args:
            all_vars |= term.atoms(sympy.Symbol)
        return GetVars(free=all_vars - assume_quantified,
                       bound=all_vars & assume_quantified)

    def subs(self, substitution: dict) -> AtomicFormula:
        """Implements the abstract method :meth:`.firstorder.Formula.subs`.
        """
        args = (arg.subs(substitution, simultaneous=True) for arg in self.args)
        return self.func(*args)


class BinaryAtomicFormula(AtomicFormula):
    r"""A class whose instances are binary formulas in the sense that both
    their m-arity and their p-arity is 2.

    Let `R` be a subclass of :class:`BinaryAtomicFormula` implementing atomic
    formulas with a binary relation symbol :math:`r`. For instance, if `R` is
    :class:`Eq`, then :math:`r` stands for the equality relation :math:`=` in
    the following discussion:

    >>> Eq(0, 0).func
    <class 'logic1.atomlib.sympy.Eq'>
    >>> Eq.func
    <class 'logic1.atomlib.sympy.Eq'>

    Assume that :math:`r` is defined on a domain :math:`D`. Then the
    *complement relation* of :math:`r` is defined as :math:`\overline{r} = D^2
    \setminus r`. It is avaialable as a class property `R.complement_func`,
    e.g.:

    >>> rels = (Eq, Ne, Ge, Le, Gt, Lt)
    >>> tuple(r.complement_func for r in rels)
    (<class 'logic1.atomlib.sympy.Ne'>,
     <class 'logic1.atomlib.sympy.Eq'>,
     <class 'logic1.atomlib.sympy.Lt'>,
     <class 'logic1.atomlib.sympy.Gt'>,
     <class 'logic1.atomlib.sympy.Le'>,
     <class 'logic1.atomlib.sympy.Ge'>)

    Since :math:`\overline{r}(s, t)` is equivalent to :math:`\neg r(s, t)`, the
    availability of complement relations is relevant for the computation of
    positive negation normal forms; compare :meth:`.firstorder.Formula.to_nnf`
    with keyword argument `to_positive=True`.

    The *converse relation* of :math:`r` is defined as
    :math:`r^{-1} = \{ (x, y) \in D : (y, x) \in r \}`.
    It is avaialable as a class property `R.converse_func`, e.g.:

    >>> tuple(r.converse_func for r in rels)
    (<class 'logic1.atomlib.sympy.Eq'>,
     <class 'logic1.atomlib.sympy.Ne'>,
     <class 'logic1.atomlib.sympy.Le'>,
     <class 'logic1.atomlib.sympy.Ge'>,
     <class 'logic1.atomlib.sympy.Lt'>,
     <class 'logic1.atomlib.sympy.Gt'>)

    The converse relation is the inverse with respect to composition.

    Finally, the *dual relation* of :math:`r` is defined as
    :math:`\overline{r}^{-1}`. It is available as a class property
    `R.dual_func`. Generally, :math:`\overline{r}^{-1} = \overline{r^{-1}}`,
    e.g.:

    >>> tuple(r.dual_func for r in rels)
    (<class 'logic1.atomlib.sympy.Ne'>,
     <class 'logic1.atomlib.sympy.Eq'>,
     <class 'logic1.atomlib.sympy.Gt'>,
     <class 'logic1.atomlib.sympy.Lt'>,
     <class 'logic1.atomlib.sympy.Ge'>,
     <class 'logic1.atomlib.sympy.Le'>)
    >>> all(r.dual_func == r.complement_func.converse_func for r in rels)
    True
    >>> all(r.dual_func == r.converse_func.complement_func for r in rels)
    True

    In the context of orderings, dualization turns strict inequalities into
    weak inequalities, and vice versa. Note that we also have duality and
    corresponding properties with Boolean functions, which is defined
    differently.

    All those operators on relations are involutive:

    >>> all(r.complement_func.complement_func == r for r in rels)
    True
    >>> all(r.converse_func.converse_func == r for r in rels)
    True
    >>> all(r.dual_func.dual_func == r for r in rels)
    True
    """

    @classproperty
    def dual_func(cls):
        """A class property yielding the dual class of this class or of the
        derived subclass.

        There is an implicit assumption that there are abstract class
        properties `complement_func` and `converse_func` specified, which is
        technically not possible at the moment.
        """
        return cls.complement_func.converse_func

    # The following would be abstract class variables, which are not available
    # at the moment.
    latex_symbol: str  #: :meta private:
    text_symbol: str  #: :meta private:

    # Similarly the following would be an abstract instance variable:
    args: tuple[Term, Term]

    @property
    def lhs(self) -> Term:
        """The left-hand side of the :class:`BinaryAtomicFormula`.
        """
        return self.args[0]

    @property
    def rhs(self) -> Term:
        """The right-hand side of the :class:`BinaryAtomicFormula`.
        """
        return self.args[1]

    # Instance methods
    def __init__(self, *args):
        if len(args) != 2:
            raise ValueError(f'bad number of arguments for binary relation')
        args_ = []
        for arg in args:
            arg_ = sympy.Integer(arg) if isinstance(arg, int) else arg
            if not isinstance(arg_, Term):
                raise ValueError(f"{arg!r} is not a Term")
            args_.append(arg_)
        super().__init__(*args_)

    def _sprint(self, mode: str) -> str:
        if mode == 'latex':
            symbol = self.__class__.latex_symbol
            lhs = sympy.latex(self.lhs)
            rhs = sympy.latex(self.rhs)
            spacing = self.__class__.latex_symbol_spacing
        else:
            assert mode == 'text'
            symbol = self.__class__.text_symbol
            lhs = self.lhs.__str__()
            rhs = self.rhs.__str__()
            spacing = self.__class__.text_symbol_spacing
        return f'{lhs}{spacing}{symbol}{spacing}{rhs}'


class Eq(BinaryAtomicFormula):
    """A class whose instances are equations in the sense that their toplevel
    operator represents the relation symbol :math:`=`.

    >>> from sympy import exp, I, pi
    >>> from sympy.abc import t
    >>>
    >>> equation = Eq(exp(t * I * pi, evaluate=False), -1)
    >>> equation
    Eq(exp(I*pi*t), -1)
    >>> equation.lhs
    exp(I*pi*t)
    >>> equation.rhs
    -1
    """

    # Class variables
    latex_symbol = '='
    """A class variable holding a LaTeX symbol for :class:`Eq`.

    This is used with :meth:`.firstorder.Formula.to_latex`, which is in turn
    used for the output in Jupyter notebooks.
    """

    text_symbol = '=='
    """A class variable holding a representation of :class:`Eq` suitable for
    string representation.

    This is used for string conversions, e.g., explicitly with the constructor
    of :class:`str` or implicitly with :func:`print`.
    """

    func: type[Eq]
    """A type annotation for the class property `func` inherited from
    :attr:`.firstorder.AtomicFormula.func`.
    """

    @classproperty
    def complement_func(cls):
        """A class property yielding the complement class :class:`Ne` of
        :class:'Eq'.
        """
        return Ne

    @classproperty
    def converse_func(cls):
        """A class property yielding the converse class :class:`Eq` of
        :class:'Eq'.
        """
        return Eq

    # Instance methods
    def simplify(self):
        """Compare the parent method :meth:`.firstorder.Formula.simplify`.

        >>> from sympy.abc import x, y
        >>>
        >>> Eq(x, x)
        Eq(x, x)
        >>> Eq(x, x).simplify()
        T
        >>> Eq(x, y).simplify()
        Eq(x, y)
        """
        if self.lhs == self.rhs:
            return T
        return self


class Ne(BinaryAtomicFormula):
    r"""A class whose instances are inequations in the sense that their
    toplevel operator represents the relation symbol :math:`\neq`.

    >>> Ne(1, 0)
    Ne(1, 0)
    """

    # Class variables
    latex_symbol = '\\neq'
    """A class variable holding a LaTeX symbol for :class:`Ne`.

    This is used with :meth:`.firstorder.Formula.to_latex`, which is in turn
    used for the output in Jupyter notebooks.
    """

    text_symbol = '!='
    """A class variable holding a representation of :class:`Ne` suitable for
    string representation.

    This is used for string conversions, e.g., explicitly with the constructor
    of :class:`str` or implicitly with :func:`print`.
    """

    func: type[Ne]
    """A type annotation for the class property `func` inherited from
    :attr:`.firstorder.AtomicFormula.func`.
    """

    @classproperty
    def complement_func(cls):
        """A class property yielding the complement class :class:`Eq` of
        :class:'Ne'.
        """
        return Eq

    @classproperty
    def converse_func(cls):
        """A class property yielding the converse class :class:`Ne` of
        :class:'Ne'.
        """
        return Ne

    # Instance methods
    def simplify(self):
        """Compare the parent method :meth:`.firstorder.Formula.simplify`.

        >>> from sympy.abc import x, y
        >>>
        >>> Ne(x, x)
        Ne(x, x)
        >>> Ne(x, x).simplify()
        F
        >>> Ne(x, y).simplify()
        Ne(x, y)
        """
        if self.lhs == self.rhs:
            return F
        return self


class Ge(BinaryAtomicFormula):
    r"""A class whose instances are inequalities where the toplevel operator
    represents the relation symbol :math:`\geq`.

    >>> Ge(1, 0)
    Ge(1, 0)
    """

    # Class variables
    latex_symbol = '\\geq'
    """A class variable holding a LaTeX symbol for :class:`Ge`.

    This is used with :meth:`.firstorder.Formula.to_latex`, which is in turn
    used for the output in Jupyter notebooks.
    """

    text_symbol = '>='
    """A class variable holding a representation of :class:`Ge` suitable for
    string representation.

    This is used for string conversions, e.g., explicitly with the constructor
    of :class:`str` or implicitly with :func:`print`.
    """

    func: type[Ge]
    """A type annotation for the class property `func` inherited from
    :attr:`.firstorder.AtomicFormula.func`.
    """

    @classproperty
    def complement_func(cls):
        """A class property yielding the complement class :class:`Lt` of
        :class:'Ge'.
        """
        return Lt

    @classproperty
    def converse_func(cls):
        """A class property yielding the converse class :class:`Le` of
        :class:'Ge'.
        """
        return Le


class Le(BinaryAtomicFormula):
    r"""A class whose instances are inequalities where the toplevel operator
    represents the relation symbol :math:`\leq`.

    >>> Le(1, 0)
    Le(1, 0)
    """

    # Class variables
    latex_symbol = '\\leq'
    """A class variable holding a LaTeX symbol for :class:`Le`.

    This is used with :meth:`.firstorder.Formula.to_latex`, which is in turn
    used for the output in Jupyter notebooks.
    """

    text_symbol = '<='
    """A class variable holding a representation of :class:`Le` suitable for
    string representation.

    This is used for string conversions, e.g., explicitly with the constructor
    of :class:`str` or implicitly with :func:`print`.
    """

    func: type[Le]
    """A type annotation for the class property `func` inherited from
    :attr:`.firstorder.AtomicFormula.func`.
    """

    @classproperty
    def complement_func(cls):
        """A class property yielding the complement class :class:`Gt` of
        :class:'Le'.
        """
        return Gt

    @classproperty
    def converse_func(cls):
        """A class property yielding the converse class :class:`Ge` of
        :class:'Le'.
        """
        return Ge


class Gt(BinaryAtomicFormula):
    r"""A class whose instances are inequalities where the toplevel operator
    represents the relation symbol :math:`>`.

    >>> Gt(1, 0)
    Gt(1, 0)
    """
    # Class variables
    latex_symbol = '>'
    """A class variable holding a LaTeX symbol for :class:`Gt`.

    This is used with :meth:`.firstorder.Formula.to_latex`, which is in turn
    used for the output in Jupyter notebooks.
    """

    text_symbol = '>'
    """A class variable holding a representation of :class:`Gt` suitable for
    string representation.

    This is used for string conversions, e.g., explicitly with the constructor
    of :class:`str` or implicitly with :func:`print`.
    """

    func: type[Gt]
    """A type annotation for the class property `func` inherited from
    :attr:`.firstorder.AtomicFormula.func`.
    """

    @classproperty
    def complement_func(cls):
        """A class property yielding the complement class :class:`Le` of
        :class:'Gt'.
        """
        return Le

    @classproperty
    def converse_func(cls):
        """A class property yielding the converse class :class:`Lt` of
        :class:'Gt'.
        """
        return Lt


class Lt(BinaryAtomicFormula):
    r"""A class whose instances are inequalities where the toplevel operator
    represents the relation symbol :math:`<`.

    >>> Lt(1, 0)
    Lt(1, 0)
    """

    # Class variables
    latex_symbol = '<'
    """A class variable holding a LaTeX symbol for :class:`Lt`.

    This is used with :meth:`.firstorder.Formula.to_latex`, which is in turn
    used for the output in Jupyter notebooks.
    """

    text_symbol = '<'
    """A class variable holding a representation of :class:`Lt` suitable for
    string representation.

    This is used for string conversions, e.g., explicitly with the constructor
    of :class:`str` or implicitly with :func:`print`.
    """

    func: type[Lt]
    """A type annotation for the class property `func` inherited from
    :attr:`.firstorder.AtomicFormula.func`.
    """

    @classproperty
    def complement_func(cls):
        """A class property yielding the complement class :class:`Ge` of
        :class:'Lt'.
        """
        return Ge

    @classproperty
    def converse_func(cls):
        """A class property yielding the converse class :class:`Gt` of
        :class:'Lt'.
        """
        return Gt


class IndexedConstantAtomicFormula(AtomicFormula):
    r"""A class whose instances form a family of atomic formulas with m-arity
    0. Their p-arity is 1, where the one argument of the constructor is the
    index.
    """
    # Instance variables
    @property
    def index(self) -> Any:
        """The index of the :class:`IndexedConstantAtomicFormula`.
        """
        return self.args[0]

    # Instance methods
    def get_vars(self, assume_quantified: set = set()) -> GetVars:
        """Implements the abstract method :meth:`.firstorder.Formula.get_vars`.
        """
        return GetVars()


class C(IndexedConstantAtomicFormula):
    r"""A class whose instances are cardinality constraints in the sense that
    their toplevel operator represents a constant relation symbol :math:`C_n`
    where :math:`n \in \mathbb{N} \cup \{\infty\}`. A typical interpretation in
    a domain :math:`D` is that :math:`C_n` holds iff :math:`|D| \geq n`.

    The class constructor takes one argument, which is the index `n`. It takes
    care that instance with equal indices are identical.

    >>> c_0_1 = C(0)
    >>> c_0_2 = C(0)
    >>> c_oo = C(oo)
    >>> c_0_1 is c_0_2
    True
    >>> c_0_1 == c_oo
    False
    """

    # Class variables
    func: type[C]  #: :meta private:
    """A type annotation for the class property `func` inherited from
    :attr:`.firstorder.AtomicFormula.func`.
    """

    @classproperty
    def complement_func(cls):
        """A class property yielding the complement class :class:`C_` of
        :class:`C`.
        """
        return C_

    _instances: ClassVar[dict] = {}
    """A private class variable, which is a dictionary holding unique instances
    of `C(n)` with key `n`.
    """

    # Instance variables
    args: tuple[int]  #: :meta private:
    """A type annotation for the property `func` inherited from
    :attr:`.firstorder.AtomicFormula.func`.

    :meta private:
    """

    # Class methods
    def __new__(cls, *args):
        if len(args) != 1:
            raise ValueError(f"bad number of arguments")
        n = args[0]
        if not isinstance(n, (int, sympy.core.numbers.Infinity)) or n < 0:
            raise ValueError(f"{n!r} is not an admissible cardinality")
        if n not in cls._instances:
            cls._instances[n] = super().__new__(cls)
        return cls._instances[n]

    # Instance methods
    def __repr__(self):
        return f'C({self.index})'

    def _sprint(self, mode: str) -> str:
        if mode == 'text':
            return repr(self)
        assert mode == 'latex', f'bad print mode {mode!r}'
        k = str(self.index) if isinstance(self.index, int) else '\\infty'
        return f'C_{{{k}}}'


class C_(IndexedConstantAtomicFormula):
    r"""A class whose instances are cardinality constraints in the sense that
    their toplevel operator represents a constant relation symbol
    :math:`\bar{C}_n` where :math:`n \in \mathbb{N} \cup \{\infty\}`. A typical
    interpretation in a domain :math:`D` is that :math:`\bar{C}_n` holds iff
    :math:`|D| < n`.

    The class constructor takes one argument, which is the index `n`. It takes
    care that instance with equal indices are identical.

    >>> c_0_1 = C_(0)
    >>> c_0_2 = C_(0)
    >>> c_oo = C_(oo)
    >>> c_0_1 is c_0_2
    True
    >>> c_0_1 == c_oo
    False
    """

    # Class variables
    func: type[C_]  #: :meta private:
    """A type annotation for the class property `func` inherited from
    :attr:`.firstorder.AtomicFormula.func`.
    """

    @classproperty
    def complement_func(cls):
        """A class property yielding the complement class :class:`C` of
        :class:'C_'.
        """
        return C

    _instances: ClassVar[dict] = {}
    """A private class variable, which is a dictionary holding unique instances
    of `C_(n)` with key `n`.
    """

    # Instance variables
    args: tuple[int]
    """A type annotation for the property `func` inherited from
    :attr:`.firstorder.AtomicFormula.func`.

    :meta private:
    """

    # Class methods
    def __new__(cls, *args):
        if len(args) != 1:
            raise ValueError(f"bad number of arguments")
        n = args[0]
        if not isinstance(n, (int, sympy.core.numbers.Infinity)) or n < 0:
            raise ValueError(f"{n!r} is not an admissible cardinality")
        if n not in cls._instances:
            cls._instances[n] = super().__new__(cls)
        return cls._instances[n]

    # Instance methods
    def __repr__(self) -> str:
        return f'C_({self.index})'

    def _sprint(self, mode: str) -> str:
        if mode == 'text':
            return repr(self)
        assert mode == 'latex', f'bad print mode {mode!r}'
        k = str(self.index) if isinstance(self.index, int) else '\\infty'
        return f'\\overline{{C_{{{k}}}}}'
