"""This module provides a library of atomic formulas based on SymPy terms.
"""

from __future__ import annotations

from typing import ClassVar, Union

import sympy

from ..firstorder import atomic
from ..firstorder import T, F
from ..support.containers import GetVars
from ..support.decorators import classproperty
from ..support.renaming import rename

# Type alias
Card = Union[int, sympy.core.numbers.Infinity]

Term = sympy.Expr
Variable = sympy.Symbol

oo = sympy.oo


class TermMixin():

    @staticmethod
    def term_type() -> type[Term]:
        return Term

    @staticmethod
    def term_get_vars(term: Term) -> set[Variable]:
        return sympy.S(term).atoms(Variable)

    @staticmethod
    def term_to_latex(term: Term) -> str:
        return sympy.latex(term)

    @staticmethod
    def term_to_sympy(term: Term) -> sympy.Basic:
        return term

    @staticmethod
    def variable_type() -> type[Variable]:
        return Variable

    @staticmethod
    def rename_var(variable: Variable) -> Variable:
        return rename(variable)


class AtomicFormula(TermMixin, atomic.AtomicFormula):
    """Atomic Formula with Sympy Terms. All terms are sympy.Expr.
    """

    # Instance methods
    def get_vars(self, assume_quantified: set = set()) -> GetVars:
        all_vars = set()
        for term in self.args:
            all_vars |= term.atoms(sympy.Symbol)
        return GetVars(free=all_vars - assume_quantified,
                       bound=all_vars & assume_quantified)

    def subs(self, substitution: dict) -> AtomicFormula:
        args = (arg.subs(substitution, simultaneous=True) for arg in self.args)
        return self.func(*args)


class BinaryAtomicFormula(AtomicFormula):
    """A base class for atomic formulas obtained from binary relations.

    The binary relation R is available as both an instance property and a class
    property of the respective derived classes:

    >>> Eq(0, 0).func
    <class 'logic1.atomlib.sympy.Eq'>
    >>> Eq.func
    <class 'logic1.atomlib.sympy.Eq'>

    Assume that R is defined on a Cartesian product P. Then the complement
    relation of R is R' = P - R.  It is avaialable as both an instance and a
    class property `complement_func`, e.g.:

    >>> rels = (Eq, Ne, Ge, Le, Gt, Lt)
    >>> tuple(r.complement_func for r in rels)
    (<class 'logic1.atomlib.sympy.Ne'>,
     <class 'logic1.atomlib.sympy.Eq'>,
     <class 'logic1.atomlib.sympy.Lt'>,
     <class 'logic1.atomlib.sympy.Gt'>,
     <class 'logic1.atomlib.sympy.Le'>,
     <class 'logic1.atomlib.sympy.Ge'>)

    Since R'(s, t) is equivalent to Not(R(s, t)), the availability of
    complement relations is relevant for the computation of positive negation
    normal forms.

    If R is defined on S x T, then the converse relation R ** (-1) of R is
    defined as R ** (-1) = { (x, y) in T x S | (y, x) in R }. It is avaialable
    as both an instance and a class property `converse_func`, e.g.:

    >>> tuple(r.converse_func for r in rels)
    (<class 'logic1.atomlib.sympy.Eq'>,
     <class 'logic1.atomlib.sympy.Ne'>,
     <class 'logic1.atomlib.sympy.Le'>,
     <class 'logic1.atomlib.sympy.Ge'>,
     <class 'logic1.atomlib.sympy.Lt'>,
     <class 'logic1.atomlib.sympy.Gt'>)

    The converse relation is the inverse with respect to composition.

    Finally, the dual relation (R') ** (-1) of R is availalble as both an
    instance and a class property `dual_func`. The dual (R') ** (-1) generally
    equals (R ** (-1))', e.g.:

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

    In the context of orderings, dualization turns ``strict`` inequalities into
    ``weak`` inequalities, and vice versa. Note that we also have duality and
    corresponding properties with Boolean functions, which is defined
    differently.

    All those operators are involutive:

    >>> all(r.complement_func.complement_func == r for r in rels)
    True
    >>> all(r.converse_func.converse_func == r for r in rels)
    True
    >>> all(r.dual_func.dual_func == r for r in rels)
    True
    """

    # Class variables
    latex_symbol: ClassVar[str]
    text_symbol: ClassVar[str]

    @classproperty
    def dual_func(cls):
        """The dual relation.
        """
        return cls.complement_func.converse_func

    # Instance variables
    args: tuple[Term, Term]

    @property
    def lhs(self) -> Term:
        """The left-hand side of the BinaryAtomicFormula."""
        return self.args[0]

    @property
    def rhs(self) -> Term:
        """The right-hand side of the BinaryAtomicFormula."""
        return self.args[1]

    # Instance methods
    def __init__(self, *args):
        if len(args) != 2:
            raise ValueError(f'bad number of arguments for binary relation')
        args_ = []
        for arg in args:
            arg_ = (sympy.Integer(arg) if isinstance(arg, int) else arg)
            if not isinstance(arg_, self.term_type()):
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
    """Represent equations as atomic formulas.

    >>> from sympy import exp, I, pi
    >>> from sympy.abc import t
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
    sympy_func = sympy.Eq
    text_symbol = '=='

    func: type[Eq]

    @classproperty
    def complement_func(cls):
        """The complement relation Ne of Eq.
        """
        return Ne

    @classproperty
    def converse_func(cls):
        """The converse relation Eq of Eq.
        """
        return Eq

    # Instance methods
    def simplify(self, Theta=None):
        if self.lhs == self.rhs:
            return T
        return self


class Ne(BinaryAtomicFormula):
    """
    >>> Ne(1, 0)
    Ne(1, 0)
    """

    # Class variables
    latex_symbol = '\\neq'
    sympy_func = sympy.Ne
    text_symbol = '!='

    func: type[Ne]

    @classproperty
    def complement_func(cls):
        """The complement relation Eq of Ne.
        """
        return Eq

    @classproperty
    def converse_func(cls):
        return Ne

    # Instance methods
    def simplify(self, Theta=None):
        if self.lhs == self.rhs:
            return F
        return self


class Ge(BinaryAtomicFormula):

    # Class variables
    latex_symbol = '\\geq'
    sympy_func = sympy.Ge
    text_symbol = '>='

    func: type[Ge]

    @classproperty
    def complement_func(cls):
        """The complement relation Lt of Ge.
        """
        return Lt

    @classproperty
    def converse_func(cls):
        return Le


class Le(BinaryAtomicFormula):

    # Class variables
    latex_symbol = '\\leq'
    sympy_func = sympy.Le
    text_symbol = '<='

    func: type[Le]

    @classproperty
    def complement_func(cls):
        """The complement relation Gt of Le.
        """
        return Gt

    @classproperty
    def converse_func(cls):
        return Ge


class Gt(BinaryAtomicFormula):
    """A class holding binary atomic formulas with the relation `>`.
    """
    # Class variables
    latex_symbol = '>'
    sympy_func = sympy.Gt
    text_symbol = '>'

    func: type[Gt]

    @classproperty
    def complement_func(cls):
        """The complement relation Le of Gt.
        """
        return Le

    @classproperty
    def converse_func(cls):
        return Lt


class Lt(BinaryAtomicFormula):

    # Class variables
    latex_symbol = '<'
    sympy_func = sympy.Lt
    text_symbol = '<'

    func: type[Lt]

    @classproperty
    def complement_func(cls):
        """The complement relation Ge of Lt.
        """
        return Ge

    @classproperty
    def converse_func(cls):
        return Gt


class Cardinality(AtomicFormula):

    # Instance variables
    @property
    def index(self):
        return self.args[0]

    # Instance methods
    def get_vars(self, assume_quantified: set = set()) -> GetVars:
        return GetVars()


class C(Cardinality):
    """
    >>> c_0_1 = C(0)
    >>> c_0_2 = C(0)
    >>> c_oo = C(oo)
    >>> c_0_1 is c_0_2
    True
    >>> c_0_1 == c_oo
    False
    """

    # Class variables
    func: type[C]

    @classproperty
    def complement_func(cls):
        """The complement relation _C_ of _C.
        """
        return C_

    _instances: ClassVar[dict] = {}

    # Instance variables
    args: tuple[int]

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
        return f'C_{k}'


class C_(Cardinality):
    """
    >>> c_0_1 = C_(0)
    >>> c_0_2 = C_(0)
    >>> c_oo = C_(oo)
    >>> c_0_1 is c_0_2
    True
    >>> c_0_1 == c_oo
    False
    """

    # Class variables
    func: type[C_]

    @classproperty
    def complement_func(cls):
        """The complement relation C of C_.
        """
        return C

    _instances: ClassVar[dict] = {}

    # Instance variables
    args: tuple[int]

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
        return f'\\overline{{C_{k}}}'
