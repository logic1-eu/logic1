from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar, final, Union

import sympy

from ..firstorder import atomic
from ..support.containers import GetVars
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

    func: type[AtomicFormula]
    args: tuple

    @classmethod
    def interactive_new(cls, *args):
        args_ = []
        for arg in args:
            arg_ = (sympy.Integer(arg) if isinstance(arg, int) else arg)
            if not isinstance(arg_, cls.term_type()):
                raise TypeError(f"{arg} is not a Term")
            args_.append(arg_)
        return cls(*args_)

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

    func: type[BinaryAtomicFormula]
    args: tuple[Term, Term]

    @property
    def lhs(self) -> Term:
        """The left-hand side of the BinaryAtomicFormula."""
        return self.args[0]

    @property
    def rhs(self) -> Term:
        """The right-hand side of the BinaryAtomicFormula."""
        return self.args[1]

    # Relations
    @staticmethod
    @abstractmethod
    def rel_converse(conditional: bool = True) -> type[BinaryAtomicFormula]:
        """Returns the converse R ** (-1) of a relation R derived from
        BinaryAtomicFormula if conditional is True, else R.

        If R is defined on S x T, then R ** (-1) = { (x, y) in T x S | (y, x)
        in R }. For instance,

        >>> Le.rel_converse()
        <class 'logic1.atomlib.sympy.Ge'>
        >>> Eq.rel_converse()
        <class 'logic1.atomlib.sympy.Eq'>
        >>> Le.rel_complement(9 ** 2 < 80)
        <class 'logic1.atomlib.sympy.Le'>

        Compare the notions of the complement relation R' of R, and
        of the dual relation (R') ** (-1), which equals (R ** (-1))'.
        """
        ...

    @staticmethod
    @abstractmethod
    def rel_dual(conditional: bool = True) -> type[BinaryAtomicFormula]:
        """Returns the dual (R') ** (-1) of a relation R derived from
        BinaryAtomicFormula if conditional is True, else R.

        For instance,

        >>> Le.rel_dual()
        <class 'logic1.atomlib.sympy.Lt'>
        >>> Eq.rel_dual()
        <class 'logic1.atomlib.sympy.Ne'>
        >>> Le.rel_dual(9 ** 2 < 80)
        <class 'logic1.atomlib.sympy.Le'>

        Note that (R') ** (-1)  equals (R ** (-1))'. Compare the notions of the
        converse relation R ** (-1) and of the complement relation R' of R.
        """
        ...

    # Instance methods
    def __init__(self, lhs: Term, rhs: Term) -> None:
        self.func = self.__class__
        self.args = (lhs, rhs)

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

    @final
    def to_converse(self, conditional: bool = True) -> BinaryAtomicFormula:
        # Do not pass on but check conditional in order to avoid construction
        # in case of False.
        if conditional:
            return self.func.rel_converse()(*self.args)
        return self

    @final
    def to_dual(self, conditional: bool = True) -> BinaryAtomicFormula:
        # Do not pass on but check conditional in order to avoid construction
        # in case of False.
        if conditional:
            return self.func.rel_dual()(*self.args)
        return self


class Eq(BinaryAtomicFormula):
    """
    >>> from sympy import exp, I, pi
    >>> Eq(exp(I * pi, evaluate=False), -1)
    Eq(exp(I*pi), -1)
    """
    latex_symbol: ClassVar[str] = '='
    sympy_func: ClassVar[type[sympy.Eq]] = sympy.Eq
    text_symbol: ClassVar[str] = '='

    func: type[Eq]

    @staticmethod
    def rel_complement(conditional: bool = True) -> type[BinaryAtomicFormula]:
        if conditional:
            return Ne
        return Eq

    @staticmethod
    def rel_converse(conditional: bool = True) -> type[BinaryAtomicFormula]:
        return Eq

    @staticmethod
    def rel_dual(conditional: bool = True) -> type[BinaryAtomicFormula]:
        if conditional:
            return Ne
        return Eq


EQ = Eq.interactive_new


class Ne(BinaryAtomicFormula):
    """
    >>> Ne(1, 0)
    Ne(1, 0)
    """
    latex_symbol: ClassVar[str] = '\\neq'
    sympy_func: ClassVar[type[sympy.Ne]] = sympy.Ne
    text_symbol: ClassVar[str] = '!='

    func: type[Ne]

    @staticmethod
    def rel_complement(conditional: bool = True) -> type[BinaryAtomicFormula]:
        if conditional:
            return Eq
        return Ne

    @staticmethod
    def rel_converse(conditional: bool = True) -> type[BinaryAtomicFormula]:
        return Ne

    @staticmethod
    def rel_dual(conditional: bool = True) -> type[BinaryAtomicFormula]:
        if conditional:
            return Eq
        return Ne


NE = Ne.interactive_new


class Ge(BinaryAtomicFormula):

    latex_symbol: ClassVar[str] = '\\geq'
    sympy_func: ClassVar[type[sympy.Ge]] = sympy.Ge
    text_symbol: ClassVar[str] = '>='

    func: type[Ge]

    @staticmethod
    def rel_complement(conditional: bool = True) -> type[BinaryAtomicFormula]:
        if conditional:
            return Lt
        return Ge

    @staticmethod
    def rel_converse(conditional: bool = True) -> type[BinaryAtomicFormula]:
        if conditional:
            return Le
        return Ge

    @staticmethod
    def rel_dual(conditional: bool = True) -> type[BinaryAtomicFormula]:
        if conditional:
            return Gt
        return Ge


GE = Ge.interactive_new


class Le(BinaryAtomicFormula):

    latex_symbol: ClassVar[str] = '\\leq'
    sympy_func: ClassVar[type[sympy.Le]] = sympy.Le
    text_symbol: ClassVar[str] = '<='

    func: type[Le]

    @staticmethod
    def rel_complement(conditional: bool = True) -> type[BinaryAtomicFormula]:
        if conditional:
            return Gt
        return Le

    @staticmethod
    def rel_converse(conditional: bool = True) -> type[BinaryAtomicFormula]:
        if conditional:
            return Ge
        return Le

    @staticmethod
    def rel_dual(conditional: bool = True) -> type[BinaryAtomicFormula]:
        if conditional:
            return Lt
        return Le


LE = Le.interactive_new


class Gt(BinaryAtomicFormula):

    latex_symbol: ClassVar[str] = '>'
    sympy_func: ClassVar[type[sympy.Gt]] = sympy.Gt
    text_symbol: ClassVar[str] = '>'

    func: type[Gt]

    @staticmethod
    def rel_complement(conditional: bool = True) -> type[BinaryAtomicFormula]:
        if conditional:
            return Le
        return Gt

    @staticmethod
    def rel_converse(conditional: bool = True) -> type[BinaryAtomicFormula]:
        if conditional:
            return Lt
        return Gt

    @staticmethod
    def rel_dual(conditional: bool = True) -> type[BinaryAtomicFormula]:
        if conditional:
            return Ge
        return Gt


GT = Gt.interactive_new


class Lt(BinaryAtomicFormula):

    latex_symbol: ClassVar[str] = '<'
    sympy_func: ClassVar[type[sympy.Lt]] = sympy.Lt
    text_symbol: ClassVar[str] = '<'

    func: type[Lt]

    @staticmethod
    def rel_complement(conditional: bool = True) -> type[BinaryAtomicFormula]:
        if conditional:
            return Ge
        return Lt

    @staticmethod
    def rel_converse(conditional: bool = True) -> type[BinaryAtomicFormula]:
        if conditional:
            return Gt
        return Lt

    @staticmethod
    def rel_dual(conditional: bool = True) -> type[BinaryAtomicFormula]:
        if conditional:
            return Le
        return Lt


LT = Lt.interactive_new


class Cardinality(AtomicFormula):

    @classmethod
    def interactive_new(cls, *args):
        if len(args) != 1:
            raise TypeError(f"bad number of arguments")
        n = args[0]
        if not isinstance(n, (int, sympy.core.numbers.Infinity)):
            raise TypeError(f"{n!r} is not a Cardinality")
        return cls(n)


class _C(Cardinality):
    """
    >>> c_0_1 = C(0)
    >>> c_0_2 = C(0)
    >>> c_oo = C(oo)
    >>> c_0_1 is c_0_2
    True
    >>> c_0_1 == c_oo
    False
    """
    _instances: ClassVar[dict] = {}

    def __new__(cls, n: Card):
        if n not in cls._instances:
            cls._instances[n] = super().__new__(cls)
        return cls._instances[n]

    @staticmethod
    def rel_complement(conditional: bool = True) -> type[AtomicFormula]:
        if conditional:
            return _C_bar
        return _C

    def __init__(self, n: Card) -> None:
        self.func = _C
        self.args = (n,)
        self.n = n

    def __repr__(self):
        return f'C({self.n})'

    def _sprint(self, mode: str) -> str:
        if mode == 'text':
            return repr(self)
        assert mode == 'latex', f'bad print mode {mode!r}'
        k = str(self.n) if isinstance(self.n, int) else '\\infty'
        return f'C_{k}'


C = _C


class _C_bar(Cardinality):
    """
    >>> c_0_1 = C_bar(0)
    >>> c_0_2 = C_bar(0)
    >>> c_oo = C_bar(oo)
    >>> c_0_1 is c_0_2
    True
    >>> c_0_1 == c_oo
    False
    """
    _instances: ClassVar[dict] = {}

    def __new__(cls, n: Card):
        if n not in cls._instances:
            cls._instances[n] = super().__new__(cls)
        return cls._instances[n]

    @staticmethod
    def rel_complement(conditional: bool = True) -> type[AtomicFormula]:
        if conditional:
            return _C
        return _C_bar

    def __init__(self, n: Card) -> None:
        self.func = _C_bar
        self.args = (n,)
        self.n = n

    def __repr__(self) -> str:
        return f'C_bar({self.n})'

    def _sprint(self, mode: str) -> str:
        if mode == 'text':
            return repr(self)
        assert mode == 'latex', f'bad print mode {mode!r}'
        k = str(self.n) if isinstance(self.n, int) else '\\infty'
        return f'\\overline{{C_{k}}}'


C_bar = _C_bar
