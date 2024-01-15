from __future__ import annotations

from abc import ABCMeta, abstractmethod
import sympy
from typing import Final, TypeAlias

from .. import firstorder
from ..support.containers import GetVars
from ..support.renaming import rename
from . import generic

Index: TypeAlias = sympy.Integer | sympy.core.numbers.Infinity
Term: TypeAlias = sympy.Expr
Variable: TypeAlias = sympy.Symbol

oo: Final = sympy.oo


class TermMixin:

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

    def __le__(self, other: firstorder.Formula) -> bool:
        match other:
            case AtomicFormula():
                if len(self.args) != len(other.args):
                    return len(self.args) < len(other.args)
                for i in range(len(self.args)):
                    if self.args[i] != other.args[i]:
                        return self.args[i].sort_key() <= other.args[i].sort_key()
                L = self.relations()
                return L.index(self.func) <= L.index(other.func)
            case _:
                return True

    def get_vars(self, assume_quantified: set = set()) -> GetVars:
        """Implements the abstract method :meth:`.firstorder.Formula.get_vars`.
        """
        all_vars = set()
        for term in self.args:
            all_vars |= term.atoms(sympy.Symbol)
        return GetVars(free=all_vars - assume_quantified,
                       bound=all_vars & assume_quantified)

    @abstractmethod
    def relations(self) -> list[ABCMeta]:
        ...

    def subs(self, substitution: dict) -> AtomicFormula:
        """Implements the abstract method :meth:`.firstorder.Formula.subs`.
        """
        args = (arg.subs(substitution, simultaneous=True) for arg in self.args)
        return self.func(*args)


class BinaryAtomicFormula(generic.BinaryAtomicFormulaMixin, AtomicFormula):
    """A class whose instances are binary formulas in the sense that both
    their m-arity and their p-arity is 2.
    """

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


class IndexedConstantAtomicFormula(AtomicFormula):
    r"""A class whose instances form a family of atomic formulas with m-arity
    0. Their p-arity is 1, where the one argument of the constructor is the
    index.
    """
    def __new__(cls, *args):
        if len(args) != 1:
            raise ValueError(f"bad number of arguments")
        n = args[0]
        if not isinstance(n, (int, sympy.Integer, sympy.core.numbers.Infinity)) or n < 0:
            raise ValueError(f"{n!r} is not an admissible cardinality")
        if n not in cls._instances:
            cls._instances[n] = super().__new__(cls)
        return cls._instances[n]

    def __init__(self, index: object, chk: bool = True):
        if chk:
            match index:
                case sympy.Integer() | sympy.core.numbers.Infinity():
                    super().__init__(index)
                case int():
                    super().__init__(sympy.Integer(index))
                case _:
                    raise ValueError(f'{index} invalid as index')
        else:
            super().__init__(index)

    @property
    def index(self) -> sympy.Integer:
        """The index of the :class:`IndexedConstantAtomicFormula`.
        """
        return self.args[0]

    def get_vars(self, assume_quantified: set = set()) -> GetVars:
        """Implements the abstract method :meth:`.firstorder.Formula.get_vars`.
        """
        return GetVars()
