from abc import ABC, abstractmethod
from pyeda.boolalg import expr, minimization  # type: ignore

from ..firstorder import (
    All, And, AtomicFormula, BooleanFormula, Equivalent, Ex, _F, F, Formula,
    Implies, Not, Or, _T, T)

from ..support.tracing import trace  # noqa


class DisjunctiveNormalForm(ABC):

    def __call__(self, f: Formula) -> Formula:
        if self.dualize:
            return self.simplify(Not(self.dnf(Not(f))).to_nnf())
        return self.dnf(f)

    def __init__(self, dualize: bool = False):
        self.logic1_to_pyeda = {Equivalent: expr.Equal, Implies: expr.Implies,
                                And: expr.And, Or: expr.Or, Not: expr.Not,
                                _T: expr._Zero, _F: expr._One}
        self.dualize = dualize
        self.index = 0
        self.atoms_to_pyeda: dict[AtomicFormula, expr.Literal] = {}
        self.pyeda_to_atoms: dict[expr.Literal, AtomicFormula] = {}

    def dnf(self, f: Formula) -> Formula:
        f = self.pnf(f)
        f = self.simplify(f)
        quantifiers = []
        mtx = f
        while isinstance(mtx, (Ex, All)):
            Q = mtx.op
            v = mtx.var
            quantifiers.append((Q, v))
            mtx = mtx.arg
        match mtx:
            case And() | Or():
                dnf: Formula = self.dnf_and_or(mtx)
            case AtomicFormula() | _F() | _T():
                return f
            case Not(arg=arg):
                assert isinstance(arg, AtomicFormula)
                return f
            case _:
                assert False
        for Q, v in reversed(quantifiers):
            dnf = Q(v, dnf)
        dnf = self.simplify(dnf)
        return dnf

    def dnf_and_or(self, f: And | Or) -> AtomicFormula | BooleanFormula:
        f_as_pyeda = self.to_pyeda(f)
        dnf_as_pyeda = f_as_pyeda.to_dnf()
        if not isinstance(dnf_as_pyeda, expr.Constant):
            dnf_as_pyeda, = minimization.espresso_exprs(dnf_as_pyeda)
        dnf = self.from_pyeda(dnf_as_pyeda)
        return dnf

    def to_pyeda(self, f) -> expr:
        match f:
            case AtomicFormula():
                if f in self.atoms_to_pyeda:
                    return self.atoms_to_pyeda[f]
                cf = f.to_complement()
                if cf in self.atoms_to_pyeda:
                    return expr.Not(self.atoms_to_pyeda[cf])
                new_exprvar = expr.exprvar('a', self.index)
                self.index += 1
                self.atoms_to_pyeda[f] = new_exprvar
                self.pyeda_to_atoms[new_exprvar] = f
                return new_exprvar
            case And(args=args) | Or(args=args):
                name = self.logic1_to_pyeda[f.op]
                xs = (self.to_pyeda(arg) for arg in args)
                return name(*xs, simplify=False)
            case _:
                assert False

    def from_pyeda(self, f: expr) -> AtomicFormula | BooleanFormula:
        xs: expr
        match f:
            case expr.Variable():
                return self.pyeda_to_atoms[f]
            case expr.Complement():
                # Complement of a variable is different from logical Not, and
                # it is not covered by our dictionary
                return self.pyeda_to_atoms[~ f].to_complement()
            case expr.AndOp(xs=xs):
                args = (self.from_pyeda(x) for x in xs)
                return And(*args)
            case expr.OrOp(xs=xs):
                args = (self.from_pyeda(x) for x in xs)
                return Or(*args)
            case expr._Zero():
                return F
            case expr._One():
                return T
            case _:
                assert False

    @abstractmethod
    def pnf(self, f: Formula) -> Formula:
        ...

    @abstractmethod
    def simplify(self, f: Formula) -> Formula:
        ...
