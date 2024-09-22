"""This module :mod:`logic1.abc.bnf` provides a generic abstract implementation
of boolean normal form computations using the famous Espresso algorithm in
combination with boolean abstraction. Techincally, we use the python package
`PyEDA <https://pyeda.readthedocs.io/en/latest/index.html>`_, which in turns
wraps a `C extension
<https://ptolemy.berkeley.edu/projects/embedded/pubs/downloads/espresso/_index.htm>`_
of the famous Berkeley Espresso library [BraytonEtAl-1984]_.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from pyeda.boolalg import expr, minimization  # type: ignore
from typing import ClassVar, Generic, TypeVar

from .. import firstorder
from ..firstorder import (
    And, AtomicFormula, BooleanFormula, _F, Formula, Not, Or, _T, Term, Variable)

from ..support.tracing import trace  # noqa

α = TypeVar('α', bound='AtomicFormula')
τ = TypeVar('τ', bound='Term')
χ = TypeVar('χ', bound='Variable')
σ = TypeVar('σ')


@dataclass
class BooleanNormalForm(Generic[α, τ, χ, σ]):
    """Boolean normal form computation.
    """

    _logic1_to_pyeda: ClassVar[dict[type[Formula], expr]] = {
        firstorder.Equivalent: expr.Equal,
        firstorder.Implies: expr.Implies,
        firstorder.And: expr.And,
        firstorder.Or: expr.Or,
        firstorder.Not: expr.Not,
        firstorder._T: expr._Zero,
        firstorder._F: expr._One}

    _index: int = 0
    _atoms_to_pyeda: dict[AtomicFormula, expr.Literal] = field(default_factory=dict)
    _pyeda_to_atoms: dict[expr.Literal, AtomicFormula] = field(default_factory=dict)

    def cnf(self, f: Formula[α, τ, χ, σ]) -> Formula[α, τ, χ, σ]:
        """Compute a conjunctive normal form. If `f` contains quantifiers, then
        the result is a prenex normal form whose matrix is in CNF.
        """
        return self.simplify(Not(self._dnf(Not(f))).to_nnf())

    def dnf(self, f: Formula[α, τ, χ, σ]) -> Formula[α, τ, χ, σ]:
        """Compute a disjunctive normal form. If `f` contains quantifiers, then
        the result is a prenex normal form whose matrix is in DNF.
        """
        return self.simplify(self._dnf(f))

    def _dnf(self, f: Formula[α, τ, χ, σ]) -> Formula[α, τ, χ, σ]:
        f = self.simplify(f.to_pnf())
        mat, prefix = f.matrix()
        match mat:
            case And() | Or():
                dnf: Formula = self._dnf_and_or(mat)
            case AtomicFormula() | _F() | _T():
                return f
            case Not(arg=arg):
                assert isinstance(arg, AtomicFormula)
                return f
            case _:
                assert False
        return dnf.quantify(prefix)

    def _dnf_and_or(self, f: And[α, τ, χ, σ] | Or[α, τ, χ, σ]) \
            -> AtomicFormula[α, τ, χ, σ] | BooleanFormula[α, τ, χ, σ]:
        f_as_pyeda = self._to_pyeda(f)
        dnf_as_pyeda = f_as_pyeda.to_dnf()
        if not isinstance(dnf_as_pyeda, expr.Constant):
            dnf_as_pyeda, = minimization.espresso_exprs(dnf_as_pyeda)
        dnf = self._from_pyeda(dnf_as_pyeda)
        return dnf

    def _to_pyeda(self, f: AtomicFormula[α, τ, χ, σ] | And[α, τ, χ, σ] | Or[α, τ, χ, σ]) -> expr:
        match f:
            case AtomicFormula():
                if f in self._atoms_to_pyeda:
                    return self._atoms_to_pyeda[f]
                cf = f.to_complement()
                if cf in self._atoms_to_pyeda:
                    return expr.Not(self._atoms_to_pyeda[cf])
                new_exprvar = expr.exprvar('a', self._index)
                self._index += 1
                self._atoms_to_pyeda[f] = new_exprvar
                self._pyeda_to_atoms[new_exprvar] = f
                return new_exprvar
            case And(args=args) | Or(args=args):
                name = self._logic1_to_pyeda[f.op]
                xs = (self._to_pyeda(arg) for arg in args)
                return name(*xs, simplify=False)
            case _:
                assert False

    def _from_pyeda(self, f: expr) -> AtomicFormula[α, τ, χ, σ] | BooleanFormula[α, τ, χ, σ]:
        xs: expr
        match f:
            case expr.Variable():
                return self._pyeda_to_atoms[f]
            case expr.Complement():
                # Complement of a variable is different from logical Not, and
                # it is not covered by our dictionary
                return self._pyeda_to_atoms[~ f].to_complement()
            case expr.AndOp(xs=xs):
                args = (self._from_pyeda(x) for x in xs)
                return And(*args)
            case expr.OrOp(xs=xs):
                args = (self._from_pyeda(x) for x in xs)
                return Or(*args)
            case expr._Zero():
                return _F()
            case expr._One():
                return _T()
            case _:
                assert False

    @abstractmethod
    def simplify(self, f: Formula[α, τ, χ, σ]) -> Formula[α, τ, χ, σ]:
        """Compute a simplified equivalent of `f`.
        """
        ...
