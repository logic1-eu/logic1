"""Boolean normal forms are computed by applying the normal form algorithms
provided by `PyEDA <https://pyeda.readthedocs.io/>`_ to Boolean abstractions of
the formulas. The results are then converted back to formulas.
"""
import itertools
from typing import Literal, NewType, TypeGuard

from logic1.firstorder import And, Or, Not, _F, _T, BooleanFormula
from logic1 import abc
from logic1.theories.RCF.term import Term, Variable
from logic1.theories.RCF.atomic import AtomicFormula
from logic1.theories.RCF.simplify import simplify as _simplify
from logic1.theories.RCF.types import Formula, Number


AndOr = NewType("AndOr", Formula)


def is_and_or(f: Formula) -> TypeGuard[AndOr]:
    if isinstance(f, (_F, _T, AtomicFormula)):
        return True
    if isinstance(f, (And, Or)):
        return all(is_and_or(arg) for arg in f.args)
    return False


class CartesianBooleanNormalForm:

    def cnf(self, f: Formula) -> Formula:
        """Compute a conjunctive normal form. If ``f`` contains quantifiers,
        then the result is a prenex normal form whose matrix is in CNF.
        """
        return self.final_simplify(Not(self._dnf(Not(f))).to_nnf())

    def dnf(self, f: Formula) -> Formula:
        """Compute a disjunctive normal form. If ``f`` contains quantifiers,
        then the result is a prenex normal form whose matrix is in DNF.
        """
        return self.final_simplify(self._dnf(f))

    def _dnf(self, f: Formula) -> Formula:
        f = self.simplify(f.to_pnf())
        matrix, prefix = f.matrix()
        assert is_and_or(matrix)
        dnf = self._dnf_and_or(matrix)
        return dnf.quantify(prefix)

    def _dnf_and_or(self, f: AndOr) -> Formula:
        sdnf = self._strict_dnf(f)
        dnf = Or(*(And(*(atom for atom in conj)) for conj in sdnf))
        return dnf

    def final_simplify(self, f: Formula) -> Formula:
        return _simplify(f, explode_always=False)

    def simplify(self, f: Formula) -> Formula:
        return _simplify(f)

    def _strict_dnf(self, f: AndOr) -> set[frozenset[AtomicFormula]]:
        if isinstance(f, _F):
            return set()
        if isinstance(f, _T):
            return {frozenset()}
        if isinstance(f, AtomicFormula):
            return {frozenset((f,))}
        if isinstance(f, And):
            strict_dnfs = (self._strict_dnf(arg) for arg in f.args)
            result = set()
            for tuples in itertools.product(*strict_dnfs):
                result.add(frozenset.union(*tuples))
            return result
        if isinstance(f, Or):
            result = set()
            for arg in f.args:
                result.update(self._strict_dnf(arg))
            return result
        raise ValueError(f"Unexpected formula type: {type(f)}")


class PyedaBooleanNormalForm(abc.bnf.BooleanNormalForm[AtomicFormula, Term, Variable, Number]):
    """Implements the abstract methods :meth:`simplify
    <.abc.bnf.BooleanNormalForm.simplify>` and :meth:`final_simplify
    <.abc.bnf.BooleanNormalForm.final_simplify>` of its super class
    :class:`.abc.bnf.BooleanNormalForm`. Inherits the methods :meth:`cnf
    <.abc.bnf.BooleanNormalForm.cnf>` and :meth:`dnf
    <.abc.bnf.BooleanNormalForm.dnf>`, which should be called via the functions
    :func:`.cnf` and :func:`.dnf`, respectively.
    """

    def simplify(self, f: Formula) -> Formula:
        """Implements the abstract method :meth:`.abc.bnf.BooleanNormalForm.simplify`.
        """
        return _simplify(f)

    def final_simplify(self, f: Formula) -> Formula:
        """Implements the abstract method :meth:`.abc.bnf.BooleanNormalForm.final_simplify`.
        """
        return _simplify(f, explode_always=False)


_cartesian_bnf = CartesianBooleanNormalForm()

_pyeda_bnf = PyedaBooleanNormalForm()
"""One instance of the PyedaBooleanNormalForm class used for both CNF and DNF
computations.
"""


def cnf(f: Formula, method: Literal["cartesian", "pyeda"] = "pyeda") -> Formula:
    """User interface for the computation of a conjunctive normal form.
    """
    if method == "cartesian":
        return _cartesian_bnf.cnf(f)
    if method == "pyeda":
        return _pyeda_bnf.cnf(f)
    else:
        raise ValueError(f"Unknown method: {method}")


def dnf(f: Formula, method: Literal["cartesian", "pyeda"] = "pyeda") -> Formula:
    """User interface for the computation of a disjunctive normal form.
    """
    if method == "cartesian":
        return _cartesian_bnf.dnf(f)
    if method == "pyeda":
        return _pyeda_bnf.dnf(f)
    else:
        raise ValueError(f"Unknown method: {method}")
