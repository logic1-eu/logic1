from ... import abc
from .atomic import AtomicFormula, Term, Variable
from .simplify import simplify as _simplify
from .typing import Formula


class BooleanNormalForm(abc.bnf.BooleanNormalForm[AtomicFormula, Term, Variable, int]):
    """Implements the abstract method :meth:`simplify
    <.abc.bnf.BooleanNormalForm.simplify>` of its super class
    :class:`.abc.bnf.BooleanNormalForm`. In addition, this class inherits
    :meth:`cnf <.abc.bnf.BooleanNormalForm.cnf>` and :meth:`dnf
    <.abc.bnf.BooleanNormalForm.dnf>`, which should be called via
    :func:`.cnf` and :func:`.dnf` as described below, respectively.
    """

    def simplify(self, f: Formula) -> Formula:
        """Implements the abstract method
        :meth:`.abc.bnf.BooleanNormalForm.simplify`.
        """
        return _simplify(f)


cnf = BooleanNormalForm().cnf
"""User interface for the computation of a conjunctive normal form.
"""

dnf = BooleanNormalForm().dnf
"""User interface for the computation of a disjunctive normal form.
"""
