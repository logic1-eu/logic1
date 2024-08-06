from ... import abc
from .atomic import AtomicFormula, Variable
from .simplify import simplify as _simplify
from .typing import Formula, Never


class DisjunctiveNormalForm(
        abc.bnf.DisjunctiveNormalForm[AtomicFormula, Variable, Variable, Never]):

    def simplify(self, f: Formula) -> Formula:
        return _simplify(f)


dnf = DisjunctiveNormalForm()
cnf = DisjunctiveNormalForm(dualize=True)
