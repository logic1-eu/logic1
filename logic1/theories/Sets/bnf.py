from ... import abc
from .atomic import AtomicFormula, Variable
from .simplify import simplify as _simplify
from .typing import Sets_Formula


class DisjunctiveNormalForm(abc.bnf.DisjunctiveNormalForm[AtomicFormula, Variable, Variable]):

    def simplify(self, f: Sets_Formula) -> Sets_Formula:
        return _simplify(f)


dnf = DisjunctiveNormalForm()
cnf = DisjunctiveNormalForm(dualize=True)
