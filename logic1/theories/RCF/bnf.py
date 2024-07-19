from ... import abc
from .atomic import AtomicFormula, Term, Variable
from .simplify import simplify as _simplify
from .typing import RCF_Formula


class DisjunctiveNormalForm(abc.bnf.DisjunctiveNormalForm[AtomicFormula, Term, Variable]):

    def simplify(self, f: RCF_Formula) -> RCF_Formula:
        return _simplify(f)


dnf = DisjunctiveNormalForm()
cnf = DisjunctiveNormalForm(dualize=True)
