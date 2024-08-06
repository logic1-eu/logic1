from ... import abc
from .atomic import AtomicFormula, Term, Variable
from .simplify import simplify as _simplify
from .typing import Formula


class DisjunctiveNormalForm(abc.bnf.DisjunctiveNormalForm[AtomicFormula, Term, Variable, int]):

    def simplify(self, f: Formula) -> Formula:
        return _simplify(f)


dnf = DisjunctiveNormalForm()
cnf = DisjunctiveNormalForm(dualize=True)
