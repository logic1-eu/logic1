from ...firstorder import Formula
from ... import abc
from .simplify import simplify as _simplify


class DisjunctiveNormalForm(abc.bnf.DisjunctiveNormalForm):

    def pnf(self, f: Formula) -> Formula:
        return f.to_pnf()

    def simplify(self, f: Formula) -> Formula:
        return _simplify(f)


dnf = DisjunctiveNormalForm()
cnf = DisjunctiveNormalForm(dualize=True)
