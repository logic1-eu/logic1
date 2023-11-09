from ...firstorder import Formula
from ... import abc
from .pnf import pnf as _pnf


class DisjunctiveNormalForm(abc.bnf.DisjunctiveNormalForm):

    def pnf(self, f: Formula) -> Formula:
        return _pnf(f)

    def simplify(self, f: Formula) -> Formula:
        return f.simplify()


dnf = DisjunctiveNormalForm()
