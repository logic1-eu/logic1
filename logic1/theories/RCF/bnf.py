from ...firstorder import Formula
from ... import abc
from .pnf import pnf as _pnf
from .simplify import simplify as _simplify


class DisjunctiveNormalForm(abc.bnf.DisjunctiveNormalForm):

    def pnf(self, f: Formula) -> Formula:
        return _pnf(f)

    def simplify(self, f: Formula) -> Formula:
        return _simplify(f)


dnf = DisjunctiveNormalForm()
