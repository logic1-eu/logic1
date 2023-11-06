from ... import abc

from ...firstorder import Formula
from ...support import renaming
from .zmodm import Variable

from ...support.tracing import trace  # noqa


class PrenexNormalForm(abc.pnf.PrenexNormalForm):

    def __call__(self, f: Formula, prefer_universal: bool = False, is_nnf: bool = False):
        return self.pnf(f, prefer_universal=prefer_universal, is_nnf=is_nnf)

    def rename(self, v: Variable) -> Variable:
        return renaming.rename(v)


pnf = PrenexNormalForm()
