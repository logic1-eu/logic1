from ... import abc

from ...firstorder import Formula
from .rcf import ring, Variable

from ...support.tracing import trace  # noqa


class PrenexNormalForm(abc.pnf.PrenexNormalForm):

    def __call__(self, f: Formula, prefer_universal: bool = False, is_nnf: bool = False):
        return self.pnf(f, prefer_universal=prefer_universal, is_nnf=is_nnf)

    def rename(self, v: Variable) -> Variable:
        i = 0
        vars_as_str = tuple(str(v) for v in ring.get_vars())
        v_as_str = str(v)
        while v_as_str in vars_as_str:
            i += 1
            v_as_str = str(v) + "_R" + str(i)
        v = ring.add_var(v_as_str)
        return v


pnf = PrenexNormalForm()
