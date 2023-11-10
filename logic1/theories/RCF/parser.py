import ast
from typing import Any

from ... import abc
from ...firstorder import *
from .rcf import *

from ...support.tracing import trace  # noqa


class L1Parser(abc.parser.L1Parser):

    def __call__(self, s: str):
        self.globals = {str(v): v for v in ring.get_vars()}
        a = ast.parse(s, mode='eval')
        # print(ast.dump(a, indent=4))
        assert isinstance(a, ast.Expression)
        return self.process(a.body)

    def _process_atom(self, a: Any):
        try:
            return self._process_atom1(a)
        except (TypeError, NameError) as exc:
            raise abc.parser.ParserError(f'{exc.__str__()}')

    def _process_atom1(self, a: Any):
        match a:
            case ast.Compare(ops=ops, left=left, comparators=comparators):
                eval_left = self._process_term(left)
                L: list[RcfAtomicFormula] = []
                assert len(ops) == len(comparators)
                for op, right in zip(ops, comparators):
                    eval_right = self._process_term(right)
                    match op:
                        case ast.Eq():
                            L.append(Eq(eval_left, eval_right))
                        case ast.GtE():
                            L.append(Ge(eval_left, eval_right))
                        case ast.LtE():
                            L.append(Le(eval_left, eval_right))
                        case ast.Gt():
                            L.append(Gt(eval_left, eval_right))
                        case ast.Lt():
                            L.append(Lt(eval_left, eval_right))
                        case ast.NotEq():
                            L.append(Ne(eval_left, eval_right))
                        case _:
                            raise TypeError(f'unknown operator {ast.dump(op)} '
                                            f'in {ast.unparse(a)}')
                    eval_left = eval_right
                return And(*L)
            case _:
                raise TypeError(f'cannot parse {ast.unparse(a)}')

    def _process_term(self, t: Any):
        try:
            return eval(ast.unparse(t), self.globals)
        except NameError as inst:
            raise NameError(f'{inst.__str__()} in {ast.unparse(t)}')

    def _process_var(self, v: Any):
        # v is the first argument of a quantifier.
        match v:
            case ast.Name():
                return ring(v.id)
            case _:
                raise TypeError(f'{ast.unparse(v)} invalid as quantifed variable')


L1 = L1Parser()
