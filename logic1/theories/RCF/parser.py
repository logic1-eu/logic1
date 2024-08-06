import ast
from typing import Any

from ... import abc
from ...firstorder import And, Formula
from .atomic import Eq, Ne, Le, Lt, Gt, Ge, Term, AtomicFormula, Variable, VV


class L1Parser(abc.parser.L1Parser[AtomicFormula, Term, Variable, int]):

    def __call__(self, s: str) -> Formula:
        self.globals: dict[str, Variable] = dict()
        f = self.process(s)
        return f

    def _declare_variable(self, v: str):
        self.globals[v] = VV[v]

    def process_atom(self, a: Any) -> Formula:
        try:
            return self._process_atom(a)
        except (TypeError, NameError) as exc:
            raise abc.parser.ParserError(f'{exc.__str__()}')

    def _process_atom(self, a: Any) -> Formula:
        match a:
            case ast.Compare(ops=ops, left=left, comparators=comparators):
                eval_left = self.process_term(left)
                L: list[AtomicFormula] = []
                assert len(ops) == len(comparators)
                for op, right in zip(ops, comparators):
                    eval_right = self.process_term(right)
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

    def process_term(self, t: Any) -> Term:
        for node in ast.walk(t):
            if isinstance(node, ast.Name):
                self._declare_variable(node.id)
        try:
            return eval(ast.unparse(t), self.globals)
        except NameError as inst:
            raise abc.parser.ParserError(f'{inst.__str__()} in {ast.unparse(t)}') from None

    def process_var(self, v: Any) -> Variable:
        # v is the first argument of a quantifier.
        match v:
            case ast.Name():
                self._declare_variable(v.id)
                return eval(ast.unparse(v), self.globals)
            case _:
                raise TypeError(f'{ast.unparse(v)} invalid as quantifed variable')


l1 = L1Parser()
