from abc import abstractmethod
import ast
from typing import Generic

from ..firstorder import All, And, Equivalent, Ex, _F, Implies, Not, Or, _T
from ..firstorder.formula import α, τ, χ, σ, Formula
from ..support.excepthook import NoTraceException

from ..support.tracing import trace  # noqa


class ParserError(Exception):
    pass


class L1Parser(Generic[α, τ, χ, σ]):

    def process(self, s: str) -> Formula[α, τ, χ, σ]:
        try:
            a = ast.parse(s, mode='eval')
            # print(ast.dump(a, indent=4))
            assert isinstance(a, ast.Expression)
            return self._process(a.body)
        except (NameError, ParserError, SyntaxError, TypeError) as exc:
            raise NoTraceException(*exc.args)

    def _process(self, a: ast.expr) -> Formula[α, τ, χ, σ]:
        match a:
            case ast.Call(func=func, args=args, keywords=keywords):
                if keywords:
                    raise ParserError("expression cannot contain assignment. "
                                      "Maybe you meant '==' instead of '='?")
                assert isinstance(func, ast.Name)
                match func.id:
                    case 'Ex' | 'ex':
                        var = self.process_var(args[0])
                        return Ex(var, *(self._process(arg) for arg in args[1:]))
                    case 'All' | 'all':
                        var = self.process_var(args[0])
                        return All(var, *(self._process(arg) for arg in args[1:]))
                    case 'Or':
                        return Or(*(self._process(arg) for arg in args))
                    case 'And':
                        return And(*(self._process(arg) for arg in args))
                    case 'Implies':
                        return Implies(*(self._process(arg) for arg in args))
                    case 'Equivalent':
                        return Equivalent(*(self._process(arg) for arg in args))
                    case _:
                        raise ParserError(f'cannot parse {ast.unparse(a)}')
            case ast.BoolOp(op=op, values=args):
                match op:
                    case ast.Or():
                        return Or(*(self._process(arg) for arg in args))
                    case ast.And():
                        return And(*(self._process(arg) for arg in args))
                    case _:
                        raise ParserError(f'unknown operator {ast.dump(op)} in {ast.unparse(a)}')
            case ast.BinOp(op=op, left=left, right=right):
                match op:
                    case ast.BitOr():
                        return Or(self._process(left), self._process(right))
                    case ast.BitAnd():
                        return And(self._process(left), self._process(right))
                    case ast.RShift():
                        return Implies(self._process(left), self._process(right))
                    case _:
                        raise ParserError(f'unknown operator {ast.dump(op)} in {ast.unparse(a)}')
            case ast.UnaryOp(op=op, operand=operand):
                match op:
                    case ast.Invert() | ast.Not():
                        return Not(self._process(operand))
                    case _:
                        raise ParserError(f'unknown operator {ast.dump(op)} in {ast.unparse(a)}')
            case ast.Name(id=id):
                match id:
                    case 'T' | 'true':
                        return _T()
                    case 'F' | 'false':
                        return _F()
                    case _:
                        raise ParserError(f'cannot parse {ast.unparse(a)}')
            case _:
                return self.process_atom(a)

    @abstractmethod
    def process_atom(self, a: ast.expr) -> Formula[α, τ, χ, σ]:
        ...

    @abstractmethod
    def process_var(self, v: ast.expr) -> χ:
        ...
