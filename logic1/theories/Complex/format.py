

from typing import ClassVar

from gmpy2 import mpz

from logic1.theories.Complex.ast import _I, AST, ASTVisitor, Add, Conj, Im, Mul, Neg, Pow, Re, Var, Rat


class BaseFormatter(ASTVisitor[str]):

    symbols: ClassVar[dict[type[AST], str]] = {}

    def _omit_mul_symbol(self, ast1: AST, ast2: AST) -> bool:
        return False

    def visit_rat(self, num: Rat) -> str:
        return str(num.value)

    def visit_i(self, _: _I) -> str:
        return self.symbols.get(_I, 'i')

    def visit_var(self, var: Var) -> str:
        return var.name

    def visit_add(self, add: Add) -> str:
        symbol_plus = self.symbols.get(Add, '+')
        symbol_minus = self.symbols.get(Neg, '-')
        result = []
        for i, arg in enumerate(add.args):
            if i > 0:
                if isinstance(arg, Neg):
                    result.append(symbol_minus)
                    arg = arg.arg
                else:
                    result.append(symbol_plus)
            if i > 0 and isinstance(arg, Neg):
                result.append(f'({arg.accept(self)})')
            else:
                result.append(arg.accept(self))
        return " ".join(result)

    def visit_mul(self, mul: Mul) -> str:
        symbol = self.symbols.get(Mul, '*')
        factors = []
        result = []
        for i, arg in enumerate(mul.args):
            factors.append(arg)
            if len(factors) > 1 and not self._omit_mul_symbol(factors[-2], factors[-1]):
                result.append(symbol)
            if isinstance(arg, Add) or (i > 0 and isinstance(arg, Neg)):
                result.append(f'({arg.accept(self)})')
            else:
                result.append(arg.accept(self))
        return f" ".join(result)

    def visit_pow(self, pow: Pow) -> str:
        symbol = self.symbols.get(Pow, '^')
        if isinstance(pow.base, (Add, Mul, Neg, Pow, Conj)):
            return f'({pow.base.accept(self)}){symbol}{pow.exponent}'
        return f'{pow.base.accept(self)}{symbol}{pow.exponent}'

    def visit_neg(self, neg: Neg) -> str:
        symbol = self.symbols.get(Neg, '-')
        if isinstance(neg.arg, (Add, Mul)):
            return f'{symbol}({neg.arg.accept(self)})'
        return f'{symbol}{neg.arg.accept(self)}'

    def visit_conj(self, conj: Conj) -> str:
        symbol = self.symbols.get(Conj, '~')
        if isinstance(conj.arg, (Add, Mul)):
            return f'{symbol}({conj.arg.accept(self)})'
        return f'{symbol}{conj.arg.accept(self)}'

    def visit_re(self, re: Re) -> str:
        symbol = self.symbols.get(Re, 'Re')
        return f'{symbol}({re.arg.accept(self)})'

    def visit_im(self, im: Im) -> str:
        symbol = self.symbols.get(Im, 'Im')
        return f'{symbol}({im.arg.accept(self)})'


class ReprFormatter(BaseFormatter):

    symbols = {
        _I: 'I',
        Pow: '**'
    }


class StrFormatter(BaseFormatter):
    pass


class LatexFormatter(BaseFormatter):

    symbols = {
        Mul: '\\cdot',
        Re: '\\Re',
        Im: '\\Im',
    }

    def _omit_mul_symbol(self, ast1: AST, ast2: AST) -> bool:
        while isinstance(ast1, (Conj, Pow)):
            if isinstance(ast1, Conj):
                ast1 = ast1.arg
            else:
                ast1 = ast1.base
        while isinstance(ast2, (Conj, Pow)):
            if isinstance(ast2, Conj):
                ast2 = ast2.arg
            else:
                ast2 = ast2.base
        if isinstance(ast1, (Var, Im, Re)) and isinstance(ast2, (Var, Im, Re)):
            return True
        if isinstance(ast1, Rat) and isinstance(ast2, _I):
            return True
        return False

    def visit_rat(self, num: Rat) -> str:
        a = num.value.numerator
        b = num.value.denominator
        if a == mpz(0) or b == mpz(1):
            return str(a)
        else:
            return f'\\frac{{{str(a)}}}{{{str(b)}}}'

    def visit_var(self, var: Var) -> str:
        def format_name(name: str) -> str:
            return name if len(name) == 1 else f'\\mathrm{{{name}}}'
        if "_" in var.name:
            base, *indices = var.name.split("_")
            grouped = "_".join(f'{{{idx}}}' for idx in indices)
            return f'{format_name(base)}_{grouped}'
        else:
            base = var.name.rstrip('0123456789')
            index = var.name[len(base):]
            if index:
                return f'{format_name(base)}_{{{str(index)}}}'
            else:
                return format_name(base)

    def visit_conj(self, conj: Conj) -> str:
        return f'\\overline{{{conj.arg.accept(self)}}}'

    def visit_pow(self, pow: Pow) -> str:
        if isinstance(pow.base, (Add, Mul, Neg, Pow)):
            return f'({pow.base.accept(self)})^{{{pow.exponent}}}'
        return f'{pow.base.accept(self)}^{{{pow.exponent}}}'
