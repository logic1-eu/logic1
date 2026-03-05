
from logic1.theories.Complex.atomic import _I, Add, Im, Mul, Neg, Pow, Rational, Re, Term, TermVisitor, Variable
from gmpy2 import mpq


class BaseFormatter(TermVisitor[str]):

    symbols: dict[type[Term], str] = {}
    
    def visit_rational(self, num: Rational) -> str:
        return str(num.value)
    
    def visit_i(self, _: _I) -> str:
        return self.symbols.get(_I, 'i')
    
    def visit_variable(self, var: Variable) -> str:
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
                elif isinstance(arg, Rational) and arg.value < mpq(0):
                    result.append(symbol_minus)
                    arg = Rational(-arg.value)
                else:
                    result.append(symbol_plus)
            if i > 0 and isinstance(arg, Neg):
                result.append(f'({arg.accept(self)})')
            else:
                result.append(arg.accept(self))
        return " ".join(result)

    def visit_mul(self, mul: Mul) -> str:
        symbol = self.symbols.get(Mul, '*')
        result = []
        for i, arg in enumerate(mul.args):
            if isinstance(arg, Add) or (i > 0 and isinstance(arg, Neg)): 
                result.append(f'({arg.accept(self)})')
            else:
                result.append(arg.accept(self))
        return f" {symbol} ".join(result)

    def visit_pow(self, pow: Pow) -> str:
        symbol = self.symbols.get(Pow, '^')
        if isinstance(pow.base, (Add, Mul, Neg, Pow)):
            return f'({pow.base.accept(self)}){symbol}{pow.exponent}'
        return f'{pow.base.accept(self)}{symbol}{pow.exponent}'

    def visit_neg(self, neg: Neg) -> str:
        symbol = self.symbols.get(Neg, '-')
        if isinstance(neg.arg, (Add, Mul, Neg)):
            return f'{symbol}({neg.arg.accept(self)})'
        return f'{symbol}{neg.arg.accept(self)}'

    def visit_re(self, re: Re) -> str:
        symbol = self.symbols.get(Re, 'Re')
        return f'{symbol}({re.arg.accept(self)})'

    def visit_im(self, im: Im) -> str:
        symbol = self.symbols.get(Im, 'Im')
        return f'{symbol}({im.arg.accept(self)})'


class ReprFormatter(BaseFormatter):
    
    symbols: dict[type[Term], str] = {
        _I: 'I',
        Pow: '**'
    }


class StrFormatter(BaseFormatter):
    pass


class LatexFormatter(BaseFormatter):
    pass