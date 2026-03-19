
from logic1.theories.Complex.atomic import AtomicFormula, AtomicFormulaVisitor, Eq, Ge, Gt, Le, Lt, Ne
from logic1.theories.Complex.term import Add, Conj, IdentityTermVisitor, _I, I, Im, Mul, Neg, Number, Pow, Rational, Re, Term, TermVisitor, Variable, VV
from gmpy2 import mpq, mpz


class BaseFormatter(TermVisitor[str], AtomicFormulaVisitor[str]):

    symbols: dict[type[Term] | type[AtomicFormula], str] = {}

    def group(self, s: str) -> str:
        return s
    
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
        if isinstance(neg.arg, (Add, Mul, Neg, Conj)):  # discuss -- for repr?
            return f'{symbol}({neg.arg.accept(self)})'
        return f'{symbol}{neg.arg.accept(self)}'

    def visit_conj(self, conj: Conj) -> str:
        symbol = self.symbols.get(Conj, '~')
        if isinstance(conj.arg, (Add, Mul, Neg, Conj)):  # discuss -- for repr?
            return f'{symbol}({conj.arg.accept(self)})'
        return f'{symbol}{conj.arg.accept(self)}'

    def visit_re(self, re: Re) -> str:
        symbol = self.symbols.get(Re, 'Re')
        return f'{symbol}({re.arg.accept(self)})'

    def visit_im(self, im: Im) -> str:
        symbol = self.symbols.get(Im, 'Im')
        return f'{symbol}({im.arg.accept(self)})'

    def visit_eq(self, eq: Eq) -> str:
        symbol = self.symbols.get(Eq, '==')
        return f'{eq.lhs.accept(self)} {symbol} {eq.rhs.accept(self)}'

    def visit_ne(self, ne: Ne) -> str:
        symbol = self.symbols.get(Ne, '!=')
        return f'{ne.lhs.accept(self)} {symbol} {ne.rhs.accept(self)}'

    def visit_le(self, le: Le) -> str:
        symbol = self.symbols.get(Le, '<=')
        return f'{le.lhs.accept(self)} {symbol} {le.rhs.accept(self)}'

    def visit_lt(self, lt: Lt) -> str:
        symbol = self.symbols.get(Lt, '<')
        return f'{lt.lhs.accept(self)} {symbol} {lt.rhs.accept(self)}'

    def visit_ge(self, ge: Ge) -> str:
        symbol = self.symbols.get(Ge, '>=')
        return f'{ge.lhs.accept(self)} {symbol} {ge.rhs.accept(self)}'

    def visit_gt(self, gt: Gt) -> str:
        symbol = self.symbols.get(Gt, '>')
        return f'{gt.lhs.accept(self)} {symbol} {gt.rhs.accept(self)}'


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
        Re: '\\operatorname{Re}',
        Im: '\\operatorname{Im}',
        Eq: '=',
        Ne: '\\neq',
        Le: '\\leq',
        Lt: '<',
        Ge: '\\geq',
        Gt: '>'
    }

    def group(self, s: str) -> str:  # TODO: fix i^1000
        return f'{{{s}}}'

    def visit_rational(self, num: Rational) -> str:
        a = num.value.numerator
        b = num.value.denominator
        if a == mpz(0) or b == mpz(1):
            return str(a)
        else:
            return f'\\frac{{{str(a)}}}{{{str(b)}}}'
        
    def visit_variable(self, var: Variable) -> str:
        if "_" in var.name:
            base, *indices = var.name.split("_")
            grouped = "_".join(f'{{{idx}}}' for idx in indices)
            return f'\\mathit{{{base}}}_{grouped}'
        else:
            base = var.name.rstrip('0123456789')
            index = var.name[len(base):]
            if index:
                return f'\\mathit{{{base}}}_{{{str(index)}}}'
            else:
                return f'\\mathit{{{base}}}'

    def visit_conj(self, conj: Conj) -> str:
        return f'\\overline{{{conj.arg.accept(self)}}}'
        

    

    