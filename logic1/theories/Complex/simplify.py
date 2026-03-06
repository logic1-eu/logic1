

from matplotlib.pylab import add

from logic1.theories.Complex.atomic import _I, Add, Im, Mul, Neg, Pow, Rational, Re, Term, TermVisitor, Variable
from gmpy2 import mpq


class Evaluator(TermVisitor[tuple[mpq, mpq]]):
    """Visitor that evaluates a term if it is constant, returning the 
    real and imaginary part as a pair of rational numbers. Raises ValueError if the term is not constant.
    """
    
    def visit_rational(self, num: Rational) -> tuple[mpq, mpq]:
        return num.value, mpq(0)

    def visit_i(self, i: _I) -> tuple[mpq, mpq]:
        return mpq(0), mpq(1)

    def visit_variable(self, var: Variable) -> tuple[mpq, mpq]:
        raise ValueError(f'Cannot evaluate variable')

    def visit_add(self, add: Add) -> tuple[mpq, mpq]:
        a, b = mpq(0), mpq(0)
        for arg in add.args:
            x, y = arg.accept(self)
            a, b = a + x, b + y
        return a, b

    def visit_mul(self, mul: Mul) -> tuple[mpq, mpq]:
        a, b = mpq(1), mpq(0)
        for arg in mul.args:
            x, y = arg.accept(self)
            a, b = a * x - b * y, a * y + b * x
        return a, b

    def visit_pow(self, pow: Pow) -> tuple[mpq, mpq]:
        if pow.exponent == 0:
            return mpq(1), mpq(0)
        elif pow.exponent % 2 == 0:    
            a, b = Pow(pow.base, pow.exponent // 2).accept(self)
            return a * a - b * b, mpq(2) * a * b
        else:
            a, b = pow.base.accept(self)
            x, y = Pow(pow.base, pow.exponent - 1).accept(self)
            return a * x - b * y, a * y + b * x

    def visit_neg(self, neg: Neg) -> tuple[mpq, mpq]:
        a, b = neg.arg.accept(self)
        return -a, -b
    
    def visit_re(self, re: Re) -> tuple[mpq, mpq]:
        a, _ = re.arg.accept(self)
        return a, mpq(0)

    def visit_im(self, im: Im) -> tuple[mpq, mpq]:
        _, b = im.arg.accept(self)
        return b, mpq(0)


class Normalizer(TermVisitor[Term]):
    """Visitor that normalizes a term.
    """
    
    def visit_rational(self, num: Rational) -> Term:
        return num
    
    def visit_i(self, i: _I) -> Term:
        return i
    
    def visit_variable(self, var: Variable) -> Term:
        return var
    
    def visit_add(self, add: Add) -> Term:
        args = [arg.accept(self) for arg in add.args]
        # collect all products and the absolute constants
        constant: Term = Rational(mpq(0))
        products: dict[Term, Term] = {}
        while args:
            arg = args.pop()
            if isinstance(arg, Add):
                args.extend(arg.args)
                continue
            if arg.is_constant():
                constant = Add(constant, arg)
                continue
            coeff: Term = Rational(mpq(1))
            if isinstance(arg, Mul) and arg.args[0].is_constant():
                coeff = arg.args[0]
                arg = Mul(*arg.args[1:])
            exisiting_coeff = products.get(arg, Rational(mpq(0)))
            products[arg] = Add(exisiting_coeff, coeff)
        # put all products and constants back together and simplify if possible
        result = []
        for prod in sorted(products, key=Term.sort_key):
            coeff = products[prod]
            a, b = coeff.eval_constant()
            if a == mpq(0) and b == mpq(0):
                continue
            elif a == mpq(1) and b == mpq(0):
                result.append(prod)
            else:
                result.append(Mul(Term.from_real_imag(a, b), prod))
        a, b = constant.eval_constant()
        if not (a == mpq(0) and b == mpq(0)):
            result.append(Term.from_real_imag(a, b))
        return Add(*result)
    
    def visit_mul(self, mul: Mul) -> Term:
        args = [arg.accept(self) for arg in mul.args]
        # resolve sums and negs as args
        for i, arg in enumerate(args):
            if isinstance(arg, Add):
                arg_args = [Mul(*args[:i], arg_arg, *args[i + 1:]) for arg_arg in arg.args]
                return Add(*arg_args).accept(self)
            if isinstance(arg, Neg):
                return Mul(*args[:i], Rational(mpq(-1)), arg.arg, *args[i + 1:]).accept(self)
        # collect constant/non-constant factors
        consts = []
        factors: dict[Term, int] = {}
        while args:
            arg = args.pop()
            if isinstance(arg, Mul):
                args.extend(arg.args)
                continue
            if arg.is_constant():
                consts.append(arg)
                continue
            exp = 1
            if isinstance(arg, Pow):
                exp = arg.exponent
                arg = arg.base
            assert isinstance(arg, (Variable, Re, Im)), (type(arg), args, mul)
            factors[arg] = factors.get(arg, 0) + exp
        # regroup factors
        result = []
        for factor in sorted(factors, key=Term.sort_key):
            exp = factors[factor]
            if exp == 0:
                continue
            elif exp == 1:
                result.append(factor)
            else:
                result.append(Pow(factor, exp))
        # evaluate constant and build final product
        a, b = Mul(*consts).eval_constant()
        if a == mpq(0) and b == mpq(0):
            return Rational(mpq(0))
        if a == mpq(1) and b == mpq(0):
            return Mul(*result)
        return Mul(Term.from_real_imag(a, b), *result)

    def visit_pow(self, pow: Pow) -> Term:
        if pow.exponent == 0:
            return Rational(mpq(1))  # note: 0^0 = 1 as for mpq
        base = pow.base.accept(self)
        if base.is_constant():
            a, b = Pow(base, pow.exponent).eval_constant()
            return Term.from_real_imag(a, b)
        elif isinstance(base, (Variable, Re, Im)):
            return Pow(base, pow.exponent)
        else:
            return Mul(*[base] * pow.exponent).accept(self)

    def visit_neg(self, neg: Neg) -> Term:
        arg = neg.arg.accept(self)
        if isinstance(arg, Rational):
            return Rational(-arg.value)
        elif isinstance(arg, _I):
            return Neg(arg)
        elif isinstance(arg, Variable):
            return Neg(arg)
        elif isinstance(arg, Add):
            return Add(*map(Neg, arg.args)).accept(self)
        elif isinstance(arg, Mul):
            return Mul(Neg(arg.args[0]), *arg.args[1:]).accept(self)
        elif isinstance(arg, Pow):
            return Neg(arg)
        elif isinstance(arg, Neg):
            return arg.arg
        elif isinstance(arg, Re):
            return Neg(arg)
        elif isinstance(arg, Im):
            return Neg(arg)
        else:
            assert False, type(arg)

    def visit_re(self, re: Re) -> Term:
        arg = re.arg.accept(self)
        if isinstance(arg, Pow):
            arg = Mul(*[arg.base] * arg.exponent) # TODO optimize?
        if isinstance(arg, Rational):
            return arg
        elif isinstance(arg, _I):
            return Rational(mpq(0))
        elif isinstance(arg, Variable):
            return Re(arg)
        elif isinstance(arg, Add):
            return Add(*map(Re, arg.args)).accept(self)
        elif isinstance(arg, Mul):
            x, xs = arg.args[0], Mul(*arg.args[1:])
            return (Re(x) * Re(xs) - Im(x) * Im(xs)).accept(self)
        elif isinstance(arg, Pow):
            assert False
        elif isinstance(arg, Neg):
            return Neg(Re(arg.arg)).accept(self)
        elif isinstance(arg, Re):
            return arg
        elif isinstance(arg, Im):
            return arg
        else:
            assert False, type(arg) 

    def visit_im(self, im: Im) -> Term:
        arg = im.arg.accept(self)
        if isinstance(arg, Pow):
            arg = Mul(*[arg.base] * arg.exponent) # TODO optimize?
        if isinstance(arg, Rational):
            return Rational(mpq(0))
        elif isinstance(arg, _I):
            return Rational(mpq(1))
        elif isinstance(arg, Variable):
            return Im(arg)
        elif isinstance(arg, Add):
            return Add(*map(Im, arg.args)).accept(self)
        elif isinstance(arg, Mul):
            x, xs = arg.args[0], Mul(*arg.args[1:])
            return (Re(x) * Im(xs) + Im(x) * Re(xs)).accept(self)
        elif isinstance(arg, Pow):
            assert False # return Im(Mul(*([arg.base] * arg.exponent))).accept(self)
        elif isinstance(arg, Neg):
            return Neg(Im(arg.arg)).accept(self)
        elif isinstance(arg, Re):
            return Rational(mpq(0))
        elif isinstance(arg, Im):
            return Rational(mpq(0))
        else:
            assert False, type(arg)
