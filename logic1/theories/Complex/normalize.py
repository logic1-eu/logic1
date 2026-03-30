"""This module defines visitors for evaluating and normalizing terms in
the theory of complex numbers.
"""

from abc import abstractmethod
from typing import TypeVar

from gmpy2 import mpq

from logic1.theories.Complex.ast import _I, AST, I, Add, ASTVisitor, Conj, IdentityASTVisitor, Im, Mul, Neg, Pow, Rat, Re, Var

α = TypeVar('α')
"""Type variable for the result type of `ASTVisitor` methods."""

class ArithmeticEvaluator(ASTVisitor[α]):
    """Abstract visitor that evaluates a AST to an element of α, 
    given implementations of addition, negation and multiplication.
    """

    @abstractmethod
    def _add(self, a: α, b: α) -> α:
        """Adds two elements. Must be implemented by subclasses.
        """
        ...

    @abstractmethod
    def _neg(self, a: α) -> α:
        """Negates an element. Must be implemented by subclasses.
        """
        ...

    @abstractmethod
    def _mul(self, a: α, b: α) -> α:
        """Multiplies two elements. Must be implemented by subclasses.
        """
        ...

    def visit_add(self, add: Add) -> α:
        """Evaluates a sum by using the `_add` method.
        """
        result = Rat(0).accept(self)
        for arg in add.args:
            result = self._add(result, arg.accept(self))
        return result

    def visit_mul(self, mul: Mul) -> α:
        """Evaluates a product by using the `_mul` method.
        """
        result = Rat(1).accept(self)
        for arg in mul.args:
            result = self._mul(result, arg.accept(self))
        return result

    def visit_neg(self, neg: Neg) -> α:
        """Evaluates a negation by using the `_neg` method.
        """
        return self._neg(neg.arg.accept(self))
    
    def visit_pow(self, pow: Pow) -> α:
        """Evaluates a power.
        """
        if pow.exponent == 0:
            return Rat(1).accept(self)
        elif pow.exponent % 2 == 0:    
            a = Pow(pow.base, pow.exponent // 2).accept(self)
            return self._mul(a, a)
        else:
            a = pow.base.accept(self)
            b = Pow(pow.base, pow.exponent - 1).accept(self)
            return self._mul(a, b)


class ConstantEvaluator(ArithmeticEvaluator[tuple[mpq, mpq]]):
    """Visitor that evaluates a AST to a constant. The result is a pair 
    of rational numbers representing the real and imaginary parts of the 
    complex number. Raises a ValueError if the AST contains variables.
    """

    def _add(self, a: tuple[mpq, mpq], b: tuple[mpq, mpq]) -> tuple[mpq, mpq]:
        """Adds two complex numbers represented as pairs of rational
        numbers (real, imag). Implements the abstract method 
        :meth:`.ArithmeticEvaluator._add`.

        >>> ConstantEvaluator()._add((mpq(1), mpq(2)), (mpq(3), mpq(4)))
        (mpq(4,1), mpq(6,1))
        """
        return a[0] + b[0], a[1] + b[1]
    
    def _neg(self, a: tuple[mpq, mpq]) -> tuple[mpq, mpq]:
        """Negates a complex number represented as a pair of rational
        numbers (real, imag). Implements the abstract method 
        :meth:`.ArithmeticEvaluator._neg`.

        >>> ConstantEvaluator()._neg((mpq(1), mpq(2)))
        (mpq(-1,1), mpq(-2,1))
        """
        return -a[0], -a[1]

    def _mul(self, a: tuple[mpq, mpq], b: tuple[mpq, mpq]) -> tuple[mpq, mpq]:
        """Multiplies two complex numbers represented as pairs of
        rational numbers (real, imag). Implements the abstract method 
        :meth:`.ArithmeticEvaluator._mul`.

        >>> ConstantEvaluator()._mul((mpq(1), mpq(2)), (mpq(3), mpq(4)))
        (mpq(-5,1), mpq(10,1))
        """
        return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]

    def visit_rat(self, num: Rat) -> tuple[mpq, mpq]:
        """Evaluates a rational number to a complex number
        represented as a pair of rational numbers (real, imag).
        Implements the abstract method :meth:`.TermVisitor.visit_rat`.

        >>> ConstantEvaluator().visit_rat(Rat(mpq(1, 2)))
        (mpq(1,2), mpq(0,1))
        """
        return num.value, mpq(0)

    def visit_i(self, i: _I) -> tuple[mpq, mpq]:
        """Evaluates the imaginary unit term to a complex number
        represented as a pair of rational numbers (real, imag).
        Implements the abstract method :meth:`.TermVisitor.visit_i`.
        
        >>> ConstantEvaluator().visit_i(I)
        (mpq(0,1), mpq(1,1))
        """
        return mpq(0), mpq(1)

    def visit_var(self, var: Var) -> tuple[mpq, mpq]:
        """Raises a ValueError since variables cannot be evaluated to constants.
        Implements the abstract method :meth:`.TermVisitor.visit_var`.

        >>> from logic1.theories.Complex.ast import *
        >>> x = Var('x')
        >>> ConstantEvaluator().visit_var(x)
        Traceback (most recent call last):
          ...
        ValueError: Cannot evaluate variable x
        """
        raise ValueError(f'Cannot evaluate variable {var}')

    def visit_conj(self, conj: Conj) -> tuple[mpq, mpq]:
        """Evaluates a conjugation to a complex number represented
        as a pair of rational numbers (real, imag).
        Implements the abstract method :meth:`.TermVisitor.visit_conj`.

        >>> ConstantEvaluator().visit_conj(Conj(1 + 2 * I))
        (mpq(1,1), mpq(-2,1))
        """
        a, b = conj.arg.accept(self)
        return a, -b

    def visit_re(self, re: Re) -> tuple[mpq, mpq]:
        """Evaluates a real part to a complex number represented as
        a pair of rational numbers (real, imag).
        Implements the abstract method :meth:`.TermVisitor.visit_re`.

        >>> ConstantEvaluator().visit_re(Re(1 + 2 * I))
        (mpq(1,1), mpq(0,1))
        """
        a, _ = re.arg.accept(self)
        return a, mpq(0)

    def visit_im(self, im: Im) -> tuple[mpq, mpq]:
        """Evaluates an imaginary part to a complex number represented as
        a pair of rational numbers (real, imag).
        Implements the abstract method :meth:`.TermVisitor.visit_im`.
        
        >>> ConstantEvaluator().visit_im(Im(1 + 2 * I))
        (mpq(2,1), mpq(0,1))
        """
        _, b = im.arg.accept(self)
        return b, mpq(0)


class WeakNormalizer(IdentityASTVisitor):
    """Visitor that normalizes a AST by rearranging sums and products, 
    and applying obvious simplifications, but not expanding any nodes.
    """
    
    def visit_add(self, add: Add) -> AST:
        """Normalizes a sum by collecting constant terms and
        rearranging non-constant terms in a canonical order.
        
        >>> from logic1.theories.Complex.ast import *
        >>> x, y, z = Var('x'), Var('y'), Var('z')
        >>> (y + x + z + x - z).accept(WeakNormalizer())
        2 * x + y
        >>> (2 + x - 3).accept(WeakNormalizer())
        x - 1
        """
        args = [arg.accept(self) for arg in add.args]
        # collect all products with their coefficients and the absolute constants
        constant: AST = Rat(0)
        products: dict[AST, AST] = {}
        while args:
            arg = args.pop()
            if isinstance(arg, Add):
                args.extend(arg.args)
                continue
            if arg.is_constant():
                constant = Add(constant, arg)
                continue
            # extract the coefficient of the product
            if isinstance(arg, Neg):
                coeff: AST = Rat(-1)
                arg = arg.arg
            else:
                coeff = Rat(1)
            while isinstance(arg, Mul) and arg.args[0].is_constant():
                coeff = coeff * arg.args[0]
                arg = Mul(*arg.args[1:])
            if isinstance(arg, Mul) and isinstance(arg.args[0], Neg):
                coeff = coeff * Rat(-1)
                arg = Mul(arg.args[0].arg, *arg.args[1:])
            products[arg] = products.get(arg, 0) + coeff 
        # put all products with their coefficients back together and simplify if possible
        result = []
        for i, prod in enumerate(sorted(products, key=AST.sort_key)):
            assert not isinstance(prod, Mul) or not prod.args[0].is_constant(), (prod, add)
            a, b = products[prod].eval()
            if a == mpq(0) and b == mpq(0):
                continue
            elif a == mpq(1) and b == mpq(0): 
                result.append(prod)
            elif a == mpq(-1) and b == mpq(0):
                if i == 0 and isinstance(prod, Mul):
                    result.append(Mul(Neg(prod.args[0]), *prod.args[1:]))
                else:
                    result.append(Neg(prod))
            elif i > 0 and (a < mpq(0) or (a == mpq(0) and b < mpq(0))):
                coeff = AST.from_real_imag(-a, -b)
                result.append(Neg(coeff * prod))
            else:
                coeff = AST.from_real_imag(a, b)
                result.append(coeff * prod)
        # add the abolute constant if it is not zero
        a, b = constant.eval()
        if a == mpq(0) and b == mpq(0):
            pass
        elif len(result) > 0 and (a < mpq(0) or (a == mpq(0) and b < mpq(0))):
            result.append(Neg(AST.from_real_imag(-a, -b)))
        else:
            result.append(AST.from_real_imag(a, b))
        return Add(*result)
    
    def visit_mul(self, mul: Mul) -> AST:
        """
        Normalizes a product by collecting constant factors and
        rearranging non-constant factors in a canonical order.

        >>> from logic1.theories.Complex.ast import *
        >>> x, y = Var('x'), Var('y')
        >>> WeakNormalizer().visit_mul(y * x * y)
        x * y**2
        >>> WeakNormalizer().visit_mul(2 * x * -I)
        -2 * I * x
        >>> WeakNormalizer().visit_mul(x * Re(x) * Im(x) * Conj(x))
        x * ~x * Re(x) * Im(x)
        >>> WeakNormalizer().visit_mul(Re(x) * Re(y) * (x + y))
        Re(x) * Re(y) * (x + y)
        >>> WeakNormalizer().visit_mul(0 * x)
        0
        >>> WeakNormalizer().visit_mul(1 * x)
        x
        >>> WeakNormalizer().visit_mul(-1 * x)
        -x
        """
        negated = False
        args = [arg.accept(self) for arg in mul.args]
        # collect constant/non-constant factors
        constant: AST = Rat(1)
        factors: dict[AST, int] = {}
        while args:
            arg = args.pop()
            if isinstance(arg, Mul):
                args.extend(arg.args)
                continue
            #if isinstance(arg, Neg):
            #    negated = not negated
            #    args.append(arg.arg)
            #    continue
            if arg.is_constant():
                constant = Mul(constant, arg)
                continue
            exp = 1
            if isinstance(arg, Pow):
                exp = arg.exponent
                arg = arg.base
            factors[arg] = factors.get(arg, 0) + exp
        # regroup factors
        result = []
        for factor in factors:
            exp = factors[factor]
            if exp == 0:
                continue
            if isinstance(factor, Neg):
                negated = not negated if exp % 2 == 1 else negated
                factor = factor.arg  
            if exp == 1:
                result.append(factor)
            else:
                result.append(Pow(factor, exp))
        result.sort(key=AST.sort_key)
        # evaluate constant and build final product
        a, b = constant.eval()
        if negated:
            a, b = -a, -b
        if len(result) == 0:
            return AST.from_real_imag(a, b)
        if a == mpq(0) and b == mpq(0):
            return Rat(0)
        if a == mpq(1) and b == mpq(0):
            return Mul(*result)
        if a == mpq(-1) and b == mpq(0):
            result[0] = Neg(result[0])
        else:
            result = [AST.from_real_imag(a, b)] + result
        return Mul(*result)

    def visit_pow(self, pow: Pow) -> AST:
        """Normalizes a power by evaluating it if the base is constant,
        and simplifying if the exponent is 0 or 1. Note that 0^0 is
        defined to be 1 as for mpq.

        >>> from logic1.theories.Complex.ast import *
        >>> x, y = Var('x'), Var('y')
        >>> WeakNormalizer().visit_pow((x + y)**0)
        1
        >>> WeakNormalizer().visit_pow((x + y)**1)
        x + y
        >>> WeakNormalizer().visit_pow((x + y)**2)
        (x + y)**2
        >>> WeakNormalizer().visit_pow(I**2)
        -1
        >>> WeakNormalizer().visit_pow((I - I)**0)
        1
        """
        if pow.exponent == 0:
            return Rat(1)
        base = pow.base.accept(self)
        if base.is_constant():
            a, b = Pow(base, pow.exponent).eval()
            return AST.from_real_imag(a, b)
        elif pow.exponent == 1:
            return base
        else:
            return Pow(base, pow.exponent)

    def visit_neg(self, neg: Neg) -> AST:
        """Normalizes a negation by evaluating constants, simplifying double 
        negations and moving the negation inside products.

        >>> from logic1.theories.Complex.ast import *
        >>> x, y = Var('x'), Var('y')
        >>> WeakNormalizer().visit_neg(-(1 + I))
        -1 - I
        >>> WeakNormalizer().visit_neg(-(-x))
        x
        >>> WeakNormalizer().visit_neg(-(x * y))
        -x * y
        """
        arg = neg.arg.accept(self)
        if arg.is_constant():
            a, b = arg.eval()
            return AST.from_real_imag(-a, -b)
        elif isinstance(arg, Neg):
            return arg.arg
        elif isinstance(arg, Mul):
            return Mul(Neg(arg.args[0]), *arg.args[1:]).accept(self)
        else:
            return Neg(arg)

    def visit_conj(self, conj: Conj) -> AST:
        """Normalizes a conjugation by evaluating constants and simplifying 
        double conjugations, and simplifying conjugations of real and imaginary parts.

        >>> from logic1.theories.Complex.ast import *
        >>> x, y = Var('x'), Var('y')
        >>> WeakNormalizer().visit_conj(Conj(1 + I))
        1 - I
        >>> WeakNormalizer().visit_conj(Conj(Conj(x)))
        x
        >>> WeakNormalizer().visit_conj(Conj(Re(x)))
        Re(x)
        >>> WeakNormalizer().visit_conj(Conj(Im(x)))
        Im(x)
        """
        arg = conj.arg.accept(self)
        if arg.is_constant():
            a, b = arg.eval()
            return AST.from_real_imag(a, -b)
        elif isinstance(arg, Conj):
            return arg.arg
        elif isinstance(arg, (Re, Im)):
            return arg
        else:
            return Conj(arg)
    
    def visit_re(self, re: Re) -> AST:
        """Normalizes a real part by evaluating constants and simplifying
        real parts of real and imaginary parts, and of conjugates.
        
        >>> from logic1.theories.Complex.ast import *
        >>> x = Var('x')
        >>> WeakNormalizer().visit_re(Re(1 + I))
        1
        >>> WeakNormalizer().visit_re(Re(Re(x)))
        Re(x)
        >>> WeakNormalizer().visit_re(Re(Im(x)))
        Im(x)
        >>> WeakNormalizer().visit_re(Re(Conj(x)))
        Re(x)
        """
        arg = re.arg.accept(self)
        if arg.is_constant():
            a, _ = arg.eval()
            return Rat(a)
        elif isinstance(arg, (Re, Im)):
            return arg
        elif isinstance(arg, Conj):
            return Re(arg.arg).accept(self)
        else:
            return Re(arg)

    def visit_im(self, im: Im) -> AST:
        """Normalizes an imaginary part by evaluating constants and simplifying
        imaginary parts of real and imaginary parts, and of conjugates.
        
        >>> from logic1.theories.Complex.ast import *
        >>> x = Var('x')
        >>> WeakNormalizer().visit_im(Im(1 + I))
        1
        >>> WeakNormalizer().visit_im(Im(Re(x)))
        0
        >>> WeakNormalizer().visit_im(Im(Im(x)))
        0
        >>> WeakNormalizer().visit_im(Im(Conj(x)))
        -Im(x)
        """
        arg = im.arg.accept(self)
        if arg.is_constant():
            _, b = arg.eval()
            return Rat(b)
        elif isinstance(arg, (Re, Im)):
            return Rat(0)
        elif isinstance(arg, Conj):
            return Neg(Im(arg.arg)).accept(self)
        else:
            return Im(arg)


class Normalizer(WeakNormalizer):
    """Visitor that normalizes a AST as in `WeakNormalizer`, but also
    expands nodes and propagates `Re`, `Im` and `Conj`.
    """

    def visit_mul(self, mul: Mul) -> AST:
        """Expands a product by distributing it over sums and normalizing the 
        factors recursively.
        
        >>> from logic1.theories.Complex.ast import *
        >>> x, y, z = Var('x'), Var('y'), Var('z')
        >>> Normalizer().visit_mul(x * (y + z))
        x * y + x * z
        """
        args = [arg.accept(self) for arg in mul.args]
        for i, arg in enumerate(args):
            if isinstance(arg, Add):
                arg_args = [Mul(*args[:i], arg_arg, *args[i + 1:]) for arg_arg in arg.args]
                return Add(*arg_args).accept(self)
        result = super().visit_mul(mul)
        return result
    
    def visit_pow(self, pow: Pow) -> AST:
        """Expands powers of sums and products and normalizes them recursively.
        
        >>> from logic1.theories.Complex.ast import *
        >>> x, y = Var('x'), Var('y')
        >>> Normalizer().visit_pow((x + y)**2)
        x**2 + y**2 + 2 * x * y
        >>> Normalizer().visit_pow((x * y)**2)
        x**2 * y**2
        >>> Normalizer().visit_pow((-x)**3)
        -x**3
        """
        node = super().visit_pow(pow)
        if isinstance(node, Pow) and not isinstance(node.base, (Var, Re, Im, Conj)):
            result = Mul(*[node.base] * node.exponent).accept(self)
            return result
        return node

    def visit_neg(self, neg: Neg) -> AST:
        """Expands a negation by distributing it over sums and normalizing the 
        argument recursively.
        
        >>> from logic1.theories.Complex.ast import *
        >>> x, y = Var('x'), Var('y')
        >>> Normalizer().visit_neg(-(x + y))
        -x - y
        """
        node = super().visit_neg(neg)
        if isinstance(node, Neg) and isinstance(node.arg, Add):
            return Add(*map(Neg, node.arg.args)).accept(self)
        return node

    def visit_conj(self, conj: Conj) -> AST:
        """Propagates a conjugation by distributing it over sums and products 
        and normalizing the argument recursively.
        
        >>> from logic1.theories.Complex.ast import *
        >>> x, y = Var('x'), Var('y')
        >>> Normalizer().visit_conj(Conj(x + y))
        ~x + ~y
        >>> Normalizer().visit_conj(Conj(x * y))
        ~x * ~y
        >>> Normalizer().visit_conj(Conj(-x))
        -~x
        >>> Normalizer().visit_conj(Conj(x**2))
        (~x)**2
        """
        node = super().visit_conj(conj)
        if isinstance(node, Conj):
            if isinstance(node.arg, Add):
                return Add(*map(Conj, node.arg.args)).accept(self)
            if isinstance(node.arg, Mul):
                return Mul(*map(Conj, node.arg.args)).accept(self)
            if isinstance(node.arg, Neg):
                return Neg(Conj(node.arg.arg)).accept(self)
            if isinstance(node.arg, Pow):
                return Pow(Conj(node.arg.base), node.arg.exponent).accept(self)
        return node

    def visit_re(self, re: Re) -> AST:
        """Propagates a real part by distributing it over sums and products 
        and normalizing the argument recursively.

        >>> from logic1.theories.Complex.ast import *
        >>> x, y = Var('x'), Var('y')
        >>> Normalizer().visit_re(Re(x + y))
        Re(x) + Re(y)
        >>> Normalizer().visit_re(Re(x * y))
        Re(x) * Re(y) - Im(x) * Im(y)
        >>> Normalizer().visit_re(Re(-x))
        -Re(x)
        >>> Normalizer().visit_re(Re(x**2))
        Re(x)**2 - Im(x)**2
        """
        node = super().visit_re(re)
        if isinstance(node, Re):
            if isinstance(node.arg, Add):
                return Add(*map(Re, node.arg.args)).accept(self)
            if isinstance(node.arg, Mul):
                x, xs = node.arg.args[0], Mul(*node.arg.args[1:])
                return (Re(x) * Re(xs) - Im(x) * Im(xs)).accept(self)
            if isinstance(node.arg, Neg):
                return Neg(Re(node.arg.arg)).accept(self)
            if isinstance(node.arg, Pow):
                y, ys = node.arg.base, Pow(node.arg.base, node.arg.exponent - 1)
                return (Re(y) * Re(ys) - Im(y) * Im(ys)).accept(self)
        return node

    def visit_im(self, im: Im) -> AST:
        """Propagates an imaginary part by distributing it over sums and 
        products and normalizing the argument recursively.
        
        >>> from logic1.theories.Complex.ast import *
        >>> x, y = Var('x'), Var('y')
        >>> Normalizer().visit_im(Im(x + y))
        Im(x) + Im(y)
        >>> Normalizer().visit_im(Im(x * y))
        Re(x) * Im(y) + Re(y) * Im(x)
        >>> Normalizer().visit_im(Im(-x))
        -Im(x)
        >>> Normalizer().visit_im(Im(x**2))
        2 * Re(x) * Im(x)
        """
        node = super().visit_im(im)
        if isinstance(node, Im):
             if isinstance(node.arg, Add):
                return Add(*map(Im, node.arg.args)).accept(self)
             if isinstance(node.arg, Mul):
                x, xs = node.arg.args[0], Mul(*node.arg.args[1:])
                return (Re(x) * Im(xs) + Im(x) * Re(xs)).accept(self)
             if isinstance(node.arg, Neg):
                return Neg(Im(node.arg.arg)).accept(self)
             if isinstance(node.arg, Pow):
                y, ys = node.arg.base, Pow(node.arg.base, node.arg.exponent - 1)
                return (Re(y) * Im(ys) + Im(y) * Re(ys)).accept(self)
        return node


class ComplexNormalizer(Normalizer):
    """Visitor that normalizes a AST as in `Normalizer`, but also
    replaces all occurrences of `Re` and `Im`.
    This yields a unique normal form.
    
    >>> from logic1.theories.Complex.ast import *
    >>> z = Var('z')
    >>> (Re(z) + I * Im(z)).accept(ComplexNormalizer())
    z
    >>> (Re(z)**2 + Im(z)**2).accept(ComplexNormalizer())
    z * ~z 
    """

    def visit_re(self, re: Re) -> AST:
        """Replaces a real part with its equivalent expression in
        terms of its argument and its conjugate.
        
        >>> from logic1.theories.Complex.ast import *
        >>> z = Var('z')
        >>> ComplexNormalizer().visit_re(Re(z))
        1/2 * z + 1/2 * ~z
        """
        return ((re.arg + Conj(re.arg)) / 2).accept(self)

    def visit_im(self, im: Im) -> AST:
        """Replaces an imaginary part with its equivalent
        expression in terms of its argument and its conjugate.

        >>> from logic1.theories.Complex.ast import *
        >>> z = Var('z')
        >>> ComplexNormalizer().visit_im(Im(z))
        -1/2 * I * z + 1/2 * I * ~z
        """
        return ((im.arg - Conj(im.arg)) / (2 * I)).accept(self)

    