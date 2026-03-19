"""This module defines visitors for simplifying and normalizing terms in
the theory of complex numbers.
"""

from abc import abstractmethod
from typing import TypeVar

from logic1.theories.Complex.atomic import AtomicFormula, AtomicFormulaVisitor, Eq, Ge, Gt, Le, Lt, Ne
from logic1.theories.Complex.term import Add, Conj, IdentityTermVisitor, _I, I, Im, Mul, Neg, Number, Pow, Rational, Re, Term, TermVisitor, Variable
from gmpy2 import mpq

α = TypeVar('α')
"""Type variable for the result type of `TermVisitor` methods."""

class ArithmeticEvaluator(TermVisitor[α]):

    @abstractmethod
    def _add(self, a: α, b: α) -> α:
        ...

    @abstractmethod
    def _neg(self, a: α) -> α:
        ...

    @abstractmethod
    def _mul(self, a: α, b: α) -> α:
        ...

    def visit_add(self, add: Add) -> α:
        result = Rational(mpq(0)).accept(self)
        for arg in add.args:
            result = self._add(result, arg.accept(self))
        return result

    def visit_mul(self, mul: Mul) -> α:
        result = Rational(mpq(1)).accept(self)
        for arg in mul.args:
            result = self._mul(result, arg.accept(self))
        return result

    def visit_neg(self, neg: Neg) -> α:
        return self._neg(neg.arg.accept(self))
    
    def visit_pow(self, pow):
        if pow.exponent == 0:
            return Rational(mpq(1)).accept(self)
        elif pow.exponent % 2 == 0:    
            a = Pow(pow.base, pow.exponent // 2).accept(self)
            return self._mul(a, a)
        else:
            a = pow.base.accept(self)
            b = Pow(pow.base, pow.exponent - 1).accept(self)
            return self._mul(a, b)


class ConstantEvaluator(ArithmeticEvaluator[tuple[mpq, mpq]], AtomicFormulaVisitor[bool]):
    """Visitor that evaluates a term to a constant under a given
    variable assignment. The result is a pair of rational numbers 
    representing the real and imaginary parts of the complex number. 
    Raises a ValueError if the term contains variables that are not 
    in the assignment.
    """

    _variables: dict[Variable, Number]

    def __init__(self, variables: dict[Variable, Number] = {}) -> None:
        """Initializes the evaluator with a variable assignment.
        The variable assignment is a dictionary mapping variables to numbers.
        """
        self._variables = variables

    def _add(self, a: tuple[mpq, mpq], b: tuple[mpq, mpq]) -> tuple[mpq, mpq]:
        """Adds two complex numbers represented as pairs of rational
        numbers (real, imag). Implements the abstract method 
        :meth:`.ArithmeticEvaluator._add`.
        """
        return a[0] + b[0], a[1] + b[1]
    
    def _neg(self, a: tuple[mpq, mpq]) -> tuple[mpq, mpq]:
        """Negates a complex number represented as a pair of rational
        numbers (real, imag). Implements the abstract method 
        :meth:`.ArithmeticEvaluator._neg`.
        """
        return -a[0], -a[1]

    def _mul(self, a: tuple[mpq, mpq], b: tuple[mpq, mpq]) -> tuple[mpq, mpq]:
        """Multiplies two complex numbers represented as pairs of
        rational numbers (real, imag). Implements the abstract method 
        :meth:`.ArithmeticEvaluator._mul`.
        """
        return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]

    def visit_rational(self, num: Rational) -> tuple[mpq, mpq]:
        """Evaluates a rational number term to a complex number
        represented as a pair of rational numbers (real, imag).
        Implements the abstract method :meth:`.TermVisitor.visit_rational`.
        """
        return num.value, mpq(0)

    def visit_i(self, i: _I) -> tuple[mpq, mpq]:
        """Evaluates the imaginary unit term to a complex number
        represented as a pair of rational numbers (real, imag).
        Implements the abstract method :meth:`.TermVisitor.visit_i`.
        """
        return mpq(0), mpq(1)

    def visit_variable(self, var: Variable) -> tuple[mpq, mpq]:
        """Evaluates a variable term to a complex number represented as
        a pair of rational numbers (real, imag) under the variable
        assignment given in the constructor. Implements the abstract
        method :meth:`.TermVisitor.visit_variable`.
        """
        try:
            return Term.from_number(self._variables[var]).accept(self)
        except KeyError:
            raise ValueError(f'Cannot evaluate variable {var}')

    def visit_conj(self, conj: Conj) -> tuple[mpq, mpq]:
        """Evaluates a conjugation term to a complex number represented
        as a pair of rational numbers (real, imag).
        Implements the abstract method :meth:`.TermVisitor.visit_conj`.
        """
        a, b = conj.arg.accept(self)
        return a, -b

    def visit_re(self, re: Re) -> tuple[mpq, mpq]:
        """Evaluates a real part term to a complex number represented as
        a pair of rational numbers (real, imag).
        Implements the abstract method :meth:`.TermVisitor.visit_re`.
        """
        a, _ = re.arg.accept(self)
        return a, mpq(0)

    def visit_im(self, im: Im) -> tuple[mpq, mpq]:
        """Evaluates an imaginary part term to a complex number
        represented as a pair of rational numbers (real, imag).
        Implements the abstract method :meth:`.TermVisitor.visit_im`.
        """
        _, b = im.arg.accept(self)
        return b, mpq(0)

    def visit_eq(self, eq: Eq) -> bool:
        """Evaluates an equality formula to a boolean value under the
        variable assignment given in the constructor.
        Implements the abstract method :meth:`.AtomicFormulaVisitor.visit_eq`.
        """
        return eq.lhs.accept(self) == eq.rhs.accept(self)
    
    def visit_ne(self, ne: Ne) -> bool:
        """Evaluates an inequality formula to a boolean value under the
        variable assignment given in the constructor.
        Implements the abstract method :meth:`.AtomicFormulaVisitor.visit_ne`.
        """
        return ne.lhs.accept(self) != ne.rhs.accept(self)
    
    def visit_ge(self, ge: Ge) -> bool:
        """Evaluates a greater-than-or-equal formula to a boolean value
        under the variable assignment given in the constructor.
        Implements the abstract method :meth:`.AtomicFormulaVisitor.visit_ge`.
        """
        a1, b1 = ge.lhs.accept(self)
        a2, b2 = ge.rhs.accept(self)
        return a1 >= a2 and b1 == 0 and b2 == 0

    def visit_le(self, le: Le) -> bool:
        """Evaluates a less-than-or-equal formula to a boolean value
        under the variable assignment given in the constructor.
        Implements the abstract method :meth:`.AtomicFormulaVisitor.visit_le`.
        """
        a1, b1 = le.lhs.accept(self)
        a2, b2 = le.rhs.accept(self)
        return a1 <= a2 and b1 == 0 and b2 == 0

    def visit_gt(self, gt: Gt) -> bool:
        """Evaluates a greater-than formula to a boolean value under the
        variable assignment given in the constructor.
        Implements the abstract method :meth:`.AtomicFormulaVisitor.visit_gt`.
        """
        a1, b1 = gt.lhs.accept(self)
        a2, b2 = gt.rhs.accept(self)
        return a1 > a2 and b1 == 0 and b2 == 0

    def visit_lt(self, lt: Lt) -> bool:
        """Evaluates a less-than formula to a boolean value under the
        variable assignment given in the constructor.
        Implements the abstract method :meth:`.AtomicFormulaVisitor.visit_lt`.
        """
        a1, b1 = lt.lhs.accept(self)
        a2, b2 = lt.rhs.accept(self)
        return a1 < a2 and b1 == 0 and b2 == 0


class WeakNormalizer(IdentityTermVisitor, AtomicFormulaVisitor[AtomicFormula]):
    """Visitor that normalizes a term by rearranging sums and products, 
    and applying obvious simplifications, but not expanding any terms.
    """
    
    def visit_add(self, add: Add) -> Term:
        """Normalizes a sum by collecting constant terms and
        rearranging non-constant terms in a canonical order.
        
        >>> from logic1.theories.Complex import *
        >>> x, y, z = VV.get('x', 'y', 'z')
        >>> (y + x + z + x - z).accept(WeakNormalizer())
        2 * x + y
        >>> (2 + x - 3).accept(WeakNormalizer())
        x - 1
        """
        args = [arg.accept(self) for arg in add.args]
        # collect all products with their coefficients and the absolute constants
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
            # extract the coefficient of the product
            if isinstance(arg, Neg):
                coeff: Term = Rational(mpq(-1))
                arg = arg.arg
            else:
                coeff = Rational(mpq(1))
            while isinstance(arg, Mul) and arg.args[0].is_constant():
                coeff = coeff * arg.args[0]
                arg = Mul(*arg.args[1:])
            if isinstance(arg, Mul) and isinstance(arg.args[0], Neg):
                coeff = coeff * Rational(mpq(-1))
                arg = Mul(arg.args[0].arg, *arg.args[1:])
            products[arg] = products.get(arg, 0) + coeff 
        # put all products with their coefficients back together and simplify if possible
        result = []
        for i, prod in enumerate(sorted(products, key=Term.sort_key)):
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
                coeff = Term.from_real_imag(-a, -b)
                result.append(Neg(coeff * prod))
            else:
                coeff = Term.from_real_imag(a, b)
                result.append(coeff * prod)
        # add the abolute constant if it is not zero
        a, b = constant.eval()
        if a == mpq(0) and b == mpq(0):
            pass
        elif len(result) > 0 and (a < mpq(0) or (a == mpq(0) and b < mpq(0))):
            result.append(Neg(Term.from_real_imag(-a, -b)))
        else:
            result.append(Term.from_real_imag(a, b))
        return Add(*result)
    
    def visit_mul(self, mul: Mul) -> Term:
        """
        Normalizes a product by collecting constant factors and
        rearranging non-constant factors in a canonical order.

        >>> from logic1.theories.Complex import *
        >>> x, y, z = VV.get('x', 'y', 'z')
        >>> WeakNormalizer().visit_mul(y * x * z * y)
        x * y**2 * z
        >>> WeakNormalizer().visit_mul(2 * x * -I)
        -2 * I * x**3
        >>> WeakNormalizer().visit_mul(x * Re(x) * Im(x) * Conj(x))
        Conj(x) * Im(x) * Re(x) * x
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
        constant: Term = Rational(mpq(1))
        factors: dict[Term, int] = {}
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
        for factor in sorted(factors, key=Term.sort_key):
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
        # evaluate constant and build final product
        a, b = constant.eval()
        if negated:
            a, b = -a, -b
        if len(result) == 0:
            return Term.from_real_imag(a, b)
        if a == mpq(0) and b == mpq(0):
            return Rational(mpq(0))
        if a == mpq(1) and b == mpq(0):
            return Mul(*result)
        if a == mpq(-1) and b == mpq(0):
            result[0] = Neg(result[0])
        else:
            result = [Term.from_real_imag(a, b)] + result
        return Mul(*result)

    def visit_pow(self, pow: Pow) -> Term:
        """Normalizes a power by evaluating it if the base is constant,
        and simplifying if the exponent is 0 or 1.

        >>> from logic1.theories.Complex import *
        >>> x, y = VV.get('x', 'y')
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
            return Rational(mpq(1))  # note: 0^0 = 1 as for mpq
        base = pow.base.accept(self)
        if base.is_constant():
            a, b = Pow(base, pow.exponent).eval()
            return Term.from_real_imag(a, b)
        elif pow.exponent == 1:
            return base
        else:
            return Pow(base, pow.exponent)

    def visit_neg(self, neg: Neg) -> Term:
        """Normalizes a negation by simplifying double negations.

        >>> from logic1.theories.Complex import *
        >>> x = VV['x']
        >>> WeakNormalizer().visit_neg(-(-x))
        x
        """
        arg = neg.arg.accept(self)
        if isinstance(arg, Neg):
            return arg.arg
        elif isinstance(arg, Mul):
            return Mul(Neg(arg.args[0]), *arg.args[1:]).accept(self)
        else:
            return Neg(arg)

    def visit_conj(self, conj: Conj) -> Term:
        arg = conj.arg.accept(self)
        if isinstance(arg, (Rational, Re, Im)):
            return arg
        elif isinstance(arg, _I):
            return Neg(arg)
        elif isinstance(arg, Conj):
            return arg.arg
        else:
            return Conj(arg)
    
    def visit_re(self, re: Re) -> Term:
        arg = re.arg.accept(self)
        if isinstance(arg, (Rational, Re, Im)):
            return arg
        elif isinstance(arg, _I):
            return Rational(mpq(0))
        elif isinstance(arg, Conj):
            return Re(arg.arg).accept(self)
        else:
            return Re(arg)

    def visit_im(self, im: Im) -> Term:
        arg = im.arg.accept(self)
        if isinstance(arg, (Rational, Re, Im)):
            return Rational(mpq(0))
        elif isinstance(arg, _I):
            return Rational(mpq(1))
        elif isinstance(arg, Conj):
            return Neg(Im(arg.arg)).accept(self)
        else:
            return Im(arg)

    def _visit_equality(self, eq: Eq | Ne) -> AtomicFormula:
        lhs: Term = (eq.lhs - eq.rhs).accept(self)
        a, b = lhs.lc().eval()
        if (a, b) != (mpq(0), mpq(0)):
            lhs = (lhs / Term.from_real_imag(a, b)).accept(self)
        return eq.op(lhs, 0)

    def _visit_inequality(self, ieq: Le | Ge | Lt | Gt) -> AtomicFormula:
        lhs: Term = (ieq.lhs - ieq.rhs).accept(self)
        a, b = lhs.lc().eval()
        if a == mpq(0) and b != mpq(0):
            c = b
        elif a != mpq(0):
            c = a
        else:
            return ieq.op(lhs, 0)
        lhs = (lhs / Rational(c)).accept(self)
        if c < mpq(0):
            return ieq.op.converse()(lhs, 0)
        else:
            return ieq.op(lhs, 0)

    def visit_eq(self, eq: Eq) -> AtomicFormula:
        return self._visit_equality(eq)
        
    def visit_ne(self, ne: Ne) -> AtomicFormula:
        return self._visit_equality(ne)

    def visit_ge(self, ge: Ge) -> AtomicFormula:
        return self._visit_inequality(ge)

    def visit_le(self, le: Le) -> AtomicFormula:
        return self._visit_inequality(le)
    
    def visit_gt(self, gt: Gt) -> AtomicFormula:
        return self._visit_inequality(gt)

    def visit_lt(self, lt: Lt) -> AtomicFormula:
        return self._visit_inequality(lt)


class Normalizer(WeakNormalizer):
    """Visitor that normalizes a term as in `WeakNormalizer`, but also
    expands terms and propagates `Re`, `Im` and `Conj`.

    >>> from logic1.theories.Complex import *
    >>> x, y, z = VV.get('x', 'y', 'z')
    >>> (x * (y + z)).accept(Normalizer())
    x * y + x * z
    >>> ((x + I)**2).accept(Normalizer())
    x**2 + 2 * x * I - 1
    >>> Re(x * y + z).accept(Normalizer())
    Re(x) * Re(y) - Im(x) * Im(y) + Re(z)
    >>> Im(x * y + z).accept(Normalizer())
    Re(x) * Im(y) + Im(x) * Re(y) + Im(z)
    >>> Conj(x * y + z).accept(Normalizer())
    Conj(x) * Conj(y) + Conj(z)
    """

    def visit_mul(self, mul: Mul) -> Term:
        args = [arg.accept(self) for arg in mul.args]
        for i, arg in enumerate(args):
            if isinstance(arg, Add):
                arg_args = [Mul(*args[:i], arg_arg, *args[i + 1:]) for arg_arg in arg.args]
                return Add(*arg_args).accept(self)
        result = super().visit_mul(mul)
        return result
    
    def visit_pow(self, pow: Pow) -> Term:
        term = super().visit_pow(pow)
        if isinstance(term, Pow) and not isinstance(term.base, (Variable, Re, Im, Conj)):
            result = Mul(*[term.base] * term.exponent).accept(self)
            return result
        return term

    def visit_neg(self, neg: Neg) -> Term:
        term = super().visit_neg(neg)
        if isinstance(term, Neg) and isinstance(term.arg, Add):
            return Add(*map(Neg, term.arg.args)).accept(self)
        return term

    def visit_conj(self, conj: Conj) -> Term:
        term = super().visit_conj(conj)
        if isinstance(term, Conj):
            if isinstance(term.arg, Add):
                return Add(*map(Conj, term.arg.args)).accept(self)
            if isinstance(term.arg, Mul):
                return Mul(*map(Conj, term.arg.args)).accept(self)
            if isinstance(term.arg, Neg):
                return Neg(Conj(term.arg.arg)).accept(self)
            if isinstance(term.arg, Pow):
                return Pow(Conj(term.arg.base), term.arg.exponent).accept(self)
        return term

    def visit_re(self, re: Re) -> Term:
        term = super().visit_re(re)
        if isinstance(term, Re):
            if isinstance(term.arg, Add):
                return Add(*map(Re, term.arg.args)).accept(self)
            if isinstance(term.arg, Mul):
                x, xs = term.arg.args[0], Mul(*term.arg.args[1:])
                return (Re(x) * Re(xs) - Im(x) * Im(xs)).accept(self)
            if isinstance(term.arg, Neg):
                return Neg(Re(term.arg.arg)).accept(self)
            if isinstance(term.arg, Pow):
                y, ys = term.arg.base, Pow(term.arg.base, term.arg.exponent - 1)
                return (Re(y) * Re(ys) - Im(y) * Im(ys)).accept(self)
        return term

    def visit_im(self, im: Im) -> Term:
        term = super().visit_im(im)
        if isinstance(term, Im):
             if isinstance(term.arg, Add):
                return Add(*map(Im, term.arg.args)).accept(self)
             if isinstance(term.arg, Mul):
                x, xs = term.arg.args[0], Mul(*term.arg.args[1:])
                return (Re(x) * Im(xs) + Im(x) * Re(xs)).accept(self)
             if isinstance(term.arg, Neg):
                return Neg(Im(term.arg.arg)).accept(self)
             if isinstance(term.arg, Pow):
                y, ys = term.arg.base, Pow(term.arg.base, term.arg.exponent - 1)
                return (Re(y) * Im(ys) + Im(y) * Re(ys)).accept(self)
        return term
    

class ComplexNormalizer(Normalizer):
    """Visitor that normalizes a term as in `Normalizer`, but also
    replaces all occurrences of `Re` and `Im`.
    This yields a unique normal form for terms.
    
    >>> from logic1.theories.Complex import *
    >>> z = VV['z']
    >>> (Re(z) + I * Im(z)).accept(ComplexNormalizer())
    z
    >>> (Re(z)**2 + Im(z)**2).accept(ComplexNormalizer())
    z * Conj(z) 
    """

    def visit_re(self, re: Re) -> Term:
        """Replaces a real part term with its equivalent expression in
        terms of its argument and its conjugate.
        
        >>> from logic1.theories.Complex import *
        >>> z = VV['z']
        >>> ComplexNormalizer().visit_re(Re(z))
        (z + Conj(z)) * 1/2
        """
        return ((re.arg + Conj(re.arg)) / 2).accept(self)

    def visit_im(self, im: Im) -> Term:
        """Replaces an imaginary part term with its equivalent
        expression in terms of its argument and its conjugate.

        >>> from logic1.theories.Complex import *
        >>> z = VV['z']
        >>> ComplexNormalizer().visit_im(Im(z))
        (z - Conj(z)) * -I * 1/2
        """
        return ((im.arg - Conj(im.arg)) / (2 * I)).accept(self)

    