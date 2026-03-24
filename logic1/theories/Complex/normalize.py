"""This module defines visitors for evaluating and normalizing terms in
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
    """Abstract visitor that evaluates a term to an element of α, 
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
        result = Rational(0).accept(self)
        for arg in add.args:
            result = self._add(result, arg.accept(self))
        return result

    def visit_mul(self, mul: Mul) -> α:
        """Evaluates a product by using the `_mul` method.
        """
        result = Rational(1).accept(self)
        for arg in mul.args:
            result = self._mul(result, arg.accept(self))
        return result

    def visit_neg(self, neg: Neg) -> α:
        """Evaluates a negation by using the `_neg` method.
        """
        return self._neg(neg.arg.accept(self))
    
    def visit_pow(self, pow):
        """Evaluates a power.
        """
        if pow.exponent == 0:
            return Rational(1).accept(self)
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

    def visit_rational(self, num: Rational) -> tuple[mpq, mpq]:
        """Evaluates a rational number term to a complex number
        represented as a pair of rational numbers (real, imag).
        Implements the abstract method :meth:`.TermVisitor.visit_rational`.

        >>> ConstantEvaluator().visit_rational(Rational(mpq(1, 2)))
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

    def visit_variable(self, var: Variable) -> tuple[mpq, mpq]:
        """Evaluates a variable term to a complex number represented as
        a pair of rational numbers (real, imag) under the variable
        assignment given in the constructor. Implements the abstract
        method :meth:`.TermVisitor.visit_variable`.

        >>> from logic1.theories.Complex import *
        >>> x = VV['x']
        >>> ConstantEvaluator({x: mpq(1, 2)}).visit_variable(x)
        (mpq(1,2), mpq(0,1))
        >>> ConstantEvaluator().visit_variable(x)
        Traceback (most recent call last):
          ...
        ValueError: Cannot evaluate variable x
        """
        try:
            return Term.from_number(self._variables[var]).accept(self)
        except KeyError:
            raise ValueError(f'Cannot evaluate variable {var}')

    def visit_conj(self, conj: Conj) -> tuple[mpq, mpq]:
        """Evaluates a conjugation term to a complex number represented
        as a pair of rational numbers (real, imag).
        Implements the abstract method :meth:`.TermVisitor.visit_conj`.

        >>> ConstantEvaluator().visit_conj(Conj(1 + 2 * I))
        (mpq(1,1), mpq(-2,1))
        """
        a, b = conj.arg.accept(self)
        return a, -b

    def visit_re(self, re: Re) -> tuple[mpq, mpq]:
        """Evaluates a real part term to a complex number represented as
        a pair of rational numbers (real, imag).
        Implements the abstract method :meth:`.TermVisitor.visit_re`.

        >>> ConstantEvaluator().visit_re(Re(1 + 2 * I))
        (mpq(1,1), mpq(0,1))
        """
        a, _ = re.arg.accept(self)
        return a, mpq(0)

    def visit_im(self, im: Im) -> tuple[mpq, mpq]:
        """Evaluates an imaginary part term to a complex number
        represented as a pair of rational numbers (real, imag).
        Implements the abstract method :meth:`.TermVisitor.visit_im`.
        
        >>> ConstantEvaluator().visit_im(Im(1 + 2 * I))
        (mpq(2,1), mpq(0,1))
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
        constant: Term = Rational(0)
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
                coeff: Term = Rational(-1)
                arg = arg.arg
            else:
                coeff = Rational(1)
            while isinstance(arg, Mul) and arg.args[0].is_constant():
                coeff = coeff * arg.args[0]
                arg = Mul(*arg.args[1:])
            if isinstance(arg, Mul) and isinstance(arg.args[0], Neg):
                coeff = coeff * Rational(-1)
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
        >>> x, y = VV.get('x', 'y')
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
        constant: Term = Rational(1)
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
        result.sort(key=Term.sort_key)
        # evaluate constant and build final product
        a, b = constant.eval()
        if negated:
            a, b = -a, -b
        if len(result) == 0:
            return Term.from_real_imag(a, b)
        if a == mpq(0) and b == mpq(0):
            return Rational(0)
        if a == mpq(1) and b == mpq(0):
            return Mul(*result)
        if a == mpq(-1) and b == mpq(0):
            result[0] = Neg(result[0])
        else:
            result = [Term.from_real_imag(a, b)] + result
        return Mul(*result)

    def visit_pow(self, pow: Pow) -> Term:
        """Normalizes a power by evaluating it if the base is constant,
        and simplifying if the exponent is 0 or 1. Note that 0^0 is
        defined to be 1 as for mpq.

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
            return Rational(1)
        base = pow.base.accept(self)
        if base.is_constant():
            a, b = Pow(base, pow.exponent).eval()
            return Term.from_real_imag(a, b)
        elif pow.exponent == 1:
            return base
        else:
            return Pow(base, pow.exponent)

    def visit_neg(self, neg: Neg) -> Term:
        """Normalizes a negation by evaluating constants, simplifying double 
        negations and moving the negation inside products.

        >>> from logic1.theories.Complex import *
        >>> x, y = VV.get('x', 'y')
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
            return Term.from_real_imag(-a, -b)
        elif isinstance(arg, Neg):
            return arg.arg
        elif isinstance(arg, Mul):
            return Mul(Neg(arg.args[0]), *arg.args[1:]).accept(self)
        else:
            return Neg(arg)

    def visit_conj(self, conj: Conj) -> Term:
        """Normalizes a conjugation by evaluating constants and simplifying 
        double conjugations, and simplifying conjugations of real and imaginary parts.

        >>> from logic1.theories.Complex import *
        >>> x, y = VV.get('x', 'y')
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
            return Term.from_real_imag(a, -b)
        elif isinstance(arg, Conj):
            return arg.arg
        elif isinstance(arg, (Re, Im)):
            return arg
        else:
            return Conj(arg)
    
    def visit_re(self, re: Re) -> Term:
        """Normalizes a real part by evaluating constants and simplifying
        real parts of real and imaginary parts, and of conjugates.
        
        >>> from logic1.theories.Complex import *
        >>> x = VV['x']
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
            return Rational(a)
        elif isinstance(arg, (Re, Im)):
            return arg
        elif isinstance(arg, Conj):
            return Re(arg.arg).accept(self)
        else:
            return Re(arg)

    def visit_im(self, im: Im) -> Term:
        """Normalizes an imaginary part by evaluating constants and simplifying
        imaginary parts of real and imaginary parts, and of conjugates.
        
        >>> from logic1.theories.Complex import *
        >>> x = VV['x']
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
            return Rational(b)
        elif isinstance(arg, (Re, Im)):
            return Rational(0)
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
    """

    def visit_mul(self, mul: Mul) -> Term:
        """Expands a product by distributing it over sums and normalizing the factors recursively.
        
        >>> from logic1.theories.Complex import *
        >>> x, y, z = VV.get('x', 'y', 'z')
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
    
    def visit_pow(self, pow: Pow) -> Term:
        """Expands powers of sums and products and normalizes them recursively.
        
        >>> from logic1.theories.Complex import *
        >>> x, y = VV.get('x', 'y')
        >>> Normalizer().visit_pow((x + y)**2)
        x**2 + y**2 + 2 * x * y
        >>> Normalizer().visit_pow((x * y)**2)
        x**2 * y**2
        >>> Normalizer().visit_pow((-x)**3)
        -x**3
        """
        term = super().visit_pow(pow)
        if isinstance(term, Pow) and not isinstance(term.base, (Variable, Re, Im, Conj)):
            result = Mul(*[term.base] * term.exponent).accept(self)
            return result
        return term

    def visit_neg(self, neg: Neg) -> Term:
        """Expands a negation by distributing it over sums and normalizing the argument recursively.
        
        >>> from logic1.theories.Complex import *
        >>> x, y = VV.get('x', 'y')
        >>> Normalizer().visit_neg(-(x + y))
        -x - y
        """
        term = super().visit_neg(neg)
        if isinstance(term, Neg) and isinstance(term.arg, Add):
            return Add(*map(Neg, term.arg.args)).accept(self)
        return term

    def visit_conj(self, conj: Conj) -> Term:
        """Propagates a conjugation by distributing it over sums and products and normalizing the argument recursively.
        
        >>> from logic1.theories.Complex import *
        >>> x, y = VV.get('x', 'y')
        >>> Normalizer().visit_conj(Conj(x + y))
        ~x + ~y
        >>> Normalizer().visit_conj(Conj(x * y))
        ~x * ~y
        >>> Normalizer().visit_conj(Conj(-x))
        -~x
        >>> Normalizer().visit_conj(Conj(x**2))
        (~x)**2
        """
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
        """Propagates a real part by distributing it over sums and products and normalizing the argument recursively.

        >>> from logic1.theories.Complex import *
        >>> x, y = VV.get('x', 'y')
        >>> Normalizer().visit_re(Re(x + y))
        Re(x) + Re(y)
        >>> Normalizer().visit_re(Re(x * y))
        Re(x) * Re(y) - Im(x) * Im(y)
        >>> Normalizer().visit_re(Re(-x))
        -Re(x)
        >>> Normalizer().visit_re(Re(x**2))
        Re(x)**2 - Im(x)**2
        """
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
        """Propagates an imaginary part by distributing it over sums and products and normalizing the argument recursively.
        
        >>> from logic1.theories.Complex import *
        >>> x, y = VV.get('x', 'y')
        >>> Normalizer().visit_im(Im(x + y))
        Im(x) + Im(y)
        >>> Normalizer().visit_im(Im(x * y))
        Re(x) * Im(y) + Re(y) * Im(x)
        >>> Normalizer().visit_im(Im(-x))
        -Im(x)
        >>> Normalizer().visit_im(Im(x**2))
        2 * Re(x) * Im(x)
        """
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
    z * ~z 
    """

    def visit_re(self, re: Re) -> Term:
        """Replaces a real part term with its equivalent expression in
        terms of its argument and its conjugate.
        
        >>> from logic1.theories.Complex import *
        >>> z = VV['z']
        >>> ComplexNormalizer().visit_re(Re(z))
        1/2 * z + 1/2 * ~z
        """
        return ((re.arg + Conj(re.arg)) / 2).accept(self)

    def visit_im(self, im: Im) -> Term:
        """Replaces an imaginary part term with its equivalent
        expression in terms of its argument and its conjugate.

        >>> from logic1.theories.Complex import *
        >>> z = VV['z']
        >>> ComplexNormalizer().visit_im(Im(z))
        -1/2 * I * z + 1/2 * I * ~z
        """
        return ((im.arg - Conj(im.arg)) / (2 * I)).accept(self)

    