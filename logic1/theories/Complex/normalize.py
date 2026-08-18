"""Visitors and functions for evaluating and normalizing complex ASTs.
"""

from abc import abstractmethod
from dataclasses import dataclass
from functools import total_ordering
from typing import Self

from gmpy2 import mpq

from logic1.theories.Complex.types import α
from logic1.theories.Complex.ast import (
    Add, AST, ASTVisitor, Conj, IdentityASTVisitor, _I, I, Im, Mul, Neg, Pow,
    Rat, Re, Var)


class ArithmeticEvaluator(ASTVisitor[α]):
    """Abstract visitor that evaluates an AST to an element of
    :class:`.α`, given implementations of addition, negation and
    multiplication.

    .. seealso::
        :class:`.ConstantEvaluator`, :class:`.qe.RCF_Evaluator`
    """

    @abstractmethod
    def add(self, a: α, b: α) -> α:
        """Add two elements. Must be implemented by subclasses.
        """
        ...

    @abstractmethod
    def neg(self, a: α) -> α:
        """Negate an element. Must be implemented by subclasses.
        """
        ...

    @abstractmethod
    def mul(self, a: α, b: α) -> α:
        """Multiply two elements. Must be implemented by subclasses.
        """
        ...

    def visit_add(self, add: Add) -> α:
        """Evaluate a sum by using the :meth:`.add` method.
        """
        result = Rat(0).accept(self)
        for arg in add.args:
            result = self.add(result, arg.accept(self))
        return result

    def visit_mul(self, mul: Mul) -> α:
        """Evaluate a product by using the :meth:`.mul` method.
        """
        result = Rat(1).accept(self)
        for arg in mul.args:
            result = self.mul(result, arg.accept(self))
        return result

    def visit_neg(self, neg: Neg) -> α:
        """Evaluate a negation by using the :meth:`.neg` method.
        """
        return self.neg(neg.arg.accept(self))

    def visit_pow(self, pow: Pow) -> α:
        """Evaluate a power using the :meth:`.mul` method.
        """
        if pow.exponent == 0:
            return Rat(1).accept(self)
        elif pow.exponent % 2 == 0:
            a = Pow(pow.base, pow.exponent // 2).accept(self)
            return self.mul(a, a)
        else:
            a = pow.base.accept(self)
            b = Pow(pow.base, pow.exponent - 1).accept(self)
            return self.mul(a, b)


class ConstantEvaluator(ArithmeticEvaluator[tuple[mpq, mpq]]):
    """Visitor based on :class:`.ArithmeticEvaluator` that evaluates an AST to
    a constant. The result is a complex number represented as a pair of real and
    imaginary parts. Raises a :class:`ValueError` if the AST contains variables.

    >>> (1 + 2 * I).accept(ConstantEvaluator())
    (mpq(1,1), mpq(2,1))
    """

    def add(self, a: tuple[mpq, mpq], b: tuple[mpq, mpq]) -> tuple[mpq, mpq]:
        """Add two complex numbers represented as pairs of real and imaginary
        parts. Implements the abstract method :meth:`.ArithmeticEvaluator.add`.

        >>> ConstantEvaluator().add((mpq(1), mpq(2)), (mpq(3), mpq(4)))
        (mpq(4,1), mpq(6,1))
        """
        return a[0] + b[0], a[1] + b[1]

    def neg(self, a: tuple[mpq, mpq]) -> tuple[mpq, mpq]:
        """Negate a complex number represented as a pair of real and imaginary
        parts. Implements the abstract method :meth:`.ArithmeticEvaluator.neg`.

        >>> ConstantEvaluator().neg((mpq(1), mpq(2)))
        (mpq(-1,1), mpq(-2,1))
        """
        return -a[0], -a[1]

    def mul(self, a: tuple[mpq, mpq], b: tuple[mpq, mpq]) -> tuple[mpq, mpq]:
        """Multiply two complex numbers represented as pairs of real and
        imaginary parts. Implements the abstract method
        :meth:`.ArithmeticEvaluator.mul`.

        >>> ConstantEvaluator().mul((mpq(1), mpq(2)), (mpq(3), mpq(4)))
        (mpq(-5,1), mpq(10,1))
        """
        return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]

    def visit_rat(self, num: Rat) -> tuple[mpq, mpq]:
        """Evaluate a rational number. Implements the abstract method
        :meth:`.ASTVisitor.visit_rat`.

        >>> ConstantEvaluator().visit_rat(Rat(mpq(1, 2)))
        (mpq(1,2), mpq(0,1))
        """
        return num.value, mpq(0)

    def visit_i(self, i: _I) -> tuple[mpq, mpq]:
        """Evaluate the imaginary unit. Implements the abstract method
        :meth:`.ASTVisitor.visit_i`.

        >>> ConstantEvaluator().visit_i(I)
        (mpq(0,1), mpq(1,1))
        """
        return mpq(0), mpq(1)

    def visit_var(self, var: Var) -> tuple[mpq, mpq]:
        """Raise a :class:`ValueError` since variables cannot be evaluated to
        constants. Implements the abstract method
        :meth:`.ASTVisitor.visit_var`.

        >>> x = Var('x')
        >>> ConstantEvaluator().visit_var(x)
        Traceback (most recent call last):
          ...
        ValueError: Cannot evaluate variable x
        """
        raise ValueError(f'Cannot evaluate variable {var}')

    def visit_conj(self, conj: Conj) -> tuple[mpq, mpq]:
        """Evaluate a complex conjugation. Implements the abstract method
        :meth:`.ASTVisitor.visit_conj`.

        >>> ConstantEvaluator().visit_conj(Conj(1 + 2 * I))
        (mpq(1,1), mpq(-2,1))
        """
        a, b = conj.arg.accept(self)
        return a, -b

    def visit_re(self, re: Re) -> tuple[mpq, mpq]:
        """Evaluate a real part. Implements the abstract method
        :meth:`.ASTVisitor.visit_re`.

        >>> ConstantEvaluator().visit_re(Re(1 + 2 * I))
        (mpq(1,1), mpq(0,1))
        """
        a, _ = re.arg.accept(self)
        return a, mpq(0)

    def visit_im(self, im: Im) -> tuple[mpq, mpq]:
        """Evaluate an imaginary part. Implements the abstract method
        :meth:`.ASTVisitor.visit_im`.

        >>> ConstantEvaluator().visit_im(Im(1 + 2 * I))
        (mpq(2,1), mpq(0,1))
        """
        _, b = im.arg.accept(self)
        return b, mpq(0)


@dataclass
@total_ordering
class AddSortKey:
    """A sort key for canonically ordering sums in :class:`.WeakNormalizer`.
    It compares two AST nodes first by their total degree and then
    lexicographically by their factors.

    >>> x, y = Var('x'), Var('y')
    >>> AddSortKey(x * y) <= AddSortKey(x**2)
    True
    """

    ast: AST
    """The AST node for which this is a sort key.
    """

    total_degree: int
    """The total degree of the AST node.
    """

    factors: list[tuple[AST, int]]
    """The factors of the AST node, each represented as a tuple of the factor
    and its degree.
    """

    def __init__(self, ast: AST) -> None:
        """Initialize the sort key of an AST node.
        """
        if isinstance(ast, Neg):
            ast = ast.arg
        self.ast = ast
        self.total_degree = 0
        self.factors = []
        for factor in self.ast.factors():
            if factor.is_constant():
                continue
            if isinstance(factor, Neg):
                factor = factor.arg
            degree = 1
            if isinstance(factor, Pow):
                degree = factor.exponent
                factor = factor.base
            self.factors.append((factor, degree))
            self.total_degree += degree

    def __le__(self, other: Self) -> bool:
        """Compare the underlying AST nodes first by their total degree and then
        lexicographically by their factors using :class:`.MulSortKey`. The
        remaining comparison operators are derived from this using
        :func:`functools.total_ordering`.

        >>> x, y, z = Var('x'), Var('y'), Var('z')
        >>> AddSortKey(x * y) <= AddSortKey(x**2)
        True
        >>> AddSortKey(x**2) <= AddSortKey(x * y)
        False
        """
        if self.total_degree != other.total_degree:
            return self.total_degree <= other.total_degree
        for i in range(len(self.factors)):
            if i == len(other.factors):
                return False
            if self.factors[i] != other.factors[i]:
                factor1, degree1 = self.factors[i]
                factor2, degree2 = other.factors[i]
                if factor1 != factor2:
                    return not (MulSortKey(factor1) <= MulSortKey(factor2))
                else:
                    return degree1 <= degree2
        return True


@dataclass
@total_ordering
class MulSortKey:
    """A sort key for canonically ordering AST nodes inside products in
    :class:`.WeakNormalizer`. It compares two AST nodes first by their operator
    and then by their arguments and exponent.

    >>> z = Var('z')
    >>> MulSortKey(z**2) <= MulSortKey(z)
    False
    >>> MulSortKey(z) <= MulSortKey(Re(z))
    True
    """

    ast: AST
    """The AST node for which this is a sort key.
    """

    degree: int = 1
    """The exponent of the AST node.
    """

    @property
    def args(self) -> tuple[object, ...]:
        """Return the arguments of the AST node, replacing any AST nodes with
        their corresponding sort keys.
        """
        return tuple(MulSortKey(arg) if isinstance(arg, AST) else arg for arg in self.ast.args)

    def __init__(self, ast: AST) -> None:
        """Initialize the sort key of an AST node. The AST node must not be a
        constant, a negation or a product.
        """
        assert not ast.is_constant()
        assert not isinstance(ast, (Neg, Mul))
        if isinstance(ast, Pow):
            self.ast = ast.base
            self.degree = ast.exponent
        else:
            self.ast = ast
            self.degree = 1

    def __le__(self, other: Self) -> bool:
        """Compare the underlying AST nodes first by their operator and then by
        their arguments and exponent. The remaining comparison operators are
        derived from this using :func:`functools.total_ordering`.

        >>> z = Var('z')
        >>> MulSortKey(z**2) <= MulSortKey(z)
        False
        >>> MulSortKey(z) <= MulSortKey(Re(z))
        True
        """
        ORDER = (Var, Conj, Re, Im, Add)
        assert self.ast.op in ORDER, self.ast.op
        assert other.ast.op in ORDER, other.ast.op
        if self.ast.op == other.ast.op:
            return (self.args, self.degree) <= (other.args, other.degree)
        else:
            return ORDER.index(self.ast.op) <= ORDER.index(other.ast.op)


class WeakNormalizer(IdentityASTVisitor):
    """Visitor that normalizes an AST by rearranging sums and products,
    and applying local simplifications, but not expanding any nodes.
    """

    def visit_add(self, add: Add) -> AST:
        """Normalize a sum by collecting constant terms and
        rearranging non-constant terms in a canonical order according to
        :class:`AddSortKey`.

        >>> x, y, z = Var('x'), Var('y'), Var('z')
        >>> print(WeakNormalizer().visit_add(y + x + z + x - z))
        2 * x + y
        >>> print(WeakNormalizer().visit_add(2 + x - 3))
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
        sorted_products = sorted(products, key=AddSortKey, reverse=True)
        for i, prod in enumerate(sorted_products):
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
        Normalize a product by collecting constant factors and
        rearranging non-constant factors in a canonical order according to
        :class:`MulSortKey`.

        >>> x, y = Var('x'), Var('y')
        >>> print(WeakNormalizer().visit_mul(y * x * y))
        x * y^2
        >>> print(WeakNormalizer().visit_mul(2 * x * -I))
        -2 * i * x
        >>> print(WeakNormalizer().visit_mul(x * Re(x) * Im(x) * Conj(x)))
        x * ~x * Re(x) * Im(x)
        >>> print(WeakNormalizer().visit_mul(Re(x) * Re(y) * (x + y)))
        Re(x) * Re(y) * (x + y)
        >>> print(WeakNormalizer().visit_mul(0 * x))
        0
        >>> print(WeakNormalizer().visit_mul(1 * x))
        x
        >>> print(WeakNormalizer().visit_mul(-1 * x))
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
        result.sort(key=MulSortKey)
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
        """Normalize a power by evaluating it if the base is constant,
        and simplifying if the exponent is :code:`0` or :code:`1`.
        Note that :code:`0 ** 0` is defined to be :code:`1` as for
        :class:`gmpy2.mpq`.

        >>> x, y = Var('x'), Var('y')
        >>> print(WeakNormalizer().visit_pow((x + y)**0))
        1
        >>> print(WeakNormalizer().visit_pow((x + y)**1))
        x + y
        >>> print(WeakNormalizer().visit_pow((x + y)**2))
        (x + y)^2
        >>> print(WeakNormalizer().visit_pow(I**2))
        -1
        >>> print(WeakNormalizer().visit_pow((I - I)**0))
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
        """Normalize a negation by evaluating constants, simplifying double
        negations and moving the negation inside products.

        >>> x, y = Var('x'), Var('y')
        >>> print(WeakNormalizer().visit_neg(-(1 + I)))
        -1 - i
        >>> print(WeakNormalizer().visit_neg(-(-x)))
        x
        >>> print(WeakNormalizer().visit_neg(-(x * y)))
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
        """Normalize a conjugation by evaluating constants and by simplifying
        double conjugations and conjugations of real and imaginary parts.

        >>> x, y = Var('x'), Var('y')
        >>> print(WeakNormalizer().visit_conj(Conj(1 + I)))
        1 - i
        >>> print(WeakNormalizer().visit_conj(Conj(Conj(x))))
        x
        >>> print(WeakNormalizer().visit_conj(Conj(Re(x))))
        Re(x)
        >>> print(WeakNormalizer().visit_conj(Conj(Im(x))))
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
        """Normalize a real part by evaluating constants and by simplifying
        real parts of real parts, imaginary parts and conjugates.

        >>> x = Var('x')
        >>> print(WeakNormalizer().visit_re(Re(1 + I)))
        1
        >>> print(WeakNormalizer().visit_re(Re(Re(x))))
        Re(x)
        >>> print(WeakNormalizer().visit_re(Re(Im(x))))
        Im(x)
        >>> print(WeakNormalizer().visit_re(Re(Conj(x))))
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
        """Normalize an imaginary part by evaluating constants and by simplifying
        imaginary parts of real parts, imaginary parts and conjugates.

        >>> x = Var('x')
        >>> print(WeakNormalizer().visit_im(Im(1 + I)))
        1
        >>> print(WeakNormalizer().visit_im(Im(Re(x))))
        0
        >>> print(WeakNormalizer().visit_im(Im(Im(x))))
        0
        >>> print(WeakNormalizer().visit_im(Im(Conj(x))))
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
    """Visitor based on :class:`.WeakNormalizer` that also
    expands sums and products, and propagates :class:`.ast.Re`, :class:`.ast.Im`
    and :class:`.ast.Conj`.
    """

    def visit_mul(self, mul: Mul) -> AST:
        """Expand a product by distributing it over sums and normalizing the
        factors recursively.

        >>> x, y, z = Var('x'), Var('y'), Var('z')
        >>> print(Normalizer().visit_mul(x * (y + z)))
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
        """Expand powers of sums and products, and normalize them recursively.

        >>> x, y = Var('x'), Var('y')
        >>> print(Normalizer().visit_pow((x + y)**2))
        x^2 + 2 * x * y + y^2
        >>> print(Normalizer().visit_pow((x * y)**2))
        x^2 * y^2
        >>> print(Normalizer().visit_pow((-x)**3))
        -x^3
        """
        node = super().visit_pow(pow)
        if isinstance(node, Pow) and not isinstance(node.base, (Var, Re, Im, Conj)):
            result = Mul(*[node.base] * node.exponent).accept(self)
            return result
        return node

    def visit_neg(self, neg: Neg) -> AST:
        """Expand a negation by distributing it over sums and normalizing the
        argument recursively.

        >>> x, y = Var('x'), Var('y')
        >>> print(Normalizer().visit_neg(-(x + y)))
        -x - y
        """
        node = super().visit_neg(neg)
        if isinstance(node, Neg) and isinstance(node.arg, Add):
            return Add(*map(Neg, node.arg.args)).accept(self)
        return node

    def visit_conj(self, conj: Conj) -> AST:
        """Propagate a conjugation by distributing it over sums and products,
        and normalizing the argument recursively.

        >>> x, y = Var('x'), Var('y')
        >>> print(Normalizer().visit_conj(Conj(x + y)))
        ~x + ~y
        >>> print(Normalizer().visit_conj(Conj(x * y)))
        ~x * ~y
        >>> print(Normalizer().visit_conj(Conj(-x)))
        -~x
        >>> print(Normalizer().visit_conj(Conj(x**2)))
        (~x)^2
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
        """Propagate a real part by distributing it over sums and products,
        and normalizing the argument recursively.

        >>> x, y = Var('x'), Var('y')
        >>> print(Normalizer().visit_re(Re(x + y)))
        Re(x) + Re(y)
        >>> print(Normalizer().visit_re(Re(x * y)))
        Re(x) * Re(y) - Im(x) * Im(y)
        >>> print(Normalizer().visit_re(Re(-x)))
        -Re(x)
        >>> print(Normalizer().visit_re(Re(x**2)))
        Re(x)^2 - Im(x)^2
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
        """Propagate an imaginary part by distributing it over sums and
        products, and normalizing the argument recursively.

        >>> x, y = Var('x'), Var('y')
        >>> print(Normalizer().visit_im(Im(x + y)))
        Im(x) + Im(y)
        >>> print(Normalizer().visit_im(Im(x * y)))
        Re(x) * Im(y) + Re(y) * Im(x)
        >>> print(Normalizer().visit_im(Im(-x)))
        -Im(x)
        >>> print(Normalizer().visit_im(Im(x**2)))
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


class ConjugateNormalizer(Normalizer):
    """Visitor based on :class:`.Normalizer` that additionally replaces all
    occurrences of :class:`.Re` and :class:`.Im`. This yields a unique
    normal form.

    >>> z = Var('z')
    >>> normalizer = ConjugateNormalizer()
    >>> print((Re(z) + I * Im(z)).accept(normalizer))
    z
    >>> print((Re(z)**2 + Im(z)**2).accept(normalizer))
    z * ~z
    """

    def visit_re(self, re: Re) -> AST:
        """Replace a real part with its equivalent expression in
        terms of its argument and its conjugate.

        >>> z = Var('z')
        >>> print(ConjugateNormalizer().visit_re(Re(z)))
        1/2 * z + 1/2 * ~z
        """
        return ((re.arg + Conj(re.arg)) / 2).accept(self)

    def visit_im(self, im: Im) -> AST:
        """Replace an imaginary part with its equivalent
        expression in terms of its argument and its conjugate.

        >>> z = Var('z')
        >>> print(ConjugateNormalizer().visit_im(Im(z)))
        -1/2 * i * z + 1/2 * i * ~z
        """
        return ((im.arg - Conj(im.arg)) / (2 * I)).accept(self)


def conjugate_normal_form(ast: AST) -> AST:
    """Return the conjugate normal form of an AST which is a polynomial
    expression in the variables and their conjugates. It is a unique
    normal form.

    >>> z = Var('z')
    >>> print(conjugate_normal_form(z))
    z
    >>> print(conjugate_normal_form(Re(z)))
    1/2 * z + 1/2 * ~z
    """
    return ast.accept(ConjugateNormalizer())


def cartesian_normal_form(ast: AST) -> AST:
    """Return the Cartesian normal form of an AST which is of the form
    :code:`f + I * g` where :code:`f` and :code:`g` are polynomial expressions
    in the real and imaginary parts of variables. It is a unique normal form.

    >>> z = Var('z')
    >>> print(cartesian_normal_form(z))
    Re(z) + i * Im(z)
    >>> print(cartesian_normal_form(z**2))
    Re(z)^2 - Im(z)^2 + i * 2 * Re(z) * Im(z)
    """
    normalizer = Normalizer()
    re = Re(ast).accept(normalizer)
    im = Im(ast).accept(normalizer)
    if isinstance(re, Rat) and re.value == 0 and isinstance(im, Rat) and im.value == 0:
        return Rat(0)
    if isinstance(re, Rat) and re.value == 0:
        return I * im
    if isinstance(im, Rat) and im.value == 0:
        return re
    return re + I * im
