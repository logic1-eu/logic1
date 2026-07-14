from typing import ClassVar

from typing import ClassVar

from gmpy2 import mpz

from logic1.theories.Complex.ast import (
    Add, AST, ASTVisitor, Conj, _I, Im, Mul, Neg, Pow, Re, Var, Rat)


class ReprFormatter(ASTVisitor[str]):
    """Formatter for AST nodes that produces a more human-readable string
    representation that is valid Python code and allows for the
    reconstruction of the original expression.

    >>> from logic1.theories.Complex.ast import *
    >>> z = Var('z')
    >>> (z**3 + 2 * I).accept(ReprFormatter())
    'z**3 + 2 * I'
    """

    symbols: ClassVar[dict[type[AST], str]] = {}
    """Mapping of AST node types to their corresponding symbols used in the
    string representation. This mapping can be overridden in subclasses to
    customize the symbols.
    """

    def _omit_mul_symbol(self, ast1: AST, ast2: AST) -> bool:
        """Return :obj:`True` if the multiplication symbol should be omitted
        between the two given AST nodes. This method can be overridden in
        subclasses to customize the behavior.

        >>> from logic1.theories.Complex.ast import *
        >>> z = Var('z')
        >>> ReprFormatter()._omit_mul_symbol(z, z + 1)
        False
        """
        return False

    def visit_rat(self, num: Rat) -> str:
        """Return the string representation of a rational number.

        >>> from logic1.theories.Complex.ast import *
        >>> ReprFormatter().visit_rat(Rat(mpq(3, 4)))
        '3/4'
        """
        return str(num.value)

    def visit_i(self, _: _I) -> str:
        """Return the string representation of the imaginary unit.

        >>> from logic1.theories.Complex.ast import *
        >>> ReprFormatter().visit_i(I)
        'I'
        """
        return self.symbols.get(_I, 'I')

    def visit_var(self, var: Var) -> str:
        """Return the string representation of a variable.

        >>> from logic1.theories.Complex.ast import *
        >>> z = Var('z')
        >>> ReprFormatter().visit_var(z)
        'z'
        """
        return var.name

    def visit_add(self, add: Add) -> str:
        """Return the string representation of an addition.

        >>> from logic1.theories.Complex.ast import *
        >>> z = Var('z')
        >>> ReprFormatter().visit_add(z + 1 - I)
        'z + 1 - I'
        """
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
        """Return the string representation of a multiplication.

        >>> from logic1.theories.Complex.ast import *
        >>> z = Var('z')
        >>> ReprFormatter().visit_mul(z * (z + 1))
        'z * (z + 1)'
        """
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
        """Return the string representation of a power.

        >>> from logic1.theories.Complex.ast import *
        >>> z = Var('z')
        >>> ReprFormatter().visit_pow(z**2)
        'z**2'
        """
        symbol = self.symbols.get(Pow, '**')
        if isinstance(pow.base, (Add, Mul, Neg, Pow, Conj)):
            return f'({pow.base.accept(self)}){symbol}{pow.exponent}'
        return f'{pow.base.accept(self)}{symbol}{pow.exponent}'

    def visit_neg(self, neg: Neg) -> str:
        """Return the string representation of a negation.

        >>> from logic1.theories.Complex.ast import *
        >>> z = Var('z')
        >>> ReprFormatter().visit_neg(-z)
        '-z'
        """
        symbol = self.symbols.get(Neg, '-')
        if isinstance(neg.arg, (Add, Mul)):
            return f'{symbol}({neg.arg.accept(self)})'
        return f'{symbol}{neg.arg.accept(self)}'

    def visit_conj(self, conj: Conj) -> str:
        """Return the string representation of a conjugation.

        >>> from logic1.theories.Complex.ast import *
        >>> z = Var('z')
        >>> ReprFormatter().visit_conj(~z)
        '~z'
        """
        symbol = self.symbols.get(Conj, '~')
        if isinstance(conj.arg, (Add, Mul)):
            return f'{symbol}({conj.arg.accept(self)})'
        return f'{symbol}{conj.arg.accept(self)}'

    def visit_re(self, re: Re) -> str:
        """Return the string representation of a real part.

        >>> from logic1.theories.Complex.ast import *
        >>> z = Var('z')
        >>> ReprFormatter().visit_re(Re(z))
        'Re(z)'
        """
        symbol = self.symbols.get(Re, 'Re')
        return f'{symbol}({re.arg.accept(self)})'

    def visit_im(self, im: Im) -> str:
        """Return the string representation of an imaginary part.

        >>> from logic1.theories.Complex.ast import *
        >>> z = Var('z')
        >>> ReprFormatter().visit_im(Im(z))
        'Im(z)'
        """
        symbol = self.symbols.get(Im, 'Im')
        return f'{symbol}({im.arg.accept(self)})'


class StrFormatter(ReprFormatter):
    """Formatter for AST nodes that produces a more human-readable string
    representation but does not necessarily allow the reconstruction
    of the original expression.

    >>> from logic1.theories.Complex.ast import *
    >>> z = Var('z')
    >>> (z**3 + 2 * I).accept(StrFormatter())
    'z^3 + 2 * i'
    """

    symbols = {
        _I: 'i',
        Pow: '^'
    }
    """Custom mapping of AST node types to their corresponding symbols."""


class LatexFormatter(ReprFormatter):
    """Formatter for AST nodes that produces a LaTeX representation.

    >>> from logic1.theories.Complex.ast import *
    >>> z = Var('z')
    >>> (z**3 + 2 * I).accept(LatexFormatter())
    'z^{3} + 2 i'
    """

    symbols = {
        _I: 'i',
        Mul: '\\cdot',
        Re: '\\Re',
        Im: '\\Im',
    }
    """Custom mapping of AST node types to their corresponding symbols."""

    def _omit_mul_symbol(self, ast1: AST, ast2: AST) -> bool:
        """Return :obj:`True` if the multiplication symbol should be omitted
        between the two given AST nodes.

        >>> from logic1.theories.Complex.ast import *
        >>> z = Var('z')
        >>> LatexFormatter()._omit_mul_symbol(z, z + 1)
        False
        >>> LatexFormatter()._omit_mul_symbol(Rat(mpq(2)), I)
        True
        """
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
        """Return the LaTeX representation of a rational number as integer or
        fraction.

        >>> from logic1.theories.Complex.ast import *
        >>> LatexFormatter().visit_rat(Rat(mpq(2)))
        '2'
        >>> LatexFormatter().visit_rat(Rat(mpq(3, 4)))
        '\\\\frac{3}{4}'
        """
        a = num.value.numerator
        b = num.value.denominator
        if a == mpz(0) or b == mpz(1):
            return str(a)
        else:
            return f'\\frac{{{str(a)}}}{{{str(b)}}}'

    def visit_var(self, var: Var) -> str:
        """Return the LaTeX representation of a variable.

        >>> from logic1.theories.Complex.ast import *
        >>> LatexFormatter().visit_var(Var('z'))
        'z'
        >>> LatexFormatter().visit_var(Var('z1'))
        'z_{1}'
        >>> LatexFormatter().visit_var(Var('z_re'))
        'z_{re}'
        """
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
        """Return the LaTeX representation of a conjugation.

        >>> from logic1.theories.Complex.ast import *
        >>> z = Var('z')
        >>> LatexFormatter().visit_conj(~z)
        '\\\\overline{z}'
        """
        return f'\\overline{{{conj.arg.accept(self)}}}'

    def visit_pow(self, pow: Pow) -> str:
        """Return the LaTeX representation of a power.

        >>> from logic1.theories.Complex.ast import *
        >>> z = Var('z')
        >>> LatexFormatter().visit_pow(z**2)
        'z^{2}'
        """
        if isinstance(pow.base, (Add, Mul, Neg, Pow)):
            return f'({pow.base.accept(self)})^{{{pow.exponent}}}'
        return f'{pow.base.accept(self)}^{{{pow.exponent}}}'
