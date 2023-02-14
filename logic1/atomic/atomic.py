import sympy
from ..formulas import BinaryAtomicFormula


class Eq(BinaryAtomicFormula):
    """
    >>> from sympy.abc import x
    >>> Eq(x, x)
    Eq(x, x)
    """
    _text_symbol = '='
    _latex_symbol = '='

    _sympy_func = sympy.Eq

    @staticmethod
    def dualize(conditional: bool = True):
        if conditional:
            return Ne
        return Eq

    def __init__(self, lhs, rhs):
        self.func = Eq
        self.args = (lhs, rhs)


EQ = Eq.interactive_new


class Ne(BinaryAtomicFormula):
    """
    >>> Ne(1, 0)
    Ne(1, 0)
    """
    _text_symbol = '!='
    _latex_symbol = '\\neq'

    _sympy_func = sympy.Ne

    @staticmethod
    def dualize(conditional: bool = True):
        if conditional:
            return Eq
        return Ne

    def __init__(self, lhs, rhs):
        self.func = Ne
        self.args = (lhs, rhs)


NE = Ne.interactive_new


class Ge(BinaryAtomicFormula):

    _text_symbol = '>='
    _latex_symbol = '\\geq'

    _sympy_func = sympy.Ge

    def __init__(self, lhs, rhs):
        self.func = Ge
        self.args = (lhs, rhs)


GE = Ge.interactive_new


class Le(BinaryAtomicFormula):

    _text_symbol = '<='
    _latex_symbol = '\\leq'

    _sympy_func = sympy.Le

    def __init__(self, lhs, rhs):
        self.func = Le
        self.args = (lhs, rhs)


LE = Le.interactive_new


class Gt(BinaryAtomicFormula):

    _text_symbol = '>'
    _latex_symbol = '>'

    _sympy_func = sympy.Gt

    def __init__(self, lhs, rhs):
        self.func = Gt
        self.args = (lhs, rhs)


GT = Gt.interactive_new


class Lt(BinaryAtomicFormula):

    _text_symbol = '<'
    _latex_symbol = '<'

    _sympy_func = sympy.Lt

    def __init__(self, lhs, rhs):
        self.func = Lt
        self.args = (lhs, rhs)


LT = Lt.interactive_new
