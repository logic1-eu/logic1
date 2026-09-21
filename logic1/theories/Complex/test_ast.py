from logic1.firstorder import *
from logic1.theories.Complex.ast import *


def test_from_real_imag():
    assert AST.from_real_imag(mpq(2), mpq(0)) == Rat(2)
    assert AST.from_real_imag(mpq(0), mpq(1)) == I
    assert AST.from_real_imag(mpq(0), mpq(-1)) == Neg(I)
    assert AST.from_real_imag(mpq(0), mpq(3)) == Mul(Rat(3), I)
    assert AST.from_real_imag(mpq(2), mpq(1)) == Add(Rat(2), I)
    assert AST.from_real_imag(mpq(2), mpq(-1)) == Add(Rat(2), Neg(I))
    assert AST.from_real_imag(mpq(2), mpq(3)) == Add(Rat(2), Mul(Rat(3), I))

def test_from_number():
    assert AST.from_number(2) == Rat(2)
    assert AST.from_number(3.5) == Rat(mpq(7, 2))
    assert AST.from_number(Fraction(1, 3)) == Rat(mpq(1, 3))
    assert AST.from_number(mpq(1, 4)) == Rat(mpq(1, 4))
    assert AST.from_number(2 + 3j) == Add(Rat(2), Mul(Rat(3), I))
    try:
        AST.from_number("x")
    except ValueError as e:
        assert str(e) == "expected one of int, float, Fraction, mpq, complex; x is <class 'str'>"
