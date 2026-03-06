from gmpy2 import mpq
import pytest

from logic1.theories.Complex.atomic import Term, I, Re, Im, VV


def test_eval_constant():
    
    assert Term.from_number(5).eval_constant() == (mpq(5), mpq(0))

    assert (-I).eval_constant() == (mpq(0), mpq(-1))

    assert (2 + 7 * I).eval_constant() == (mpq(2), mpq(7))

    assert (1 - I / 2).eval_constant() == (mpq(1), mpq(-1, 2))

    assert (I ** 10).eval_constant() == (mpq(-1), mpq(0))

    assert ((3 + I) * (1 - I)).eval_constant() == (mpq(4), mpq(-2))

    assert ((2 + 0 * I) ** 3).eval_constant() == (mpq(8), mpq(0))

    assert Re(9 + 5 * I).eval_constant() == (mpq(9), mpq(0))

    assert Im(9 + 5 * I).eval_constant() == (mpq(5), mpq(0))

    x = VV['x']
    with pytest.raises(ValueError):
        ((1 - x) ** 2).eval_constant()
    

def test_normalize():
    ...  # TODO