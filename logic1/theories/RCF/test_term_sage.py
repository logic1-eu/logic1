import pytest

from logic1.theories.RCF.term import *

def test_regression_variable_init():
    with pytest.raises(NotImplementedError):
        Variable("x")

def test_regression_wrong_variable_order():
    x2, x10 = VV.get('x2', 'x10')
    p = x2 + 2*x10
    assert p.lc() == mpq(1,1)
    assert str(p) == 'x2 + 2*x10'
    VV.fresh()
    assert p.lc() == mpq(1,1)
    assert str(p) == 'x2 + 2*x10'