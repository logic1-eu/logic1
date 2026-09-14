from logic1.theories.Complex.ast import Var
from logic1.theories.Complex.normalize import WeakNormalizer
from logic1.theories.Complex.term import VV


def test_regression_normalform_neg():
    """Regression test for for correct handling of negated terms when
    normalizing products in :meth:`WeakNormalizer.visit_mul`.
    """
    z = Var("z")
    a = (z * (-z)).accept(WeakNormalizer())
    b = (-(z**2)).accept(WeakNormalizer())
    assert a == b

def test_regression_nested_powers():
    """Regression test for correct handling of nested powers in
    :meth:`WeakNormalizer.visit_pow`.
    """
    z = Var("z")
    a = ((z**2)**3).accept(WeakNormalizer())
    b = (z**6).accept(WeakNormalizer())
    assert a == b

    x, y = VV.get('x', 'y')
    assert (-x**2)**2 == x**4
    assert (-x**2 * y)**2 == x**4 * y**2
