from logic1.theories.Complex.ast import Var
from logic1.theories.Complex.normalize import WeakNormalizer


def test_regression_normalform_neg():
    """Regression test for for correct handling of negated terms when
    normalizing products in :meth:`WeakNormalizer.visit_mul`.
    """
    z = Var("z")
    a = (z * (-z)).accept(WeakNormalizer())
    b = (-(z**2)).accept(WeakNormalizer())
    assert a == b