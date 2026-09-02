from logic1.firstorder import *


def test_regression_traverse_implies_not():
    """Regression test for :class:`Implies` and :class:`Not` in
    :meth:`.Formula.traverse`.
    """
    assert Not(T).traverse() == Not(T)
    assert Implies(T, T).traverse() == Implies(T, T)


def test_regression_sort_quantifiers():
    """Regression test for :meth:`.QuantifiedFormula.__le__`

    Comparison of the quantified variables threw an error in :mod:`Sets` and
    :mod:`Complex` since variables were not comparable.
    """
    from logic1.theories import Complex

    x1, y1 = Complex.VV.get('sortx', 'sorty')
    result1 = sorted([All(x1, T), All(y1, T)])
    assert list(result1) == [All(x1, T), All(y1, T)]

    from logic1.theories import Sets

    x2, y2 = Sets.VV.get('x', 'y')
    result2 = sorted([All(x2, T), All(y2, T)])
    assert list(result2) == [All(x2, T), All(y2, T)]


def test_regression_pnf_negated_atom():
    """Regression test for :meth:`.Formula.to_pnf` on negated atoms in NNF.

    Treatment of explicitly negated atoms in NNF was missing in earlier
    versions.
    """
    from logic1.theories.RCF import Eq, VV

    x, y = VV.get('pnfx', 'pnfy')
    f = Not(Eq(x, y))
    assert f.to_nnf(to_positive=False) == f
    assert f.to_pnf(is_nnf=True) == f