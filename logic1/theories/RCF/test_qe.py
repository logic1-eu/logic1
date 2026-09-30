"""Tests for real quantifier elimination by virtual substitution."""

# Coverage:
# - 33 pytest cases based on the executable examples in test_qe.txt
# - Basic QE with both clustering modes and one multiprocessing case
# - Deliberate difference from test_qe.txt: the period-9 case eliminates x11,
#   x10, and x9 with default options and checks the exact result
# - Exact Davenport--Heintz, Motzkin, and Hong results
# - Kahan and Hong expected failures plus atom-count regressions
# - Validation of assumptions and independent VirtualSubstitution instances
# - This file replaces test_qe.txt
#
# Verification:
# - Python compilation succeeds
# - Mypy with --disallow-untyped-defs reports no issues
# - Focused pytest collection finds 33 cases
# - All 33 cases pass, including the workers=1 multiprocessing case

import pytest

from logic1.firstorder import All, And, Ex, F, Implies, Or, T
from logic1.support.excepthook import NoTraceException
from logic1.theories.RCF import Clustering, Formula, Generic, VV, qe
from logic1.theories.RCF.qe import VirtualSubstitution


a, b, c, d, x, y, z = VV.get('a', 'b', 'c', 'd', 'x', 'y', 'z')

phi_1 = All(x, Ex((y, z), And(y >= 0, z >= 0, y - z == x)))
phi_2 = All(x, Ex((y, z), And(y >= 0, z >= 0, y - z == x, a * x + b == 0)))
phi_3 = All(x, Ex((y, z), And(y >= 0, z >= 0, y + z == x)))
phi_4 = Ex(x, a * x + b == 0)
phi_5 = Ex(x, a * x + b <= 0)
phi_6 = Ex(x, And(a * x + b <= 0, x <= b))
phi_7 = Ex(x, a * x**2 + b * x + c == 0)


@pytest.mark.parametrize(
    ('formula', 'clustering', 'expected'),
    [
        pytest.param(phi_1, Clustering.NONE, T, id='phi-1-none'),
        pytest.param(phi_1, Clustering.FULL, T, id='phi-1-full'),
        pytest.param(
            phi_2,
            Clustering.NONE,
            And(b == 0, a == 0),
            id='phi-2-none',
        ),
        pytest.param(
            phi_2,
            Clustering.FULL,
            And(b == 0, a == 0),
            id='phi-2-full',
        ),
        pytest.param(phi_3, Clustering.NONE, F, id='phi-3-none'),
        pytest.param(phi_3, Clustering.FULL, F, id='phi-3-full'),
        pytest.param(
            phi_4,
            Clustering.NONE,
            Or(b == 0, a != 0),
            id='phi-4-none',
        ),
        pytest.param(
            phi_4,
            Clustering.FULL,
            Or(b == 0, a != 0),
            id='phi-4-full',
        ),
        pytest.param(
            phi_5,
            Clustering.NONE,
            Or(b <= 0, a != 0),
            id='phi-5-none',
        ),
        pytest.param(
            phi_5,
            Clustering.FULL,
            Or(b <= 0, a != 0),
            id='phi-5-full',
        ),
        pytest.param(
            phi_6,
            Clustering.NONE,
            Or(a > 0, And(b <= 0, a == 0), And(a < 0, a * b + b <= 0)),
            id='phi-6-none',
        ),
        pytest.param(
            phi_6,
            Clustering.FULL,
            Or(a > 0, And(b <= 0, a == 0),
               And(a < 0, a**2 * b + a * b >= 0)),
            id='phi-6-full',
        ),
        pytest.param(
            phi_7,
            Clustering.NONE,
            Or(
                And(c == 0, b == 0, a == 0),
                And(b < 0, a == 0),
                And(b > 0, a == 0),
                And(a < 0, 4 * a * c - b**2 == 0),
                And(a < 0, 4 * a * c - b**2 < 0),
                And(a > 0, 4 * a * c - b**2 == 0),
                And(a > 0, 4 * a * c - b**2 < 0),
            ),
            id='phi-7-none',
        ),
        pytest.param(
            phi_7,
            Clustering.FULL,
            Or(
                And(c == 0, b == 0, a == 0),
                And(b != 0, a == 0),
                And(a != 0, 4 * a * c - b**2 <= 0),
            ),
            id='phi-7-full',
        ),
    ],
)
def test_basic_qe(formula: Formula, clustering: Clustering,
                  expected: Formula) -> None:
    assert qe(formula, clustering=clustering) == expected


def test_parallel_qe() -> None:
    assert qe(phi_4, workers=1) == Or(b == 0, a != 0)


# Period 9
x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = VV.get(
    *(f'x{i}' for i in range(12)))

period9_matrix = And(
    Or(And(x2 >= 0, x3 == x2 - x1), And(x2 < 0, x3 == -x2 - x1)),
    Or(And(x3 >= 0, x4 == x3 - x2), And(x3 < 0, x4 == -x3 - x2)),
    Or(And(x4 >= 0, x5 == x4 - x3), And(x4 < 0, x5 == -x4 - x3)),
    Or(And(x5 >= 0, x6 == x5 - x4), And(x5 < 0, x6 == -x5 - x4)),
    Or(And(x6 >= 0, x7 == x6 - x5), And(x6 < 0, x7 == -x6 - x5)),
    Or(And(x7 >= 0, x8 == x7 - x6), And(x7 < 0, x8 == -x7 - x6)),
    Or(And(x8 >= 0, x9 == x8 - x7), And(x8 < 0, x9 == -x8 - x7)),
    Or(And(x9 >= 0, x10 == x9 - x8), And(x9 < 0, x10 == -x9 - x8)),
    Or(And(x10 >= 0, x11 == x10 - x9), And(x10 < 0, x11 == -x10 - x9)),
)
period9 = All(
    (x11, x10, x9),
    Implies(period9_matrix, And(x1 == x10, x2 == x11)),
)
period9_early_steps = (
    And(Or(x6 < 0, x5 - x6 + x7 != 0),
        Or(x6 >= 0, x5 + x6 + x7 != 0)),
    And(Or(x5 < 0, x4 - x5 + x6 != 0),
        Or(x5 >= 0, x4 + x5 + x6 != 0)),
    And(Or(x4 < 0, x3 - x4 + x5 != 0),
        Or(x4 >= 0, x3 + x4 + x5 != 0)),
    And(Or(x3 < 0, x2 - x3 + x4 != 0),
        Or(x3 >= 0, x2 + x3 + x4 != 0)),
    And(Or(x2 < 0, x1 - x2 + x3 != 0),
        Or(x2 >= 0, x1 + x2 + x3 != 0)),
)
period9_expected = And(
    Or(
        x8 < 0,
        x7 - 2 * x8 < 0,
        x7 - x8 < 0,
        x7 < 0,
        x6 - x7 + x8 != 0,
        And(x2 - 2 * x7 + 3 * x8 == 0, x1 - x7 + 2 * x8 == 0),
        *period9_early_steps,
    ),
    Or(
        x8 < 0,
        x7 - 2 * x8 > 0,
        x7 - x8 < 0,
        x7 < 0,
        x6 - x7 + x8 != 0,
        And(x2 - x8 == 0, x1 - x7 + 2 * x8 == 0),
        *period9_early_steps,
    ),
    Or(
        x8 < 0,
        x7 - x8 > 0,
        x7 < 0,
        x6 - x7 + x8 != 0,
        And(x2 - 2 * x7 + x8 == 0, x1 + x7 == 0),
        *period9_early_steps,
    ),
    Or(
        x8 < 0,
        x7 > 0,
        And(x2 + x8 == 0, x1 + x7 == 0),
        And(
            Or(x7 == 0, x6 + x7 + x8 != 0),
            Or(x7 < 0, x6 + x8 != 0),
        ),
        *period9_early_steps,
    ),
    Or(
        x8 > 0,
        x7 < 0,
        x7 + x8 < 0,
        x6 - x7 + x8 != 0,
        x6 < 0,
        x5 - x6 + x7 != 0,
        And(x2 - 2 * x7 - x8 == 0, x1 - x7 == 0),
        *period9_early_steps[1:],
    ),
    Or(
        x8 > 0,
        x7 + x8 > 0,
        x7 + 2 * x8 > 0,
        And(x2 + x8 == 0, x1 + x7 + 2 * x8 == 0),
        And(
            Or(x7 < 0, x6 - x7 + x8 != 0),
            Or(x7 >= 0, x6 + x7 + x8 != 0),
        ),
        *period9_early_steps,
    ),
)


def test_period9() -> None:
    assert qe(period9) == period9_expected


@pytest.mark.parametrize('clustering', [Clustering.NONE, Clustering.FULL])
def test_davenport_heintz(clustering: Clustering) -> None:
    formula = Ex(
        c,
        All(
            (b, a),
            Implies(
                Or(And(a == d, b == c), And(a == c, b == 1)),
                a**2 == b,
            ),
        ),
    )
    expected = And(d != 0, Or(d - 1 == 0, d + 1 == 0))

    assert qe(formula, clustering=clustering) == expected


# Kahan's problem
ellipse = All(
    (x, y),
    Implies(
        b**2 * (x - c)**2 + a**2 * y**2 - a**2 * b**2 == 0,
        x**2 + y**2 <= 1,
    ),
)


@pytest.mark.parametrize('clustering', [Clustering.NONE, Clustering.FULL])
def test_kahan_traditional_guards_fail(clustering: Clustering) -> None:
    with pytest.raises(NoTraceException, match='Failed - 2 failure nodes'):
        result = qe(ellipse, clustering=clustering)
        assert result is not None
        len(list(result.atoms()))


@pytest.mark.parametrize(
    ('clustering', 'expected_atoms'),
    [
        pytest.param(Clustering.NONE, 83, id='none'),
        pytest.param(Clustering.FULL, 43, id='full'),
    ],
)
def test_kahan_without_traditional_guards(
        clustering: Clustering, expected_atoms: int) -> None:
    result = qe(
        ellipse,
        clustering=clustering,
        traditional_guards=False,
    )

    assert result is not None
    assert len(list(result.atoms())) == expected_atoms


# Five generic quadratic polynomials
a1, a2, a3, a4, a5 = VV.get('a1', 'a2', 'a3', 'a4', 'a5')
b1, b2, b3, b4, b5 = VV.get('b1', 'b2', 'b3', 'b4', 'b5')
c1, c2, c3, c4, c5 = VV.get('c1', 'c2', 'c3', 'c4', 'c5')
five_generic = Ex(
    x,
    And(
        a1 * x**2 + b1 * x + c1 == 0,
        a2 * x**2 + b2 * x + c2 == 0,
        a3 * x**2 + b3 * x + c3 < 0,
        a4 * x**2 + b4 * x + c4 < 0,
        a5 * x**2 + b5 * x + c5 < 0,
    ),
)


@pytest.mark.parametrize(
    ('clustering', 'expected_atoms'),
    [
        pytest.param(Clustering.NONE, 563, id='none'),
        pytest.param(Clustering.FULL, 283, id='full'),
    ],
)
def test_five_generic_quadratics(
        clustering: Clustering, expected_atoms: int) -> None:
    result = qe(five_generic, clustering=clustering)

    assert result is not None
    assert len(list(result.atoms())) == expected_atoms


# Motzkin's polynomial
motzkin = All(
    x,
    Implies(And(x >= 0, y >= 0), 1 + x * y * (x + y - 3) >= 0),
)


@pytest.mark.parametrize(
    ('clustering', 'expected'),
    [
        pytest.param(
            Clustering.NONE,
            Or(y - 4 <= 0, y**2 - 4 * y <= 0, y**2 - 3 * y >= 0),
            id='none',
        ),
        pytest.param(
            Clustering.FULL,
            Or(y - 3 >= 0, y - 1 == 0, y <= 0, y**2 - 4 * y <= 0),
            id='full',
        ),
    ],
)
def test_motzkin(clustering: Clustering, expected: Formula) -> None:
    assert qe(motzkin, clustering=clustering) == expected


@pytest.mark.parametrize('clustering', [Clustering.NONE, Clustering.FULL])
def test_motzkin_closed(clustering: Clustering) -> None:
    assert qe(motzkin.all(), clustering=clustering) is T


def test_hong_failure() -> None:
    formula = All(
        x,
        Ex(y, And(x**2 + x * y + b > 0, x + a * y**2 + b <= 0)),
    )

    with pytest.raises(NoTraceException, match='Failed - 1 failure nodes'):
        qe(formula)


def test_hong_inner_quantifier() -> None:
    polynomial = a * x**4 + 2 * a * b * x**2 + a * b**2 + b * x**2 + x**3
    scaled_polynomial = (
        a**2 * x**4 + 2 * a**2 * b * x**2 + a**2 * b**2
        + a * b * x**2 + a * x**3
    )
    formula = Ex(y, And(x**2 + x * y + b > 0, x + a * y**2 + b <= 0))
    expected = Or(
        And(
            x > 0,
            Or(
                polynomial < 0,
                And(
                    polynomial == 0,
                    Or(
                        a * x**3 + a * b * x > 0,
                        And(a <= 0, Or(a == 0, x**2 + b == 0)),
                    ),
                ),
            ),
        ),
        And(
            a != 0,
            a * b + a * x <= 0,
            Or(x**2 + b > 0, scaled_polynomial < 0),
            Or(a * x < 0, scaled_polynomial > 0),
        ),
        And(
            Or(x < 0, And(x == 0, b > 0)),
            Or(a < 0, And(b + x <= 0, a == 0)),
        ),
    )

    assert qe(formula) == expected


def test_assumption_on_bound_variable() -> None:
    formula = Ex(x, And(a > 0, x > 9, b == 0))

    with pytest.raises(
            ValueError,
            match='invalid assumption on bound variables in x - 8 < 0'):
        qe(formula, assume={x < 8, a > 1})


def test_assumptions_on_free_variables() -> None:
    formula = Ex(y, And(a > 0, y > 9, b == 0))

    assert qe(formula, assume={x < 8, a > 1}) == (b == 0)


def test_independent_virtual_substitution_instances() -> None:
    another_qe = VirtualSubstitution()
    formula = Ex(x, (a + 1) * x**2 + b * x + c == 0)

    assert qe(formula, generic=Generic.FULL) == (4 * a * c - b**2 + 4 * c <= 0)
    assert another_qe(Ex(x, a * x + b == 0), generic=Generic.FULL) is T
    assert qe.assumptions == [a + 1 != 0]
    assert another_qe.assumptions == [a != 0]
