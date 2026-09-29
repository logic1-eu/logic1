"""Pytest migration of the former lines 1--166 of ``test_simplify.txt``.

Those lines have been removed from the source file. All 55 executable doctest
checks in that range were migrated:

- 2 DS97 examples;
- 6 negation and constant cases;
- 2 extended-Boolean cases;
- 3 quantifier and assumption cases;
- 7 ``explode_always`` cases;
- 2 ``implicit_ranges`` cases;
- 2 ``lift`` cases;
- 9 ``prefer_order`` cases;
- 9 ``prefer_weak`` cases;
- 2 equality-versus-order preference cases;
- 3 ``substitute`` cases;
- 4 ``is_valid`` cases; and
- 4 internal regression cases.

Expected outputs became assertions, related examples became parametrized
tests, and the explanatory comments and provenance were retained beside the
corresponding tests. Calls that exercised default arguments still omit those
arguments; explicit option values remain explicit. The ``VV.push()`` scopes
are protected by ``try``/``finally`` so failures cannot leak variable state.

The unused ``x0`` through ``x11`` setup from former line 5 was not copied
because none of those variables occurred in a check within the migrated range.
No executable check was omitted. At migration time all 55 pytest cases passed,
and mypy with ``--disallow-untyped-defs`` reported no issues.

The module additionally tests motor series4 with all six combinations of
``implicit_ranges`` and ``substitute`` used by the motor benchmark.
"""

from typing import Any

import pytest

from benchmarks.theories.RCF.motor_series.motor_inputs import series4
from logic1 import All, And, Equivalent, Ex, F, Not, Or, T
from logic1.theories.RCF import AtomicFormula, Eq, Formula, VV, is_valid, simplify


a, b, c, d, w, x, y, z = VV.get('a', 'b', 'c', 'd', 'w', 'x', 'y', 'z')


@pytest.mark.parametrize(
    ('formula', 'expected'),
    [
        pytest.param(
            And(a == 0, Or(b != 0, And(c <= 0, Or(d > 0, a == 0)))),
            And(a == 0, Or(c <= 0, b != 0)),
            id='ds97-section-5.3-example-1',
        ),
        pytest.param(
            And(a == 0, Or(b == 0, And(c == 0, d >= 0)), Or(d != 0, a != 0)),
            And(d != 0, a == 0, Or(b == 0, And(d > 0, c == 0))),
            id='ds97-section-5.3-example-2',
        ),
    ],
)
def test_ds97_examples(formula: Formula, expected: Formula) -> None:
    """Test DS97, Section 5.3 (doi:10.1006/jsco.1997.0123)."""
    assert simplify(formula) == expected


@pytest.mark.parametrize(
    ('formula', 'expected'),
    [
        pytest.param(T, T, id='true'),
        pytest.param(Not(T), F, id='not-true'),
        pytest.param(Eq(1, 0), F, id='false-constant-atom'),
        pytest.param(Not(Eq(1, 0)), T, id='not-false-constant-atom'),
        pytest.param(a == 0, a == 0, id='atom'),
        pytest.param(Not(a == 0), a != 0, id='not-atom'),
    ],
)
def test_implicit_negation(formula: Formula, expected: Formula) -> None:
    assert simplify(formula) == expected


@pytest.mark.parametrize(
    ('formula', 'expected'),
    [
        pytest.param(
            And(a > b, Equivalent(a == 0, b > 0)),
            And(b <= 0, a - b > 0, a != 0),
            id='conjunction',
        ),
        pytest.param(
            Or(a > b, Equivalent(a == 0, b > 0)),
            Or(a - b > 0, And(Or(b <= 0, a == 0), Or(b > 0, a < 0))),
            id='disjunction',
        ),
    ],
)
def test_extended_boolean_operator(formula: Formula, expected: Formula) -> None:
    assert simplify(formula) == expected


def test_quantifier() -> None:
    VV.push()
    try:
        local_a, local_b, _, _ = VV.get('a', 'b', 'c', 'd')
        formula = Or(
            local_a > local_b,
            Equivalent(local_a == 0, Ex(local_a, local_a > local_b)),
        )
        expected = Or(
            local_a - local_b > 0,
            And(
                Or(local_a == 0, All(local_a, local_a - local_b <= 0)),
                Or(local_a != 0, Ex(local_a, local_a - local_b > 0)),
            ),
        )

        assert simplify(formula) == expected
    finally:
        VV.pop()


def test_assumption_with_quantifier() -> None:
    VV.push()
    try:
        local_a, local_b, _, _ = VV.get('a', 'b', 'c', 'd')
        formula = Or(
            local_a > local_b,
            Equivalent(local_a == 0, Ex(local_a, local_a > local_b)),
        )
        expected = Or(local_b < 0, Ex(local_a, local_a - local_b > 0))

        assert simplify(formula, assume=[local_a == 0]) == expected
    finally:
        VV.pop()


def test_assumptions_do_not_affect_bound_variables() -> None:
    formula = Ex(a, And(a > 5, b > 10))

    assert simplify(formula, assume=[a > 10, b > 20]) == Ex(a, a - 5 > 0)


@pytest.mark.parametrize(
    ('formula', 'options', 'expected'),
    [
        pytest.param(
            And(a * b == 0, c == 0),
            {},
            And(c == 0, Or(b == 0, a == 0)),
            id='conjunction-default',
        ),
        pytest.param(
            And(a * b == 0, c == 0),
            {'explode_always': False},
            And(c == 0, a * b == 0),
            id='conjunction-disabled',
        ),
        pytest.param(
            Or(a * b == 0, c == 0),
            {'explode_always': False},
            Or(c == 0, b == 0, a == 0),
            id='disjunction-disabled',
        ),
        pytest.param(
            a * b == 0,
            {},
            Or(b == 0, a == 0),
            id='product-atom-default',
        ),
        pytest.param(
            a * b == 0,
            {'explode_always': False},
            a * b == 0,
            id='product-atom-disabled',
        ),
        pytest.param(
            a**2 + b**2 == 0,
            {},
            And(b == 0, a == 0),
            id='sum-of-squares-default',
        ),
        pytest.param(
            a**2 + b**2 == 0,
            {'explode_always': False},
            a**2 + b**2 == 0,
            id='sum-of-squares-disabled',
        ),
    ],
)
def test_explode_always(formula: Formula, options: dict[str, Any],
                        expected: Formula) -> None:
    assert simplify(formula, **options) == expected


@pytest.mark.parametrize(
    ('implicit_ranges', 'expected'),
    [
        pytest.param(True, And(x - 100 < 0, x - 10 > 0), id='enabled'),
        pytest.param(
            False,
            And(x - 100 < 0, x - 10 > 0, Or(z == 0, x * y**2 + x**2 > 0)),
            id='disabled',
        ),
    ],
)
def test_implicit_ranges(implicit_ranges: bool, expected: Formula) -> None:
    formula = And(10 < x, x < 100, Or(z == 0, And(x * y**2 + x**2 > 0)))

    assert simplify(formula, implicit_ranges=implicit_ranges) == expected


@pytest.mark.parametrize(
    ('lift', 'expected'),
    [
        pytest.param(None, Or(2 * x - 1 == 0, 2 * x + 1 == 0), id='default'),
        pytest.param(False, Or(x - 1 / 2 == 0, x + 1 / 2 == 0), id='disabled'),
    ],
)
def test_lift(lift: bool | None, expected: Formula) -> None:
    formula = 4 * x**2 == 1

    result = simplify(formula) if lift is None else simplify(formula, lift=lift)

    assert result == expected


@pytest.mark.parametrize(
    ('formula', 'prefer_order', 'expected'),
    [
        pytest.param(
            Or(a > 0, And(b == 0, a < 0)),
            True,
            Or(a > 0, And(b == 0, a < 0)),
            id='or-order-input-order-output',
        ),
        pytest.param(
            Or(a > 0, And(b == 0, a != 0)),
            True,
            Or(a > 0, And(b == 0, a < 0)),
            id='or-disequality-input-order-output',
        ),
        pytest.param(
            Or(a > 0, And(b == 0, a < 0)),
            False,
            Or(a > 0, And(b == 0, a != 0)),
            id='or-order-input-disequality-output',
        ),
        pytest.param(
            Or(a > 0, And(b == 0, a != 0)),
            False,
            Or(a > 0, And(b == 0, a != 0)),
            id='or-disequality-input-disequality-output',
        ),
        pytest.param(
            And(a >= 0, Or(b == 0, a > 0)),
            True,
            And(a >= 0, Or(b == 0, a > 0)),
            id='and-order-input-order-output',
        ),
        pytest.param(
            And(a >= 0, Or(b == 0, a != 0)),
            True,
            And(a >= 0, Or(b == 0, a > 0)),
            id='and-disequality-input-order-output',
        ),
        pytest.param(
            And(a >= 0, Or(b == 0, a > 0)),
            False,
            And(a >= 0, Or(b == 0, a != 0)),
            id='and-order-input-disequality-output',
        ),
        pytest.param(
            And(a >= 0, Or(b == 0, a != 0)),
            False,
            And(a >= 0, Or(b == 0, a != 0)),
            id='and-disequality-input-disequality-output',
        ),
    ],
)
def test_prefer_order(formula: Formula, prefer_order: bool,
                      expected: Formula) -> None:
    assert simplify(formula, prefer_order=prefer_order) == expected


def test_prefer_order_default() -> None:
    formula = Or(a > 0, And(b == 0, a != 0))

    assert simplify(formula) == simplify(formula, prefer_order=True)


@pytest.mark.parametrize(
    ('formula', 'prefer_weak', 'expected'),
    [
        pytest.param(
            And(a != 0, Or(b == 0, a >= 0)),
            False,
            And(a != 0, Or(b == 0, a > 0)),
            id='and-weak-input-strict-output',
        ),
        pytest.param(
            And(a != 0, Or(b == 0, a > 0)),
            False,
            And(a != 0, Or(b == 0, a > 0)),
            id='and-strict-input-strict-output',
        ),
        pytest.param(
            And(a != 0, Or(b == 0, a >= 0)),
            True,
            And(a != 0, Or(b == 0, a >= 0)),
            id='and-weak-input-weak-output',
        ),
        pytest.param(
            And(a != 0, Or(b == 0, a > 0)),
            True,
            And(a != 0, Or(b == 0, a >= 0)),
            id='and-strict-input-weak-output',
        ),
        pytest.param(
            Or(a == 0, And(b == 0, a >= 0)),
            False,
            Or(a == 0, And(b == 0, a > 0)),
            id='or-weak-input-strict-output',
        ),
        pytest.param(
            Or(a == 0, And(b == 0, a > 0)),
            False,
            Or(a == 0, And(b == 0, a > 0)),
            id='or-strict-input-strict-output',
        ),
        pytest.param(
            Or(a == 0, And(b == 0, a >= 0)),
            True,
            Or(a == 0, And(b == 0, a >= 0)),
            id='or-weak-input-weak-output',
        ),
        pytest.param(
            Or(a == 0, And(b == 0, a > 0)),
            True,
            Or(a == 0, And(b == 0, a >= 0)),
            id='or-strict-input-weak-output',
        ),
    ],
)
def test_prefer_weak(formula: Formula, prefer_weak: bool,
                     expected: Formula) -> None:
    assert simplify(formula, prefer_weak=prefer_weak) == expected


def test_prefer_weak_default() -> None:
    formula = And(a != 0, Or(b == 0, a >= 0))

    assert simplify(formula) == simplify(formula, prefer_weak=False)


@pytest.mark.parametrize(
    ('formula', 'expected'),
    [
        pytest.param(
            And(a <= 0, Or(b != 0, a == 0)),
            And(a <= 0, Or(b != 0, a == 0)),
            id='equality-input',
        ),
        pytest.param(
            And(a <= 0, Or(b != 0, a >= 0)),
            And(a <= 0, Or(b != 0, a == 0)),
            id='weak-order-input',
        ),
    ],
)
def test_do_not_prefer_order_over_equality(formula: Formula,
                                           expected: Formula) -> None:
    assert simplify(formula, prefer_order=True) == expected


@pytest.mark.parametrize(
    ('substitute', 'expected'),
    [
        pytest.param(
            0,
            And(d - 2 == 0, 4 * b - 3 * c == 0, a + b + c + d >= 0),
            id='disabled',
        ),
        pytest.param(
            1,
            And(d - 2 == 0, 4 * b - 3 * c == 0, a + b + c + 2 >= 0),
            id='values',
        ),
        pytest.param(
            2,
            And(d - 2 == 0, 4 * b - 3 * c == 0, 4 * a + 7 * c + 8 >= 0),
            id='values-and-monomials',
        ),
    ],
)
def test_substitute(substitute: int, expected: Formula) -> None:
    formula = And(d == 2, 4 * b - 3 * c == 0, a + b + c + d >= 0)

    assert simplify(formula, substitute=substitute) == expected


@pytest.mark.parametrize(
    ('formula', 'assume', 'expected'),
    [
        pytest.param(3 * b**2 + c**2 >= 0, None, True, id='valid'),
        pytest.param(3 * b**2 + c**2 < 0, None, False, id='invalid'),
        pytest.param(
            a * b**2 + c**2 >= 0,
            [a > 0],
            True,
            id='valid-under-assumption',
        ),
        pytest.param(a * b**2 + c**2 >= 0, None, None, id='unknown'),
    ],
)
def test_is_valid(formula: Formula, assume: list[AtomicFormula] | None,
                  expected: bool | None) -> None:
    result = is_valid(formula) if assume is None else is_valid(formula, assume=assume)

    assert result is expected


# Examples by NF from the development of simpl_and_or.
@pytest.mark.parametrize(
    'formula',
    [
        pytest.param(
            And(x + y == 0, Or(And(x == 0, y != 0), And(x != 0, y == 0))),
            id='linear-equation',
        ),
        pytest.param(
            And(x**2 + y == 0, Or(And(x == 0, y != 0), And(x != 0, y == 0))),
            id='nonlinear-equation',
        ),
    ],
)
def test_inconsistent_substitution_cases(formula: Formula) -> None:
    assert simplify(formula) is F


def test_reset_internal_representation_after_substitution() -> None:
    # The internal representation must be reset in simpl_and_or when new
    # substitutions have been found.
    formula = And(x**2 + y**2 + z > 0, z == 1)

    assert simplify(formula) == (z - 1 == 0)


def test_knowledge_intersection_discovers_substitution() -> None:
    # The non-trivial _Knowledge.get() intersection in
    # InternalRepresentation.add() discovers a substitution.
    formula = And(x >= 0, y >= 0, x + y <= 0)

    assert simplify(formula) == And(y == 0, x == 0)


@pytest.mark.parametrize(
    ('implicit_ranges', 'substitute', 'expected_output_atoms'),
    [
        pytest.param(False, 0, 259, id='implicit-ranges-false-substitute-0'),
        pytest.param(False, 1, 283, id='implicit-ranges-false-substitute-1'),
        pytest.param(False, 2, 190, id='implicit-ranges-false-substitute-2'),
        pytest.param(True, 0, 250, id='implicit-ranges-true-substitute-0'),
        pytest.param(True, 1, 274, id='implicit-ranges-true-substitute-1'),
        pytest.param(True, 2, 184, id='implicit-ranges-true-substitute-2'),
    ],
)
def test_motor_series4(implicit_ranges: bool, substitute: int,
                       expected_output_atoms: int) -> None:
    """Check motor series4 with every benchmarked option combination."""
    assert len(list(series4.atoms())) == 292

    result = simplify(series4, implicit_ranges=implicit_ranges,
                      substitute=substitute)

    assert len(list(result.atoms())) == expected_output_atoms
