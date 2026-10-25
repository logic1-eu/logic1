"""Tests for the Cartesian normal form simplification schedules.
"""

from collections.abc import Callable

import pytest

from logic1.firstorder import And, Ex, F, Not, Or, T
from logic1.theories.RCF import AtomicFormula, Formula, Simplification, VV, cnf, dnf
from logic1.theories.RCF.bnf import (AfterStepSimplifyingCartesianBooleanNormalForm,
                                    DuringStepSimplifyingCartesianBooleanNormalForm,
                                    FinalSimplifyingCartesianBooleanNormalForm,
                                    SimplifyingCartesianBooleanNormalForm, is_and_or)


x, y = VV.get('x', 'y')


def cube(*atoms: AtomicFormula) -> frozenset[AtomicFormula]:
    return frozenset(atoms)


@pytest.mark.parametrize(('sdnf', 'expected'), [
        # The empty outer set represents false.
        (set(),
         set()),
        # An empty cube represents true.
        ({cube(), cube(x > 0)},
         {cube()}),
        # A shorter cube subsumes a longer one.
        ({cube(x > 0), cube(x > 0, y > 0)},
         {cube(x > 0)}),
        # A cut can replace two relations by a weaker relation.
        ({cube(x < 0, y == 0), cube(x == 0, y == 0)},
         {cube(x <= 0, y == 0)}),
        # Relations covering every sign remove their common polynomial.
        ({cube(x >= 0, y == 0), cube(x <= 0, y == 0)},
         {cube(y == 0)}),
        # Strict positive and negative together mean nonzero.
        ({cube(x > 0), cube(x < 0)},
         {cube(x != 0)})])
def test_atomic_simplification(sdnf: set[frozenset[AtomicFormula]],
                               expected: set[frozenset[AtomicFormula]]) -> None:
    normal_form = SimplifyingCartesianBooleanNormalForm()
    result: set[frozenset[AtomicFormula]] = set()
    for conjunction in sdnf:
        normal_form._insert(result, conjunction)
    assert result == expected


@pytest.mark.parametrize(('cls', 'expected_cubes'), [
        pytest.param(FinalSimplifyingCartesianBooleanNormalForm, 4, id='final'),
        pytest.param(AfterStepSimplifyingCartesianBooleanNormalForm, 1, id='after-step'),
        pytest.param(DuringStepSimplifyingCartesianBooleanNormalForm, 1, id='during-step')])
def test_construction_schedule(cls: type[SimplifyingCartesianBooleanNormalForm],
                               expected_cubes: int) -> None:
    formula = And(Or(x < 0, x == 0), Or(y > 0, y == 0))
    normal_form = cls(max_pair_checks=0)
    assert is_and_or(formula)
    # Final-only leaves recursive construction unreduced; all schedules reduce
    # the completed result, even with semantic comparisons disabled.
    assert len(normal_form._strict_dnf(formula)) == expected_cubes
    assert normal_form._dnf_and_or(formula) == And(y >= 0, x <= 0)


@pytest.mark.parametrize(('conjunction', 'expected'), [
        # Intersecting strict and weak bounds keeps the strict bound.
        (cube(x < 0, x <= 0),
         cube(x < 0)),
        # Incompatible signs reject the cube.
        (cube(x < 0, x > 0),
         None)])
def test_conjunction_on_one_polynomial(conjunction: frozenset[AtomicFormula],
                                       expected: frozenset[AtomicFormula] | None) -> None:
    normal_form = SimplifyingCartesianBooleanNormalForm()
    assert normal_form._simplify_cube(conjunction) == expected


def test_semantic_subsumption() -> None:
    sdnf = {cube(x - 1 > 0), cube(x > 0)}
    assert SimplifyingCartesianBooleanNormalForm()._simplify_strict_dnf(sdnf) == {cube(x > 0)}
    limited = SimplifyingCartesianBooleanNormalForm(max_pair_checks=0)
    assert limited._simplify_strict_dnf(sdnf) == sdnf


def test_pair_check_budget_resets_for_each_computation() -> None:
    normal_form = SimplifyingCartesianBooleanNormalForm(max_pair_checks=1)
    sdnf = {cube(x - 1 > 0), cube(x > 0)}
    expected = {cube(x > 0)}
    assert normal_form._simplify_strict_dnf(sdnf) == expected
    assert normal_form._simplify_strict_dnf(sdnf) == expected


def test_invalid_limits_are_rejected() -> None:
    with pytest.raises(ValueError):
        SimplifyingCartesianBooleanNormalForm(max_pair_checks=-1)
    with pytest.raises(TypeError):
        SimplifyingCartesianBooleanNormalForm(max_pair_checks=1.5)  # type: ignore[arg-type]


def test_limits_require_simplified_method() -> None:
    with pytest.raises(ValueError):
        dnf(x > 0, method='pyeda', simplification=Simplification.NONE, max_pair_checks=0)
    with pytest.raises(ValueError):
        cnf(x > 0, method='cartesian', simplification=Simplification.NONE, max_pair_checks=0)


@pytest.mark.parametrize('normal_form', [dnf, cnf])
@pytest.mark.parametrize('simplification', list(Simplification))
def test_integer_schedules(normal_form: Callable[..., Formula],
                           simplification: Simplification) -> None:
    formula = Or(And(x > 0, y == 0), And(x <= 0, y == 0))
    expected = normal_form(formula, method='cartesian', simplification=simplification)
    assert normal_form(formula, method='cartesian', simplification=int(simplification)) == expected


@pytest.mark.parametrize('normal_form', [dnf, cnf])
def test_defaults_are_cartesian_and_final(normal_form: Callable[..., Formula]) -> None:
    a, b, c, d, e = VV.get('a', 'b', 'c', 'd', 'e')
    formula: Formula = Or(And(a > 0, c > 0), And(a > 0, b > 0, c > 0),
                          And(a <= 0, b > 0, c > 0), And(b <= 0, c > 0),
                          And(d > 0, e > 0), And(d <= 0, e > 0))
    if normal_form is cnf:
        formula = Not(formula)
    expected = normal_form(formula, method='cartesian', simplification=Simplification.FINAL)
    assert normal_form(formula) == expected
    plain = normal_form(formula, method='cartesian', simplification=Simplification.NONE)
    assert expected != plain


@pytest.mark.parametrize(('value', 'error'), [
        (-1, ValueError), (4, ValueError), (True, TypeError), (1.5, TypeError)])
def test_invalid_schedules(value: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        dnf(x > 0, method='cartesian', simplification=value)  # type: ignore[arg-type]


@pytest.mark.parametrize('simplification', list(Simplification)[1:])
def test_pyeda_rejects_simplification_schedules(simplification: Simplification) -> None:
    with pytest.raises(ValueError):
        dnf(x > 0, method='pyeda', simplification=simplification)


@pytest.mark.parametrize('simplification', list(Simplification)[1:])
@pytest.mark.parametrize(('formula', 'expected'), [
        # Conjunctions on the same polynomial reduce to one relation.
        (And(x < 0, x <= 0),
         x < 0),
        # A contradictory product vanishes.
        (And(x < 0, x > 0),
         F),
        # Two child DNFs are multiplied and merged.
        (And(Or(x < 0, x == 0), Or(y > 0, y == 0)),
         And(y >= 0, x <= 0)),
        # A cut over all signs produces truth.
        (Or(x < 0, x >= 0),
         T)])
def test_dnf(formula: Formula, expected: Formula, simplification: Simplification) -> None:
    result = dnf(formula, method='cartesian', simplification=simplification, max_pair_checks=0)
    assert result == expected


@pytest.mark.parametrize('simplification', list(Simplification)[1:])
def test_dnf_public_dispatch_preserves_quantifier(simplification: Simplification) -> None:
    formula = Ex(x, And(Or(x > 0, y == 0), Or(x <= 0, y == 0)))
    expected = Ex(x, y == 0)
    assert dnf(formula, method='cartesian', simplification=simplification) == expected
    result = dnf(formula, method='cartesian', simplification=simplification, max_pair_checks=None)
    assert result == expected


@pytest.mark.parametrize('simplification', list(Simplification)[1:])
@pytest.mark.parametrize(('formula', 'expected'), [
        # Complementary inequalities leave the common equation.
        (Or(And(x > 0, y == 0), And(x <= 0, y == 0)),
         y == 0),
        # Quantifiers remain in front while the matrix becomes CNF.
        (Ex(x, Or(And(x > 0, y == 0), And(x <= 0, y == 0))),
         Ex(x, y == 0))])
def test_cnf(formula: Formula, expected: Formula, simplification: Simplification) -> None:
    assert cnf(formula, method='cartesian', simplification=simplification) == expected
