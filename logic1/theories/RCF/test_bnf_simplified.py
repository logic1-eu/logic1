"""Tests for early and completed-subtree Cartesian normal form simplification.
"""

from typing import Literal

import pytest

from logic1.firstorder import And, Ex, F, Or, T
from logic1.theories.RCF import AtomicFormula, Formula, VV, cnf, dnf
from logic1.theories.RCF.bnf import (SimplifyingCartesianBooleanNormalForm,
                                    SubtreeSimplifyingCartesianBooleanNormalForm, is_and_or)


Method = Literal['cartesian-simplified', 'cartesian-subtree-simplified']
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


@pytest.mark.parametrize('cls', [
        pytest.param(SimplifyingCartesianBooleanNormalForm, id='early'),
        pytest.param(SubtreeSimplifyingCartesianBooleanNormalForm, id='subtree')])
def test_recursive_simplification(cls: type[SimplifyingCartesianBooleanNormalForm]) -> None:
    formula = And(Or(x < 0, x == 0), Or(y > 0, y == 0))
    normal_form = cls()
    assert is_and_or(formula)
    assert normal_form._strict_dnf(formula) == {cube(x <= 0, y >= 0)}


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
        dnf(x > 0, method='pyeda', max_pair_checks=0)
    with pytest.raises(ValueError):
        cnf(x > 0, method='cartesian', max_pair_checks=0)


@pytest.mark.parametrize('method', ['cartesian-simplified', 'cartesian-subtree-simplified'])
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
def test_dnf(formula: Formula, expected: Formula, method: Method) -> None:
    assert dnf(formula, method=method, max_pair_checks=0) == expected


@pytest.mark.parametrize('method', ['cartesian-simplified', 'cartesian-subtree-simplified'])
def test_dnf_public_dispatch_preserves_quantifier(method: Method) -> None:
    formula = Ex(x, And(Or(x > 0, y == 0), Or(x <= 0, y == 0)))
    expected = Ex(x, y == 0)
    assert dnf(formula, method=method) == expected
    assert dnf(formula, method=method, max_pair_checks=None) == expected


@pytest.mark.parametrize('method', ['cartesian-simplified', 'cartesian-subtree-simplified'])
@pytest.mark.parametrize(('formula', 'expected'), [
        # Complementary inequalities leave the common equation.
        (Or(And(x > 0, y == 0), And(x <= 0, y == 0)),
         y == 0),
        # Quantifiers remain in front while the matrix becomes CNF.
        (Ex(x, Or(And(x > 0, y == 0), And(x <= 0, y == 0))),
         Ex(x, y == 0))])
def test_cnf(formula: Formula, expected: Formula, method: Method) -> None:
    assert cnf(formula, method=method) == expected
