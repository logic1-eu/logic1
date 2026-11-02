"""Tests for cubes, strict DNFs, and Cartesian simplification schedules.
"""

from collections.abc import Callable
from dataclasses import FrozenInstanceError
import pickle

import pytest

from logic1.firstorder import And, Ex, F, Not, Or, T
from logic1.theories.RCF import AtomicFormula, Formula, Simplification, VV, cnf, dnf
from logic1.theories.RCF.bnf import (AfterStepSimplifyingCartesianBnf, CartesianBnf, Cube,
                                   DuringStepSimplifyingCartesianBnf, FinalSimplifyingCartesianBnf,
                                   SimplifyingCartesianBnf, StrictDnf, is_and_or)


type SimplifyingBNF = (FinalSimplifyingCartesianBnf
                      | AfterStepSimplifyingCartesianBnf
                      | DuringStepSimplifyingCartesianBnf)

SIMPLIFYING_CLASSES = (FinalSimplifyingCartesianBnf,
                       AfterStepSimplifyingCartesianBnf,
                       DuringStepSimplifyingCartesianBnf)


x, y = VV.get('x', 'y')


def cube(*atoms: AtomicFormula) -> Cube:
    return Cube(atoms)


def test_cube_is_an_immutable_structural_value() -> None:
    atoms = [x > 0, y == 0, x > 0]
    value = Cube(atoms)
    reordered = cube(y == 0, x > 0)
    assert value == reordered
    assert hash(value) == hash(reordered)
    assert len({value, reordered}) == 1
    assert len(value) == 2
    assert set(value) == {x > 0, y == 0}
    atoms.clear()
    assert value == reordered
    assert not hasattr(value, '__dict__')
    with pytest.raises(FrozenInstanceError):
        setattr(value, '_atoms', frozenset())
    # Logical equivalence does not make structurally different cubes equal.
    assert cube(x > 0, x >= 0) != cube(x > 0)


def test_cube_pickle_preserves_set_membership() -> None:
    value = cube(x > 0, y == 0)
    restored = pickle.loads(pickle.dumps(value))
    assert isinstance(restored, Cube)
    assert restored == value
    assert restored in {value}


def test_cube_conjoin_preserves_explicit_normalization() -> None:
    combined = cube(x > 0).conjoin(cube(x < 0))
    assert isinstance(combined, Cube)
    assert combined == cube(x > 0, x < 0)
    assert combined.normalize() is None


@pytest.mark.parametrize(('cubes', 'expected'), [
        # An empty conjunction is true.
        ((),
         cube()),
        # Repeated atoms are stored once.
        ((cube(x > 0), cube(x > 0, y == 0)),
         cube(x > 0, y == 0)),
        # Combining cubes does not detect contradictions automatically.
        ((cube(x > 0), cube(x < 0)),
         cube(x > 0, x < 0))])
def test_cube_combine(cubes: tuple[Cube, ...], expected: Cube) -> None:
    combined = Cube.combine(c for c in cubes)
    assert isinstance(combined, Cube)
    assert combined == expected


@pytest.mark.parametrize(('left', 'right', 'expected'), [
        # Complementary relations yield the empty cube, representing truth.
        (cube(x > 0), cube(x <= 0),
         cube()),
        # Different polynomials cannot be merged by the single-relation rule.
        (cube(x > 0), cube(y > 0),
         None),
        # Two differing relations do not fit the single-relation rule.
        (cube(x > 0, y > 0), cube(x <= 0, y <= 0),
         None),
        # Different cube sizes do not fit the rule either.
        (cube(x > 0, y > 0), cube(x <= 0),
         None)])
def test_cube_merge(left: Cube, right: Cube, expected: Cube | None) -> None:
    result = left.merge(right)
    assert result == expected
    assert result is None or isinstance(result, Cube)


@pytest.mark.parametrize(('premise', 'conclusion', 'expected'), [
        # An empty conclusion is true.
        (cube(x > 0), cube(),
         True),
        # Sign masks establish implications on the same polynomial.
        (cube(x > 0), cube(x >= 0),
         True),
        # RCF simplification can compare constraints on different polynomials.
        (cube(x - 1 > 0), cube(x > 0),
         True),
        # The converse implication does not hold.
        (cube(x > 0), cube(x - 1 > 0),
         False)])
def test_cube_implies(premise: Cube, conclusion: Cube, expected: bool) -> None:
    assert premise.implies(conclusion) is expected


def test_strict_dnf_owns_its_set_without_implicit_simplification() -> None:
    cubes = {cube(x > 0), cube(x >= 0)}
    value = StrictDnf(cubes)
    cubes.clear()
    assert len(value) == 2
    value.add(cube(x > 0))
    assert len(value) == 2
    other = StrictDnf((cube(),))
    value.update(other)
    other.add(cube(y > 0))
    assert len(value) == 3
    assert cube() in value
    assert cube(y > 0) not in value
    assert value != StrictDnf((cube(),))
    assert value.syntactic_reduction() == StrictDnf((cube(),))
    with pytest.raises(TypeError):
        hash(value)


def test_strict_dnf_copy_and_pickle_preserve_independence() -> None:
    value = StrictDnf((cube(x > 0),))
    copied = StrictDnf(value)
    restored = pickle.loads(pickle.dumps(value))
    assert restored == copied == value
    copied.add(cube(y > 0))
    restored.add(cube())
    assert len(value) == 1
    assert copied != restored


def test_strict_dnf_insert_stops_at_truth() -> None:
    value = StrictDnf()
    assert not value
    value.insert(cube(x > 0))
    value.insert(cube(x <= 0))
    assert value == StrictDnf((cube(),))
    assert value
    value.insert(cube(y > 0))
    assert value == StrictDnf((cube(),))


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
         {cube(x != 0)}),
        # Normalize multiple relations on one polynomial before insertion.
        ({cube(x < 0, x <= 0)},
         {cube(x < 0)}),
        # Reject a contradictory cube while retaining the other disjunct.
        ({cube(x < 0, x > 0), cube(y == 0)},
         {cube(y == 0)})])
def test_syntactic_reduction(sdnf: set[Cube], expected: set[Cube]) -> None:
    original = StrictDnf(sdnf)
    assert original.syntactic_reduction() == StrictDnf(expected)
    assert set(original) == sdnf

    # The same rules can be applied incrementally as cubes become available.
    result = StrictDnf()
    for conjunction in sdnf:
        normalized = conjunction.normalize()
        if normalized is not None:
            result.insert(normalized)
    assert result == StrictDnf(expected)


@pytest.mark.parametrize(('cls', 'expected_cubes'), [
        pytest.param(FinalSimplifyingCartesianBnf, 4, id='final'),
        pytest.param(AfterStepSimplifyingCartesianBnf, 1, id='after-step'),
        pytest.param(DuringStepSimplifyingCartesianBnf, 1, id='during-step')])
def test_construction_schedule(cls: type[SimplifyingBNF],
                               expected_cubes: int) -> None:
    formula = And(Or(x < 0, x == 0), Or(y > 0, y == 0))
    normal_form = cls(max_pair_checks=0)
    assert is_and_or(formula)
    # Final-only leaves recursive construction unsimplified; all schedules
    # simplify the completed result, even with semantic comparisons disabled.
    assert len(normal_form._strict_dnf(formula)) == expected_cubes
    assert normal_form._dnf_and_or(formula) == And(y >= 0, x <= 0)


@pytest.mark.parametrize(('conjunction', 'expected'), [
        # Intersecting strict and weak bounds keeps the strict bound.
        (cube(x < 0, x <= 0),
         cube(x < 0)),
        # Incompatible signs reject the cube.
        (cube(x < 0, x > 0),
         None)])
def test_cube_normalize(conjunction: Cube, expected: Cube | None) -> None:
    assert conjunction.normalize() == expected


def test_semantic_subsumption() -> None:
    sdnf = StrictDnf((cube(x - 1 > 0), cube(x > 0)))
    original = StrictDnf(sdnf)
    assert sdnf.semantic_subsumption(500) == StrictDnf((cube(x > 0),))
    assert sdnf == original
    skipped = sdnf.semantic_subsumption(0)
    assert skipped == sdnf
    assert skipped is not sdnf


def test_pair_check_budget_resets_for_each_computation() -> None:
    sdnf = StrictDnf((cube(x - 1 > 0), cube(x > 0)))
    expected = StrictDnf((cube(x > 0),))
    assert sdnf.semantic_subsumption(1) == expected
    assert sdnf.semantic_subsumption(1) == expected


@pytest.mark.parametrize('cls', SIMPLIFYING_CLASSES)
def test_invalid_limits_are_rejected(cls: type[SimplifyingBNF]) -> None:
    with pytest.raises(ValueError):
        cls(max_pair_checks=-1)
    with pytest.raises(TypeError):
        cls(max_pair_checks=1.5)  # type: ignore[arg-type]


@pytest.mark.parametrize('cls', SIMPLIFYING_CLASSES)
@pytest.mark.parametrize('max_pair_checks', [0, 1, 500, None])
def test_inherited_budget_and_final_semantic_pass(cls: type[SimplifyingBNF],
                                                 max_pair_checks: int | None) -> None:
    normal_form = cls(max_pair_checks=max_pair_checks)
    assert isinstance(normal_form, SimplifyingCartesianBnf)
    assert normal_form.max_pair_checks == max_pair_checks
    sdnf = StrictDnf((cube(x - 1 > 0), cube(x > 0)))
    expected = sdnf if max_pair_checks == 0 else StrictDnf((cube(x > 0),))
    assert normal_form._finish_strict_dnf(sdnf) == expected
    assert len(sdnf) == 2


def test_plain_cartesian_finishing_keeps_strict_dnf() -> None:
    sdnf = StrictDnf((cube(x - 1 > 0), cube(x > 0)))
    assert CartesianBnf()._finish_strict_dnf(sdnf) is sdnf


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
        # Conjunctions on the same polynomial normalize to one relation.
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
