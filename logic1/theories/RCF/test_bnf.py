"""Boolean normal form examples for both RCF implementations."""

from typing import Literal

import pytest

from benchmarks.theories.RCF.motor_series.motor_inputs import series3
from logic1.firstorder import All, And, Equivalent, Ex, F, Or, T
from logic1.theories.RCF import AtomicFormula, Formula, VV, cnf, dnf


Method = Literal['cartesian', 'pyeda']
a, b, c = VV.get('a', 'b', 'c')


@pytest.mark.parametrize('method', ['cartesian', 'pyeda'])
@pytest.mark.parametrize(('formula', 'expected'), [
        # Truth remains truth.
        (T,
         T),
        # A conjunction of true constants collapses to truth.
        (And(T, T),
         T),
        # Falsehood remains falsehood.
        (F,
         F),
        # An equation is already in DNF.
        (a == 0,
         a == 0),
        # A strict inequality is already in DNF.
        (a > 0,
         a > 0),
        # Equivalence expands into its two satisfying cases.
        (Equivalent(a == 0, b == 0),
         Or(And(b == 0, a == 0), And(b != 0, a != 0))),
        # A conjunctive encoding of equivalence has the same DNF.
        (And(Or(a == 0, b != 0), Or(a != 0, b == 0)),
         Or(And(b == 0, a == 0), And(b != 0, a != 0))),
        # Complementary inequalities leave the common equation.
        (And(Or(a > 0, b == 0), Or(a <= 0, b == 0)),
         b == 0),
        # Quantifiers remain in front while the matrix becomes DNF.
        (Ex((a, b), All(c, Equivalent(c == 0, And(a == 0, b == 0)))),
         Ex((a, b), All(c, Or(And(c == 0, b == 0, a == 0),
                              And(c != 0, b != 0),
                              And(c != 0, a != 0))))),
])
def test_dnf(formula: Formula, expected: Formula, method: Method) -> None:
    assert dnf(formula, method=method) == expected


@pytest.mark.parametrize(('method', 'expected'), [
        # Cartesian expansion retains a redundant middle clause.
        ('cartesian', Or(And(c <= 0, b != 0), And(c <= 0, a > 0), And(b == 0, a > 0))),
        # PyEDA removes that redundant clause.
        ('pyeda', Or(And(c <= 0, b != 0), And(b == 0, a > 0))),
])
def test_dnf_mixed_clauses(method: Method, expected: Formula) -> None:
    formula = And(Or(a > 0, b != 0), Or(c <= 0, b == 0))
    assert dnf(formula, method=method) == expected


@pytest.mark.parametrize(('method', 'expected_clauses', 'expected_atoms'), [
        ('cartesian', 14, 112),
        ('pyeda', 7, 49),
])
def test_dnf_motor_series3(method: Method, expected_clauses: int, expected_atoms: int) -> None:
    """Check the DNF structure and size of the third motor input."""
    result = dnf(series3, method=method)
    assert isinstance(result, Or)
    assert len(result.args) == expected_clauses
    assert all(isinstance(clause, And) for clause in result.args)
    assert all(isinstance(atom, AtomicFormula) for clause in result.args for atom in clause.args)
    assert len(list(result.atoms())) == expected_atoms


@pytest.mark.parametrize('method', ['cartesian', 'pyeda'])
def test_dnf_renames_bound_variables(method: Method) -> None:
    formula = Equivalent(Ex((a, b), And(a == 0, b == 0)), All(c, And(a == 0, c == 0)))
    result = dnf(formula, method=method)
    q0, q1, q2, q3, q4, q5 = result.qvars()
    expected_matrix = Or(And(q5 == 0, a == 0, q1 == 0, q2 == 0),
                         And(q5 == 0, a == 0, q0 != 0),
                         And(q4 != 0, a != 0),
                         And(q4 != 0, q1 == 0, q2 == 0),
                         And(q4 != 0, q0 != 0),
                         And(a != 0, q3 != 0),
                         And(q1 == 0, q2 == 0, q3 != 0),
                         And(q0 != 0, q3 != 0))
    expected = Ex((q0, q1, q2), All((q3, q4, q5), expected_matrix))
    matrix, prefix = result.matrix()
    _, expected_prefix = expected.matrix()
    assert prefix == expected_prefix
    assert isinstance(matrix, Or)
    assert len(matrix.args) == len(expected_matrix.args)
    assert all(isinstance(clause, And) for clause in matrix.args)
    assert {frozenset(clause.args) for clause in matrix.args} == {
        frozenset(clause.args) for clause in expected_matrix.args
    }


@pytest.mark.parametrize('method', ['cartesian', 'pyeda'])
@pytest.mark.parametrize(('formula', 'expected'), [
        # Truth remains truth.
        (T,
         T),
        # A conjunction of true constants collapses to truth.
        (And(T, T),
         T),
        # Falsehood remains falsehood.
        (F,
         F),
        # An equation is already in CNF.
        (a == 0,
         a == 0),
        # A strict inequality is already in CNF.
        (a > 0,
         a > 0),
        # Equivalence expands into two clauses.
        (Equivalent(a == 0, b == 0),
         And(Or(b == 0, a != 0), Or(b != 0, a == 0))),
        # Exclusive-or expands into two clauses.
        (Or(And(a == 0, b != 0), And(a != 0, b == 0)),
         And(Or(b == 0, a == 0), Or(b != 0, a != 0))),
        # Complementary inequalities leave the common equation.
        (Or(And(a > 0, b == 0), And(a <= 0, b == 0)),
         b == 0),
        # Quantifiers remain in front while the matrix becomes CNF.
        (Ex((a, b), All(c, Equivalent(c == 0, And(a == 0, b == 0)))),
         Ex((a, b), All(c, And(Or(c == 0, b != 0, a != 0),
                               Or(c != 0, b == 0),
                               Or(c != 0, a == 0))))),
])
def test_cnf(formula: Formula, expected: Formula, method: Method) -> None:
    assert cnf(formula, method=method) == expected


@pytest.mark.parametrize('method', ['cartesian', 'pyeda'])
def test_cnf_renames_bound_variables(method: Method) -> None:
    formula = Equivalent(Ex((a, b), And(a == 0, b == 0)), All(c, And(a == 0, c == 0)))
    result = cnf(formula, method=method)
    q0, q1, q2, q3, q4, q5 = result.qvars()
    expected_matrix = And(Or(q2 == 0, q1 != 0, q0 != 0),
                          Or(q1 != 0, a == 0, q0 != 0),
                          Or(a != 0, q4 == 0, q3 != 0),
                          Or(a != 0, q3 != 0, q5 == 0))
    expected = All((q0, q1, q2), Ex((q3, q4, q5), expected_matrix))
    matrix, prefix = result.matrix()
    _, expected_prefix = expected.matrix()
    assert prefix == expected_prefix
    assert isinstance(matrix, And)
    assert len(matrix.args) == len(expected_matrix.args)
    assert all(isinstance(clause, Or) for clause in matrix.args)
    assert {frozenset(clause.args) for clause in matrix.args} == {
        frozenset(clause.args) for clause in expected_matrix.args
    }
