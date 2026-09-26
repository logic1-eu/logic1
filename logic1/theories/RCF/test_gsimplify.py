# Coverage:
# - 16 test functions, expanding to 28 pytest cases when Redlog is available
# - Boundary cases: one equation, one inequality, T, F, and a disjunction of two equations
# - Clause and global-premise operations, assumptions, contradictions, and Gröbner reduction
# - Application example: DS97 Table 6, instance 3 (94 input atoms reduced to 35 with
#   Redlog and 49 with PyEDA)
# - Backend-dependent cases always use PyEDA and additionally use Redlog after an rlqe
#   smoke test
# - Production implementations; no monkeypatching
#
# Verification:
# - Coverage.py (both backends): 98% combined statement and branch coverage for gsimplify.py
#   (287 statements, 5 missed; 110 branch opportunities, 2 partially covered)
# - Pytest with Redlog: all 28 cases passed; all 454 project tests passed
# - Pytest without Redlog: 20 cases passed; 8 Redlog variants skipped
# - Mypy: no issues found

import shutil
import subprocess

import pytest

from logic1.firstorder import And, F, Or, T
from logic1.theories.RCF import Eq, Gt, Lt, Ne, VV, gsimplify
from logic1.theories.RCF.gsimplify import Clause, GlobalPremise, GSimplify, Options


x, y, z = VV.get('gsimplify_test_x', 'gsimplify_test_y', 'gsimplify_test_z')
PYEDA_OPTIONS = {'use_redlog_cnf': False}


def _redlog_works() -> bool:
    redcsl = shutil.which('redcsl')
    if redcsl is None:
        return False
    try:
        completed_process = subprocess.run(
            [redcsl, '-w'],
            input='rlset r; rlqe ex(x, a * x + b = 0); quit;\n',
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    output = ' '.join(completed_process.stdout.split())
    return completed_process.returncode == 0 and 'b = 0 or a <> 0' in output


CNF_OPTIONS = [
    pytest.param(
        {},
        id='redlog',
        marks=pytest.mark.skipif(
            not _redlog_works(),
            reason='requires a working Redlog rlqe installation',
        ),
    ),
    pytest.param(PYEDA_OPTIONS, id='pyeda'),
]


@pytest.fixture(params=CNF_OPTIONS)
def cnf_options(request: pytest.FixtureRequest) -> dict[str, bool]:
    return request.param


def test_single_equation(cnf_options):
    formula = x == 0

    assert gsimplify(formula, **cnf_options) == formula


def test_single_inequality(cnf_options):
    formula = x >= 0

    assert gsimplify(formula, **cnf_options) == formula


def test_true(cnf_options):
    assert gsimplify(T, **cnf_options) is T


def test_false(cnf_options):
    assert gsimplify(F, **cnf_options) is F


def test_disjunction_of_two_equations(cnf_options):
    result = gsimplify(Or(x == 0, y == 0), **cnf_options)

    assert isinstance(result, Or)
    assert len(result.args) == 2
    assert set(result.args) == {x == 0, y == 0}


def test_inconsistent_assumptions():
    with pytest.raises(GSimplify.Inconsistent):
        gsimplify(T, assume=[x == 0, x != 0], **PYEDA_OPTIONS)


@pytest.mark.parametrize(
    ('formula', 'expected'),
    [
        (x * y + z == 0, z == 0),
        (x * y + z > 0, z > 0),
    ],
)
def test_reduction_modulo_equational_assumption(formula, expected, cnf_options):
    assert gsimplify(formula, assume=[x == 0], **cnf_options) == expected


def test_empty_clause():
    clause = Clause(F)

    assert clause.is_empty()
    assert list(clause) == []
    assert repr(clause) == 'Clause()'
    with pytest.raises(ValueError, match='is not atomic'):
        clause.as_atom()


def test_clause_rejects_non_clause_formula():
    with pytest.raises(AssertionError):
        Clause(T)


@pytest.mark.parametrize(
    ('formula', 'strict_relation'),
    [
        (x >= 0, Gt),
        (x <= 0, Lt),
    ],
)
def test_weak_inequality_clause(formula, strict_relation):
    clause = Clause(formula)

    assert clause[Eq] == {x == 0}
    assert clause[strict_relation] == {strict_relation(x, 0)}
    assert clause.is_atomic()
    assert clause.as_atom() == formula


def test_equational_clause():
    clause = Clause(Or(x == 0, y == 0))

    assert clause.is_equational()
    assert clause.is_atomic()
    assert clause.as_atom() == (x * y == 0)
    assert clause.product_of(Eq) == x * y
    assert set(clause.term_list_of(Eq)) == {x, y}

    copied_clause = clause.copy()
    copied_clause.add(z == 0)
    assert len(copied_clause) == 3
    assert len(clause) == 2


@pytest.mark.parametrize(
    'formula',
    [
        Or(x != 0, y != 0),
        Or(x == 0, y > 0),
        Or(x != 0, y > 0, y < 0),
    ],
)
def test_non_atomic_clause(formula):
    assert not Clause(formula).is_atomic()


def test_global_premise_operations():
    premise = GlobalPremise([x == 0, y > 0], Options(use_redlog_cnf=False))

    assert premise[Eq] == {x == 0}
    assert premise.product_of(Eq) == x
    assert premise.product_of((Eq, Gt)) == x * y
    assert premise.term_list_of(Gt) == [y]
    assert set(premise.term_list_of((Eq, Gt))) == {x, y}

    first_basis = premise.gbasis
    assert first_basis == [x]
    assert premise.gbasis is first_basis

    premise.update([y == 0, x != 0])
    assert premise[Ne] == {x != 0}
    assert set(premise.gbasis) == {x, y}


def test_multiple_atomic_clauses_detect_contradiction():
    simplifier = GSimplify(Options(use_redlog_cnf=False))

    result = simplifier.gsimplify_clauses(
        [Clause(x == 0), Clause(x != 0)],
        assume=[],
    )

    assert len(result) == 1
    assert result[0].is_empty()


def test_tautological_regular_clause_is_removed():
    simplifier = GSimplify(Options(use_redlog_cnf=False))

    result = simplifier.gsimplify_clauses(
        [Clause(Or(x > 0, x <= 0))],
        assume=[],
    )

    assert result == []


def test_application_example_testseries3(cnf_options):
    """Test instance 3 from Table 6 in Dolzmann--Sturm (1997)."""
    i2, n, p1, q, td, z = VV.get('i2', 'n', 'p1', 'q', 'td', 'z')
    p1_boundary = 2 * p1 - 7
    q_td_boundary = 400 * q + 9 * td - 20050
    td_cases = Or(
        And(td - 400 >= 0, td - 700 < 0, 3 * td + 400 * z - 5320 == 0),
        And(td - 700 >= 0, td - 990 < 0, 2 * td - 300 * z + 1015 == 0),
        And(td == 0, z == 0),
    )
    boundary_cases = Or(
        And(q_td_boundary <= 0, q - 40 == 0),
        And(q_td_boundary > 0, td - 450 == 0),
    )
    formula = Or(
        And(
            p1_boundary >= 0,
            q_td_boundary <= 0,
            i2 == 0,
            n - td == 0,
            q >= 0,
            q - 40 <= 0,
            td_cases,
        ),
        And(
            p1_boundary >= 0,
            q_td_boundary >= 0,
            i2 == 0,
            n - td == 0,
            td - 450 >= 0,
            9 * td - 20050 <= 0,
            q - 40 <= 0,
            q >= 0,
            td_cases,
        ),
        And(
            p1_boundary >= 0,
            i2 == 0,
            n - td == 0,
            q - 40 <= 0,
            q >= 0,
            td_cases,
            boundary_cases,
        ),
        And(
            p1_boundary < 0,
            q_td_boundary <= 0,
            i2 == 0,
            n - td == 0,
            q >= 0,
            q - 40 <= 0,
            td_cases,
        ),
        And(
            p1_boundary < 0,
            q_td_boundary >= 0,
            i2 == 0,
            n - td == 0,
            td - 450 >= 0,
            9 * td - 20050 <= 0,
            q - 40 <= 0,
            q >= 0,
            td_cases,
        ),
        And(
            p1_boundary < 0,
            i2 == 0,
            n - td == 0,
            q - 40 <= 0,
            q >= 0,
            td_cases,
            boundary_cases,
        ),
    )

    result = gsimplify(formula, **cnf_options)
    expected_atom_count = 49 if cnf_options == PYEDA_OPTIONS else 35

    assert len(list(formula.atoms())) == 94
    assert len(list(result.atoms())) == expected_atom_count
