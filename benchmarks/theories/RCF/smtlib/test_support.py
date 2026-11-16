"""Checks for SMT-LIB translation, DNF validation, and benchmark workers.

Worker checks use synthetic computations to exercise errors and timeouts
without running expensive DNF benchmarks or requiring Reduce.
"""

from collections.abc import Callable, Iterator
from dataclasses import replace
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
from time import sleep

import pytest

from logic1.firstorder import And, Ex, F, Not, Or, T
from logic1.theories.RCF import VV, redlog
from logic1.theories.RCF.atomic import Lt
from logic1.theories.RCF.term import Term
from logic1.theories.RCF.types import Formula

from smtlib_dnf import DnfCase, DnfRun, MAIN_CASES, count_dnf_atoms
from smtlib_inputs import SMTLIB_CASES, _load, load_input


@pytest.fixture(autouse=True)
def isolate_variables_and_workers() -> Iterator[None]:
    """Preserve the parent variable set and detect leaked worker processes.
    """
    children = {child.pid for child in multiprocessing.active_children()}
    VV.push()
    try:
        yield
    finally:
        VV.pop()
        assert {child.pid for child in multiprocessing.active_children()} == children


def _first_atom(formula: Formula) -> Formula:
    return next(formula.atoms())


def _raise_error(formula: Formula) -> Formula:
    raise ValueError("deliberate worker error")


def _exit_worker(formula: Formula) -> Formula:
    os._exit(1)


def _non_dnf(formula: Formula) -> Formula:
    return Not(next(formula.atoms()))


def _slow_result(formula: Formula) -> Formula:
    sleep(5)
    return _first_atom(formula)


def _result_with_child(formula: Formula, started: str, finished: str) -> Formula:
    subprocess.run([sys.executable, "-c",
                    "from pathlib import Path; import sys, time; "
                    "Path(sys.argv[1]).touch(); time.sleep(2); Path(sys.argv[2]).touch()",
                    started, finished], check=True)
    return _first_atom(formula)


@pytest.fixture
def case() -> DnfCase:
    """Use a small existing input with a synthetic computation.
    """
    return replace(MAIN_CASES[0], dnf_function=_first_atom, options={})


def read_formula(tmp_path: Path, commands: str) -> Formula:
    path = tmp_path / "input.smt2"
    path.write_text("(set-logic QF_NRA)\n" + commands, encoding="utf-8")
    return _load(path)


def test_let_bindings_are_simultaneous(tmp_path: Path) -> None:
    x, y = VV.get("x", "y")
    formula = read_formula(tmp_path, """
        (declare-const x Real)
        (declare-const y Real)
        (assert (let ((x y) (y x)) (< x y)))
    """)
    assert formula == Lt(y - x, 0)


def test_nested_let_shadowing_preserves_outer_scope(tmp_path: Path) -> None:
    x, y = VV.get("x", "y")
    formula = read_formula(tmp_path, """
        (declare-const x Real)
        (declare-const y Real)
        (assert (let ((x (+ x 1)))
                  (let ((x (+ x 1)) (y x)) (= x y))))
        (assert (< x y))
    """)
    assert formula == And(Term(1) == 0, x - y < 0)


def test_definitions_keep_their_lexical_scope(tmp_path: Path) -> None:
    x, y = VV.get("x", "y")
    formula = read_formula(tmp_path, """
        (declare-fun x () Real)
        (declare-fun y () Real)
        (define-fun square () Real (* x x))
        (define-fun guarded () Bool (and (> x 0) (= square y)))
        (assert (let ((x y) (square (+ y 1))) (and guarded (> square x))))
    """)
    assert formula == And(x > 0, x*x - y == 0, Term(1) > 0)


def test_division_by_nonzero_constants_is_exact(tmp_path: Path) -> None:
    x = VV["x"]
    formula = read_formula(tmp_path, """
        (declare-const x Real)
        (assert (< (/ (* x x) 6.0) (/ 1 3)))
    """)
    assert formula == (x*x / 6 - Term(1) / 3 < 0)


def test_boolean_bindings_and_negation(tmp_path: Path) -> None:
    x = VV["x"]
    formula = read_formula(tmp_path, """
        (declare-const x Real)
        (assert (let ((p (< x 0))) (not (or p false))))
    """)
    assert formula == Not(Or(x < 0, F))
    assert read_formula(tmp_path, "(assert true)") is T
    assert read_formula(tmp_path, "(assert false)") is F


@pytest.mark.parametrize("commands", [
    pytest.param("(assert (= (/ 1 0) 0))", id="zero-denominator"),
    pytest.param("(declare-const x Real) (assert (= (/ (* x x) x) x))",
                 id="variable-denominator"),
    pytest.param("(declare-const p Bool) (assert p)", id="free-boolean"),
    pytest.param("(assert (ite true true false))", id="ite"),
    pytest.param("(assert (mystery 1 2))", id="unknown-operator"),
    pytest.param("(assert (not true false))", id="wrong-not-arity"),
    pytest.param("(define-fun p () Bool 1) (assert p)", id="wrong-definition-sort"),
    pytest.param("(define-fun p () Real true) (assert true)", id="wrong-real-sort"),
    pytest.param("(assert (let ((a 1) (b a)) (= b 1)))", id="sibling-let-binding"),
    pytest.param("(assert (let ((a 1) (a 2)) (= a 1)))", id="duplicate-let-binding"),
    pytest.param("(define-fun f ((x Real)) Real x) (assert true)",
                 id="non-nullary-definition")])
def test_unsupported_or_ill_sorted_input_is_rejected(tmp_path: Path, commands: str) -> None:
    with pytest.raises(ValueError):
        read_formula(tmp_path, commands)


@pytest.mark.parametrize(("relative_path", "expected_atoms"),
                         [pytest.param(path, atoms, id=name)
                          for name, path, atoms in SMTLIB_CASES])
def test_recorded_input_atom_count(relative_path: Path, expected_atoms: int) -> None:
    formula = load_input(relative_path)
    atoms = list(formula.atoms())
    assert len(atoms) == expected_atoms
    assert all(atom.rhs.is_zero() for atom in atoms)


@pytest.mark.parametrize("formula", (T, F), ids=("true", "false"))
def test_truth_values_have_no_atoms(formula: Formula) -> None:
    assert count_dnf_atoms(formula, frozenset()) == 0


def test_atom_occurrences_are_counted_across_cubes() -> None:
    x, y = VV.get("x", "y")
    variables = frozenset((x, y))
    assert count_dnf_atoms(x > 0, variables) == 1
    assert count_dnf_atoms(And(x > 0, y > 0), variables) == 2
    assert count_dnf_atoms(Or(And(x > 0, y > 0), And(x > 0, y < 0)), variables) == 4


@pytest.mark.parametrize("kind", ("undistributed", "negated", "quantified"))
def test_malformed_dnf_is_rejected(kind: str) -> None:
    x, y = VV.get("x", "y")
    formula: Formula
    match kind:
        case "undistributed":
            formula = And(x > 0, Or(y > 0, y < 0))
        case "negated":
            formula = Not(x > 0)
        case "quantified":
            formula = Ex(x, x > 0)
        case _:
            raise AssertionError(kind)
    with pytest.raises(AssertionError, match="not a quantifier-free DNF"):
        count_dnf_atoms(formula, frozenset((x, y)))


def test_new_output_variable_is_rejected() -> None:
    x, y = VV.get("x", "y")
    with pytest.raises(AssertionError, match="variable absent from the input"):
        count_dnf_atoms(And(x > 0, y > 0), frozenset((x,)))


def test_redlog_lowercase_names_are_accepted_only_when_enabled() -> None:
    upper, lower = VV.get("V", "v")
    variables = frozenset((upper,))
    assert count_dnf_atoms(lower > 0, variables, lowercase_variables=True) == 1
    with pytest.raises(AssertionError, match="variable absent from the input"):
        count_dnf_atoms(lower > 0, variables)


@pytest.mark.parametrize("formula", (None, T, F), ids=("atom", "true", "false"))
def test_redlog_lowercase_collisions_are_rejected(formula: Formula | None) -> None:
    upper, lower = VV.get("X", "x")
    if formula is None:
        formula = lower > 0
    with pytest.raises(AssertionError, match="lowercasing merges distinct input variable names"):
        count_dnf_atoms(formula, frozenset((upper, lower)), lowercase_variables=True)


def test_redlog_lowercase_mode_still_rejects_unexpected_variables() -> None:
    original, unexpected = VV.get("V", "w")
    with pytest.raises(AssertionError, match="variable absent from the input"):
        count_dnf_atoms(unexpected > 0, frozenset((original,)), lowercase_variables=True)


def test_redlog_case_collisions_fail_before_computation(case: DnfCase, tmp_path: Path) -> None:
    source = tmp_path / "case-collision.smt2"
    source.write_text("(set-logic QF_NRA)\n(declare-const X Real)\n(declare-const x Real)\n"
                      "(assert (and (> X 0) (< x 0)))\n", encoding="utf-8")
    run = DnfRun(replace(case, relative_path=source, input_atoms=2, dnf_function=redlog.dnf,
                         options={"bnfsac": False, "bnfsm": False}), 60)
    with pytest.raises(RuntimeError, match="lowercasing merges distinct input variable names"):
        with run:
            pytest.fail("Case collisions must be rejected before launching Reduce")
    assert run.status == "error"
    assert run.phase == "startup"
    assert run.output_atoms is None
    assert run.seconds is None


@pytest.mark.parametrize("timeout", (60, float("inf")))
def test_success_retains_validated_count_and_time(case: DnfCase, timeout: float) -> None:
    run = DnfRun(case, timeout)
    with run:
        run.compute()
        assert run.finish() == 1
    assert run.status == "ok"
    assert run.output_atoms == 1
    assert run.seconds is not None and run.seconds >= 0


@pytest.mark.parametrize(("function", "phase", "message"), (
    pytest.param(_raise_error, "computation", "deliberate worker error", id="exception"),
    pytest.param(_exit_worker, "computation", "disconnected", id="worker-exit"),
    pytest.param(_non_dnf, "validation", "not a quantifier-free DNF", id="invalid-result")))
def test_failures_retain_phase_and_unknown_size(case: DnfCase,
                                               function: Callable[..., Formula],
                                               phase: str, message: str) -> None:
    run = DnfRun(replace(case, dnf_function=function), 60)
    with pytest.raises(RuntimeError, match=message):
        with run:
            run.compute()
            run.finish()
    assert run.status == "error"
    assert run.phase == phase
    assert run.output_atoms is None
    assert run.seconds is not None and run.seconds >= 0


def test_changed_input_count_fails_before_computation(case: DnfCase) -> None:
    run = DnfRun(replace(case, input_atoms=case.input_atoms + 1), 60)
    with pytest.raises(AssertionError, match="Input atom count changed"):
        with run:
            pytest.fail("The input mismatch was not detected on context entry")
    assert run.status == "error"
    assert run.phase == "startup"
    assert run.output_atoms is None
    assert run.seconds is None


def test_computation_timeout_cleans_up_before_next_run(case: DnfCase) -> None:
    run = DnfRun(replace(case, dnf_function=_slow_result), 0.05)
    with pytest.raises(TimeoutError, match="computation exceeded"):
        with run:
            run.compute()
    assert run.status == "timeout"
    assert run.phase == "computation"
    assert run.limit == 0.05
    assert run.output_atoms is None
    assert run.seconds is not None and run.seconds >= run.limit

    # A failed computation must not prevent the next worker from succeeding.
    with DnfRun(case, 60) as next_run:
        next_run.compute()
        assert next_run.finish() == 1


def test_timeout_terminates_child_process(case: DnfCase, tmp_path: Path) -> None:
    started, finished = tmp_path / "started", tmp_path / "finished"
    run = DnfRun(replace(case, dnf_function=_result_with_child,
                         options={"started": str(started), "finished": str(finished)}), 0.5)
    with pytest.raises(TimeoutError, match="computation exceeded"):
        with run:
            run.compute()
    assert started.exists(), "The child must have started before the timeout"
    sleep(2)
    assert not finished.exists(), "The child survived the computation timeout"
