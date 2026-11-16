"""Shared cases and isolated execution for the SMT-LIB DNF pytest services.

Each worker loads one input, computes one DNF, then validates and counts its
atoms. The parent times only computation and the pipe round trip. Loading
and validation have separate limits. This module does not require
pytest-benchmark; both pytest services use the same computations.
"""

from collections.abc import Callable
from dataclasses import dataclass
import math
import multiprocessing
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
import os
from pathlib import Path
import signal
from time import perf_counter
from types import TracebackType
from typing import Self

import pytest

from logic1.theories.RCF import redlog
from logic1.theories.RCF.atomic import AtomicFormula
from logic1.theories.RCF.bnf import CartesianBnf, Simplification, dnf as logic1_dnf
from logic1.theories.RCF.simplify import simplify as rcf_simplify
from logic1.theories.RCF.term import Variable
from logic1.theories.RCF.types import Formula

from smtlib_inputs import SMTLIB_CASES, load_input


type DNFOptions = dict[str, bool | int | str]


@dataclass(frozen=True)
class DnfCase:
    """One input and configuration, independent of execution limits.

    ``name`` is the local input ID, and ``relative_path`` identifies the
    original SMT-LIB file. ``configuration`` names a table column; ``options``
    records the arguments supplied to ``dnf_function``.
    """

    name: str
    relative_path: Path
    input_atoms: int
    configuration: str
    dnf_function: Callable[..., Formula]
    options: DNFOptions


class DnfRun:
    """Run one case in a fresh process, retaining its outcome for reporting.

    ``timeout`` limits computation in seconds, excluding loading and validation.
    Infinity permits unlimited computation; a finite deadline kills the worker
    process group, including any Reduce subprocess, when the context exits.
    Entering the context loads the input and checks its recorded atom count.
    Call ``compute`` once, then ``finish`` to validate the result and obtain
    its atom count. Only ``compute`` belongs inside a timing benchmark.
    Context exit also terminates descendant processes left by a failed call.

    ``seconds`` measures the computation, including the pipe round trip;
    it remains ``None`` if computation never started. ``output_atoms`` stays
    ``None`` unless validation succeeds. ``status``, ``phase``, and ``limit``
    distinguish failures during startup, computation, or validation and remain
    available after exceptions. ``limit`` is the current phase's time limit.
    """

    def __init__(self, case: DnfCase, timeout: float) -> None:
        self.case = case
        self.timeout = timeout
        self.seconds: float | None = None
        self.output_atoms: int | None = None
        self.status = "error"
        self.phase = "startup"
        self.limit: float = 30
        self._parent: Connection | None = None
        self._worker: BaseProcess | None = None

    def __enter__(self) -> Self:
        context = multiprocessing.get_context("spawn")
        self._parent, child = context.Pipe()
        self._worker = context.Process(target=_compute_dnf, args=(child, self.case))
        try:
            self._worker.start()
            child.close()
            input_atoms = self._receive("ready")
            assert input_atoms == self.case.input_atoms, (
                f"Input atom count changed: expected {self.case.input_atoms}, got {input_atoms}")
            return self
        except BaseException:
            self.close()
            raise
        finally:
            child.close()

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 traceback: TracebackType | None) -> None:
        self.close()

    def close(self) -> None:
        """Release the pipe and terminate the worker and any descendants.

        Before ``setsid`` has run, the worker may not yet have its own process
        group. In that case, terminate the worker directly. After startup,
        killing the group also covers a Reduce process left by a worker exit.
        """
        if self._worker is not None and self._worker.pid is not None:
            # Reap an exited worker before signaling its group. On macOS a
            # group containing only an unreaped zombie can reject killpg.
            self._worker.join(timeout=0)
            try:
                os.killpg(self._worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if self._worker.is_alive():
                self._worker.kill()
            self._worker.join(timeout=5)
        self._worker = None
        if self._parent is not None:
            self._parent.close()
        self._parent = None

    def compute(self) -> None:
        """Compute the DNF once, excluding loading and result validation.

        Errors propagate as test failures. The elapsed time is retained even
        if computation fails; a timeout also records its phase and limit.
        """
        assert self._parent is not None
        self.phase = "computation"
        self.limit = self.timeout
        start = perf_counter()
        try:
            self._parent.send(None)
            assert self._receive("ok") is None
        finally:
            self.seconds = perf_counter() - start

    def finish(self) -> int:
        """Wait for DNF validation and return the number of atom occurrences.

        The worker checks DNF structure and output variable names, allowing
        Redlog's lowercasing only when distinct input names remain distinct.
        These checks do not prove equivalence or require a historical size.
        """
        self.phase = "validation"
        self.limit = 30
        output_atoms = self._receive("atoms")
        assert isinstance(output_atoms, int) and output_atoms >= 0
        self.output_atoms = output_atoms
        self.status = "ok"
        return output_atoms

    def _receive(self, expected: str) -> object:
        """Receive one worker message with the current phase's time limit.
        """
        assert self._parent is not None
        if not self._parent.poll(None if math.isinf(self.limit) else self.limit):
            self.status = "timeout"
            raise TimeoutError(f"DNF {self.phase} exceeded {self.limit:g} seconds")
        try:
            status, value = self._parent.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError(f"DNF worker disconnected during {self.phase}") from exc
        if status == "error":
            raise RuntimeError(f"DNF worker failed during {self.phase}: {value}")
        assert status == expected, f"Expected {expected!r}, received {status!r}"
        return value


def _compute_dnf(connection: Connection, case: DnfCase) -> None:
    """Load, compute, and validate one case using separate protocol messages.
    """
    os.setsid()
    try:
        formula = load_input(case.relative_path)
        input_variables = frozenset(formula.fvars())
        lowercase_variables = case.dnf_function is redlog.dnf
        if lowercase_variables:
            _input_variable_names(input_variables, lowercase_variables=True)
        connection.send(("ready", sum(1 for _ in formula.atoms())))
        connection.recv()
        result = case.dnf_function(formula, **case.options)
        connection.send(("ok", None))
        connection.send(("atoms", count_dnf_atoms(result, input_variables,
                                                  lowercase_variables=lowercase_variables)))
    except Exception as exc:
        try:
            connection.send(("error", repr(exc)))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def count_dnf_atoms(formula: Formula, input_variables: frozenset[Variable],
                    *, lowercase_variables: bool = False) -> int:
    """Validate quantifier-free DNF structure and count atom occurrences.

    Truth values contain no atoms. Otherwise every disjunct must be an atom
    or a conjunction of atoms, using only input variable names. Set
    ``lowercase_variables`` for Redlog output: expected names are lowercased,
    and collisions are rejected even if the result is a truth value. Repeated
    atoms in different disjuncts are counted separately without creating a
    list of all atoms or a copy of the formula.
    """
    input_names = _input_variable_names(input_variables, lowercase_variables)
    if Formula.is_true(formula) or Formula.is_false(formula):
        return 0
    cubes = formula.args if Formula.is_or(formula) else (formula,)
    count = 0
    for cube in cubes:
        atoms = cube.args if Formula.is_and(cube) else (cube,)
        for atom in atoms:
            assert isinstance(atom, AtomicFormula), "Result is not a quantifier-free DNF"
            assert all(str(var) in input_names for var in atom.fvars()), (
                "Result contains a variable absent from the input")
            count += 1
    return count


def _input_variable_names(input_variables: frozenset[Variable],
                          lowercase_variables: bool) -> frozenset[str]:
    """Return expected names, rejecting collisions caused by lowercasing.
    """
    names = frozenset(str(var) for var in input_variables)
    if lowercase_variables:
        names = frozenset(name.lower() for name in names)
        assert len(names) == len(input_variables), (
            "Redlog lowercasing merges distinct input variable names")
    return names


def _dnf_without_initial_implicit_ranges(f: Formula) -> Formula:
    """Disable implicit ranges only for DURING_STEP preprocessing.
    """
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(CartesianBnf, "simplify", _simplify_without_implicit_ranges)
        return logic1_dnf(f, method="cartesian", simplification=Simplification.DURING_STEP,
                         max_pair_checks=500)


def _simplify_without_implicit_ranges(self: CartesianBnf, f: Formula) -> Formula:
    """Preserve the usual preprocessing options except for implicit ranges.
    """
    return rcf_simplify(f, explode_always=False, substitute=1, implicit_ranges=False)


DNF_CONFIGURATIONS: tuple[tuple[str, Callable[..., Formula], DNFOptions], ...] = (
    ("redlog-off", redlog.dnf, {"bnfsac": False, "bnfsm": False}),
    ("redlog-on", redlog.dnf, {"bnfsac": True, "bnfsm": True}),
    ("none", logic1_dnf,
     {"method": "cartesian", "simplification": Simplification.NONE}),
    ("final", logic1_dnf,
     {"method": "cartesian", "simplification": Simplification.FINAL,
      "max_pair_checks": 500}),
    ("after-step", logic1_dnf,
     {"method": "cartesian", "simplification": Simplification.AFTER_STEP,
      "max_pair_checks": 500}),
    ("during-step", logic1_dnf,
     {"method": "cartesian", "simplification": Simplification.DURING_STEP,
      "max_pair_checks": 500}),
)

# Membership follows the completed-input timings in dnf_results.md, recorded
# on 2026-10-22, 2026-11-04, and 2026-11-05. Every standard configuration must
# have succeeded below the threshold. These sets do not impose execution limits.
UNDER_60_INPUTS = frozenset(('beyer-figure4', 'bugtest1', 'const1', 'economics-0016c',
                             'economics-0052c', 'economics-0063c', 'economics-0086c',
                             'economics-1001c', 'half-2', 'hendecagon', 'hhk2008', 'issac10-ex3e',
                             'issac10-ex3f', 'issac10-ex4h', 'kissing-3-9', 'kissing-4-14',
                             'kissing-4-25', 'knopp-schur-b', 'kochanski', 'levin-rhsc',
                             'lex-index', 'max-in-array', 'mbo-e11e14', 'mbo-e14', 'mbo-e4e11',
                             'mbo-e7e10', 'mgc-10', 'mitrinovic-3-29', 'mitrinovic-4-14',
                             'monadic-invariant', 'nutz-transform05', 'partial-init',
                             'pentagon-diagonal', 'pine-pendulum', 'pine-quadratic',
                             'polypaver-0326', 'ppda-escape25', 'ppda-mod5', 'ppda-mod7',
                             'right-triangle', 'sumcracker-14b', 'sumcracker-14c',
                             'sumcracker-14d', 'triangle-bottema', 'triangle-sym321', 'two-index'))

UNDER_600_INPUTS = UNDER_60_INPUTS | frozenset(('bug-hoare', 'cggmp2005', 'impulse', 'nested-c',
                                                'ppda-schelling', 'simple1'))

MAIN_CASES = tuple(
    DnfCase(name, path, atoms, configuration, function, dict(options))
    for name, path, atoms in SMTLIB_CASES
    for configuration, function, options in DNF_CONFIGURATIONS
)

IMPLICIT_RANGE_CASES = tuple(
    DnfCase(name, path, atoms, "during-step-no-implicit-ranges",
            _dnf_without_initial_implicit_ranges, {})
    for name, path, atoms in SMTLIB_CASES
    if name == "levin-rhsc"
)

PAIR_CASES = tuple(
    DnfCase(name, path, atoms, configuration, logic1_dnf,
            {"method": "cartesian", "simplification": simplification, "max_pair_checks": budget})
    for name, path, atoms in SMTLIB_CASES
    if name == "sumcracker-14e"
    for budget in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
    for configuration, simplification in (("during-step", Simplification.DURING_STEP),
                                         ("after-step", Simplification.AFTER_STEP))
)

MAIN_PARAMETERS = tuple(
    pytest.param(case, id=f"{case.name}-{case.configuration}",
                 marks=(pytest.mark.dnf_main,
                        *([pytest.mark.dnf_under_60s] if case.name in UNDER_60_INPUTS else []),
                        *([pytest.mark.dnf_under_600s] if case.name in UNDER_600_INPUTS else []),
                        pytest.mark.benchmark(group=f"dnf-smtlib-{case.configuration}")))
    for case in MAIN_CASES
)

IMPLICIT_RANGE_PARAMETERS = tuple(
    pytest.param(case, id=f"{case.name}-{case.configuration}",
                 marks=(pytest.mark.dnf_implicit_ranges,
                        pytest.mark.benchmark(group="dnf-smtlib-during-step")))
    for case in IMPLICIT_RANGE_CASES
)

PAIR_PARAMETERS = tuple(
    pytest.param(case, id=f"{case.options['max_pair_checks']}-{case.configuration}",
                 marks=(pytest.mark.dnf_pair_checks,
                        pytest.mark.benchmark(group="dnf-smtlib-pair-checks")))
    for case in PAIR_CASES
)
