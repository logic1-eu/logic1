"""DNF benchmarks on selected SMT-LIB QF_NRA formulas.

Run with the ``pytest-benchmark`` plugin installed::

    pytest test_dnf_smtlib.py::test_dnf_smtlib --benchmark-only

Inputs are from the SMT-LIB 2025 non-incremental release,
https://doi.org/10.5281/zenodo.16740866. Original files, credits, licenses,
and selection criteria are recorded in ``inputs`` and ``NOTES.md``.

``RL off`` uses Redlog with ``bnfsac=False, bnfsm=False``; ``RL on``
enables both. The other columns use ``method="cartesian"`` with
``Simplification.NONE``, ``FINAL``, ``AFTER_STEP``, or ``DURING_STEP``.
AFTER and DURING abbreviate the last two schedules in the tables.
The three simplifying schedules use ``max_pair_checks=500``.

Each main case computes one DNF in a fresh spawned worker with a 30-second
computation limit. Startup, parsing, and output counting are outside timing;
the timed call includes a small pipe round trip. Atom counts include
repetitions across cubes. ``t/o`` denotes a timeout with no result size.

Input and result atom counts measured on 2026-10-22:

+-----------------+-------+--------+-------+-------+-------+-------+--------+
| case            | input | RL off | RL on | NONE  | FINAL | AFTER | DURING |
+=================+=======+========+=======+=======+=======+=======+========+
| mitrinovic-4-14 |    16 |      8 |     8 |     8 |     8 |     8 |      8 |
+-----------------+-------+--------+-------+-------+-------+-------+--------+
| mitrinovic-3-29 |    20 |      7 |     7 |     7 |     7 |     7 |      7 |
+-----------------+-------+--------+-------+-------+-------+-------+--------+
| knopp-schur-b   |    29 |      9 |     9 |     9 |     9 |     9 |      9 |
+-----------------+-------+--------+-------+-------+-------+-------+--------+
| half-2          |    81 |    288 |   288 |   280 |   280 |   280 |    280 |
+-----------------+-------+--------+-------+-------+-------+-------+--------+
| issac10-ex4h    |    39 |     10 |    10 |    10 |    10 |    10 |     10 |
+-----------------+-------+--------+-------+-------+-------+-------+--------+
| levin-rhsc      |    46 |     88 |    88 |    88 |    88 |    88 |     88 |
+-----------------+-------+--------+-------+-------+-------+-------+--------+
| sumcracker-14d  |    44 |   7168 |  7168 |  7168 |  7168 |  7168 |   7168 |
+-----------------+-------+--------+-------+-------+-------+-------+--------+
| sumcracker-14e  |    54 |    t/o |   t/o | 34816 |   t/o |   t/o |    t/o |
+-----------------+-------+--------+-------+-------+-------+-------+--------+

Computation times from the same run, in seconds (one call per case):

+-----------------+--------+--------+--------+-------+-------+--------+
| case            | RL off | RL on  | NONE   | FINAL | AFTER | DURING |
+=================+========+========+========+=======+=======+========+
| mitrinovic-4-14 |  0.122 |  0.125 |  0.016 | 0.013 | 0.014 |  0.014 |
+-----------------+--------+--------+--------+-------+-------+--------+
| mitrinovic-3-29 |  0.114 |  0.115 |  0.019 | 0.017 | 0.017 |  0.018 |
+-----------------+--------+--------+--------+-------+-------+--------+
| knopp-schur-b   |  0.122 |  0.123 |  0.049 | 0.049 | 0.044 |  0.044 |
+-----------------+--------+--------+--------+-------+-------+--------+
| half-2          |  0.221 |  0.239 |  0.439 | 0.742 | 0.760 |  0.753 |
+-----------------+--------+--------+--------+-------+-------+--------+
| issac10-ex4h    |  0.110 |  0.109 |  0.046 | 0.046 | 0.048 |  0.046 |
+-----------------+--------+--------+--------+-------+-------+--------+
| levin-rhsc      |  0.137 |  0.139 |  0.138 | 0.284 | 0.292 |  0.293 |
+-----------------+--------+--------+--------+-------+-------+--------+
| sumcracker-14d  | 17.445 | 22.753 |  2.787 | 9.657 | 9.572 |  9.950 |
+-----------------+--------+--------+--------+-------+-------+--------+
| sumcracker-14e  |    t/o |    t/o | 14.397 |   t/o |   t/o |    t/o |
+-----------------+--------+--------+--------+-------+-------+--------+

Informational PyEDA results from a separate run on 2026-10-21, using
``method="pyeda"`` with default options and a 30-second computation limit.
These are not benchmark cases implemented in this file. PyEDA was not
benchmarked again. Each measurement excludes startup, parsing, and counting:

+-----------------+-------+--------------+---------+
| case            | input | result atoms | seconds |
+=================+=======+==============+=========+
| mitrinovic-4-14 |    16 |            8 |   0.015 |
+-----------------+-------+--------------+---------+
| mitrinovic-3-29 |    20 |            7 |   0.017 |
+-----------------+-------+--------------+---------+
| knopp-schur-b   |    29 |            9 |   0.052 |
+-----------------+-------+--------------+---------+
| half-2          |    81 |        28520 |  12.573 |
+-----------------+-------+--------------+---------+
| issac10-ex4h    |    39 |           10 |   0.045 |
+-----------------+-------+--------------+---------+
| levin-rhsc      |    46 |           88 |   0.137 |
+-----------------+-------+--------------+---------+
| sumcracker-14d  |    44 |         7168 |   2.840 |
+-----------------+-------+--------------+---------+
| sumcracker-14e  |    54 |        34816 |  15.338 |
+-----------------+-------+--------------+---------+

Additional levin-rhsc comparison using ``Simplification.DURING_STEP``,
measured on 2026-10-22 with one call per setting and a 30-second limit.
Only initial ``implicit_ranges`` is changed; semantic implication checks
and final simplification retain their defaults. Run the two cases with::

    pytest test_dnf_smtlib.py -k levin-rhsc-during-step --benchmark-only

+-------------------------+--------------+---------+
| initial implicit_ranges | result atoms | seconds |
+=========================+==============+=========+
| True (default)          |           88 |   0.293 |
+-------------------------+--------------+---------+
| False                   |           16 |   0.203 |
+-------------------------+--------------+---------+

In levin-rhsc, enabling implicit ranges reduces the initial formula but
collapses explicit variable sign cases. The current simplifier cannot
recover those signs from product constraints, leaving six contradictory
cubes in the final DNF. Disabling the option retains the cases and enables
their elimination, despite a larger intermediate DNF. This is the only
observed case so far where enabling implicit ranges increases the final
atom count.

Equivalence was checked by splitting each cube of the default result on
``V2 < 0`` and ``V2 > 0``. Every cube contains ``V2 != 0``, so this split
is exhaustive. Simplifying the resulting disjunction produced a formula
structurally identical to the result with the option disabled.

A separate sumcracker-14e benchmark doubles ``max_pair_checks`` from 1
for DURING_STEP and AFTER_STEP, with initial ``implicit_ranges=True``.
Zero-budget controls disable only final semantic comparisons. The limit
is 60 seconds. Stop each sweep at its first timeout::

    pytest test_dnf_smtlib.py -k 'pair_checks and during-step and not 0-' -x
    pytest test_dnf_smtlib.py -k 'pair_checks and after-step and not 0-' -x

Run the zero-budget cases with::

    pytest test_dnf_smtlib.py -k 'pair_checks and 0-' --benchmark-only

Selected measurements from 2026-10-21, retained without recomputation.
The former labels early and subtree now mean DURING_STEP and AFTER_STEP:

+-----------------+--------------+------------+-------------+-----------+
| max_pair_checks | DURING atoms | DURING (s) | AFTER atoms | AFTER (s) |
+=================+==============+============+=============+===========+
| 0               |        34816 |     44.241 |       34816 |    41.600 |
+-----------------+--------------+------------+-------------+-----------+
| 1               |        34816 |     43.890 |       34816 |    41.475 |
+-----------------+--------------+------------+-------------+-----------+
| 2               |        34816 |     43.357 |       34816 |    41.321 |
+-----------------+--------------+------------+-------------+-----------+
| 256             |        34816 |     45.836 |       34816 |    45.049 |
+-----------------+--------------+------------+-------------+-----------+
| 512             |        34816 |     49.177 |       34816 |    48.449 |
+-----------------+--------------+------------+-------------+-----------+
| 1024            |        34816 |     58.292 |       34816 |    56.887 |
+-----------------+--------------+------------+-------------+-----------+
| 2048            |      unknown |    timeout |     unknown |   timeout |
+-----------------+--------------+------------+-------------+-----------+

Both doubling sweeps first timed out at budget 2048. Every completed run
returned 34,816 atom occurrences, matching plain cartesian and the
zero-budget controls. Increasing the semantic pair budget gave no reduction
in the final atom count for this input.
"""

from collections.abc import Callable
import multiprocessing
from multiprocessing.connection import Connection
import os
from pathlib import Path
import signal

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from logic1.theories.RCF import redlog
from logic1.theories.RCF.bnf import (CartesianBooleanNormalForm, Simplification,
                                    dnf as logic1_dnf)
from logic1.theories.RCF.simplify import simplify as rcf_simplify
from logic1.theories.RCF.types import Formula

from smtlib_inputs import SMTLIB_CASES, load_input


type DNFOptions = dict[str, bool | int | str]


def _dnf_without_initial_implicit_ranges(f: Formula) -> Formula:
    """Compute early DNF with implicit ranges disabled during preprocessing.

    This function runs in the worker, where the temporary replacement also
    affects the inherited preprocessing of the early method.
    """
    def simplify(self: CartesianBooleanNormalForm, f: Formula) -> Formula:
        return rcf_simplify(f, explode_always=False, substitute=1, implicit_ranges=False)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(CartesianBooleanNormalForm, "simplify", simplify)
        return logic1_dnf(f, method="cartesian", simplification=Simplification.DURING_STEP)


DNF_CONFIGURATIONS: tuple[tuple[str, Callable[..., Formula], DNFOptions], ...] = (
    ("redlog-off", redlog.dnf, {"bnfsac": False, "bnfsm": False}),
    ("redlog-on", redlog.dnf, {"bnfsac": True, "bnfsm": True}),
    ("cartesian", logic1_dnf,
     {"method": "cartesian", "simplification": Simplification.NONE}),
    ("final", logic1_dnf,
     {"method": "cartesian", "simplification": Simplification.FINAL}),
    ("after-step", logic1_dnf,
     {"method": "cartesian", "simplification": Simplification.AFTER_STEP}),
    ("during-step", logic1_dnf,
     {"method": "cartesian", "simplification": Simplification.DURING_STEP}),
)

CASES = tuple(
    pytest.param(relative_path, expected_input_atoms, dnf_function, options,
                 id=f"{name}-{configuration_id}",
                 marks=pytest.mark.benchmark(group=f"dnf-smtlib-{configuration_id}"))
    for name, relative_path, expected_input_atoms in SMTLIB_CASES
    for configuration_id, dnf_function, options in DNF_CONFIGURATIONS
)

CASES += tuple(
    pytest.param(relative_path, expected_input_atoms, _dnf_without_initial_implicit_ranges, {},
                 id=f"{name}-during-step-no-implicit-ranges",
                 marks=pytest.mark.benchmark(group="dnf-smtlib-during-step"))
    for name, relative_path, expected_input_atoms in SMTLIB_CASES
    if name == "levin-rhsc"
)


def _compute_dnf(connection: Connection, relative_path: Path,
                 dnf_function: Callable[..., Formula], options: DNFOptions) -> None:
    os.setsid()
    try:
        formula = load_input(relative_path)
        connection.send(("ready", len(list(formula.atoms()))))
        connection.recv()
        result = dnf_function(formula, **options)
        connection.send(("ok", None))
        connection.send(("atoms", len(list(result.atoms()))))
    except Exception as exc:
        connection.send(("error", repr(exc)))
    finally:
        connection.close()


def _benchmark_dnf(benchmark: BenchmarkFixture, relative_path: Path, expected_input_atoms: int,
                   dnf_function: Callable[..., Formula], options: DNFOptions,
                   *, timeout: int = 30) -> None:
    """Time one DNF computation in a fresh worker and record its atom count.
    """
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    worker = context.Process(target=_compute_dnf,
                             args=(child, relative_path, dnf_function, options))
    worker.start()
    child.close()
    ready = False
    try:
        if not parent.poll(30):
            raise TimeoutError("DNF worker did not start within 30 seconds")
        status, input_atoms = parent.recv()
        if status == "error":
            raise RuntimeError(f"DNF worker failed while loading input: {input_atoms}")
        assert status == "ready"
        assert input_atoms == expected_input_atoms
        ready = True

        def run_once() -> None:
            parent.send(None)
            if not parent.poll(timeout):
                raise TimeoutError(f"DNF computation exceeded {timeout} seconds")
            status, value = parent.recv()
            if status == "error":
                raise RuntimeError(f"DNF worker failed: {value}")
            assert status == "ok"
            assert value is None

        benchmark.pedantic(run_once, rounds=1, iterations=1, warmup_rounds=0)
        if not parent.poll(30):
            raise TimeoutError("Counting atoms in the DNF result exceeded 30 seconds")
        status, output_atoms = parent.recv()
        if status == "error":
            raise RuntimeError(f"DNF worker failed while counting atoms: {output_atoms}")
        assert status == "atoms"
        assert isinstance(output_atoms, int)
        benchmark.extra_info["output_atoms"] = output_atoms
    finally:
        if worker.is_alive():
            if ready:
                try:
                    pid = worker.pid
                    assert pid is not None
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                worker.kill()
        worker.join(timeout=5)
        parent.close()


@pytest.mark.parametrize(("relative_path", "expected_input_atoms", "dnf_function", "options"),
                         CASES)
def test_dnf_smtlib(benchmark: BenchmarkFixture, relative_path: Path, expected_input_atoms: int,
                    dnf_function: Callable[..., Formula], options: DNFOptions) -> None:
    """Benchmark one DNF computation on one SMT-LIB input.
    """
    _benchmark_dnf(benchmark, relative_path, expected_input_atoms, dnf_function, options)


@pytest.mark.benchmark(group="dnf-smtlib-pair-checks")
@pytest.mark.parametrize("simplification",
                         (Simplification.DURING_STEP, Simplification.AFTER_STEP),
                         ids=("during-step", "after-step"))
@pytest.mark.parametrize("max_pair_checks",
                         (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048))
def test_dnf_sumcracker_pair_checks(benchmark: BenchmarkFixture,
                                    simplification: Simplification, max_pair_checks: int) -> None:
    """Compare zero and doubling semantic pair budgets on sumcracker-14e.
    """
    relative_path, expected_input_atoms = next((path, atoms)
                                              for name, path, atoms in SMTLIB_CASES
                                              if name == "sumcracker-14e")
    _benchmark_dnf(benchmark, relative_path, expected_input_atoms, logic1_dnf,
                   {"method": "cartesian", "simplification": simplification,
                    "max_pair_checks": max_pair_checks}, timeout=60)
