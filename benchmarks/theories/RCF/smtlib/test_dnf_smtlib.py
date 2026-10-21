"""DNF benchmarks on selected SMT-LIB QF_NRA formulas.

Run with the ``pytest-benchmark`` plugin installed::

    pytest test_dnf_smtlib.py --benchmark-only

The inputs come from the SMT-LIB 2025 non-incremental release,
https://doi.org/10.5281/zenodo.16740866. Their original source files,
individual credits, licenses, and selection criteria are recorded in
``inputs`` and ``NOTES.md``.

``Redlog off`` uses ``bnfsac=False, bnfsm=False``; ``Redlog on`` uses
both options set to ``True``. ``cartesian`` uses the Cartesian
method. ``early`` uses ``cartesian-simplified``; ``subtree`` uses
``cartesian-subtree-simplified``. The latter two use the default limit of
500 final semantic pair checks. Each case in the main comparison times one
DNF computation in a spawned worker, with a 30-second computation limit.

An additional levin-rhsc case uses early with ``implicit_ranges=False``
only during initial simplification. Semantic implication checks and final
simplification retain their defaults. Compare the two early cases with::

    pytest test_dnf_smtlib.py -k levin-rhsc-early --benchmark-only

Input and DNF result atom counts from the full run after setting initial
Cartesian simplification to ``explode_always=False, substitute=1``:

Counts include repeated atom occurrences. ``timeout`` means no result was
available within 30 seconds.

+-----------------+-------+------------+-----------+-----------+---------+---------+
| case            | input | Redlog off | Redlog on | cartesian | early   | subtree |
+=================+=======+============+===========+===========+=========+=========+
| mitrinovic-4-14 |    16 |          8 |         8 |         8 |       8 |       8 |
+-----------------+-------+------------+-----------+-----------+---------+---------+
| mitrinovic-3-29 |    20 |          7 |         7 |         7 |       7 |       7 |
+-----------------+-------+------------+-----------+-----------+---------+---------+
| knopp-schur-b   |    29 |          9 |         9 |         9 |       9 |       9 |
+-----------------+-------+------------+-----------+-----------+---------+---------+
| half-2          |    81 |        288 |       288 |       280 |     280 |     280 |
+-----------------+-------+------------+-----------+-----------+---------+---------+
| issac10-ex4h    |    39 |         10 |        10 |        10 |      10 |      10 |
+-----------------+-------+------------+-----------+-----------+---------+---------+
| levin-rhsc      |    46 |         88 |        88 |        88 |      88 |      88 |
+-----------------+-------+------------+-----------+-----------+---------+---------+
| sumcracker-14d  |    44 |       7168 |      7168 |      7168 |    7168 |    7168 |
+-----------------+-------+------------+-----------+-----------+---------+---------+
| sumcracker-14e  |    54 |    timeout |   timeout |     34816 | timeout | timeout |
+-----------------+-------+------------+-----------+-----------+---------+---------+

Computation times from the same run, in seconds (one measurement per case):

+-----------------+------------+-----------+-----------+---------+---------+
| case            | Redlog off | Redlog on | cartesian | early   | subtree |
+=================+============+===========+===========+=========+=========+
| mitrinovic-4-14 |      0.128 |     0.122 |     0.013 |   0.014 |   0.013 |
+-----------------+------------+-----------+-----------+---------+---------+
| mitrinovic-3-29 |      0.116 |     0.113 |     0.017 |   0.019 |   0.017 |
+-----------------+------------+-----------+-----------+---------+---------+
| knopp-schur-b   |      0.120 |     0.118 |     0.050 |   0.046 |   0.044 |
+-----------------+------------+-----------+-----------+---------+---------+
| half-2          |      0.235 |     0.236 |     0.490 |   0.747 |   0.730 |
+-----------------+------------+-----------+-----------+---------+---------+
| issac10-ex4h    |      0.107 |     0.151 |     0.044 |   0.047 |   0.047 |
+-----------------+------------+-----------+-----------+---------+---------+
| levin-rhsc      |      0.137 |     0.137 |     0.135 |   0.295 |   0.284 |
+-----------------+------------+-----------+-----------+---------+---------+
| sumcracker-14d  |     17.353 |    22.004 |     2.943 |   9.639 |   9.453 |
+-----------------+------------+-----------+-----------+---------+---------+
| sumcracker-14e  |    timeout |   timeout |    13.841 | timeout | timeout |
+-----------------+------------+-----------+-----------+---------+---------+

Informational PyEDA results from a separate run on 2026-10-21, using
``method="pyeda"`` with default options and a 30-second computation limit.
These measurements are not benchmark cases implemented in this file.
Each input was computed once in a fresh worker; startup, input parsing,
and output atom counting are excluded from the timings.

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

Additional comparison on levin-rhsc using
``method="cartesian-simplified"`` (early), measured on 2026-10-21 with
one computation per setting. Only initial simplification is changed:

+-------------------------+--------------+---------+
| initial implicit_ranges | result atoms | seconds |
+=========================+==============+=========+
| True (default)          |           88 |   0.299 |
+-------------------------+--------------+---------+
| False                   |           16 |   0.209 |
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

A separate sumcracker-14e benchmark tries ``max_pair_checks=1, 2, 4, ...``
for early and subtree, with the default initial ``implicit_ranges=True``.
The zero-budget cases separately disable final semantic pair comparisons.
This separate benchmark uses a 60-second computation limit.
Run each doubling sweep with ``--exitfirst`` (``-x``) to stop at its first
timeout::

    pytest test_dnf_smtlib.py -k 'pair_checks and early and not 0-' -x
    pytest test_dnf_smtlib.py -k 'pair_checks and subtree and not 0-' -x

Run the zero-budget cases with::

    pytest test_dnf_smtlib.py -k 'pair_checks and 0-' --benchmark-only

Selected results measured on 2026-10-21, with one computation per setting:

+-----------------+-------------+-----------+---------------+-------------+
| max_pair_checks | early atoms | early (s) | subtree atoms | subtree (s) |
+=================+=============+===========+===============+=============+
|               0 |       34816 |    44.241 |         34816 |      41.600 |
+-----------------+-------------+-----------+---------------+-------------+
|               1 |       34816 |    43.890 |         34816 |      41.475 |
+-----------------+-------------+-----------+---------------+-------------+
|               2 |       34816 |    43.357 |         34816 |      41.321 |
+-----------------+-------------+-----------+---------------+-------------+
|             256 |       34816 |    45.836 |         34816 |      45.049 |
+-----------------+-------------+-----------+---------------+-------------+
|             512 |       34816 |    49.177 |         34816 |      48.449 |
+-----------------+-------------+-----------+---------------+-------------+
|            1024 |       34816 |    58.292 |         34816 |      56.887 |
+-----------------+-------------+-----------+---------------+-------------+
|            2048 |     unknown |   timeout |       unknown |     timeout |
+-----------------+-------------+-----------+---------------+-------------+

Both doubling sweeps first timed out at budget 2048. Every completed run
returned 34,816 atom occurrences, matching cartesian and the zero-budget
controls. Increasing the semantic pair budget gave no reduction in the
final atom count for this input.
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
from logic1.theories.RCF.bnf import CartesianBooleanNormalForm, dnf as logic1_dnf
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
        return logic1_dnf(f, method="cartesian-simplified")


DNF_CONFIGURATIONS: tuple[tuple[str, Callable[..., Formula], DNFOptions], ...] = (
    ("redlog-off", redlog.dnf, {"bnfsac": False, "bnfsm": False}),
    ("redlog-on", redlog.dnf, {"bnfsac": True, "bnfsm": True}),
    ("cartesian", logic1_dnf, {"method": "cartesian"}),
    ("early", logic1_dnf, {"method": "cartesian-simplified"}),
    ("subtree", logic1_dnf, {"method": "cartesian-subtree-simplified"}),
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
                 id=f"{name}-early-no-implicit-ranges",
                 marks=pytest.mark.benchmark(group="dnf-smtlib-early"))
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
@pytest.mark.parametrize("method", ("cartesian-simplified", "cartesian-subtree-simplified"),
                         ids=("early", "subtree"))
@pytest.mark.parametrize("max_pair_checks",
                         (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048))
def test_dnf_sumcracker_pair_checks(benchmark: BenchmarkFixture, method: str,
                                    max_pair_checks: int) -> None:
    """Compare zero and doubling semantic pair budgets on sumcracker-14e.
    """
    relative_path, expected_input_atoms = next((path, atoms)
                                              for name, path, atoms in SMTLIB_CASES
                                              if name == "sumcracker-14e")
    _benchmark_dnf(benchmark, relative_path, expected_input_atoms, logic1_dnf,
                   {"method": method, "max_pair_checks": max_pair_checks}, timeout=60)
