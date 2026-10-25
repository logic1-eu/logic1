"""DNF benchmarks on the motor series.

Run with the ``pytest-benchmark`` plugin installed::

    pytest test_dnf_motor.py -k 'not pyeda' --benchmark-only

``RL off`` uses Redlog with ``bnfsac=False, bnfsm=False``; ``RL on``
enables both. The other columns use ``method="cartesian"`` with
``Simplification.NONE``, ``FINAL``, ``AFTER_STEP``, or ``DURING_STEP``.
The tables abbreviate the last two schedules as AFTER and DURING.
The three simplifying schedules use ``max_pair_checks=500``. NONE retains
ordinary preprocessing and conservative final simplification.

Each case computes one DNF in a fresh spawned worker with a 30-second
computation limit. Startup and output atom counting are outside timing;
the timed call includes a small pipe round trip. Counts include repeated
atom occurrences. ``t/o`` denotes a timeout with no measured result size.

Input and result atom counts measured on 2026-10-22, excluding PyEDA:

+--------+-------+--------+-------+------+-------+-------+--------+
| series | input | RL off | RL on | NONE | FINAL | AFTER | DURING |
+========+=======+========+=======+======+=======+=======+========+
| 1      |   710 |   2105 |  2105 | 1528 |  1512 |  1512 |   1512 |
+--------+-------+--------+-------+------+-------+-------+--------+
| 2      |  1420 |   2368 |  2300 | 2297 |  2263 |  2263 |   2263 |
+--------+-------+--------+-------+------+-------+-------+--------+
| 3      |    94 |    112 |    35 |  112 |    38 |    38 |     38 |
+--------+-------+--------+-------+------+-------+-------+--------+
| 4      |   292 |    389 |   243 |  219 |    87 |    87 |     87 |
+--------+-------+--------+-------+------+-------+-------+--------+
| 5      |   157 |    152 |    96 |  143 |    48 |    48 |     48 |
+--------+-------+--------+-------+------+-------+-------+--------+
| 6      |   994 |   2205 |  2142 |  802 |   761 |   761 |    761 |
+--------+-------+--------+-------+------+-------+-------+--------+
| 7      |   248 |    308 |   308 |  230 |   176 |   176 |    176 |
+--------+-------+--------+-------+------+-------+-------+--------+
| 8      |   473 |   1203 |  1203 |  662 |   621 |   621 |    621 |
+--------+-------+--------+-------+------+-------+-------+--------+
| 9      |   235 |    282 |   282 |  210 |   162 |   162 |    162 |
+--------+-------+--------+-------+------+-------+-------+--------+
| 10     |   478 |   1104 |  1104 | 1005 |  1005 |  1005 |   1005 |
+--------+-------+--------+-------+------+-------+-------+--------+
| 11     |   168 |    255 |    87 |  240 |    78 |    78 |     78 |
+--------+-------+--------+-------+------+-------+-------+--------+
| 12     |  2176 |   5020 |  2366 | 4560 |  2116 |  2116 |   2116 |
+--------+-------+--------+-------+------+-------+-------+--------+
| 13     |   358 |    792 |   792 |  717 |   717 |   717 |    717 |
+--------+-------+--------+-------+------+-------+-------+--------+

Computation times from the same run, in seconds (one call per case):

+--------+--------+-------+-------+-------+-------+--------+
| series | RL off | RL on | NONE  | FINAL | AFTER | DURING |
+========+========+=======+=======+=======+=======+========+
| 1      |  0.709 | 1.050 | 3.480 | 5.938 | 6.185 |  6.101 |
+--------+--------+-------+-------+-------+-------+--------+
| 2      |  0.503 | 1.127 | 2.628 | 5.663 | 6.058 |  5.882 |
+--------+--------+-------+-------+-------+-------+--------+
| 3      |  0.113 | 0.106 | 0.164 | 0.206 | 0.207 |  0.225 |
+--------+--------+-------+-------+-------+-------+--------+
| 4      |  0.152 | 0.147 | 0.552 | 1.771 | 1.770 |  1.746 |
+--------+--------+-------+-------+-------+-------+--------+
| 5      |  0.118 | 0.110 | 0.173 | 0.809 | 0.813 |  0.775 |
+--------+--------+-------+-------+-------+-------+--------+
| 6      |  0.424 | 0.951 | 3.669 | 5.711 | 5.782 |  5.818 |
+--------+--------+-------+-------+-------+-------+--------+
| 7      |  0.136 | 0.165 | 0.411 | 1.780 | 1.801 |  1.834 |
+--------+--------+-------+-------+-------+-------+--------+
| 8      |  0.279 | 0.449 | 0.934 | 7.060 | 6.943 |  7.081 |
+--------+--------+-------+-------+-------+-------+--------+
| 9      |  0.137 | 0.152 | 0.387 | 1.721 | 1.678 |  1.670 |
+--------+--------+-------+-------+-------+-------+--------+
| 10     |  0.288 | 0.547 | 1.362 | 4.264 | 4.278 |  4.422 |
+--------+--------+-------+-------+-------+-------+--------+
| 11     |  0.127 | 0.109 | 0.334 | 0.622 | 0.603 |  0.613 |
+--------+--------+-------+-------+-------+-------+--------+
| 12     |  1.331 | 1.462 | 5.828 | 7.869 | 8.326 |  8.664 |
+--------+--------+-------+-------+-------+-------+--------+
| 13     |  0.219 | 0.312 | 1.181 | 5.697 | 5.818 |  6.070 |
+--------+--------+-------+-------+-------+-------+--------+

PyEDA measurements retained from 2026-10-21, with the same 30-second limit.
These cases remain available but were excluded from the schedule comparison.
PyEDA was not benchmarked again:

+--------+-------------+-------------+---------+
| series | input atoms | PyEDA atoms | seconds |
+========+=============+=============+=========+
| 1      |         710 |     timeout | timeout |
+--------+-------------+-------------+---------+
| 2      |        1420 |     timeout | timeout |
+--------+-------------+-------------+---------+
| 3      |          94 |          49 |   0.102 |
+--------+-------------+-------------+---------+
| 4      |         292 |         219 |   0.480 |
+--------+-------------+-------------+---------+
| 5      |         157 |         143 |   0.182 |
+--------+-------------+-------------+---------+
| 6      |         994 |     timeout | timeout |
+--------+-------------+-------------+---------+
| 7      |         248 |         230 |   0.401 |
+--------+-------------+-------------+---------+
| 8      |         473 |     timeout | timeout |
+--------+-------------+-------------+---------+
| 9      |         235 |         210 |   0.334 |
+--------+-------------+-------------+---------+
| 10     |         478 |     timeout | timeout |
+--------+-------------+-------------+---------+
| 11     |         168 |         130 |   0.217 |
+--------+-------------+-------------+---------+
| 12     |        2176 |     timeout | timeout |
+--------+-------------+-------------+---------+
| 13     |         358 |     timeout | timeout |
+--------+-------------+-------------+---------+
"""

from collections.abc import Callable
import multiprocessing
from multiprocessing.connection import Connection
import os
import signal

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from logic1.theories.RCF import redlog
from logic1.theories.RCF.bnf import Simplification, dnf as logic1_dnf
from logic1.theories.RCF.types import Formula

from motor_inputs import MOTOR_SERIES


type DNFOptions = dict[str, bool | int | str]


DNF_CONFIGURATIONS: tuple[tuple[str, Callable[..., Formula], DNFOptions], ...] = (
    ("redlog-bnfsac-false-bnfsm-false", redlog.dnf, {"bnfsac": False, "bnfsm": False}),
    ("redlog-bnfsac-true-bnfsm-true", redlog.dnf, {"bnfsac": True, "bnfsm": True}),
    ("logic1-cartesian", logic1_dnf,
     {"method": "cartesian", "simplification": Simplification.NONE}),
    ("logic1-final", logic1_dnf,
     {"method": "cartesian", "simplification": Simplification.FINAL}),
    ("logic1-after-step", logic1_dnf,
     {"method": "cartesian", "simplification": Simplification.AFTER_STEP}),
    ("logic1-during-step", logic1_dnf,
     {"method": "cartesian", "simplification": Simplification.DURING_STEP}),
    ("logic1-pyeda", logic1_dnf,
     {"method": "pyeda", "simplification": Simplification.NONE}),
)

CASES = tuple(
    pytest.param(formula, expected_input_atoms, dnf_function, options,
                 id=f"{name}-{configuration_id}",
                 marks=pytest.mark.benchmark(group=f"dnf-motor-{configuration_id}"))
    for name, formula, expected_input_atoms in MOTOR_SERIES
    for configuration_id, dnf_function, options in DNF_CONFIGURATIONS
)


def _compute_dnf(connection: Connection, formula: Formula,
                 dnf_function: Callable[..., Formula], options: DNFOptions) -> None:
    os.setsid()
    connection.send("ready")
    connection.recv()
    try:
        result = dnf_function(formula, **options)
        connection.send(("ok", None))
        connection.send(("atoms", len(list(result.atoms()))))
    except Exception as exc:
        connection.send(("error", repr(exc)))
    finally:
        connection.close()


@pytest.mark.parametrize(("formula", "expected_input_atoms", "dnf_function", "options"), CASES)
def test_dnf_motor(benchmark: BenchmarkFixture, formula: Formula, expected_input_atoms: int,
                   dnf_function: Callable[..., Formula], options: DNFOptions) -> None:
    """Benchmark one DNF computation on one motor input."""
    assert len(list(formula.atoms())) == expected_input_atoms

    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    worker = context.Process(target=_compute_dnf, args=(child, formula, dnf_function, options))
    worker.start()
    child.close()
    ready = False
    try:
        if not parent.poll(30) or parent.recv() != "ready":
            raise TimeoutError("DNF worker did not start within 30 seconds")
        ready = True

        def run_once() -> None:
            parent.send(None)
            if not parent.poll(30):
                raise TimeoutError("DNF computation exceeded 30 seconds")
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
