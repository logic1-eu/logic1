"""Benchmarks for DNF computation on the shared motor test series.

Run with the ``pytest-benchmark`` plugin installed::

    pytest test_dnf_motor.py --benchmark-only

Each case performs one DNF computation in a worker process. The timed call
includes a small pipe round trip, and the worker is stopped if it has not
returned a result within 30 seconds.

Input and DNF result atom counts from one full benchmark run:

``Redlog off`` uses ``bnfsac=False, bnfsm=False``; ``Redlog on`` uses both
options set to ``True``. ``timeout`` means the computation did not finish
within 30 seconds, so its result size is unknown.

+--------+-------+------------+-----------+-----------+---------+
| series | input | Redlog off | Redlog on | cartesian | pyeda   |
+========+=======+============+===========+===========+=========+
|      1 |   710 |       2105 |      2105 |      1528 | timeout |
+--------+-------+------------+-----------+-----------+---------+
|      2 |  1420 |       2368 |      2300 |      2297 | timeout |
+--------+-------+------------+-----------+-----------+---------+
|      3 |    94 |        112 |        35 |       112 |      49 |
+--------+-------+------------+-----------+-----------+---------+
|      4 |   292 |        389 |       243 |       219 |     219 |
+--------+-------+------------+-----------+-----------+---------+
|      5 |   157 |        152 |        96 |       143 |     143 |
+--------+-------+------------+-----------+-----------+---------+
|      6 |   994 |       2205 |      2142 |       802 | timeout |
+--------+-------+------------+-----------+-----------+---------+
|      7 |   248 |        308 |       308 |       230 |     230 |
+--------+-------+------------+-----------+-----------+---------+
|      8 |   473 |       1203 |      1203 |       662 | timeout |
+--------+-------+------------+-----------+-----------+---------+
|      9 |   235 |        282 |       282 |       210 |     210 |
+--------+-------+------------+-----------+-----------+---------+
|     10 |   478 |       1104 |      1104 |      1005 | timeout |
+--------+-------+------------+-----------+-----------+---------+
|     11 |   168 |        255 |        87 |       240 |     130 |
+--------+-------+------------+-----------+-----------+---------+
|     12 |  2176 |       5020 |      2366 |      4560 | timeout |
+--------+-------+------------+-----------+-----------+---------+
|     13 |   358 |        792 |       792 |       717 | timeout |
+--------+-------+------------+-----------+-----------+---------+

Computation times from the same run, in seconds (one measurement per case):

+--------+------------+-----------+-----------+---------+
| series | Redlog off | Redlog on | cartesian | pyeda   |
+========+============+===========+===========+=========+
|      1 |      0.450 |     1.013 |     2.648 | timeout |
+--------+------------+-----------+-----------+---------+
|      2 |      0.514 |     1.141 |     3.668 | timeout |
+--------+------------+-----------+-----------+---------+
|      3 |      0.116 |     0.103 |     0.159 |   0.102 |
+--------+------------+-----------+-----------+---------+
|      4 |      0.145 |     0.145 |     0.425 |   0.554 |
+--------+------------+-----------+-----------+---------+
|      5 |      0.115 |     0.115 |     0.180 |   0.185 |
+--------+------------+-----------+-----------+---------+
|      6 |      0.427 |     0.889 |     2.354 | timeout |
+--------+------------+-----------+-----------+---------+
|      7 |      0.149 |     0.155 |     0.511 |   0.394 |
+--------+------------+-----------+-----------+---------+
|      8 |      0.284 |     0.454 |     1.071 | timeout |
+--------+------------+-----------+-----------+---------+
|      9 |      0.138 |     0.153 |     0.345 |   0.324 |
+--------+------------+-----------+-----------+---------+
|     10 |      0.287 |     0.505 |     1.175 | timeout |
+--------+------------+-----------+-----------+---------+
|     11 |      0.128 |     0.111 |     0.275 |   0.209 |
+--------+------------+-----------+-----------+---------+
|     12 |      1.331 |     1.462 |     6.015 | timeout |
+--------+------------+-----------+-----------+---------+
|     13 |      0.231 |     0.313 |     1.134 | timeout |
+--------+------------+-----------+-----------+---------+
"""

from collections.abc import Callable
import multiprocessing
from multiprocessing.connection import Connection
import os
import signal

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from logic1.theories.RCF import redlog
from logic1.theories.RCF.bnf import dnf as logic1_dnf
from logic1.theories.RCF.types import Formula

from motor_inputs import MOTOR_SERIES


DNF_CONFIGURATIONS: tuple[tuple[str, Callable[..., Formula], dict[str, bool | str]], ...] = (
    ("redlog-bnfsac-false-bnfsm-false", redlog.dnf, {"bnfsac": False, "bnfsm": False}),
    ("redlog-bnfsac-true-bnfsm-true", redlog.dnf, {"bnfsac": True, "bnfsm": True}),
    ("logic1-cartesian", logic1_dnf, {"method": "cartesian"}),
    ("logic1-pyeda", logic1_dnf, {"method": "pyeda"}),
)

CASES = tuple(
    pytest.param(formula, expected_input_atoms, dnf_function, options,
                 id=f"{name}-{configuration_id}",
                 marks=pytest.mark.benchmark(group=f"dnf-motor-{configuration_id}"))
    for name, formula, expected_input_atoms in MOTOR_SERIES
    for configuration_id, dnf_function, options in DNF_CONFIGURATIONS
)


def _compute_dnf(connection: Connection, formula: Formula,
                 dnf_function: Callable[..., Formula], options: dict[str, bool | str]) -> None:
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
                   dnf_function: Callable[..., Formula], options: dict[str, bool | str]) -> None:
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
