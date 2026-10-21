"""Benchmarks for DNF computation on the shared motor test series.

Run with the ``pytest-benchmark`` plugin installed::

    pytest test_dnf_motor.py --benchmark-only

Each case performs one DNF computation in a worker process. The timed call
includes a small pipe round trip, and the worker is stopped if it has not
returned a result within 30 seconds.

Input and DNF result atom counts from the full run after setting initial
Cartesian simplification to ``explode_always=False, substitute=1``:

``Redlog off`` uses ``bnfsac=False, bnfsm=False``; ``Redlog on`` uses both
options set to ``True``. ``early`` uses ``cartesian-simplified`` and
``subtree`` uses ``cartesian-subtree-simplified``. Both use the default limit
of 500 final semantic pair checks. ``timeout`` means the computation did not
finish within 30 seconds, so its result size is unknown.

+--------+-------+------------+-----------+-----------+-------+---------+---------+
| series | input | Redlog off | Redlog on | cartesian | early | subtree |   pyeda |
+========+=======+============+===========+===========+=======+=========+=========+
|      1 |   710 |       2105 |      2105 |      1528 |  1512 |    1512 | timeout |
+--------+-------+------------+-----------+-----------+-------+---------+---------+
|      2 |  1420 |       2368 |      2300 |      2297 |  2263 |    2263 | timeout |
+--------+-------+------------+-----------+-----------+-------+---------+---------+
|      3 |    94 |        112 |        35 |       112 |    38 |      38 |      49 |
+--------+-------+------------+-----------+-----------+-------+---------+---------+
|      4 |   292 |        389 |       243 |       219 |    87 |      87 |     219 |
+--------+-------+------------+-----------+-----------+-------+---------+---------+
|      5 |   157 |        152 |        96 |       143 |    48 |      48 |     143 |
+--------+-------+------------+-----------+-----------+-------+---------+---------+
|      6 |   994 |       2205 |      2142 |       802 |   761 |     761 | timeout |
+--------+-------+------------+-----------+-----------+-------+---------+---------+
|      7 |   248 |        308 |       308 |       230 |   176 |     176 |     230 |
+--------+-------+------------+-----------+-----------+-------+---------+---------+
|      8 |   473 |       1203 |      1203 |       662 |   621 |     621 | timeout |
+--------+-------+------------+-----------+-----------+-------+---------+---------+
|      9 |   235 |        282 |       282 |       210 |   162 |     162 |     210 |
+--------+-------+------------+-----------+-----------+-------+---------+---------+
|     10 |   478 |       1104 |      1104 |      1005 |  1005 |    1005 | timeout |
+--------+-------+------------+-----------+-----------+-------+---------+---------+
|     11 |   168 |        255 |        87 |       240 |    78 |      78 |     130 |
+--------+-------+------------+-----------+-----------+-------+---------+---------+
|     12 |  2176 |       5020 |      2366 |      4560 |  2116 |    2116 | timeout |
+--------+-------+------------+-----------+-----------+-------+---------+---------+
|     13 |   358 |        792 |       792 |       717 |   717 |     717 | timeout |
+--------+-------+------------+-----------+-----------+-------+---------+---------+

Computation times from the same run, in seconds (one measurement per case):

+--------+------------+-----------+-----------+-------+---------+---------+
| series | Redlog off | Redlog on | cartesian | early | subtree |   pyeda |
+========+============+===========+===========+=======+=========+=========+
|      1 |      0.457 |     1.071 |     4.137 | 6.275 |   6.160 | timeout |
+--------+------------+-----------+-----------+-------+---------+---------+
|      2 |      0.522 |     1.147 |     2.767 | 5.783 |   5.931 | timeout |
+--------+------------+-----------+-----------+-------+---------+---------+
|      3 |      0.132 |     0.108 |     0.164 | 0.207 |   0.210 |   0.102 |
+--------+------------+-----------+-----------+-------+---------+---------+
|      4 |      0.150 |     0.142 |     0.562 | 1.717 |   1.752 |   0.480 |
+--------+------------+-----------+-----------+-------+---------+---------+
|      5 |      0.116 |     0.114 |     0.152 | 0.777 |   0.805 |   0.182 |
+--------+------------+-----------+-----------+-------+---------+---------+
|      6 |      0.429 |     0.898 |     3.046 | 5.720 |   5.564 | timeout |
+--------+------------+-----------+-----------+-------+---------+---------+
|      7 |      0.141 |     0.155 |     0.485 | 1.805 |   1.846 |   0.401 |
+--------+------------+-----------+-----------+-------+---------+---------+
|      8 |      0.295 |     0.450 |     1.423 | 6.964 |   7.144 | timeout |
+--------+------------+-----------+-----------+-------+---------+---------+
|      9 |      0.140 |     0.153 |     0.417 | 1.848 |   1.664 |   0.334 |
+--------+------------+-----------+-----------+-------+---------+---------+
|     10 |      0.281 |     0.507 |     1.403 | 4.508 |   4.381 | timeout |
+--------+------------+-----------+-----------+-------+---------+---------+
|     11 |      0.129 |     0.113 |     0.284 | 0.626 |   0.646 |   0.217 |
+--------+------------+-----------+-----------+-------+---------+---------+
|     12 |      1.339 |     1.475 |     6.620 | 8.735 |   8.174 | timeout |
+--------+------------+-----------+-----------+-------+---------+---------+
|     13 |      0.238 |     0.317 |     1.010 | 5.748 |   5.728 | timeout |
+--------+------------+-----------+-----------+-------+---------+---------+
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
    ("logic1-cartesian-simplified", logic1_dnf, {"method": "cartesian-simplified"}),
    ("logic1-cartesian-subtree-simplified", logic1_dnf,
     {"method": "cartesian-subtree-simplified"}),
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
