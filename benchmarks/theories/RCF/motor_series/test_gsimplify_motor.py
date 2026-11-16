"""Benchmarks for ``gsimplify`` on the shared motor test series.

The two configurations compare ``radical=False`` and ``radical=True``.
Both use native CNF with the default options of ``RCF.bnf.cnf``.

Run with the ``pytest-benchmark`` plugin installed, for example::

    pytest test_gsimplify_motor.py --benchmark-only

Historical and current atom counts are shown below. ``Logic1/RL`` preserves
the earlier Logic1 results using Redlog CNF. ``Native CNF`` records Logic1
results verified on 2026-11-16 with the Sage backend, Cartesian CNF,
``Simplification.AFTER_STEP``, and ``max_pair_checks=500``. Both radical
settings produce identical output atom counts for all 13 series, both with
Redlog CNF and with native CNF.

+--------+--------+-------------+-------------+-----------+------------+
| Series |  Input | Redlog 1995 | Redlog 2026 | Logic1/RL | Native CNF |
+========+========+=============+=============+===========+============+
|      1 |    710 |         164 |         185 |       158 |        107 |
+--------+--------+-------------+-------------+-----------+------------+
|      2 |   1420 |         604 |         601 |       396 |        256 |
+--------+--------+-------------+-------------+-----------+------------+
|      3 |     94 |          29 |          35 |        35 |         29 |
+--------+--------+-------------+-------------+-----------+------------+
|      4 |    292 |         165 |         141 |       135 |         87 |
+--------+--------+-------------+-------------+-----------+------------+
|      5 |    157 |          96 |          96 |        87 |         48 |
+--------+--------+-------------+-------------+-----------+------------+
|      6 |    994 |         448 |         379 |       351 |        324 |
+--------+--------+-------------+-------------+-----------+------------+
|      7 |    248 |         107 |         107 |        97 |         83 |
+--------+--------+-------------+-------------+-----------+------------+
|      8 |    473 |         135 |         135 |       129 |        102 |
+--------+--------+-------------+-------------+-----------+------------+
|      9 |    235 |          96 |          96 |        87 |         75 |
+--------+--------+-------------+-------------+-----------+------------+
|     10 |    478 |         283 |         283 |       280 |        245 |
+--------+--------+-------------+-------------+-----------+------------+
|     11 |    168 |          87 |          87 |        78 |         78 |
+--------+--------+-------------+-------------+-----------+------------+
|     12 |   2176 |         489 |         777 |       482 |        349 |
+--------+--------+-------------+-------------+-----------+------------+
|     13 |    358 |         183 |         183 |       177 |        142 |
+--------+--------+-------------+-------------+-----------+------------+
"""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from logic1.theories.RCF import gsimplify
from logic1.theories.RCF.types import Formula

from motor_inputs import MOTOR_SERIES


EXPECTED_OUTPUT_ATOMS = {
    "series1": 107,
    "series2": 256,
    "series3": 29,
    "series4": 87,
    "series5": 48,
    "series6": 324,
    "series7": 83,
    "series8": 102,
    "series9": 75,
    "series10": 245,
    "series11": 78,
    "series12": 349,
    "series13": 142,
}

GSIMPLIFY_CONFIGURATIONS = (
    ("radical-false", {"radical": False}),
    ("radical-true", {"radical": True}),
)

CASES = tuple(
    pytest.param(formula, expected_input_atoms, options,
                 EXPECTED_OUTPUT_ATOMS[name],
                 id=f"{name}-{configuration_id}",
                 marks=pytest.mark.benchmark(
                     group=f"gsimplify-motor-{configuration_id}"))
    for name, formula, expected_input_atoms in MOTOR_SERIES
    for configuration_id, options in GSIMPLIFY_CONFIGURATIONS
)


@pytest.mark.parametrize(
    ("formula", "expected_input_atoms", "options", "expected_output_atoms"),
    CASES,
)
def test_gsimplify_motor(benchmark: BenchmarkFixture, formula: Formula,
                         expected_input_atoms: int,
                         options: dict[str, bool],
                         expected_output_atoms: int) -> None:
    """Benchmark simplification and verify input and output atom counts."""
    assert len(list(formula.atoms())) == expected_input_atoms

    result = benchmark.pedantic(gsimplify, args=(formula,),
                                kwargs=options,
                                rounds=1, iterations=1)

    assert len(list(result.atoms())) == expected_output_atoms
