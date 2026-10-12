"""Benchmarks for ``gsimplify`` on the shared motor test series.

Run with the ``pytest-benchmark`` plugin installed, for example::

    pytest test_gsimplify_motor.py --benchmark-only

Historical and current atom counts:

+-------------+-------------+-------------+-------------+-------------+
| series      | input       | Redlog 1995 | Redlog 2026 | Logic1      |
+=============+=============+=============+=============+=============+
|           1 |         710 |         164 |         185 |         158 |
+-------------+-------------+-------------+-------------+-------------+
|           2 |        1420 |         604 |         601 |         396 |
+-------------+-------------+-------------+-------------+-------------+
|           3 |          94 |          29 |          35 |          35 |
+-------------+-------------+-------------+-------------+-------------+
|           4 |         292 |         165 |         141 |         135 |
+-------------+-------------+-------------+-------------+-------------+
|           5 |         157 |          96 |          96 |          87 |
+-------------+-------------+-------------+-------------+-------------+
|           6 |         994 |         448 |         379 |         351 |
+-------------+-------------+-------------+-------------+-------------+
|           7 |         248 |         107 |         107 |          97 |
+-------------+-------------+-------------+-------------+-------------+
|           8 |         473 |         135 |         135 |         129 |
+-------------+-------------+-------------+-------------+-------------+
|           9 |         235 |          96 |          96 |          87 |
+-------------+-------------+-------------+-------------+-------------+
|          10 |         478 |         283 |         283 |         280 |
+-------------+-------------+-------------+-------------+-------------+
|          11 |         168 |          87 |          87 |          78 |
+-------------+-------------+-------------+-------------+-------------+
|          12 |        2176 |         489 |         777 |         482 |
+-------------+-------------+-------------+-------------+-------------+
|          13 |         358 |         183 |         183 |         177 |
+-------------+-------------+-------------+-------------+-------------+
"""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from logic1.theories.RCF import gsimplify
from logic1.theories.RCF.types import Formula

from motor_inputs import MOTOR_SERIES


EXPECTED_OUTPUT_ATOMS = {
    "series1": 158,
    "series2": 396,
    "series3": 35,
    "series4": 135,
    "series5": 87,
    "series6": 351,
    "series7": 97,
    "series8": 129,
    "series9": 87,
    "series10": 280,
    "series11": 78,
    "series12": 482,
    "series13": 177,
}

GSIMPLIFY_CONFIGURATIONS = (
    ("radical-false", {"use_redlog_cnf": True, "radical": False}),
    ("radical-true", {"use_redlog_cnf": True, "radical": True}),
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
