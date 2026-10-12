"""Benchmarks for ``simplify`` on the shared motor test series.

Run with the ``pytest-benchmark`` plugin installed, for example::

    pytest test_simplify_motor.py --benchmark-only

Historical and current atom counts:

Redlog 2024 differs from Redlog 1995 only for series 2 (935 instead of 966)
and series 6 (988 instead of 908).

The Logic1 column headings have the form ``implicit_ranges/substitute``, with
``F`` for ``False`` and ``T`` for ``True``.

+--------+-------+-------------+-----+-----+-----+-----+-----+-----+
| series | input | Redlog 1995 | F/0 | F/1 | F/2 | T/0 | T/1 | T/2 |
+========+=======+=============+=====+=====+=====+=====+=====+=====+
|      1 |   710 |         674 | 674 | 641 | 491 | 674 | 641 | 491 |
+--------+-------+-------------+-----+-----+-----+-----+-----+-----+
|      2 |  1420 |         966 | 966 | 815 | 815 | 922 | 801 | 801 |
+--------+-------+-------------+-----+-----+-----+-----+-----+-----+
|      3 |    94 |          88 |  88 |  76 |  72 |  74 |  76 |  72 |
+--------+-------+-------------+-----+-----+-----+-----+-----+-----+
|      4 |   292 |         259 | 259 | 283 | 190 | 250 | 274 | 184 |
+--------+-------+-------------+-----+-----+-----+-----+-----+-----+
|      5 |   157 |         139 | 139 | 141 | 141 | 130 | 132 | 132 |
+--------+-------+-------------+-----+-----+-----+-----+-----+-----+
|      6 |   994 |         908 | 908 | 715 | 412 | 667 | 601 | 412 |
+--------+-------+-------------+-----+-----+-----+-----+-----+-----+
|      7 |   248 |         199 | 199 | 186 | 135 | 199 | 186 | 135 |
+--------+-------+-------------+-----+-----+-----+-----+-----+-----+
|      8 |   473 |         410 | 410 | 328 | 242 | 361 | 312 | 242 |
+--------+-------+-------------+-----+-----+-----+-----+-----+-----+
|      9 |   235 |         188 | 188 | 175 | 127 | 188 | 175 | 127 |
+--------+-------+-------------+-----+-----+-----+-----+-----+-----+
|     10 |   478 |         461 | 461 | 426 | 413 | 461 | 426 | 413 |
+--------+-------+-------------+-----+-----+-----+-----+-----+-----+
|     11 |   168 |         156 | 156 | 151 | 146 | 140 | 145 | 140 |
+--------+-------+-------------+-----+-----+-----+-----+-----+-----+
|     12 |  2176 |        2100 |2100 |1977 |1918 |2054 |1977 |1914 |
+--------+-------+-------------+-----+-----+-----+-----+-----+-----+
|     13 |   358 |         342 | 342 | 303 | 294 | 342 | 303 | 294 |
+--------+-------+-------------+-----+-----+-----+-----+-----+-----+
"""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from logic1.theories.RCF.simplify import simplify
from logic1.theories.RCF.types import Formula

from motor_inputs import MOTOR_SERIES


SIMPLIFY_CONFIGURATIONS = (
    ("implicit-ranges-false-substitute-0",
     {"substitute": 0, "implicit_ranges": False}),
    ("implicit-ranges-false-substitute-1",
     {"substitute": 1, "implicit_ranges": False}),
    ("implicit-ranges-false-substitute-2",
     {"substitute": 2, "implicit_ranges": False}),
    ("implicit-ranges-true-substitute-0",
     {"substitute": 0, "implicit_ranges": True}),
    ("implicit-ranges-true-substitute-1",
     {"substitute": 1, "implicit_ranges": True}),
    ("implicit-ranges-true-substitute-2",
     {"substitute": 2, "implicit_ranges": True}),
)

# Columns follow SIMPLIFY_CONFIGURATIONS.
EXPECTED_OUTPUT_ATOMS = {
    "series1": (674, 641, 491, 674, 641, 491),
    "series2": (966, 815, 815, 922, 801, 801),
    "series3": (88, 76, 72, 74, 76, 72),
    "series4": (259, 283, 190, 250, 274, 184),
    "series5": (139, 141, 141, 130, 132, 132),
    "series6": (908, 715, 412, 667, 601, 412),
    "series7": (199, 186, 135, 199, 186, 135),
    "series8": (410, 328, 242, 361, 312, 242),
    "series9": (188, 175, 127, 188, 175, 127),
    "series10": (461, 426, 413, 461, 426, 413),
    "series11": (156, 151, 146, 140, 145, 140),
    "series12": (2100, 1977, 1918, 2054, 1977, 1914),
    "series13": (342, 303, 294, 342, 303, 294),
}

CASES = tuple(
    pytest.param(formula, expected_input_atoms, options,
                 EXPECTED_OUTPUT_ATOMS[name][configuration_index],
                 id=f"{name}-{configuration_id}",
                 marks=pytest.mark.benchmark(
                     group=f"simplify-motor-{configuration_id}"))
    for name, formula, expected_input_atoms in MOTOR_SERIES
    for configuration_index, (configuration_id, options)
    in enumerate(SIMPLIFY_CONFIGURATIONS)
)


@pytest.mark.parametrize(
    ("formula", "expected_input_atoms", "options", "expected_output_atoms"),
    CASES,
)
def test_simplify_motor(benchmark: BenchmarkFixture, formula: Formula,
                        expected_input_atoms: int,
                        options: dict[str, bool | int],
                        expected_output_atoms: int) -> None:
    """Benchmark simplification and verify input and output atom counts."""
    assert len(list(formula.atoms())) == expected_input_atoms

    result = benchmark.pedantic(simplify, args=(formula,), kwargs=options,
                                rounds=1, iterations=1)

    assert len(list(result.atoms())) == expected_output_atoms
