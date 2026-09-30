"""Benchmark period-9 quantifier elimination with several worker counts.

Run with the ``pytest-benchmark`` plugin installed, for example::

    pytest test_qe_period9.py --benchmark-only
"""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from logic1.firstorder import T
from logic1.theories.RCF import qe

from period9_inputs import PERIOD9_INPUT, PERIOD9_INPUT_ATOMS


WORKER_COUNTS = (0, 1, 2, 4, 8)

CASES = tuple(
    pytest.param(
        workers,
        id=f"workers-{workers}",
        marks=pytest.mark.benchmark(group="qe-period9"),
    )
    for workers in WORKER_COUNTS
)


@pytest.mark.parametrize("workers", CASES)
def test_qe_period9(benchmark: BenchmarkFixture, workers: int) -> None:
    """Benchmark QE and verify the input size and expected truth value."""
    assert len(list(PERIOD9_INPUT.atoms())) == PERIOD9_INPUT_ATOMS

    result = benchmark.pedantic(
        qe,
        args=(PERIOD9_INPUT,),
        kwargs={"workers": workers},
        rounds=1,
        iterations=1,
    )

    assert result is T
