"""Benchmark mtp-3 quantifier elimination with several worker counts.

Run with the ``pytest-benchmark`` plugin installed, for example::

    pytest test_qe_mtp3.py --benchmark-only
"""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from logic1.theories.RCF import qe

from mtp3_inputs import MTP3_INPUT, MTP3_INPUT_ATOMS, MTP_OUTPUT_LOWER_BOUND, MTP_OUTPUT_UPPER_BOUND


WORKER_COUNTS = (0, 1, 2, 4, 8)

CASES = tuple(
    pytest.param(
        workers,
        id=f"workers={workers}",
        marks=pytest.mark.benchmark(group="qe-mtp3"),
    )
    for workers in WORKER_COUNTS
)


@pytest.mark.parametrize("workers", CASES)
def test_qe_mtp3(benchmark: BenchmarkFixture, workers: int) -> None:
    """Benchmark QE and verify the input size and approximate output size.
    """
    assert len(list(MTP3_INPUT.atoms())) == MTP3_INPUT_ATOMS

    result = benchmark.pedantic(
        qe,
        args=(MTP3_INPUT,),
        kwargs={"workers": workers},
        rounds=1,
        iterations=1,
    )

    assert result is not None
    output_atoms = sum(1 for _ in result.atoms())
    assert MTP_OUTPUT_LOWER_BOUND < output_atoms < MTP_OUTPUT_UPPER_BOUND
