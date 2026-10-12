"""Benchmark mtp-2 quantifier elimination with increasing problem size.

Run with the ``pytest-benchmark`` plugin installed, for example::

    pytest test_qe_mtp2.py --benchmark-only
"""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture


from logic1.theories.RCF import qe, redlog
from mtp2_inputs import mtp2


K_CHOICES = range(1, 6)
EXPECTED_OUTPUT_ATOMS = (2, 8, 36, 160, 700)
EXPECTED_REDLOG_OUTPUT_ATOMS = (2, 8, 36, 160, 710)

CASES = tuple(
    pytest.param(
        k,
        EXPECTED_OUTPUT_ATOMS[k - 1],
        id=f"k={k}",
        marks=pytest.mark.benchmark(group="qe-mtp2"),
    )
    for k in K_CHOICES
)

REDLOG_CASES = tuple(
    pytest.param(
        k,
        EXPECTED_REDLOG_OUTPUT_ATOMS[k - 1],
        id=f"k={k}",
        marks=pytest.mark.benchmark(group="redlog-qe-mtp2"),
    )
    for k in K_CHOICES
)


@pytest.mark.parametrize(("k", "expected"), CASES)
def test_qe_mtp2(benchmark: BenchmarkFixture, k: int, expected: int) -> None:
    """Benchmark QE and verify the output atom count.
    """
    result = benchmark.pedantic(qe, args=(mtp2(k=k),), rounds=1, iterations=1)

    assert result is not None
    assert sum(1 for _ in result.atoms()) == expected


@pytest.mark.parametrize(("k", "expected"), REDLOG_CASES)
def test_redlog_qe_mtp2(benchmark: BenchmarkFixture, k: int, expected: int) -> None:
    """Benchmark Redlog QE and verify the output atom count.
    """
    try:
        result = benchmark.pedantic(redlog.qe, args=(mtp2(k=k),), rounds=1, iterations=1)
    except RuntimeError as exc:
        if "redcsl not found" in str(exc):
            pytest.skip("Redlog not found")
        else:
            raise

    assert result is not None
    assert sum(1 for _ in result.atoms()) == expected
