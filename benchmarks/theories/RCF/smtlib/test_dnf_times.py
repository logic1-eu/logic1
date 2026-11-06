r"""Timing benchmarks for SMT-LIB DNF computations.

Use this file for controlled timings with output checks. For atom counts
and approximate times in a terminal table, or parallel execution, use
``test_dnf_sizes.py`` instead.

Running the standard benchmark set
==================================
The standard benchmark set contains 62 SMT-LIB input formulas. They have names
such as ``half-2`` and ``levin-rhsc``, and are linked to SMT-LIB files through
the registry in ``smtlib_inputs.py``. Each input formula occupies one row in the
result tables in ``dnf_results.md``.

An input formula can be processed with any of the following six configurations:

================  ==================================================
Configuration ID  Computation
================  ==================================================
redlog-off        redlog.dnf(bnfsac=False, bnfsm=False)
redlog-on         redlog.dnf(bnfsac=True, bnfsm=True)
none              bnf.dnf(simplification=Simplification.NONE)
final             bnf.dnf(simplification=Simplification.FINAL)
after-step        bnf.dnf(simplification=Simplification.AFTER_STEP)
during-step       bnf.dnf(simplification=Simplification.DURING_STEP)
================  ==================================================

The ``redlog.dnf`` calls require ``redcsl`` in the PATH; its version can affect
results and timings. The ``bnf.dnf`` calls use ``method="cartesian"`` and
``max_pair_checks=500``.

Each of the six configurations occupies one result column in the tables in
``dnf_results.md``. This yields 62 × 6 = 372 standard computations.

Each standard computation supplies the atom count and time for one result cell
in the corresponding table in ``dnf_results.md``. Standard computations are
labelled with the marker ``dnf_main``.

These tests require ``pytest-benchmark`` and run sequentially. Omit ``-n``;
parallel execution is rejected because concurrent computations can bias
timings.

Run the commands below from ``benchmarks/theories/RCF/smtlib/``. The following
command runs all 372 standard computations without a computation time limit::

    pytest test_dnf_times.py -m dnf_main

Limiting computation time
-------------------------
``--dnf-timeout=SECONDS`` sets a wall-clock limit for each selected DNF
computation. Positive floating-point values are permitted. The default is
infinity; ``--dnf-timeout=inf`` explicitly requests no computation limit.
For example, allow half a second per computation::

    pytest test_dnf_times.py -m dnf_main --dnf-timeout=0.5

At the deadline, the worker process group is killed, including any child
Reduce process, and the outcome is reported as a timeout. The limit excludes
startup, input loading, and result validation. No limit is stored with an
input or configuration. Pytest-timeout's ``--timeout`` is a different option;
it does not set the DNF computation limit.

Selecting specific inputs and configurations
--------------------------------------------
``-m`` selects computations by labels called markers. For an input to
qualify for a runtime marker, all six standard configurations must have
completed successfully below its threshold in ``dnf_results.md``:

==============  ==============================  ======  ============
Marker          Selection                       Inputs  Computations
==============  ==============================  ======  ============
dnf_main        All standard inputs                 62           372
dnf_under_60s   All six configurations < 60 s       46           276
dnf_under_600s  All six configurations < 600 s      52           312
==============  ==============================  ======  ============

Every input in ``dnf_under_60s`` also belongs to ``dnf_under_600s``. Inputs
with unresolved timeouts or errors belong to neither runtime group. These
markers apply only to standard computations; they neither guarantee future
runtimes nor set the timeout.

For example, select inputs previously completed within 60 seconds per
configuration, allowing each new computation 90 seconds::

    pytest test_dnf_times.py -m dnf_under_60s --dnf-timeout=90

Use a single ``-m`` argument. In a quoted expression, ``and`` requires both
markers. For example, ``-m 'dnf_under_600s and not dnf_under_60s'`` selects
six inputs (36 computations) whose recorded maximum is between 60 and
600 seconds.

Within the standard benchmark set, ``-k`` selects input names and
configuration IDs. A computation must satisfy both ``-m`` and ``-k`` when both
are supplied. After selecting the standard set with ``-m dnf_main``:

=====================================================  =====
Additional filter after -m dnf_main                    Count
=====================================================  =====
-k half-2                                                  6
-k final                                                  62
-k 'half-2 and final'                                      1
-k '(half-2 or levin-rhsc) and (final or after-step)'      4
=====================================================  =====

Expressions with spaces must be quoted. Both options accept ``and``,
``or``, ``not``, and parentheses; ``-k`` matches substrings of names.
Use ``-k 'not redlog'`` to omit Redlog computations.

List selected computations without computing DNFs by adding
``--collect-only -q``::

    pytest test_dnf_times.py -m dnf_main -k half-2 --collect-only -q

Run all six standard configurations for half-2::

    pytest test_dnf_times.py -m dnf_main -k half-2 --dnf-timeout=600

Run FINAL and AFTER_STEP on both half-2 and levin-rhsc (four computations)::

    pytest test_dnf_times.py -m dnf_main --dnf-timeout=600 \
        -k '(half-2 or levin-rhsc) and (final or after-step)'

To select exactly one computation, copy its complete ID from the collection
output::

    pytest 'test_dnf_times.py::test_dnf_smtlib[half-2-final]' --dnf-timeout=600

Checks and outcomes
-------------------
Each computation runs in a fresh worker. Startup/input loading and result
validation each have a separate 30-second guard.

Checks verify the input atom count, quantifier-free DNF structure, and the
absence of new variables. Redlog lowercasing is accepted only when distinct
input names remain distinct; collisions fail before computation. Native
configurations preserve case. These checks do not prove equivalence or
require the result sizes recorded in the reference tables.

Computation timeouts are expected outcomes, shown as ``t`` in the progress
line and counted separately as ``timeout`` in the terminal summary. They
do not cause a failing exit status or stop execution with ``-x``. JUnit
reports represent them as skipped tests. Errors, failed checks, and
startup or validation timeouts remain failures; ``-x`` stops on these.
Previously timed-out or failed computations remain selectable.

Timing and terminal output
--------------------------
Pytest-benchmark times each selected computation once, with one round, one
iteration, and no warmup. The measured interval includes the parent/worker
pipe round trip; startup, input loading, and output validation/counting
are outside timing. Do not use ``--benchmark-disable`` to collect timings.

The terminal table shows timing statistics. With one sample, min, mean,
and max describe the same observation, not variability across repeated
runs. Input and result atom counts are stored as ``input_atoms`` and
``output_atoms`` in benchmark JSON metadata. For counts directly in the
terminal, use ``test_dnf_sizes.py``.

Save timings and atom counts to JSON, test outcomes to JUnit XML, and
terminal output to a log::

    pytest test_dnf_times.py -m dnf_main -k half-2 --dnf-timeout=600 \
        --benchmark-json=half-2-timing.json --junitxml=half-2-timing.xml \
        > half-2-timing.log 2>&1

Benchmark JSON can contain a completed timed call whose later validation
failed. Use the JUnit report to distinguish successful tests, expected
computation timeouts (recorded as skipped), and failures during startup,
computation, or validation.

Recorded results and discussion are in `dnf_results.md <dnf_results.md>`_.
PyEDA is not executed by these tests; its saved results are included there.
Input credits and selection history are recorded in ``NOTES.md``.


Supplementary experiments
=========================
There are 27 additional computations: one varies ``implicit_ranges`` on
levin-rhsc; 26 vary ``max_pair_checks`` on sumcracker-14e (13 budgets for
each of AFTER_STEP and DURING_STEP). Together with the 372 standard
computations, these give 399. Omitting ``-m`` selects all 399 before any
``-k`` filtering.

To select supplementary experiments, use these markers:

===================  ===============================  ============
Marker               Selection                        Computations
===================  ===============================  ============
dnf_implicit_ranges  Extra levin-rhsc setting                    1
dnf_pair_checks      sumcracker-14e: max_pair_checks            26
===================  ===============================  ============

Neither experiment has a runtime marker: the standard configurations'
recorded times do not classify these modified computations.

The standard DURING_STEP computation for levin-rhsc uses
``implicit_ranges=True``. The marker ``dnf_implicit_ranges`` selects
only the additional computation with this option disabled during initial
simplification. To compare both settings using DURING_STEP, select them
by name without restricting to ``dnf_main``::

    pytest test_dnf_times.py -k levin-rhsc-during-step --dnf-timeout=60

For sumcracker-14e, ``dnf_pair_checks`` selects budgets 0, 1, 2, 4, 8, 16,
32, 64, 128, 256, 512, 1024, and 2048 for both AFTER_STEP and DURING_STEP.
To reproduce the 60-second budget sweeps, pass ``--dnf-timeout=60``.
Run the positive budgets sequentially for each configuration. ``-x``
stops on an error, but a computation timeout does not stop the sweep::

    pytest test_dnf_times.py -m dnf_pair_checks --dnf-timeout=60 \
        -k 'during-step and not 0-' -x
    pytest test_dnf_times.py -m dnf_pair_checks --dnf-timeout=60 \
        -k 'after-step and not 0-' -x

Here ``not 0-`` excludes the zero-budget control. Run those two controls with::

    pytest test_dnf_times.py -m dnf_pair_checks -k 0- --dnf-timeout=60

To reproduce a sweep that stops at the first timeout, select and run one
budget at a time, inspecting its outcome before doubling the budget.
"""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from smtlib_dnf import (DnfCase, DnfRun, IMPLICIT_RANGE_PARAMETERS, MAIN_PARAMETERS,
                        PAIR_PARAMETERS)


pytestmark = pytest.mark.dnf_timing


def _benchmark_dnf(benchmark: BenchmarkFixture, case: DnfCase, timeout: float,
                   request: pytest.FixtureRequest) -> None:
    """Time computation, then validate and count its result outside timing.
    """
    with DnfRun(case, timeout) as run:
        try:
            benchmark.pedantic(run.compute, rounds=1, iterations=1, warmup_rounds=0)
        except TimeoutError as exc:
            request.node.user_properties.append(("dnf_timeout", str(exc)))
            pytest.skip(str(exc))
        benchmark.extra_info["input_atoms"] = case.input_atoms
        benchmark.extra_info["output_atoms"] = run.finish()


@pytest.mark.parametrize("case", MAIN_PARAMETERS + IMPLICIT_RANGE_PARAMETERS)
def test_dnf_smtlib(benchmark: BenchmarkFixture, case: DnfCase, dnf_timeout: float,
                    request: pytest.FixtureRequest) -> None:
    """Measure one input/configuration with structural output assertions.
    """
    _benchmark_dnf(benchmark, case, dnf_timeout, request)


@pytest.mark.parametrize("case", PAIR_PARAMETERS)
def test_dnf_sumcracker_pair_checks(benchmark: BenchmarkFixture, case: DnfCase,
                                    dnf_timeout: float, request: pytest.FixtureRequest) -> None:
    """Measure one semantic pair budget on sumcracker-14e.
    """
    _benchmark_dnf(benchmark, case, dnf_timeout, request)
