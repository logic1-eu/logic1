r"""Result-size comparisons for SMT-LIB DNF computations.

Use this file to obtain atom counts, approximate computation times, and
statuses in a terminal table. For controlled timing benchmarks use
``test_dnf_times.py`` instead.

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

Run the commands below from ``benchmarks/theories/RCF/smtlib/``. The following
command runs all 372 standard computations sequentially, without a
computation time limit::

    pytest test_dnf_sizes.py -m dnf_main

Parallel execution
------------------
With ``pytest-xdist`` installed, ``-n N`` runs up to N computations
concurrently, where N is any positive integer. ``-n auto`` lets
pytest-xdist choose the worker count. Omit ``-n`` for sequential execution.
For example, run the selection above with two workers::

    pytest test_dnf_sizes.py -m dnf_main -n 2

Parallel runs can reduce overall elapsed time but may bias individual
timings.

Limiting computation time
-------------------------
``--dnf-timeout=SECONDS`` sets a wall-clock limit for each selected DNF
computation. Positive floating-point values are permitted. The default is
infinity; ``--dnf-timeout=inf`` explicitly requests no computation limit.
For example, allow half a second per computation::

    pytest test_dnf_sizes.py -m dnf_main --dnf-timeout=0.5

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

    pytest test_dnf_sizes.py -m dnf_under_60s --dnf-timeout=90

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

    pytest test_dnf_sizes.py -m dnf_main -k half-2 --collect-only -q

Run all six standard configurations for half-2::

    pytest test_dnf_sizes.py -m dnf_main -k half-2 --dnf-timeout=600

Run FINAL and AFTER_STEP on both half-2 and levin-rhsc (four computations)::

    pytest test_dnf_sizes.py -m dnf_main --dnf-timeout=600 \
        -k '(half-2 or levin-rhsc) and (final or after-step)'

To select exactly one computation, copy its complete ID from the collection
output::

    pytest 'test_dnf_sizes.py::test_dnf_smtlib[half-2-final]' --dnf-timeout=600

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

Terminal output
---------------
The final table contains one row per executed computation:

=============  ========================================================
Column         Meaning
=============  ========================================================
Input          SMT-LIB input name used in this benchmark set
Configuration  Configuration ID, including budget when varied
Input atoms    Recorded input count, checked after loading
Result atoms   Atom occurrences, including repetitions across disjuncts
Seconds        Approximate computation time, or timeout limit with *
Status         ok, error, or timeout; failed phase where applicable
=============  ========================================================

Times include the parent/worker pipe round trip and exclude startup, input
loading, and output validation/counting. For an error, the elapsed
computation time is shown when available. A computation timeout shows its
limit followed by ``*``; startup and validation timeouts identify their
phase in the status column. Unknown result sizes appear as ``--``. Zero
is a measured atom count; true and false both contain zero atoms.

The table includes timeouts and partial results from failures. Save it
with any pytest failure reports by redirecting terminal output::

    pytest test_dnf_sizes.py -m dnf_main -k half-2 -n 2 --dnf-timeout=600 \
        > half-2-sizes.log 2>&1

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

    pytest test_dnf_sizes.py -k levin-rhsc-during-step --dnf-timeout=60

For sumcracker-14e, ``dnf_pair_checks`` selects budgets 0, 1, 2, 4, 8, 16,
32, 64, 128, 256, 512, 1024, and 2048 for both AFTER_STEP and DURING_STEP.
To reproduce the 60-second budget sweeps, pass ``--dnf-timeout=60``.
Run the positive budgets sequentially for each configuration. ``-x``
stops on an error, but a computation timeout does not stop the sweep::

    pytest test_dnf_sizes.py -m dnf_pair_checks --dnf-timeout=60 \
        -k 'during-step and not 0-' -x
    pytest test_dnf_sizes.py -m dnf_pair_checks --dnf-timeout=60 \
        -k 'after-step and not 0-' -x

Here ``not 0-`` excludes the zero-budget control. Run those two controls with::

    pytest test_dnf_sizes.py -m dnf_pair_checks -k 0- --dnf-timeout=60

To reproduce a sweep that stops at the first timeout, select and run one
budget at a time, inspecting its outcome before doubling the budget.
"""

import pytest

from smtlib_dnf import (DnfCase, DnfRun, IMPLICIT_RANGE_PARAMETERS, MAIN_PARAMETERS,
                        PAIR_PARAMETERS)


def _run_and_report(case: DnfCase, request: pytest.FixtureRequest, timeout: float) -> None:
    """Retain each outcome for the terminal report, including failed calls.

    User properties are transported by pytest-xdist to the controller. The
    local conftest prints them after all selected tests have finished.
    """
    configuration = case.configuration
    if request.node.get_closest_marker("dnf_pair_checks") is not None:
        configuration += f"[pairs={case.options['max_pair_checks']}]"
    run = DnfRun(case, timeout)
    try:
        with run:
            run.compute()
            run.finish()
    except TimeoutError as exc:
        if run.phase != "computation":
            raise
        request.node.user_properties.append(("dnf_timeout", str(exc)))
        pytest.skip(str(exc))
    finally:
        request.node.user_properties.append(("dnf_result", {
            "input": case.name,
            "configuration": configuration,
            "input_atoms": case.input_atoms,
            "output_atoms": run.output_atoms,
            "seconds": run.seconds,
            "status": run.status,
            "phase": run.phase,
            "limit": run.limit,
        }))


@pytest.mark.parametrize("case", MAIN_PARAMETERS + IMPLICIT_RANGE_PARAMETERS)
def test_dnf_smtlib(case: DnfCase, request: pytest.FixtureRequest, dnf_timeout: float) -> None:
    """Compute one DNF and report its size and approximate computation time.
    """
    _run_and_report(case, request, dnf_timeout)


@pytest.mark.parametrize("case", PAIR_PARAMETERS)
def test_dnf_sumcracker_pair_checks(case: DnfCase, request: pytest.FixtureRequest,
                                    dnf_timeout: float) -> None:
    """Report one semantic pair budget on sumcracker-14e.
    """
    _run_and_report(case, request, dnf_timeout)
