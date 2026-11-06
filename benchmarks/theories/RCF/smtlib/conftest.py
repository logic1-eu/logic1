"""Selection markers and terminal reporting for the SMT-LIB DNF services.
"""

import math
import re
from typing import cast, TypedDict

import pytest
from _pytest.terminal import TerminalReporter


class DnfRecord(TypedDict):
    """Serializable result attached to a size test's pytest report.
    """

    input: str
    configuration: str
    input_atoms: int
    output_atoms: int | None
    seconds: float | None
    status: str
    phase: str
    limit: float


def pytest_addoption(parser: pytest.Parser) -> None:
    """Accept a computation limit independently of benchmark selection.
    """
    parser.getgroup("SMT-LIB DNF").addoption(
        "--dnf-timeout", type=float, default=math.inf, metavar="SECONDS",
        help="Wall-clock limit per DNF computation in seconds (default: infinity).")


@pytest.fixture
def dnf_timeout(request: pytest.FixtureRequest) -> float:
    """Supply the command-line computation limit to either pytest service.
    """
    return float(request.config.getoption("dnf_timeout"))


def pytest_configure(config: pytest.Config) -> None:
    """Register selectors shared by the timing and size services.
    """
    timeout = config.getoption("dnf_timeout")
    if math.isnan(timeout) or timeout <= 0:
        raise pytest.UsageError("--dnf-timeout must be positive seconds or inf.")
    if not config.pluginmanager.hasplugin("benchmark"):
        config.addinivalue_line("markers",
                               "benchmark: timing configuration shared with size tests")
    for marker in ("dnf_timing: controlled DNF timings; requires sequential execution",
            "dnf_main: the 62 inputs and six standard DNF configurations",
            "dnf_implicit_ranges: the additional levin-rhsc configuration",
            "dnf_pair_checks: sumcracker-14e semantic pair-budget experiments",
            "dnf_under_60s: all six standard configurations completed below 60 seconds",
            "dnf_under_600s: all six standard configurations completed below 600 seconds"):
        config.addinivalue_line("markers", marker)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Reject selected parallel timing cases before fixture setup.
    """
    config = item.config
    parallel = bool(config.getoption("numprocesses", default=0)) or hasattr(config, "workerinput")
    if parallel and item.get_closest_marker("dnf_timing"):
        pytest.fail("DNF timing benchmarks require sequential execution. "
                    "Omit -n or select test_dnf_sizes.py for parallel runs.", pytrace=False)


@pytest.hookimpl(tryfirst=True)
def pytest_report_teststatus(report: pytest.TestReport) -> tuple[str, str, str] | None:
    """Count expected computation timeouts separately from test failures.

    The underlying skipped outcome preserves successful exit status and
    JUnit compatibility. A separate terminal category avoids duplicate
    failure or skip listings; size details remain in the result table.
    """
    if (report.when == "call" and report.skipped
            and any(name == "dnf_timeout" for name, _ in report.user_properties)):
        return "timeout", "t", "TIMEOUT"
    return None


def _record_key(record: DnfRecord) -> tuple[str, int, str, int]:
    """Keep each input's configurations together, with budgets ascending.
    """
    configurations = ("redlog-off", "redlog-on", "none", "final", "after-step", "during-step")
    configuration = record["configuration"]
    index = next((i for i, name in enumerate(configurations)
                  if configuration == name or configuration.startswith(name + "[")),
                 len(configurations))
    prefix, _, suffix = configuration.partition("[")
    budget = re.search(r"\d+", suffix)
    return record["input"], index, prefix, int(budget.group()) if budget else -1


def _record_row(record: DnfRecord) -> tuple[str, str, str, str, str, str]:
    """Distinguish a computation timeout from startup or validation failure.
    """
    seconds = "--" if record["seconds"] is None else f'{record["seconds"]:.3f}'
    status = record["status"]
    if status == "timeout" and record["phase"] == "computation":
        seconds = f'{record["limit"]:g}*'
    elif status != "ok":
        status += f' ({record["phase"]})'
    atoms = "--" if record["output_atoms"] is None else str(record["output_atoms"])
    return (record["input"], record["configuration"], str(record["input_atoms"]),
            atoms, seconds, status)


def pytest_terminal_summary(terminalreporter: TerminalReporter) -> None:
    """Display size-test results collected locally or from xdist workers.

    Test reports carry the same user properties in serial and parallel runs.
    Only call reports with an explicit result record contribute rows; tests
    that were deselected or never started do not acquire fabricated results.
    """
    records: dict[str, DnfRecord] = {}
    for reports in terminalreporter.stats.values():
        for report in reports:
            if not isinstance(report, pytest.TestReport) or report.when != "call":
                continue
            for name, value in report.user_properties:
                if name == "dnf_result":
                    records[report.nodeid] = cast(DnfRecord, value)
    if not records:
        return

    headers = ("Input", "Configuration", "Input atoms", "Result atoms", "Seconds", "Status")
    rows = [headers, *(_record_row(record)
                       for record in sorted(records.values(), key=_record_key))]
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    terminalreporter.write_sep("=", "DNF result sizes (approximate wall times)")
    for row in rows:
        terminalreporter.write_line("  ".join(value.ljust(width) if i in (0, 1, 5)
                                             else value.rjust(width)
                                             for i, (value, width) in enumerate(zip(row, widths))))
    terminalreporter.write_line("-- means unknown; 0 is a measured atom count.")
    terminalreporter.write_line("* marks the computation timeout limit, not a completed duration.")
    terminalreporter.write_line("Times exclude loading and validation; "
                               "parallel runs share resources.")
