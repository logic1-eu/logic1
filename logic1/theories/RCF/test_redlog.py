"""Tests for Redlog output extraction and failure diagnostics.

Subprocess calls are mocked; these tests do not require Reduce.
"""

import subprocess
from unittest.mock import patch

import pytest

from logic1.theories.RCF import redlog


@pytest.mark.parametrize(('output', 'expected'), [
        (f'{redlog._START}T{redlog._END}',
         'T'),
        (f'Reduce banner\n{redlog._START}\n  T  \n{redlog._END}\nGoodbye\n',
         '\n  T  \n'),
        (f'{redlog._START}{redlog._END}',
         '')])
def test_unwrap(output: str, expected: str) -> None:
    assert redlog._unwrap(output) == expected


@pytest.mark.parametrize('output', [
        f'Reduce banner\nT{redlog._END}',
        f'Reduce banner\n{redlog._START}T',
        'Reduce (CSL, rev 7327),  7-Mar-2026 ...\n!!! Aborting\n',
        f'Reduce banner\n{redlog._END}T{redlog._START}'])
def test_unwrap_rejects_missing_or_reversed_markers(output: str) -> None:
    with pytest.raises(RuntimeError, match='did not return a complete marked result') as exc:
        redlog._unwrap(output)
    assert output in str(exc.value)


@pytest.mark.parametrize('returncode', [1, -6])
def test_call_redlog_reports_failed_process(returncode: int) -> None:
    completed = subprocess.CompletedProcess('redcsl', returncode,
                                            b'!!! Aborting: allocator failure\n',
                                            b'process error\n')
    with patch.object(redlog.subprocess, 'run', return_value=completed):
        with pytest.raises(RuntimeError) as exc:
            redlog._call_redlog('r2py_formula true')
    message = str(exc.value)
    assert f'redcsl failed with exit code {returncode}' in message
    assert 'stdout:\n!!! Aborting: allocator failure\n' in message
    assert 'stderr:\nprocess error\n' in message


def test_call_redlog_rejects_unmarked_success() -> None:
    output = 'Reduce (CSL, rev 7327),  7-Mar-2026 ...\n***** Evaluation failed\n'
    completed = subprocess.CompletedProcess('redcsl', 0, output.encode(), b'')
    with patch.object(redlog.subprocess, 'run', return_value=completed):
        with pytest.raises(RuntimeError, match='did not return a complete marked result') as exc:
            redlog._call_redlog('r2py_formula true')
    assert output in str(exc.value)


def test_call_redlog_reports_missing_executable() -> None:
    completed = subprocess.CompletedProcess('redcsl', 127, b'', b'redcsl: command not found\n')
    with patch.object(redlog.subprocess, 'run', return_value=completed):
        with pytest.raises(RuntimeError, match='redcsl not found'):
            redlog._call_redlog('r2py_formula true')
