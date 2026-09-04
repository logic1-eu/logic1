from pathlib import Path

import pytest

from logic1.theories.RCF.term import POLYLIB


rcf_dir = Path(__file__).parent

if POLYLIB == 'SAGE':
    required_polylib = 'FLINT'
    inactive_paths = {
        rcf_dir / 'term' / 'term_flint.py',
        rcf_dir / 'test_term_flint.txt',
    }
elif POLYLIB == 'FLINT':
    required_polylib = 'SAGE'
    inactive_paths = {
        rcf_dir / 'term' / 'term_sage.py',
    }
else:
    raise ValueError(f'unknown polynomial backend {POLYLIB!r}')

collect_ignore = [str(path.relative_to(rcf_dir)) for path in inactive_paths]


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip explicitly requested tests for the inactive term backend."""
    skip_inactive = pytest.mark.skip(reason=f'requires the {required_polylib} term backend')
    for item in items:
        if item.path in inactive_paths:
            item.add_marker(skip_inactive)
