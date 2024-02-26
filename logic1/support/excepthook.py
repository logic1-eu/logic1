import sys
from typing import Any, Optional
from types import TracebackType


class NoTraceException(Exception):
    pass


def handler(exc: NoTraceException, tb: Optional[TracebackType]):
    print(f'{exc.args[0]}', file=sys.stderr, flush=True)
    # sys.stderr.write(f{err_type.__name__}: {err}\n")


# Python shell

def excepthook(exc_type: type[BaseException], exc: BaseException, tb: Optional[TracebackType]):
    if isinstance(exc, NoTraceException):
        handler(exc, tb)
    else:
        sys_excepthook(exc_type, exc, tb)


# To be executed at import:

sys_excepthook = sys.excepthook
sys.excepthook = excepthook


# iPhyton:

def ipy_custom_exec(ipy: Any, exc_type: type[NoTraceException],
                    exc: NoTraceException, tb: TracebackType, tb_offset=None):
    handler(exc, tb)


# To be executed at import:

try:
    import IPython
except ImportError:
    ipy = None
else:
    ipy = IPython.get_ipython()

if ipy is not None:
    ipy.set_custom_exc((NoTraceException,), ipy_custom_exec)
