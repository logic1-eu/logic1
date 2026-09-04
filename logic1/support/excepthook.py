import sys
from typing import Any, Optional
from types import TracebackType


class NoTraceException(Exception):
    """An exception that prints an error message and exists without a
    traceback. This can be used in situation that do not require inspection of
    the code. Examples are incorrect user input or failure of quantifier
    elimination procedures due to their mathematical incompletess. Both are
    considerd normal situations during interactive use. This exception
    typically comes with a short but informative error message for the user.
    """
    pass


def handler(exc: NoTraceException, tb: Optional[TracebackType]):
    print(f'{exc.args}', file=sys.stderr, flush=True)
    # sys.stderr.write(f{err_type.__name__}: {err}\n")


# Python shell

def excepthook(exc_type: type[BaseException], exc: BaseException, tb: Optional[TracebackType],
               sys_excepthook: Any = sys.excepthook):
    if isinstance(exc, NoTraceException):
        handler(exc, tb)
    else:
        sys_excepthook(exc_type, exc, tb)


# To be executed at import:

sys.excepthook = excepthook


# IPython:

def ipy_custom_exec(ipy: Any, exc_type: type[NoTraceException],
                    exc: NoTraceException, tb: TracebackType, tb_offset=None):
    handler(exc, tb)


# To be executed at import:

# `import IPython` would initialize all of IPython unconditionally,
# i.e., even in an ordinary Python process, thus is avoided.
#
# By contrast, `sys.modules.get('IPython')` returns the module only if it has
# been loaded *before*, which commonly is the case when running in the context
# of IPython or Jupyter.
#
# Caveat: If IPython is loaded *after* Logic1, then the custom exception handler
# will not be registered.
ipy_module = sys.modules.get('IPython')
if ipy_module is not None:
    ipy = ipy_module.get_ipython()
    if ipy is not None:
        ipy.set_custom_exc((NoTraceException,), ipy_custom_exec)