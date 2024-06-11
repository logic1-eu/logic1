# Inspired by code suggested by Vincent Fenet,
# https://stackoverflow.com/q/8315389/

import sys
from functools import wraps
import pprint


class trace(object):
    """Implements a decorator @trace(...) on functions that should be traced.
    Parenthesis must be used also when there are no arguments.
    """

    def __init__(self, stream=sys.stdout, indent_step=2, show_ret=True,
                 pretty=False):
        self.indent_step = indent_step
        self.pretty = pretty
        self.show_ret = show_ret
        self.stream = stream
        # The following is a class attribute since we want to share the
        # indentation level between different traced functions, in case they
        # call each other:
        trace.cur_indent = 0

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            indent = ' ' * trace.cur_indent
            L = []
            for a in args:
                L.append(self._format(a))
            for a, b in kwargs.items():
                L.append('%s=%s' % (a, self._format(b)))
            arg_str = ', '.join(L)
            call = f'{fn.__qualname__}({arg_str})'
            self.stream.write(f'{indent}--> {call}\n')
            trace.cur_indent += self.indent_step
            ret = fn(*args, **kwargs)
            trace.cur_indent -= self.indent_step
            if self.show_ret:
                ret_str = self._format(ret)
                result = f'{fn.__qualname__} == {ret_str}\n'
                self.stream.write(f'{indent}<-- {result}\n')
            return ret
        return wrapper

    def _format(self, obj) -> str:
        return pprint.pformat(obj) if self.pretty else repr(obj)
