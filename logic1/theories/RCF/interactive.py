from .bnf import cnf as _cnf
from .bnf import dnf as _dnf
from .parser import l1
from .pnf import pnf as _pnf
from .rcf import Term, Variable, ring, Eq, Ne, Ge, Le, Gt, Lt  # noqa
from .simplify import simplify as _simplify


def cnf(x, *args, **kwargs):
    if isinstance(x, str):
        x = l1(x)
    return _cnf(x, *args, **kwargs)


def dnf(x, *args, **kwargs):
    if isinstance(x, str):
        x = l1(x)
    return _dnf(x, *args, **kwargs)


def pnf(x, *args, **kwargs):
    if isinstance(x, str):
        x = l1(x)
    return _pnf(x, *args, **kwargs)


def simplify(x, *args, **kwargs):
    if isinstance(x, str):
        x = l1(x)
    return _simplify(x, *args, **kwargs)
