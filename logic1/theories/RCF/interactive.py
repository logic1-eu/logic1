from typing import overload

from .bnf import cnf as _cnf
from .bnf import dnf as _dnf
from .parser import l1
from .pnf import pnf as _pnf
from .qe import VirtualSubstitution as _VirtualSubstitution
from . import rcf
from .rcf import AtomicFormula, Polynomial, Variable, ring  # noqa
from .simplify import simplify as _simplify


# def patch_repr(self):
#     return f'{self.func.__name__}({self.lhs.poly}, {self.rhs.poly})'


# AtomicFormula.__repr__ = patch_repr  # type: ignore[method-assign]


def Term(arg: object) -> rcf.Term:
    match arg:
        case Polynomial():
            return rcf.Term(arg)
        case rcf.Term():
            # discuss: The user should not have to distinguish Term(1 + x) from
            # Term(x + 1).
            return arg
        case _:
            # discuss: Using ring as fallback. Better handle admissible types
            # explicitly?
            return rcf.Term(ring(arg))


def Eq(lhs: object, rhs: object) -> rcf.Eq:
    return _R(rcf.Eq, lhs, rhs)


def Ne(lhs: object, rhs: object) -> rcf.Ne:
    return _R(rcf.Ne, lhs, rhs)


def Le(lhs: object, rhs: object) -> rcf.Le:
    return _R(rcf.Le, lhs, rhs)


def Lt(lhs: object, rhs: object) -> rcf.Lt:
    return _R(rcf.Lt, lhs, rhs)


def Ge(lhs: object, rhs: object) -> rcf.Ge:
    return _R(rcf.Ge, lhs, rhs)


def Gt(lhs: object, rhs: object) -> rcf.Gt:
    return _R(rcf.Gt, lhs, rhs)


@overload
def _R(rel: type[rcf.Eq], lhs: object, rhs: object) -> rcf.Eq:
    ...


@overload
def _R(rel: type[rcf.Ne], lhs: object, rhs: object) -> rcf.Ne:
    ...


@overload
def _R(rel: type[rcf.Le], lhs: object, rhs: object) -> rcf.Le:
    ...


@overload
def _R(rel: type[rcf.Lt], lhs: object, rhs: object) -> rcf.Lt:
    ...


@overload
def _R(rel: type[rcf.Ge], lhs: object, rhs: object) -> rcf.Ge:
    ...


@overload
def _R(rel: type[rcf.Gt], lhs: object, rhs: object) -> rcf.Gt:
    ...


def _R(rel: type[AtomicFormula], lhs: object, rhs: object) -> AtomicFormula:
    if not isinstance(lhs, rcf.Term):
        lhs = Term(lhs)
    if not isinstance(rhs, rcf.Term):
        rhs = Term(rhs)
    return rel(lhs, rhs)


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


class VirtualSubstitution(_VirtualSubstitution):

    def __call__(self, x, *args, **kwargs):
        if isinstance(x, str):
            x = l1(x)
        return super().__call__(x, *args, **kwargs)


qe = virtual_substitution = VirtualSubstitution()


def simplify(x, *args, **kwargs):
    if isinstance(x, str):
        x = l1(x)
    return _simplify(x, *args, **kwargs)
