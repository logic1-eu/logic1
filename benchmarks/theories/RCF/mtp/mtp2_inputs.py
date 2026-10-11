"""Input formula for the 2-Dimensional Multiindex Transportation Problem (MTP-2)
quantifier-elimination benchmark.

Quantifier elimination for MTP-2 is discussed in:

R. Loos, V. Weispfenning. Applying Linear Quantifier Elimination.
Comput. J. 36(5): 450–462 (1993)

The problem originally appeared in:

M. Vlach. Conditions for the existence of solutions of the three-dimensional
planar transportation problem. Discrete Appl. Math. 13,61-78 (1986)

The input is constructed at import time so formula construction is excluded
from benchmark timings.
"""

from itertools import product
import logging

from logic1.firstorder import And, Ex, Formula
from logic1.theories.RCF import Term, VV


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def assign_vars(k: int, namespace: dict) -> None:
    """Assign variables for the mtp2(k) input to the given namespace.

    This function is not needed for benchmarking.

    It is useful for interactive usage when you want to have the MTP2
    variables readily available in a specific namespace.

    A typical usage would be:

    >>> assign_vars(k=3, globals())

    Note that globals() cannot be implemented as the default without using
    some form of introspection, which is generally not recommended.
    """

    def assign(v: str, namespace: dict) -> None:
        if v in namespace:
            logger.warning("Overwriting existing binding: %s", v)
        namespace[v] = VV[v]
        logger.debug("Assigned %s", v)

    r = range(1, k + 1)

    for i, j in product(r, repeat=2):
        v = f"x{i}{j}"
        assign(v, namespace)

    for p, i in product("ab", r):
        v = f"{p}{i}"
        assign(v, namespace)


def mtp2(k: int) -> Formula:
    if k < 1:
        raise ValueError("k must be at least 1")

    r = range(1, k + 1)
    indices2 = tuple(product(r, repeat=2))

    mtp2_variables = []

    # Generate the list of quantified variables for the MTP-2 input.
    for i, j in indices2:
        v = f"x{i}{j}"
        mtp2_variables.append(VV[v])

    # Generate the equations for the MTP-2 input.
    a_eqs = (sum((VV[f"x{k}{j}"] for j in r), start=Term(0)) == VV[f"a{k}"]
            for k in r)

    b_eqs = (sum((VV[f"x{i}{k}"] for i in r), start=Term(0)) == VV[f"b{k}"]
            for k in r)
    # Generate the non-negativity constraints for the MTP-2 input.
    nonnegative = (VV[f"x{i}{j}"] >= 0 for i, j in indices2)

    return Ex(mtp2_variables, And(*a_eqs, *b_eqs, *nonnegative))
