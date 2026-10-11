"""Input formula for the 3-Dimensional Multiindex Transportation Problem (MTP-3)
quantifier-elimination benchmark.

Quantifier elimination for MTP-3 is discussed in:

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
    """Assign variables for the mtp3(k) input to the given namespace.

    This function is not needed for benchmarking.

    It is useful for interactive usage when you want to have the MTP3
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

    for i, j, l in product(r, repeat=3):
        v = f"x{i}{j}{l}"
        assign(v, namespace)

    for p, i, j in product("abc", r, r):
        v = f"{p}{i}{j}"
        assign(v, namespace)


def mtp3(k: int) -> Formula:
    if k < 1:
        raise ValueError("k must be at least 1")

    r = range(1, k + 1)
    indices2 = tuple(product(r, repeat=2))
    indices3 = tuple(product(r, repeat=3))

    mtp3_variables = []

    # Generate the list of quantified variables for the MTP3 input.
    for i, j, k in indices3:
        v = f"x{i}{j}{k}"
        mtp3_variables.append(VV[v])

    # Generate the equations for the MTP3 input.
    a_eqs = (sum((VV[f"x{k}{i}{j}"] for k in r), start=Term(0)) == VV[f"a{i}{j}"]
            for i, j in indices2)

    b_eqs = (sum((VV[f"x{i}{k}{j}"] for k in r), start=Term(0)) == VV[f"b{i}{j}"]
            for i, j in indices2)

    c_eqs = (sum((VV[f"x{i}{j}{k}"] for k in r), start=Term(0)) == VV[f"c{i}{j}"]
            for i, j in indices2)

    # Generate the non-negativity constraints for the MTP3 input.
    nonnegative = (VV[f"x{i}{j}{k}"] >= 0 for i, j, k in indices3)

    return Ex(mtp3_variables, And(*a_eqs, *b_eqs, *c_eqs, *nonnegative))


MTP3_INPUT = mtp3(k=3)
MTP3_INPUT_ATOMS = sum(1 for _ in MTP3_INPUT.atoms())
MTP_OUTPUT_LOWER_BOUND = 80000
MTP_OUTPUT_UPPER_BOUND = 100000
