"""Input formula for the period-9 quantifier-elimination benchmark.

Consider the infinite real sequence defined by $x_{i+2} = |x_{i+1}| - x_{i}$.
Real quantifier elimination can check that this sequence has period 9 for all
real choices of $x_1$, $x_2$.

The period-9 problem is discussed in:

A. Colmerauer. Prolog III. Commun. ACM 33(7):70--90, July 1990.

It originally appeared in:

M. Brown. Problems and solutions. Am. Math. Monthly 90(8):569, 1983.

The input is constructed at import time so formula construction is excluded
from benchmark timings.
"""

from logic1.firstorder import All, And, Implies, Or
from logic1.theories.RCF import Term, VV
from logic1.theories.RCF.types import Formula


x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = VV.get(
    *(f"x{i}" for i in range(12)))

period9_matrix = And(
    Or(And(x2 >= 0, x3 == x2 - x1), And(x2 < 0, x3 == -x2 - x1)),
    Or(And(x3 >= 0, x4 == x3 - x2), And(x3 < 0, x4 == -x3 - x2)),
    Or(And(x4 >= 0, x5 == x4 - x3), And(x4 < 0, x5 == -x4 - x3)),
    Or(And(x5 >= 0, x6 == x5 - x4), And(x5 < 0, x6 == -x5 - x4)),
    Or(And(x6 >= 0, x7 == x6 - x5), And(x6 < 0, x7 == -x6 - x5)),
    Or(And(x7 >= 0, x8 == x7 - x6), And(x7 < 0, x8 == -x7 - x6)),
    Or(And(x8 >= 0, x9 == x8 - x7), And(x8 < 0, x9 == -x8 - x7)),
    Or(And(x9 >= 0, x10 == x9 - x8), And(x9 < 0, x10 == -x9 - x8)),
    Or(And(x10 >= 0, x11 == x10 - x9), And(x10 < 0, x11 == -x10 - x9)),
)
period9 = Implies(period9_matrix, And(x1 == x10, x2 == x11))
period9_variables = sorted(set(period9.fvars()), key=Term.sort_key)

PERIOD9_INPUT: Formula = All(period9_variables, period9)
PERIOD9_INPUT_ATOMS = 38
