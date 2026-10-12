"""Structurally factorized motor inputs for RCF simplification benchmarks.

The formulas for series1--13 exactly reproduce the original benchmark
inputs while reusing repeated boundaries, case blocks, and argument sequences.
Importing this module constructs the formulas before any benchmark call is
timed.

The data originate from:

[DS97] Andreas Dolzmann and Thomas Sturm. Simplification of quantifier-free
formulae over ordered fields. J. Symb. Comput. 24(2):209--231, August 1997.

The article contains a typo in the input atom count for series7: it states 710,
whereas the correct count is 248. Series14 in the article is literally
identical to series1 and is therefore omitted here.
"""

from logic1.firstorder import And, Or
from logic1.theories.RCF import VV


i2, n, p1, q, td, z = VV.get("i2", "n", "p1", "q", "td", "z")

# Structural notes for future edits:
# - series1--13 reproduce the original formulas exactly (including repr,
#   argument order, and the input atom counts recorded in MOTOR_SERIES). The original
#   series14 is exactly identical to series1 and is intentionally omitted.
# - Named formulas, split helpers, and starred tuples reuse exact subtrees or
#   contiguous argument sequences. Do not reorder, deduplicate, or otherwise
#   Boolean-simplify them: that would change the benchmark input tree.
# - Duplicate sibling formulas occur only in the outer Or of series7
#   (operands 4/5 and 9/10), series8 (1/3 and 6/10), and series9
#   (4/5 and 9/10). They are intentional source preservation; none occur nested.
# - Formula construction happens at import time, before benchmark invocation.
#   Polynomial expressions are kept on one physical source line.

p1_boundary = 2 * p1 - 7
q_td_boundary = 400 * q + 9 * td - 20050
motor_value = 80 * i2 + 250 * p1 + 2 * q
td_cases = Or(And(td - 400 >= 0, td - 700 < 0, 3 * td + 400 * z - 5320 == 0),
              And(td - 700 >= 0, td - 990 < 0, 2 * td - 300 * z + 1015 == 0),
              And(td == 0, z == 0))
q_td_zero_cases = Or(And(q_td_boundary <= 0, q == 0),
                     And(q_td_boundary > 0, 9 * td - 20050 == 0))
p1_motor_cases = Or(And(p1_boundary < 0, motor_value - 577 == 0), p1_boundary >= 0)
q_td_limit_cases = Or(And(q_td_boundary <= 0, q - 40 == 0),
                      And(q_td_boundary > 0, td - 450 == 0))
z_250_cases = Or(And(250 * z - 3129 < 0, 50 * z - 449 >= 0),
                 And(50 * z - 449 < 0, 125 * z - 1012 >= 0),
                 And(125 * z - 1012 < 0, 250 * z - 1361 > 0, 108437500 * z - 865858649 == 0))
z_plus_156250_cases = Or(And(156250 * z + 6937499 > 0, 156250 * z + 549999 <= 0),
                         And(156250 * z + 549999 > 0, 78125 * z - 523438 <= 0,
                             108437500 * z - 865858649 == 0),
                         And(78125 * z - 523438 > 0, 156250 * z - 5837501 < 0))
z_minus_156250_cases = Or(And(156250 * z - 10848749 < 0, 156250 * z - 3356249 >= 0,
                              108437500 * z - 865858649 == 0),
                          And(156250 * z - 3356249 < 0, 78125 * z - 741562 >= 0),
                          And(78125 * z - 741562 < 0, 156250 * z + 4136251 > 0))
n_td_z_300_cases = Or(And(20 * n - 2 * td - 300 * z - 5963 > 0,
                          20 * n - 2 * td - 300 * z - 8331 <= 0),
                      And(20 * n - 2 * td - 300 * z - 8331 > 0,
                          20 * n - 2 * td - 300 * z - 8923 <= 0,
                          4420 * n - 442 * td + 81700 * z - 3170191 == 0),
                      And(20 * n - 2 * td - 300 * z - 8923 > 0,
                          20 * n - 2 * td - 300 * z - 10699 < 0,
                          31937500 * n - 3193750 * td - 571562500 * z - 13629165033 == 0))
n_td_z_500_cases = Or(And(100 * n - 10 * td - 2500 * z - 17299 > 0,
                          20 * n - 2 * td - 500 * z - 6535 <= 0,
                          4420 * n - 442 * td + 81700 * z - 3170191 == 0),
                      And(20 * n - 2 * td - 500 * z - 6535 > 0,
                          100 * n - 10 * td - 2500 * z - 36519 <= 0),
                      And(100 * n - 10 * td - 2500 * z - 36519 > 0,
                          100 * n - 10 * td - 2500 * z - 48051 < 0,
                          187312500 * n - 18731250 * td - 4082187500 * z - 74105780531 == 0))
n_td_z_1250000_cases = Or(And(62500 * n - 6250 * td - 1250000 * z - 32509373 < 0,
                              62500 * n - 6250 * td - 1250000 * z - 27134373 >= 0,
                              31937500 * n - 3193750 * td - 571562500 * z - 13629165033 == 0),
                          And(62500 * n - 6250 * td - 1250000 * z - 27134373 < 0,
                              62500 * n - 6250 * td - 1250000 * z - 25790623 >= 0,
                              187312500 * n - 18731250 * td - 4082187500 * z - 74105780531 == 0),
                          And(62500 * n - 6250 * td - 1250000 * z - 25790623 < 0,
                              62500 * n - 6250 * td - 1250000 * z - 21759373 > 0))
n_q_td_z_1250000_cases = Or(And(62500 * n - 156250 * q - 6250 * td - 1250000 * z - 32509373 < 0,
                                62500 * n - 156250 * q - 6250 * td - 1250000 * z - 27134373 >= 0,
                                31937500 * n - 79843750 * q - 3193750 * td - 571562500 * z - 13629165033 == 0),
                            And(62500 * n - 156250 * q - 6250 * td - 1250000 * z - 27134373 < 0,
                                62500 * n - 156250 * q - 6250 * td - 1250000 * z - 25790623 >= 0,
                                187312500 * n - 468281250 * q - 18731250 * td - 4082187500 * z - 74105780531 == 0),
                            And(62500 * n - 156250 * q - 6250 * td - 1250000 * z - 25790623 < 0,
                                62500 * n - 156250 * q - 6250 * td - 1250000 * z - 21759373 > 0))


def q_td_split(nonpositive, positive):
    """Build the recurring q/td-boundary split without changing its tree."""
    return Or(And(q_td_boundary <= 0, nonpositive),
              And(q_td_boundary > 0, positive))


def p1_split(negative, nonnegative):
    """Build the recurring p1-boundary split without changing its tree."""
    return Or(And(p1_boundary < 0, negative),
              And(p1_boundary >= 0, nonnegative))


def td_split(lower, upper):
    """Build the recurring two-band td split without changing its tree."""
    return Or(And(td - 400 >= 0, td - 700 < 0, lower),
              And(td - 700 >= 0, td - 990 < 0, upper))


motor_1057_1377_band = (motor_value - 1057 > 0, motor_value - 1377 <= 0)
motor_1377_1457_band = (motor_value - 1377 > 0, motor_value - 1457 <= 0)
motor_1457_1697_band = (motor_value - 1457 > 0, motor_value - 1697 < 0)
base_operating_region = (q >= 0, q - 40 <= 0, n >= 0, n - td == 0, i2 == 0)
p1_motor_1377_td_region = (p1_boundary >= 0, motor_value - 1377 == 0, td_cases)
p1_motor_1457_td_region = (p1_boundary >= 0, motor_value - 1457 == 0, td_cases)


def make_series1():
    """Construct the factorized series1 input formula."""
    boundary_1 = 100 * n - 250 * q - 10 * td - 2500 * z - 36519
    boundary_2 = 20 * n - 50 * q - 2 * td - 300 * z - 8331
    boundary_3 = 10750000 * i2 + 500000 * n + 33593750 * p1 + 268750 * q - 21875 * td - 10000000 * z - 464765609
    boundary_4 = 10750000 * i2 + 500000 * n + 33593750 * p1 + 268750 * q - 50000 * td - 10000000 * z - 402109359
    boundary_5 = 10750000 * i2 + 500000 * n + 33593750 * p1 - 981250 * q - 50000 * td - 10000000 * z - 402109359
    boundary_6 = 1480 * i2 - 50 * n + 4625 * p1 + 162 * q + 5 * td + 750 * z - 4647
    boundary_7 = 1480 * i2 - 50 * n + 4625 * p1 + 37 * q + 5 * td + 750 * z - 4647
    boundary_8 = 153760 * i2 - 4000 * n + 480500 * p1 + 3844 * q + 175 * td + 100000 * z - 838344
    boundary_9 = 23680 * i2 - 800 * n + 74000 * p1 + 592 * q + 35 * td + 12000 * z + 25898
    boundary_10 = 76880 * i2 - 2000 * n + 240250 * p1 + 1922 * q + 200 * td + 50000 * z - 669797
    boundary_11 = 76880 * i2 - 2000 * n + 240250 * p1 + 6922 * q + 200 * td + 50000 * z - 669797
    region_1 = (q >= 0, q - 40 <= 0, 9 * td - 20050 <= 0, td - 450 >= 0,
                180 * n + 171 * td - 380950 >= 0, n - td == 0, i2 == 0, q_td_boundary >= 0)
    region_2 = (q >= 0, q - 40 <= 0, 9 * n - 380 * q >= 0, n - td == 0, i2 == 0,
                q_td_boundary <= 0)

    return Or(And(*base_operating_region, q_td_zero_cases, p1_motor_cases, td_cases,
                  Or(And(*motor_1057_1377_band, boundary_7 == 0),
                     And(*motor_1377_1457_band, boundary_10 == 0),
                     And(*motor_1457_1697_band, boundary_4 == 0))),
              And(*base_operating_region, p1_boundary >= 0, motor_value - 1377 == 0,
                  20 * n - 2 * td - 300 * z - 8331 == 0, q_td_zero_cases, td_cases),
              And(*base_operating_region, p1_boundary >= 0, motor_value - 1457 == 0,
                  100 * n - 10 * td - 2500 * z - 36519 == 0, q_td_zero_cases, td_cases),
              And(*region_2, p1_motor_cases, td_cases,
                  Or(And(*motor_1057_1377_band, boundary_6 == 0),
                     And(*motor_1377_1457_band, boundary_11 == 0),
                     And(*motor_1457_1697_band, boundary_5 == 0))),
              And(*region_2, p1_boundary >= 0, motor_value - 1377 == 0, boundary_2 == 0,
                  td_cases),
              And(*region_2, p1_boundary >= 0, motor_value - 1457 == 0, boundary_1 == 0,
                  td_cases),
              And(*region_1, p1_motor_cases, td_cases,
                  Or(And(*motor_1057_1377_band, boundary_9 == 0),
                     And(*motor_1377_1457_band, boundary_8 == 0),
                     And(*motor_1457_1697_band, boundary_3 == 0))),
              And(*region_1, p1_boundary >= 0, motor_value - 1377 == 0,
                  160 * n - 7 * td - 2400 * z - 86698 == 0, td_cases),
              And(*region_1, p1_boundary >= 0, motor_value - 1457 == 0,
                  800 * n - 35 * td - 20000 * z - 392402 == 0, td_cases),
              And(*base_operating_region, q_td_zero_cases,
                  p1_split(20 * n - 2 * td - 300 * z - 2411 == 0, boundary_7 == 0), td_cases,
                  n_td_z_300_cases),
              And(*region_2,
                  p1_split(20 * n - 50 * q - 2 * td - 300 * z - 2411 == 0, boundary_6 == 0),
                  td_cases,
                  Or(And(20 * n - 50 * q - 2 * td - 300 * z - 5963 > 0, boundary_2 <= 0),
                     And(boundary_2 > 0, 20 * n - 50 * q - 2 * td - 300 * z - 8923 <= 0,
                         4420 * n - 11050 * q - 442 * td + 81700 * z - 3170191 == 0),
                     And(20 * n - 50 * q - 2 * td - 300 * z - 8923 > 0,
                         20 * n - 50 * q - 2 * td - 300 * z - 10699 < 0,
                         31937500 * n - 79843750 * q - 3193750 * td - 571562500 * z - 13629165033 == 0))),
              And(*region_1,
                  p1_split(160 * n - 7 * td - 2400 * z - 39338 == 0, boundary_9 == 0), td_cases,
                  Or(And(160 * n - 7 * td - 2400 * z - 67754 > 0,
                         160 * n - 7 * td - 2400 * z - 86698 <= 0),
                     And(160 * n - 7 * td - 2400 * z - 86698 > 0,
                         160 * n - 7 * td - 2400 * z - 91434 <= 0,
                         35360 * n - 1547 * td + 653600 * z - 29792578 == 0),
                     And(160 * n - 7 * td - 2400 * z - 91434 > 0,
                         160 * n - 7 * td - 2400 * z - 105642 < 0,
                         255500000 * n - 11178125 * td - 4572500000 * z - 141050664014 == 0))),
              And(q >= 0, q - 40 <= 0, boundary_7 <= 0,
                  1480 * i2 - 50 * n + 4625 * p1 + 37 * q + 5 * td + 750 * z + 353 >= 0,
                  112480 * i2 - 3575 * n + 351500 * p1 + 2812 * q + 380 * td + 57000 * z - 353172 >= 0,
                  n - td == 0, i2 == 0, q_td_split(boundary_6 == 0, boundary_9 == 0),
                  p1_motor_cases, td_cases,
                  Or(And(*motor_1057_1377_band),
                     And(*motor_1377_1457_band,
                         17680 * i2 + 55250 * p1 + 442 * q + 20000 * z - 483917 == 0),
                     And(*motor_1457_1697_band,
                         25550000 * i2 + 79843750 * p1 + 638750 * q - 2500000 * z - 448579359 == 0))),
              And(q >= 0, q - 40 <= 0, 20 * n - 2 * td - 300 * z - 8331 >= 0,
                  20 * n - 2 * td - 300 * z - 10331 <= 0,
                  715 * n - 76 * td - 11400 * z - 316578 <= 0, n - td == 0, i2 == 0,
                  p1_boundary >= 0, motor_value - 1377 == 0,
                  q_td_split(boundary_2 == 0, 160 * n - 7 * td - 2400 * z - 86698 == 0),
                  td_cases),
              And(q >= 0, q - 40 <= 0, 20 * n - 2 * td - 300 * z - 8923 >= 0,
                  20 * n - 2 * td - 300 * z - 10923 <= 0,
                  715 * n - 76 * td - 11400 * z - 339074 <= 0, n - td == 0, i2 == 0,
                  p1_boundary >= 0, motor_value - 1457 == 0, 125 * z - 1012 == 0,
                  q_td_split(20 * n - 50 * q - 2 * td - 300 * z - 8923 == 0,
                             160 * n - 7 * td - 2400 * z - 91434 == 0), td_cases),
              And(*base_operating_region, q_td_zero_cases,
                  p1_split(20 * n - 2 * td - 500 * z + 1153 == 0, boundary_10 == 0), td_cases,
                  n_td_z_500_cases),
              And(*region_2,
                  p1_split(20 * n - 50 * q - 2 * td - 500 * z + 1153 == 0, boundary_11 == 0),
                  td_cases,
                  Or(And(100 * n - 250 * q - 10 * td - 2500 * z - 17299 > 0,
                         20 * n - 50 * q - 2 * td - 500 * z - 6535 <= 0,
                         4420 * n - 11050 * q - 442 * td + 81700 * z - 3170191 == 0),
                     And(20 * n - 50 * q - 2 * td - 500 * z - 6535 > 0, boundary_1 <= 0),
                     And(boundary_1 > 0, 100 * n - 250 * q - 10 * td - 2500 * z - 48051 < 0,
                         187312500 * n - 468281250 * q - 18731250 * td - 4082187500 * z - 74105780531 == 0))),
              And(*region_1,
                  p1_split(160 * n - 7 * td - 4000 * z - 10826 == 0, boundary_8 == 0), td_cases,
                  Or(And(800 * n - 35 * td - 20000 * z - 238642 > 0,
                         160 * n - 7 * td - 4000 * z - 72330 <= 0,
                         35360 * n - 1547 * td + 653600 * z - 29792578 == 0),
                     And(160 * n - 7 * td - 4000 * z - 72330 > 0,
                         800 * n - 35 * td - 20000 * z - 392402 <= 0),
                     And(800 * n - 35 * td - 20000 * z - 392402 > 0,
                         800 * n - 35 * td - 20000 * z - 484658 < 0,
                         1498500000 * n - 65559375 * td - 32657500000 * z - 780627025498 == 0))),
              And(q >= 0, q - 40 <= 0, boundary_10 <= 0,
                  76880 * i2 - 2000 * n + 240250 * p1 + 1922 * q + 200 * td + 50000 * z - 469797 >= 0,
                  1460720 * i2 - 35750 * n + 4564750 * p1 + 36518 * q + 3800 * td + 950000 * z - 12726143 >= 0,
                  n - td == 0, i2 == 0, q_td_split(boundary_11 == 0, boundary_8 == 0),
                  p1_motor_cases, td_cases,
                  Or(And(*motor_1057_1377_band,
                         17680 * i2 + 55250 * p1 + 442 * q + 20000 * z - 483917 == 0),
                     And(*motor_1377_1457_band),
                     And(*motor_1457_1697_band,
                         29970000 * i2 + 93656250 * p1 + 749250 * q + 2500000 * z - 569558609 == 0))),
              And(q >= 0, q - 40 <= 0, 20 * n - 2 * td - 500 * z - 6535 >= 0,
                  20 * n - 2 * td - 500 * z - 8535 <= 0,
                  715 * n - 76 * td - 19000 * z - 248330 <= 0, n - td == 0, i2 == 0,
                  p1_boundary >= 0, motor_value - 1377 == 0, 50 * z - 449 == 0,
                  q_td_split(20 * n - 50 * q - 2 * td - 500 * z - 6535 == 0,
                             160 * n - 7 * td - 4000 * z - 72330 == 0), td_cases),
              And(q >= 0, q - 40 <= 0, 4420 * n - 442 * td + 81700 * z - 3170191 >= 0,
                  4420 * n - 442 * td + 81700 * z - 3612191 <= 0,
                  158015 * n - 16796 * td + 3104600 * z - 120467258 <= 0, n - td == 0, i2 == 0,
                  q_td_split(4420 * n - 11050 * q - 442 * td + 81700 * z - 3170191 == 0,
                             35360 * n - 1547 * td + 653600 * z - 29792578 == 0),
                  p1_split(50 * z - 891 == 0,
                           17680 * i2 + 55250 * p1 + 442 * q + 20000 * z - 483917 == 0),
                  td_cases, z_250_cases),
              And(q >= 0, q - 40 <= 0, 100 * n - 10 * td - 2500 * z - 36519 >= 0,
                  100 * n - 10 * td - 2500 * z - 46519 <= 0,
                  3575 * n - 380 * td - 95000 * z - 1387722 <= 0, n - td == 0, i2 == 0,
                  p1_boundary >= 0, motor_value - 1457 == 0,
                  q_td_split(boundary_1 == 0, 800 * n - 35 * td - 20000 * z - 392402 == 0),
                  td_cases),
              And(*base_operating_region, q_td_zero_cases,
                  p1_split(62500 * n - 6250 * td - 1250000 * z - 40571873 == 0, boundary_4 == 0),
                  td_cases, n_td_z_1250000_cases),
              And(*region_2,
                  p1_split(62500 * n - 156250 * q - 6250 * td - 1250000 * z - 40571873 == 0,
                           boundary_5 == 0), td_cases, n_q_td_z_1250000_cases),
              And(*region_1,
                  p1_split(500000 * n - 21875 * td - 10000000 * z - 387231234 == 0,
                           boundary_3 == 0), td_cases,
                  Or(And(500000 * n - 21875 * td - 10000000 * z - 322731234 < 0,
                         500000 * n - 21875 * td - 10000000 * z - 279731234 >= 0,
                         255500000 * n - 11178125 * td - 4572500000 * z - 141050664014 == 0),
                     And(500000 * n - 21875 * td - 10000000 * z - 279731234 < 0,
                         500000 * n - 21875 * td - 10000000 * z - 268981234 >= 0,
                         1498500000 * n - 65559375 * td - 32657500000 * z - 780627025498 == 0),
                     And(500000 * n - 21875 * td - 10000000 * z - 268981234 < 0,
                         500000 * n - 21875 * td - 10000000 * z - 236731234 > 0))),
              And(q >= 0, q - 40 <= 0, boundary_4 >= 0,
                  10750000 * i2 + 500000 * n + 33593750 * p1 + 268750 * q - 50000 * td - 10000000 * z - 452109359 <= 0,
                  204250000 * i2 + 8937500 * n + 638281250 * p1 + 5106250 * q - 950000 * td - 190000000 * z - 7640077821 <= 0,
                  n - td == 0, i2 == 0, q_td_split(boundary_5 == 0, boundary_3 == 0),
                  p1_motor_cases, td_cases,
                  Or(And(*motor_1057_1377_band,
                         25550000 * i2 + 79843750 * p1 + 638750 * q - 2500000 * z - 448579359 == 0),
                     And(*motor_1377_1457_band,
                         29970000 * i2 + 93656250 * p1 + 749250 * q + 2500000 * z - 569558609 == 0),
                     And(*motor_1457_1697_band))),
              And(q >= 0, q - 40 <= 0,
                  31937500 * n - 3193750 * td - 571562500 * z - 13629165033 >= 0,
                  31937500 * n - 3193750 * td - 571562500 * z - 16822915033 <= 0,
                  1141765625 * n - 121362500 * td - 21719375000 * z - 517908271254 <= 0,
                  n - td == 0, i2 == 0,
                  q_td_split(31937500 * n - 79843750 * q - 3193750 * td - 571562500 * z - 13629165033 == 0,
                             255500000 * n - 11178125 * td - 4572500000 * z - 141050664014 == 0),
                  p1_split(156250 * z + 16518749 == 0,
                           25550000 * i2 + 79843750 * p1 + 638750 * q - 2500000 * z - 448579359 == 0),
                  td_cases, z_plus_156250_cases),
              And(q >= 0, q - 40 <= 0,
                  187312500 * n - 18731250 * td - 4082187500 * z - 74105780531 >= 0,
                  187312500 * n - 18731250 * td - 4082187500 * z - 92837030531 <= 0,
                  6696421875 * n - 711787500 * td - 155123125000 * z - 2816019660178 <= 0,
                  n - td == 0, i2 == 0,
                  q_td_split(187312500 * n - 468281250 * q - 18731250 * td - 4082187500 * z - 74105780531 == 0,
                             1498500000 * n - 65559375 * td - 32657500000 * z - 780627025498 == 0),
                  p1_split(156250 * z - 22087499 == 0,
                           29970000 * i2 + 93656250 * p1 + 749250 * q + 2500000 * z - 569558609 == 0),
                  td_cases, z_minus_156250_cases))


series1 = make_series1()


def make_series2():
    """Construct the factorized series2 input formula."""
    boundary_1 = 6 * n - 5 * td
    boundary_2 = 9 * n - 380 * q
    boundary_3 = 180 * n + 171 * td - 380950
    boundary_4 = 160 * n - 7 * td - 2400 * z - 86698
    boundary_5 = 800 * n - 35 * td - 20000 * z - 392402
    boundary_6 = 1498500000 * n - 65559375 * td - 32657500000 * z - 780627025498
    boundary_7 = 160 * n - 7 * td - 2400 * z - 91434
    boundary_8 = 160 * n - 7 * td - 4000 * z - 72330
    boundary_9 = 255500000 * n - 11178125 * td - 4572500000 * z - 141050664014
    boundary_10 = 35360 * n - 1547 * td + 653600 * z - 29792578
    boundary_11 = 100 * n - 250 * q - 10 * td - 2500 * z - 36519
    boundary_12 = 20 * n - 50 * q - 2 * td - 300 * z - 8331
    boundary_13 = 17680 * i2 + 55250 * p1 + 442 * q + 20000 * z - 483917
    boundary_14 = 20 * n - 50 * q - 2 * td - 300 * z - 8923
    boundary_15 = 20 * n - 50 * q - 2 * td - 500 * z - 6535
    boundary_16 = 25550000 * i2 + 79843750 * p1 + 638750 * q - 2500000 * z - 448579359
    boundary_17 = 29970000 * i2 + 93656250 * p1 + 749250 * q + 2500000 * z - 569558609
    boundary_18 = 4420 * n - 11050 * q - 442 * td + 81700 * z - 3170191
    boundary_19 = 187312500 * n - 468281250 * q - 18731250 * td - 4082187500 * z - 74105780531
    boundary_20 = 31937500 * n - 79843750 * q - 3193750 * td - 571562500 * z - 13629165033
    boundary_21 = 10750000 * i2 + 500000 * n + 33593750 * p1 + 268750 * q - 21875 * td - 10000000 * z - 464765609
    boundary_22 = 10750000 * i2 + 500000 * n + 33593750 * p1 + 268750 * q - 50000 * td - 10000000 * z - 402109359
    boundary_23 = 10750000 * i2 + 500000 * n + 33593750 * p1 - 981250 * q - 50000 * td - 10000000 * z - 402109359
    boundary_24 = 1480 * i2 - 50 * n + 4625 * p1 + 162 * q + 5 * td + 750 * z - 4647
    boundary_25 = 1480 * i2 - 50 * n + 4625 * p1 + 37 * q + 5 * td + 750 * z - 4647
    boundary_26 = 153760 * i2 - 4000 * n + 480500 * p1 + 3844 * q + 175 * td + 100000 * z - 838344
    boundary_27 = 23680 * i2 - 800 * n + 74000 * p1 + 592 * q + 35 * td + 12000 * z + 25898
    boundary_28 = 76880 * i2 - 2000 * n + 240250 * p1 + 1922 * q + 200 * td + 50000 * z - 669797
    boundary_29 = 76880 * i2 - 2000 * n + 240250 * p1 + 6922 * q + 200 * td + 50000 * z - 669797
    region_1 = (q_td_boundary >= 0, i2 == 0, boundary_3 >= 0, td - 450 >= 0,
                9 * td - 20050 <= 0, q - 40 <= 0, q >= 0)
    region_2 = (q_td_boundary <= 0, i2 == 0, boundary_2 >= 0, q - 40 <= 0, q >= 0)
    region_3 = (i2 == 0,
                6696421875 * n - 711787500 * td - 155123125000 * z - 2816019660178 <= 0,
                187312500 * n - 18731250 * td - 4082187500 * z - 92837030531 <= 0,
                187312500 * n - 18731250 * td - 4082187500 * z - 74105780531 >= 0, q - 40 <= 0,
                q >= 0, z_minus_156250_cases, td_cases,
                p1_split(156250 * z - 22087499 == 0, boundary_17 == 0),
                q_td_split(boundary_19 == 0, boundary_6 == 0))
    region_4 = (i2 == 0, 1141765625 * n - 121362500 * td - 21719375000 * z - 517908271254 <= 0,
                31937500 * n - 3193750 * td - 571562500 * z - 16822915033 <= 0,
                31937500 * n - 3193750 * td - 571562500 * z - 13629165033 >= 0, q - 40 <= 0,
                q >= 0, z_plus_156250_cases, td_cases,
                p1_split(156250 * z + 16518749 == 0, boundary_16 == 0),
                q_td_split(boundary_20 == 0, boundary_9 == 0))
    region_5 = (i2 == 0, 158015 * n - 16796 * td + 3104600 * z - 120467258 <= 0,
                4420 * n - 442 * td + 81700 * z - 3612191 <= 0,
                4420 * n - 442 * td + 81700 * z - 3170191 >= 0, q - 40 <= 0, q >= 0,
                z_250_cases, td_cases, p1_split(50 * z - 891 == 0, boundary_13 == 0),
                q_td_split(boundary_18 == 0, boundary_10 == 0))
    region_6 = (i2 == 0, n >= 0, q - 40 <= 0, q >= 0)
    region_7 = (i2 == 0,
                204250000 * i2 + 8937500 * n + 638281250 * p1 + 5106250 * q - 950000 * td - 190000000 * z - 7640077821 <= 0,
                10750000 * i2 + 500000 * n + 33593750 * p1 + 268750 * q - 50000 * td - 10000000 * z - 452109359 <= 0,
                boundary_22 >= 0, q - 40 <= 0, q >= 0)
    region_8 = (i2 == 0,
                1460720 * i2 - 35750 * n + 4564750 * p1 + 36518 * q + 3800 * td + 950000 * z - 12726143 >= 0,
                76880 * i2 - 2000 * n + 240250 * p1 + 1922 * q + 200 * td + 50000 * z - 469797 >= 0,
                boundary_28 <= 0, q - 40 <= 0, q >= 0)
    region_9 = (i2 == 0,
                112480 * i2 - 3575 * n + 351500 * p1 + 2812 * q + 380 * td + 57000 * z - 353172 >= 0,
                1480 * i2 - 50 * n + 4625 * p1 + 37 * q + 5 * td + 750 * z + 353 >= 0,
                boundary_25 <= 0, q - 40 <= 0, q >= 0)
    region_10 = (i2 == 0, 715 * n - 76 * td - 11400 * z - 339074 <= 0,
                 20 * n - 2 * td - 300 * z - 10923 <= 0, 20 * n - 2 * td - 300 * z - 8923 >= 0,
                 q - 40 <= 0, q >= 0, 125 * z - 1012 == 0)
    region_11 = (i2 == 0, 715 * n - 76 * td - 19000 * z - 248330 <= 0,
                 20 * n - 2 * td - 500 * z - 8535 <= 0, 20 * n - 2 * td - 500 * z - 6535 >= 0,
                 q - 40 <= 0, q >= 0, 50 * z - 449 == 0)
    region_12 = (i2 == 0, 3575 * n - 380 * td - 95000 * z - 1387722 <= 0,
                 100 * n - 10 * td - 2500 * z - 46519 <= 0,
                 100 * n - 10 * td - 2500 * z - 36519 >= 0, q - 40 <= 0, q >= 0)
    region_13 = (i2 == 0, 715 * n - 76 * td - 11400 * z - 316578 <= 0,
                 20 * n - 2 * td - 300 * z - 10331 <= 0, 20 * n - 2 * td - 300 * z - 8331 >= 0,
                 q - 40 <= 0, q >= 0)
    region_14 = (n_td_z_1250000_cases, td_cases,
                 p1_split(62500 * n - 6250 * td - 1250000 * z - 40571873 == 0, boundary_22 == 0),
                 q_td_zero_cases)
    region_15 = (n_q_td_z_1250000_cases, td_cases,
                 p1_split(62500 * n - 156250 * q - 6250 * td - 1250000 * z - 40571873 == 0, boundary_23 == 0))
    region_16 = (n_td_z_300_cases, td_cases,
                 p1_split(20 * n - 2 * td - 300 * z - 2411 == 0, boundary_25 == 0),
                 q_td_zero_cases)
    region_17 = (n_td_z_500_cases, td_cases,
                 p1_split(20 * n - 2 * td - 500 * z + 1153 == 0, boundary_28 == 0),
                 q_td_zero_cases)
    cases_1 = Or(And(*motor_1057_1377_band, boundary_24 == 0),
                 And(*motor_1377_1457_band, boundary_29 == 0),
                 And(*motor_1457_1697_band, boundary_23 == 0))
    cases_2 = Or(And(*motor_1057_1377_band, boundary_25 == 0),
                 And(*motor_1377_1457_band, boundary_28 == 0),
                 And(*motor_1457_1697_band, boundary_22 == 0))
    cases_3 = Or(And(*motor_1057_1377_band, boundary_27 == 0),
                 And(*motor_1377_1457_band, boundary_26 == 0),
                 And(*motor_1457_1697_band, boundary_21 == 0))
    cases_4 = Or(And(160 * n - 7 * td - 2400 * z - 67754 > 0, boundary_4 <= 0),
                 And(boundary_4 > 0, boundary_7 <= 0, boundary_10 == 0),
                 And(boundary_7 > 0, 160 * n - 7 * td - 2400 * z - 105642 < 0, boundary_9 == 0))
    cases_5 = Or(And(500000 * n - 21875 * td - 10000000 * z - 322731234 < 0,
                     500000 * n - 21875 * td - 10000000 * z - 279731234 >= 0, boundary_9 == 0),
                 And(500000 * n - 21875 * td - 10000000 * z - 279731234 < 0,
                     500000 * n - 21875 * td - 10000000 * z - 268981234 >= 0, boundary_6 == 0),
                 And(500000 * n - 21875 * td - 10000000 * z - 268981234 < 0,
                     500000 * n - 21875 * td - 10000000 * z - 236731234 > 0))
    cases_6 = Or(And(800 * n - 35 * td - 20000 * z - 238642 > 0, boundary_8 <= 0,
                     boundary_10 == 0), And(boundary_8 > 0, boundary_5 <= 0),
                 And(boundary_5 > 0, 800 * n - 35 * td - 20000 * z - 484658 < 0,
                     boundary_6 == 0))
    cases_7 = Or(And(100 * n - 250 * q - 10 * td - 2500 * z - 17299 > 0, boundary_15 <= 0,
                     boundary_18 == 0), And(boundary_15 > 0, boundary_11 <= 0),
                 And(boundary_11 > 0, 100 * n - 250 * q - 10 * td - 2500 * z - 48051 < 0,
                     boundary_19 == 0))
    cases_8 = Or(And(20 * n - 50 * q - 2 * td - 300 * z - 5963 > 0, boundary_12 <= 0),
                 And(boundary_12 > 0, boundary_14 <= 0, boundary_18 == 0),
                 And(boundary_14 > 0, 20 * n - 50 * q - 2 * td - 300 * z - 10699 < 0,
                     boundary_20 == 0))

    return Or(And(td == 0, *region_13, *p1_motor_1377_td_region,
                  q_td_split(boundary_12 == 0, boundary_4 == 0)),
              And(td == 0, *region_10, *p1_motor_1457_td_region,
                  q_td_split(boundary_14 == 0, boundary_7 == 0)),
              And(td == 0, *region_9,
                  Or(And(*motor_1057_1377_band), And(*motor_1377_1457_band, boundary_13 == 0),
                     And(*motor_1457_1697_band, boundary_16 == 0)), td_cases, p1_motor_cases,
                  q_td_split(boundary_24 == 0, boundary_27 == 0)),
              And(td == 0, *region_11, *p1_motor_1377_td_region,
                  q_td_split(boundary_15 == 0, boundary_8 == 0)), And(td == 0, *region_5),
              And(td == 0, *region_12, *p1_motor_1457_td_region,
                  q_td_split(boundary_11 == 0, boundary_5 == 0)),
              And(td == 0, *region_8,
                  Or(And(*motor_1057_1377_band, boundary_13 == 0), And(*motor_1377_1457_band),
                     And(*motor_1457_1697_band, boundary_17 == 0)), td_cases, p1_motor_cases,
                  q_td_split(boundary_29 == 0, boundary_26 == 0)), And(td == 0, *region_4),
              And(td == 0, *region_3),
              And(td == 0, *region_7,
                  Or(And(*motor_1057_1377_band, boundary_16 == 0),
                     And(*motor_1377_1457_band, boundary_17 == 0), And(*motor_1457_1697_band)),
                  td_cases, p1_motor_cases, q_td_split(boundary_23 == 0, boundary_21 == 0)),
              And(td == 0, *region_2, boundary_12 == 0, *p1_motor_1377_td_region),
              And(td == 0, *region_2, cases_8, td_cases,
                  p1_split(20 * n - 50 * q - 2 * td - 300 * z - 2411 == 0, boundary_24 == 0)),
              And(td == 0, *region_2, boundary_11 == 0, *p1_motor_1457_td_region),
              And(td == 0, *region_2, cases_7, td_cases,
                  p1_split(20 * n - 50 * q - 2 * td - 500 * z + 1153 == 0, boundary_29 == 0)),
              And(td == 0, *region_2, *region_15),
              And(td == 0, *region_2, cases_1, td_cases, p1_motor_cases),
              And(td == 0, *region_1, boundary_4 == 0, *p1_motor_1377_td_region),
              And(td == 0, *region_1, cases_4, td_cases,
                  p1_split(160 * n - 7 * td - 2400 * z - 39338 == 0, boundary_27 == 0)),
              And(td == 0, *region_1, boundary_5 == 0, *p1_motor_1457_td_region),
              And(td == 0, *region_1, cases_6, td_cases,
                  p1_split(160 * n - 7 * td - 4000 * z - 10826 == 0, boundary_26 == 0)),
              And(td == 0, *region_1, cases_5, td_cases,
                  p1_split(500000 * n - 21875 * td - 10000000 * z - 387231234 == 0,
                           boundary_21 == 0)),
              And(td == 0, *region_1, cases_3, td_cases, p1_motor_cases),
              And(td == 0, *region_6, 20 * n - 2 * td - 300 * z - 8331 == 0,
                  *p1_motor_1377_td_region, q_td_zero_cases),
              And(td == 0, *region_6, *region_16),
              And(td == 0, *region_6, 100 * n - 10 * td - 2500 * z - 36519 == 0,
                  *p1_motor_1457_td_region, q_td_zero_cases),
              And(td == 0, *region_6, *region_17), And(td == 0, *region_6, *region_14),
              And(td == 0, *region_6, cases_2, td_cases, p1_motor_cases, q_td_zero_cases),
              And(boundary_1 < 0, *region_13, *p1_motor_1377_td_region,
                  q_td_split(boundary_12 == 0, boundary_4 == 0)),
              And(boundary_1 < 0, *region_10, *p1_motor_1457_td_region,
                  q_td_split(boundary_14 == 0, boundary_7 == 0)),
              And(boundary_1 < 0, *region_9,
                  Or(And(*motor_1057_1377_band), And(*motor_1377_1457_band, boundary_13 == 0),
                     And(*motor_1457_1697_band, boundary_16 == 0)), td_cases, p1_motor_cases,
                  q_td_split(boundary_24 == 0, boundary_27 == 0)),
              And(boundary_1 < 0, *region_11, *p1_motor_1377_td_region,
                  q_td_split(boundary_15 == 0, boundary_8 == 0)),
              And(boundary_1 < 0, *region_5),
              And(boundary_1 < 0, *region_12, *p1_motor_1457_td_region,
                  q_td_split(boundary_11 == 0, boundary_5 == 0)),
              And(boundary_1 < 0, *region_8,
                  Or(And(*motor_1057_1377_band, boundary_13 == 0), And(*motor_1377_1457_band),
                     And(*motor_1457_1697_band, boundary_17 == 0)), td_cases, p1_motor_cases,
                  q_td_split(boundary_29 == 0, boundary_26 == 0)),
              And(boundary_1 < 0, *region_4), And(boundary_1 < 0, *region_3),
              And(boundary_1 < 0, *region_7,
                  Or(And(*motor_1057_1377_band, boundary_16 == 0),
                     And(*motor_1377_1457_band, boundary_17 == 0), And(*motor_1457_1697_band)),
                  td_cases, p1_motor_cases, q_td_split(boundary_23 == 0, boundary_21 == 0)),
              And(boundary_1 < 0, *region_2, boundary_12 == 0, *p1_motor_1377_td_region),
              And(boundary_1 < 0, *region_2, cases_8, td_cases,
                  p1_split(20 * n - 50 * q - 2 * td - 300 * z - 2411 == 0, boundary_24 == 0)),
              And(boundary_1 < 0, *region_2, boundary_11 == 0, *p1_motor_1457_td_region),
              And(boundary_1 < 0, *region_2, cases_7, td_cases,
                  p1_split(20 * n - 50 * q - 2 * td - 500 * z + 1153 == 0, boundary_29 == 0)),
              And(boundary_1 < 0, *region_2, *region_15),
              And(boundary_1 < 0, *region_2, cases_1, td_cases, p1_motor_cases),
              And(boundary_1 < 0, *region_1, boundary_4 == 0, *p1_motor_1377_td_region),
              And(boundary_1 < 0, *region_1, cases_4, td_cases,
                  p1_split(160 * n - 7 * td - 2400 * z - 39338 == 0, boundary_27 == 0)),
              And(boundary_1 < 0, *region_1, boundary_5 == 0, *p1_motor_1457_td_region),
              And(boundary_1 < 0, *region_1, cases_6, td_cases,
                  p1_split(160 * n - 7 * td - 4000 * z - 10826 == 0, boundary_26 == 0)),
              And(boundary_1 < 0, *region_1, cases_5, td_cases,
                  p1_split(500000 * n - 21875 * td - 10000000 * z - 387231234 == 0,
                           boundary_21 == 0)),
              And(boundary_1 < 0, *region_1, cases_3, td_cases, p1_motor_cases),
              And(boundary_1 < 0, *region_6, 20 * n - 2 * td - 300 * z - 8331 == 0,
                  *p1_motor_1377_td_region, q_td_zero_cases),
              And(boundary_1 < 0, *region_6, *region_16),
              And(boundary_1 < 0, *region_6, 100 * n - 10 * td - 2500 * z - 36519 == 0,
                  *p1_motor_1457_td_region, q_td_zero_cases),
              And(boundary_1 < 0, *region_6, *region_17),
              And(boundary_1 < 0, *region_6, *region_14),
              And(boundary_1 < 0, *region_6, cases_2, td_cases, p1_motor_cases, q_td_zero_cases))


series2 = make_series2()


def make_series3():
    """Construct the factorized series3 input formula."""
    region_1 = (q_td_boundary >= 0, i2 == 0, n - td == 0, td - 450 >= 0, 9 * td - 20050 <= 0,
                q - 40 <= 0, q >= 0, td_cases)

    return Or(And(p1_boundary >= 0, q_td_boundary <= 0, i2 == 0, n - td == 0, q >= 0,
                  q - 40 <= 0, td_cases), And(p1_boundary >= 0, *region_1),
              And(p1_boundary >= 0, i2 == 0, n - td == 0, q - 40 <= 0, q >= 0, td_cases,
                  q_td_limit_cases),
              And(p1_boundary < 0, q_td_boundary <= 0, i2 == 0, n - td == 0, q >= 0,
                  q - 40 <= 0, td_cases), And(p1_boundary < 0, *region_1),
              And(p1_boundary < 0, i2 == 0, n - td == 0, q - 40 <= 0, q >= 0, td_cases,
                  q_td_limit_cases))


series3 = make_series3()


def make_series4():
    """Construct the factorized series4 input formula."""
    region_1 = (i2 == 0, n - td == 0, 180 * n + 171 * td - 380950 >= 0, td - 450 >= 0,
                9 * td - 20050 <= 0, q - 40 <= 0, q >= 0, q_td_boundary >= 0, td_cases)
    region_2 = (i2 == 0, n - td == 0, 9 * n - 380 * q >= 0, q - 40 <= 0, q >= 0,
                q_td_boundary <= 0, td_cases)
    region_3 = (i2 == 0, n - td == 0, n >= 0, q - 40 <= 0, q >= 0, td_cases)

    return Or(And(*motor_1457_1697_band, *region_3, p1_motor_cases, q_td_zero_cases),
              And(*motor_1457_1697_band, *region_2, p1_motor_cases),
              And(*motor_1457_1697_band, *region_1, p1_motor_cases),
              And(p1_boundary >= 0, motor_value - 1457 == 0, *region_3, q_td_zero_cases),
              And(p1_boundary >= 0, motor_value - 1457 == 0, *region_2),
              And(p1_boundary >= 0, motor_value - 1457 == 0, *region_1),
              And(*motor_1377_1457_band, *region_3, p1_motor_cases, q_td_zero_cases),
              And(*motor_1377_1457_band, *region_2, p1_motor_cases),
              And(*motor_1377_1457_band, *region_1, p1_motor_cases),
              And(p1_boundary >= 0, motor_value - 1377 == 0, *region_3, q_td_zero_cases),
              And(p1_boundary >= 0, motor_value - 1377 == 0, *region_2),
              And(p1_boundary >= 0, motor_value - 1377 == 0, *region_1),
              And(*motor_1057_1377_band, *region_3, p1_motor_cases, q_td_zero_cases),
              And(*motor_1057_1377_band, *region_2, p1_motor_cases),
              And(*motor_1057_1377_band, *region_1, p1_motor_cases))


series4 = make_series4()


def make_series5():
    """Construct the factorized series5 input formula."""
    region_1 = (i2 == 0, 180 * n + 171 * td - 380950 >= 0, td - 450 >= 0, 9 * td - 20050 <= 0,
                q - 40 <= 0, q >= 0, q_td_boundary >= 0)
    region_2 = (i2 == 0, 9 * n - 380 * q >= 0, q - 40 <= 0, q >= 0, q_td_boundary <= 0)
    region_3 = (i2 == 0, n >= 0, q - 40 <= 0, q >= 0, p1_motor_cases, q_td_zero_cases)

    return Or(And(*motor_1457_1697_band, *region_3),
              And(*motor_1457_1697_band, *region_2, p1_motor_cases),
              And(*motor_1457_1697_band, *region_1, p1_motor_cases),
              And(p1_boundary >= 0, motor_value - 1457 == 0, i2 == 0, n >= 0, q - 40 <= 0,
                  q >= 0, q_td_zero_cases),
              And(p1_boundary >= 0, motor_value - 1457 == 0, *region_2),
              And(p1_boundary >= 0, motor_value - 1457 == 0, *region_1),
              And(*motor_1377_1457_band, *region_3),
              And(*motor_1377_1457_band, *region_2, p1_motor_cases),
              And(*motor_1377_1457_band, *region_1, p1_motor_cases),
              And(p1_boundary >= 0, motor_value - 1377 == 0, i2 == 0, n >= 0, q - 40 <= 0,
                  q >= 0, q_td_zero_cases),
              And(p1_boundary >= 0, motor_value - 1377 == 0, *region_2),
              And(p1_boundary >= 0, motor_value - 1377 == 0, *region_1),
              And(*motor_1057_1377_band, *region_3),
              And(*motor_1057_1377_band, *region_2, p1_motor_cases),
              And(*motor_1057_1377_band, *region_1, p1_motor_cases))


series5 = make_series5()


def make_series6():
    """Construct the factorized series6 input formula."""
    boundary_1 = 9 * n - 380 * q
    boundary_2 = 180 * n + 171 * td - 380950
    boundary_3 = 17680 * i2 + 55250 * p1 + 442 * q + 20000 * z - 483917
    boundary_4 = 25550000 * i2 + 79843750 * p1 + 638750 * q - 2500000 * z - 448579359
    boundary_5 = 29970000 * i2 + 93656250 * p1 + 749250 * q + 2500000 * z - 569558609
    region_1 = (boundary_1 >= 0, n - td == 0, i2 == 0, q_td_boundary <= 0)
    region_2 = (boundary_2 >= 0, n - td == 0, i2 == 0, q_td_boundary >= 0)
    region_3 = (q >= 0, q - 40 <= 0, 9 * td - 20050 <= 0, td - 450 >= 0)
    region_4 = (q >= 0, q - 40 <= 0, n >= 0, 9 * n - 15200 <= 0)
    region_5 = (n - td == 0, i2 == 0, q_td_split(boundary_1 == 0, boundary_2 == 0))
    region_6 = (9 * n - 15200 >= 0, n - td == 0, i2 == 0)
    region_7 = (p1_split(156250 * z - 22087499 == 0, boundary_5 == 0), td_cases,
                z_minus_156250_cases)
    region_8 = (p1_split(156250 * z + 16518749 == 0, boundary_4 == 0), td_cases,
                z_plus_156250_cases)
    region_9 = (p1_split(50 * z - 891 == 0, boundary_3 == 0), td_cases, z_250_cases)
    region_10 = (p1_boundary >= 0, motor_value - 1457 == 0, 125 * z - 1012 == 0)
    region_11 = (p1_boundary >= 0, motor_value - 1377 == 0, 50 * z - 449 == 0)
    cases_1 = Or(And(*motor_1057_1377_band, boundary_3 == 0), And(*motor_1377_1457_band),
                 And(*motor_1457_1697_band, boundary_5 == 0))
    cases_2 = Or(And(*motor_1057_1377_band, boundary_4 == 0),
                 And(*motor_1377_1457_band, boundary_5 == 0), And(*motor_1457_1697_band))
    cases_3 = Or(And(*motor_1057_1377_band), And(*motor_1377_1457_band, boundary_3 == 0),
                 And(*motor_1457_1697_band, boundary_4 == 0))

    return Or(And(q >= 0, q - 40 <= 0,
                  1480 * i2 - 50 * n + 4625 * p1 + 37 * q + 5 * td + 750 * z + 353 > 0,
                  *region_6, q_td_limit_cases, p1_motor_cases, td_cases, cases_3),
              And(q >= 0, q - 40 <= 0, 20 * n - 2 * td - 300 * z - 10331 < 0, *region_6,
                  p1_boundary >= 0, motor_value - 1377 == 0, q_td_limit_cases, td_cases),
              And(q >= 0, q - 40 <= 0, 20 * n - 2 * td - 300 * z - 10923 < 0, *region_6,
                  *region_10, q_td_limit_cases, td_cases),
              And(*region_4,
                  112480 * i2 - 3575 * n + 351500 * p1 + 2812 * q + 380 * td + 57000 * z - 353172 > 0,
                  *region_5, p1_motor_cases, td_cases, cases_3),
              And(*region_4, 715 * n - 76 * td - 11400 * z - 316578 < 0, n - td == 0, i2 == 0,
                  p1_boundary >= 0, motor_value - 1377 == 0,
                  q_td_split(boundary_1 == 0, boundary_2 == 0), td_cases),
              And(*region_4, 715 * n - 76 * td - 11400 * z - 339074 < 0, n - td == 0, i2 == 0,
                  *region_10, q_td_split(boundary_1 == 0, boundary_2 == 0), td_cases),
              And(q >= 0, q - 40 <= 0,
                  1480 * i2 - 50 * n + 4625 * p1 + 162 * q + 5 * td + 750 * z - 4647 > 0,
                  *region_1, p1_motor_cases, td_cases, cases_3),
              And(q >= 0, q - 40 <= 0, 20 * n - 50 * q - 2 * td - 300 * z - 8331 < 0, *region_1,
                  *p1_motor_1377_td_region),
              And(q >= 0, q - 40 <= 0, 20 * n - 50 * q - 2 * td - 300 * z - 8923 < 0, *region_1,
                  *region_10, td_cases),
              And(*region_3,
                  23680 * i2 - 800 * n + 74000 * p1 + 592 * q + 35 * td + 12000 * z + 25898 > 0,
                  *region_2, p1_motor_cases, td_cases, cases_3),
              And(*region_3, 160 * n - 7 * td - 2400 * z - 86698 < 0, *region_2,
                  *p1_motor_1377_td_region),
              And(*region_3, 160 * n - 7 * td - 2400 * z - 91434 < 0, *region_2, *region_10,
                  td_cases),
              And(q >= 0, q - 40 <= 0,
                  76880 * i2 - 2000 * n + 240250 * p1 + 1922 * q + 200 * td + 50000 * z - 469797 > 0,
                  *region_6, q_td_limit_cases, p1_motor_cases, td_cases, cases_1),
              And(q >= 0, q - 40 <= 0, 20 * n - 2 * td - 500 * z - 8535 < 0, *region_6,
                  *region_11, q_td_limit_cases, td_cases),
              And(q >= 0, q - 40 <= 0, 4420 * n - 442 * td + 81700 * z - 3612191 < 0, *region_6,
                  q_td_limit_cases, *region_9),
              And(q >= 0, q - 40 <= 0, 100 * n - 10 * td - 2500 * z - 46519 < 0, *region_6,
                  p1_boundary >= 0, motor_value - 1457 == 0, q_td_limit_cases, td_cases),
              And(*region_4,
                  1460720 * i2 - 35750 * n + 4564750 * p1 + 36518 * q + 3800 * td + 950000 * z - 12726143 > 0,
                  *region_5, p1_motor_cases, td_cases, cases_1),
              And(*region_4, 715 * n - 76 * td - 19000 * z - 248330 < 0, n - td == 0, i2 == 0,
                  *region_11, q_td_split(boundary_1 == 0, boundary_2 == 0), td_cases),
              And(*region_4, 158015 * n - 16796 * td + 3104600 * z - 120467258 < 0, *region_5,
                  *region_9),
              And(*region_4, 3575 * n - 380 * td - 95000 * z - 1387722 < 0, n - td == 0,
                  i2 == 0, p1_boundary >= 0, motor_value - 1457 == 0,
                  q_td_split(boundary_1 == 0, boundary_2 == 0), td_cases),
              And(q >= 0, q - 40 <= 0,
                  76880 * i2 - 2000 * n + 240250 * p1 + 6922 * q + 200 * td + 50000 * z - 669797 > 0,
                  *region_1, p1_motor_cases, td_cases, cases_1),
              And(q >= 0, q - 40 <= 0, 20 * n - 50 * q - 2 * td - 500 * z - 6535 < 0, *region_1,
                  *region_11, td_cases),
              And(q >= 0, q - 40 <= 0,
                  4420 * n - 11050 * q - 442 * td + 81700 * z - 3170191 < 0, *region_1,
                  *region_9),
              And(q >= 0, q - 40 <= 0, 100 * n - 250 * q - 10 * td - 2500 * z - 36519 < 0,
                  *region_1, *p1_motor_1457_td_region),
              And(*region_3,
                  153760 * i2 - 4000 * n + 480500 * p1 + 3844 * q + 175 * td + 100000 * z - 838344 > 0,
                  *region_2, p1_motor_cases, td_cases, cases_1),
              And(*region_3, 160 * n - 7 * td - 4000 * z - 72330 < 0, *region_2, *region_11,
                  td_cases),
              And(*region_3, 35360 * n - 1547 * td + 653600 * z - 29792578 < 0, *region_2,
                  *region_9),
              And(*region_3, 800 * n - 35 * td - 20000 * z - 392402 < 0, *region_2,
                  *p1_motor_1457_td_region),
              And(q >= 0, q - 40 <= 0,
                  10750000 * i2 + 500000 * n + 33593750 * p1 + 268750 * q - 50000 * td - 10000000 * z - 452109359 < 0,
                  *region_6, q_td_limit_cases, p1_motor_cases, td_cases, cases_2),
              And(q >= 0, q - 40 <= 0,
                  31937500 * n - 3193750 * td - 571562500 * z - 16822915033 < 0, *region_6,
                  q_td_limit_cases, *region_8),
              And(q >= 0, q - 40 <= 0,
                  187312500 * n - 18731250 * td - 4082187500 * z - 92837030531 < 0, *region_6,
                  q_td_limit_cases, *region_7),
              And(*region_4,
                  204250000 * i2 + 8937500 * n + 638281250 * p1 + 5106250 * q - 950000 * td - 190000000 * z - 7640077821 < 0,
                  *region_5, p1_motor_cases, td_cases, cases_2),
              And(*region_4,
                  1141765625 * n - 121362500 * td - 21719375000 * z - 517908271254 < 0,
                  *region_5, *region_8),
              And(*region_4,
                  6696421875 * n - 711787500 * td - 155123125000 * z - 2816019660178 < 0,
                  *region_5, *region_7),
              And(q >= 0, q - 40 <= 0,
                  10750000 * i2 + 500000 * n + 33593750 * p1 - 981250 * q - 50000 * td - 10000000 * z - 402109359 < 0,
                  *region_1, p1_motor_cases, td_cases, cases_2),
              And(q >= 0, q - 40 <= 0,
                  31937500 * n - 79843750 * q - 3193750 * td - 571562500 * z - 13629165033 < 0,
                  *region_1, *region_8),
              And(q >= 0, q - 40 <= 0,
                  187312500 * n - 468281250 * q - 18731250 * td - 4082187500 * z - 74105780531 < 0,
                  *region_1, *region_7),
              And(*region_3,
                  10750000 * i2 + 500000 * n + 33593750 * p1 + 268750 * q - 21875 * td - 10000000 * z - 464765609 < 0,
                  *region_2, p1_motor_cases, td_cases, cases_2),
              And(*region_3, 255500000 * n - 11178125 * td - 4572500000 * z - 141050664014 < 0,
                  *region_2, *region_8),
              And(*region_3,
                  1498500000 * n - 65559375 * td - 32657500000 * z - 780627025498 < 0,
                  *region_2, *region_7))


series6 = make_series6()


def make_series7():
    """Construct the factorized series7 input formula."""
    region_1 = (q >= 0, q - 40 <= 0, 9 * td - 20050 <= 0, td - 450 >= 0,
                180 * n + 171 * td - 380950 >= 0, n - td == 0, i2 == 0, q_td_boundary >= 0,
                p1_boundary >= 0, td_cases)
    region_2 = (q >= 0, q - 40 <= 0, 9 * n - 380 * q >= 0, n - td == 0, i2 == 0,
                q_td_boundary <= 0, p1_boundary >= 0, td_cases)
    # The original series7 contains this exact disjunct twice, as operands 4 and 5
    # of its outer Or. Both occurrences are retained below to preserve the input tree.
    case_1 = And(q >= 0, q - 40 <= 0, 100 * n - 10 * td - 2500 * z - 36519 >= 0,
                 100 * n - 10 * td - 2500 * z - 46519 <= 0,
                 3575 * n - 380 * td - 95000 * z - 1387722 <= 0, n - td == 0, i2 == 0,
                 p1_boundary >= 0,
                 q_td_split(100 * n - 250 * q - 10 * td - 2500 * z - 36519 == 0,
                            800 * n - 35 * td - 20000 * z - 392402 == 0), td_cases)
    # Likewise, the original repeats this disjunct as outer-Or operands 9 and 10.
    case_2 = And(q >= 0, q - 40 <= 0, 20 * n - 2 * td - 300 * z - 8331 >= 0,
                 20 * n - 2 * td - 300 * z - 10331 <= 0,
                 715 * n - 76 * td - 11400 * z - 316578 <= 0, n - td == 0, i2 == 0,
                 p1_boundary >= 0,
                 q_td_split(20 * n - 50 * q - 2 * td - 300 * z - 8331 == 0,
                            160 * n - 7 * td - 2400 * z - 86698 == 0), td_cases)

    return Or(And(62500 * n - 6250 * td - 1250000 * z - 25790623 < 0,
                  62500 * n - 6250 * td - 1250000 * z - 21759373 > 0, *base_operating_region,
                  p1_boundary >= 0, q_td_zero_cases, td_cases),
              And(62500 * n - 156250 * q - 6250 * td - 1250000 * z - 25790623 < 0,
                  62500 * n - 156250 * q - 6250 * td - 1250000 * z - 21759373 > 0, *region_2),
              And(500000 * n - 21875 * td - 10000000 * z - 268981234 < 0,
                  500000 * n - 21875 * td - 10000000 * z - 236731234 > 0, *region_1), case_1,
              case_1,
              And(20 * n - 2 * td - 500 * z - 6535 > 0,
                  100 * n - 10 * td - 2500 * z - 36519 <= 0, *base_operating_region,
                  p1_boundary >= 0, q_td_zero_cases, td_cases),
              And(20 * n - 50 * q - 2 * td - 500 * z - 6535 > 0,
                  100 * n - 250 * q - 10 * td - 2500 * z - 36519 <= 0, *region_2),
              And(160 * n - 7 * td - 4000 * z - 72330 > 0,
                  800 * n - 35 * td - 20000 * z - 392402 <= 0, *region_1), case_2, case_2,
              And(20 * n - 2 * td - 300 * z - 5963 > 0, 20 * n - 2 * td - 300 * z - 8331 <= 0,
                  *base_operating_region, p1_boundary >= 0, q_td_zero_cases, td_cases),
              And(20 * n - 50 * q - 2 * td - 300 * z - 5963 > 0,
                  20 * n - 50 * q - 2 * td - 300 * z - 8331 <= 0, *region_2),
              And(160 * n - 7 * td - 2400 * z - 67754 > 0,
                  160 * n - 7 * td - 2400 * z - 86698 <= 0, *region_1))


series7 = make_series7()


def make_series8():
    """Construct the factorized series8 input formula."""
    boundary_1 = 100 * n - 250 * q - 10 * td - 2500 * z - 36519
    boundary_2 = 20 * n - 50 * q - 2 * td - 300 * z - 8331
    region_1 = (i2 == 0, n - td == 0, 180 * n + 171 * td - 380950 >= 0, td - 450 >= 0,
                9 * td - 20050 <= 0, q - 40 <= 0, q >= 0, q_td_boundary >= 0)
    region_2 = (i2 == 0, n - td == 0, 9 * n - 380 * q >= 0, q - 40 <= 0, q >= 0,
                q_td_boundary <= 0)
    # The original series8 repeats this exact disjunct as operands 1 and 3
    # of its outer Or. Both occurrences are retained below to preserve the input tree.
    case_1 = And(40 * i2 + q - 251 <= 0, i2 == 0, n - td == 0,
                 715 * n - 76 * td - 11400 * z - 316578 <= 0,
                 20 * n - 2 * td - 300 * z - 10331 <= 0, 20 * n - 2 * td - 300 * z - 8331 >= 0,
                 q - 40 <= 0, q >= 0, td_cases,
                 q_td_split(boundary_2 == 0, 160 * n - 7 * td - 2400 * z - 86698 == 0))
    # Likewise, the original repeats this disjunct as outer-Or operands 6 and 10.
    case_2 = And(40 * i2 + q - 291 <= 0, i2 == 0, n - td == 0,
                 3575 * n - 380 * td - 95000 * z - 1387722 <= 0,
                 100 * n - 10 * td - 2500 * z - 46519 <= 0,
                 100 * n - 10 * td - 2500 * z - 36519 >= 0, q - 40 <= 0, q >= 0, td_cases,
                 q_td_split(boundary_1 == 0, 800 * n - 35 * td - 20000 * z - 392402 == 0))

    return Or(case_1,
              And(2960 * i2 - 100 * n + 74 * q + 10 * td + 1500 * z + 23081 <= 0, i2 == 0,
                  n - td == 0, n >= 0, q - 40 <= 0, q >= 0, n_td_z_300_cases, td_cases,
                  q_td_zero_cases), case_1,
              And(2960 * i2 - 100 * n + 324 * q + 10 * td + 1500 * z + 23081 <= 0, *region_2,
                  Or(And(20 * n - 50 * q - 2 * td - 300 * z - 5963 > 0, boundary_2 <= 0),
                     And(boundary_2 > 0, 20 * n - 50 * q - 2 * td - 300 * z - 8923 <= 0,
                         4420 * n - 11050 * q - 442 * td + 81700 * z - 3170191 == 0),
                     And(20 * n - 50 * q - 2 * td - 300 * z - 8923 > 0,
                         20 * n - 50 * q - 2 * td - 300 * z - 10699 < 0,
                         31937500 * n - 79843750 * q - 3193750 * td - 571562500 * z - 13629165033 == 0)),
                  td_cases),
              And(23680 * i2 - 800 * n + 592 * q + 35 * td + 12000 * z + 284898 <= 0, *region_1,
                  Or(And(160 * n - 7 * td - 2400 * z - 67754 > 0,
                         160 * n - 7 * td - 2400 * z - 86698 <= 0),
                     And(160 * n - 7 * td - 2400 * z - 86698 > 0,
                         160 * n - 7 * td - 2400 * z - 91434 <= 0,
                         35360 * n - 1547 * td + 653600 * z - 29792578 == 0),
                     And(160 * n - 7 * td - 2400 * z - 91434 > 0,
                         160 * n - 7 * td - 2400 * z - 105642 < 0,
                         255500000 * n - 11178125 * td - 4572500000 * z - 141050664014 == 0)),
                  td_cases), case_2,
              And(38440 * i2 - 1000 * n + 961 * q + 100 * td + 25000 * z + 85539 <= 0, i2 == 0,
                  n - td == 0, n >= 0, q - 40 <= 0, q >= 0, n_td_z_500_cases, td_cases,
                  q_td_zero_cases),
              And(40 * i2 + q - 251 <= 0, i2 == 0, n - td == 0,
                  715 * n - 76 * td - 19000 * z - 248330 <= 0,
                  20 * n - 2 * td - 500 * z - 8535 <= 0, 20 * n - 2 * td - 500 * z - 6535 >= 0,
                  q - 40 <= 0, q >= 0, 50 * z - 449 == 0, td_cases,
                  q_td_split(20 * n - 50 * q - 2 * td - 500 * z - 6535 == 0,
                             160 * n - 7 * td - 4000 * z - 72330 == 0)),
              And(8840 * i2 + 221 * q + 10000 * z - 145271 <= 0, i2 == 0, n - td == 0,
                  158015 * n - 16796 * td + 3104600 * z - 120467258 <= 0,
                  4420 * n - 442 * td + 81700 * z - 3612191 <= 0,
                  4420 * n - 442 * td + 81700 * z - 3170191 >= 0, q - 40 <= 0, q >= 0,
                  z_250_cases, td_cases,
                  q_td_split(4420 * n - 11050 * q - 442 * td + 81700 * z - 3170191 == 0,
                             35360 * n - 1547 * td + 653600 * z - 29792578 == 0)), case_2,
              And(38440 * i2 - 1000 * n + 3461 * q + 100 * td + 25000 * z + 85539 <= 0,
                  *region_2,
                  Or(And(100 * n - 250 * q - 10 * td - 2500 * z - 17299 > 0,
                         20 * n - 50 * q - 2 * td - 500 * z - 6535 <= 0,
                         4420 * n - 11050 * q - 442 * td + 81700 * z - 3170191 == 0),
                     And(20 * n - 50 * q - 2 * td - 500 * z - 6535 > 0, boundary_1 <= 0),
                     And(boundary_1 > 0, 100 * n - 250 * q - 10 * td - 2500 * z - 48051 < 0,
                         187312500 * n - 468281250 * q - 18731250 * td - 4082187500 * z - 74105780531 == 0)),
                  td_cases),
              And(153760 * i2 - 4000 * n + 3844 * q + 175 * td + 100000 * z + 843406 <= 0,
                  *region_1,
                  Or(And(800 * n - 35 * td - 20000 * z - 238642 > 0,
                         160 * n - 7 * td - 4000 * z - 72330 <= 0,
                         35360 * n - 1547 * td + 653600 * z - 29792578 == 0),
                     And(160 * n - 7 * td - 4000 * z - 72330 > 0,
                         800 * n - 35 * td - 20000 * z - 392402 <= 0),
                     And(800 * n - 35 * td - 20000 * z - 392402 > 0,
                         800 * n - 35 * td - 20000 * z - 484658 < 0,
                         1498500000 * n - 65559375 * td - 32657500000 * z - 780627025498 == 0)),
                  td_cases),
              And(102125000 * i2 + 4468750 * n + 2553125 * q - 475000 * td - 95000000 * z - 2703046723 <= 0,
                  i2 == 0, n - td == 0, 9 * n - 15200 <= 0, n >= 0, q - 40 <= 0, q >= 0,
                  Or(And(2234375 * n - 237500 * td - 47500000 * z - 1235356174 < 0,
                         2234375 * n - 237500 * td - 47500000 * z - 1031106174 >= 0,
                         1141765625 * n - 121362500 * td - 21719375000 * z - 517908271254 == 0),
                     And(2234375 * n - 237500 * td - 47500000 * z - 1031106174 < 0,
                         2234375 * n - 237500 * td - 47500000 * z - 980043674 >= 0,
                         6696421875 * n - 711787500 * td - 155123125000 * z - 2816019660178 == 0),
                     And(2234375 * n - 237500 * td - 47500000 * z - 980043674 < 0,
                         2234375 * n - 237500 * td - 47500000 * z - 826856174 > 0)), td_cases,
                  q_td_split(9 * n - 380 * q == 0, 180 * n + 171 * td - 380950 == 0)),
              And(5375000 * i2 + 250000 * n + 134375 * q - 25000 * td - 5000000 * z - 167265617 <= 0,
                  i2 == 0, n - td == 0, 9 * n - 15200 >= 0, q - 40 <= 0, q >= 0,
                  Or(And(62500 * n - 6250 * td - 1250000 * z - 38759373 < 0,
                         62500 * n - 6250 * td - 1250000 * z - 33384373 >= 0,
                         31937500 * n - 3193750 * td - 571562500 * z - 16822915033 == 0),
                     And(62500 * n - 6250 * td - 1250000 * z - 33384373 < 0,
                         62500 * n - 6250 * td - 1250000 * z - 32040623 >= 0,
                         187312500 * n - 18731250 * td - 4082187500 * z - 92837030531 == 0),
                     And(62500 * n - 6250 * td - 1250000 * z - 32040623 < 0,
                         62500 * n - 6250 * td - 1250000 * z - 28009373 > 0)), td_cases,
                  q_td_limit_cases),
              And(40 * i2 + q - 251 <= 0, i2 == 0, n - td == 0,
                  2234375 * n - 237500 * td - 47500000 * z - 1031106174 <= 0,
                  62500 * n - 6250 * td - 1250000 * z - 33384373 <= 0,
                  62500 * n - 6250 * td - 1250000 * z - 27134373 >= 0, q - 40 <= 0, q >= 0,
                  156250 * z + 549999 == 0, td_cases,
                  q_td_split(62500 * n - 156250 * q - 6250 * td - 1250000 * z - 27134373 == 0,
                             500000 * n - 21875 * td - 10000000 * z - 279731234 == 0)),
              And(12775000 * i2 + 319375 * q - 1250000 * z - 84563117 <= 0, i2 == 0,
                  n - td == 0,
                  1141765625 * n - 121362500 * td - 21719375000 * z - 517908271254 <= 0,
                  31937500 * n - 3193750 * td - 571562500 * z - 16822915033 <= 0,
                  31937500 * n - 3193750 * td - 571562500 * z - 13629165033 >= 0, q - 40 <= 0,
                  q >= 0, z_plus_156250_cases, td_cases,
                  q_td_split(31937500 * n - 79843750 * q - 3193750 * td - 571562500 * z - 13629165033 == 0,
                             255500000 * n - 11178125 * td - 4572500000 * z - 141050664014 == 0)),
              And(40 * i2 + q - 291 <= 0, i2 == 0, n - td == 0,
                  2234375 * n - 237500 * td - 47500000 * z - 980043674 <= 0,
                  62500 * n - 6250 * td - 1250000 * z - 32040623 <= 0,
                  62500 * n - 6250 * td - 1250000 * z - 25790623 >= 0, q - 40 <= 0, q >= 0,
                  78125 * z - 741562 == 0, td_cases,
                  q_td_split(62500 * n - 156250 * q - 6250 * td - 1250000 * z - 25790623 == 0,
                             500000 * n - 21875 * td - 10000000 * z - 268981234 == 0)),
              And(14985000 * i2 + 374625 * q + 1250000 * z - 120880867 <= 0, i2 == 0,
                  n - td == 0,
                  6696421875 * n - 711787500 * td - 155123125000 * z - 2816019660178 <= 0,
                  187312500 * n - 18731250 * td - 4082187500 * z - 92837030531 <= 0,
                  187312500 * n - 18731250 * td - 4082187500 * z - 74105780531 >= 0,
                  q - 40 <= 0, q >= 0, z_minus_156250_cases, td_cases,
                  q_td_split(187312500 * n - 468281250 * q - 18731250 * td - 4082187500 * z - 74105780531 == 0,
                             1498500000 * n - 65559375 * td - 32657500000 * z - 780627025498 == 0)),
              And(5375000 * i2 + 250000 * n - 490625 * q - 25000 * td - 5000000 * z - 142265617 <= 0,
                  *region_2, n_q_td_z_1250000_cases, td_cases),
              And(10750000 * i2 + 500000 * n + 268750 * q - 21875 * td - 10000000 * z - 347187484 <= 0,
                  *region_1,
                  Or(And(500000 * n - 21875 * td - 10000000 * z - 322731234 < 0,
                         500000 * n - 21875 * td - 10000000 * z - 279731234 >= 0,
                         255500000 * n - 11178125 * td - 4572500000 * z - 141050664014 == 0),
                     And(500000 * n - 21875 * td - 10000000 * z - 279731234 < 0,
                         500000 * n - 21875 * td - 10000000 * z - 268981234 >= 0,
                         1498500000 * n - 65559375 * td - 32657500000 * z - 780627025498 == 0),
                     And(500000 * n - 21875 * td - 10000000 * z - 268981234 < 0,
                         500000 * n - 21875 * td - 10000000 * z - 236731234 > 0)), td_cases))


series8 = make_series8()


def make_series9():
    """Construct the factorized series9 input formula."""
    region_1 = (q_td_boundary >= 0, i2 == 0, n - td == 0, 180 * n + 171 * td - 380950 >= 0,
                td - 450 >= 0, 9 * td - 20050 <= 0, q - 40 <= 0, q >= 0, td_cases)
    region_2 = (q_td_boundary <= 0, i2 == 0, n - td == 0, 9 * n - 380 * q >= 0, q - 40 <= 0,
                q >= 0, td_cases)
    region_3 = (i2 == 0, n - td == 0, n >= 0, q - 40 <= 0, q >= 0, td_cases, q_td_zero_cases)
    # The original series9 contains this exact disjunct twice, as operands 4 and 5
    # of its outer Or. Both occurrences are retained below to preserve the input tree.
    case_1 = And(i2 == 0, n - td == 0, 3575 * n - 380 * td - 95000 * z - 1387722 <= 0,
                 100 * n - 10 * td - 2500 * z - 46519 <= 0,
                 100 * n - 10 * td - 2500 * z - 36519 >= 0, q - 40 <= 0, q >= 0, td_cases,
                 q_td_split(100 * n - 250 * q - 10 * td - 2500 * z - 36519 == 0,
                            800 * n - 35 * td - 20000 * z - 392402 == 0))
    # Likewise, the original repeats this disjunct as outer-Or operands 9 and 10.
    case_2 = And(i2 == 0, n - td == 0, 715 * n - 76 * td - 11400 * z - 316578 <= 0,
                 20 * n - 2 * td - 300 * z - 10331 <= 0, 20 * n - 2 * td - 300 * z - 8331 >= 0,
                 q - 40 <= 0, q >= 0, td_cases,
                 q_td_split(20 * n - 50 * q - 2 * td - 300 * z - 8331 == 0,
                            160 * n - 7 * td - 2400 * z - 86698 == 0))

    return Or(And(62500 * n - 156250 * q - 6250 * td - 1250000 * z - 25790623 < 0,
                  62500 * n - 156250 * q - 6250 * td - 1250000 * z - 21759373 > 0, *region_2),
              And(500000 * n - 21875 * td - 10000000 * z - 268981234 < 0,
                  500000 * n - 21875 * td - 10000000 * z - 236731234 > 0, *region_1),
              And(62500 * n - 6250 * td - 1250000 * z - 25790623 < 0,
                  62500 * n - 6250 * td - 1250000 * z - 21759373 > 0, *region_3), case_1,
              case_1,
              And(20 * n - 50 * q - 2 * td - 500 * z - 6535 > 0,
                  100 * n - 250 * q - 10 * td - 2500 * z - 36519 <= 0, *region_2),
              And(160 * n - 7 * td - 4000 * z - 72330 > 0,
                  800 * n - 35 * td - 20000 * z - 392402 <= 0, *region_1),
              And(20 * n - 2 * td - 500 * z - 6535 > 0,
                  100 * n - 10 * td - 2500 * z - 36519 <= 0, *region_3), case_2, case_2,
              And(20 * n - 50 * q - 2 * td - 300 * z - 5963 > 0,
                  20 * n - 50 * q - 2 * td - 300 * z - 8331 <= 0, *region_2),
              And(160 * n - 7 * td - 2400 * z - 67754 > 0,
                  160 * n - 7 * td - 2400 * z - 86698 <= 0, *region_1),
              And(20 * n - 2 * td - 300 * z - 5963 > 0, 20 * n - 2 * td - 300 * z - 8331 <= 0,
                  *region_3))


series9 = make_series9()


def make_series10():
    """Construct the factorized series10 input formula."""
    boundary_1 = 100 * n - 250 * q - 10 * td - 2500 * z - 36519
    boundary_2 = 20 * n - 50 * q - 2 * td - 300 * z - 8331
    boundary_3 = 10750000 * i2 + 500000 * n + 33593750 * p1 + 268750 * q - 50000 * td - 10000000 * z - 402109359
    boundary_4 = 10750000 * i2 + 500000 * n + 33593750 * p1 - 981250 * q - 50000 * td - 10000000 * z - 402109359
    boundary_5 = 1480 * i2 - 50 * n + 4625 * p1 + 162 * q + 5 * td + 750 * z - 4647
    boundary_6 = 1480 * i2 - 50 * n + 4625 * p1 + 37 * q + 5 * td + 750 * z - 4647
    boundary_7 = 76880 * i2 - 2000 * n + 240250 * p1 + 1922 * q + 200 * td + 50000 * z - 669797
    boundary_8 = 76880 * i2 - 2000 * n + 240250 * p1 + 6922 * q + 200 * td + 50000 * z - 669797
    region_1 = (n - td == 0, 9 * n - 380 * q >= 0, q - 40 <= 0, q >= 0)
    region_2 = (n - td == 0, n >= 0, q - 40 <= 0, q > 0)

    return Or(And(boundary_2 < 0, n - td == 0, 715 * n - 76 * td - 11400 * z - 316578 <= 0,
                  20 * n - 2 * td - 300 * z - 10331 <= 0, 20 * n - 2 * td - 300 * z - 8331 >= 0,
                  q - 40 <= 0, q >= 0, *p1_motor_1377_td_region),
              And(20 * n - 50 * q - 2 * td - 300 * z - 8923 < 0, n - td == 0,
                  715 * n - 76 * td - 11400 * z - 339074 <= 0,
                  20 * n - 2 * td - 300 * z - 10923 <= 0, 20 * n - 2 * td - 300 * z - 8923 >= 0,
                  q - 40 <= 0, q >= 0, 125 * z - 1012 == 0, *p1_motor_1457_td_region),
              And(boundary_5 > 0, n - td == 0,
                  112480 * i2 - 3575 * n + 351500 * p1 + 2812 * q + 380 * td + 57000 * z - 353172 >= 0,
                  1480 * i2 - 50 * n + 4625 * p1 + 37 * q + 5 * td + 750 * z + 353 >= 0,
                  boundary_6 <= 0, q - 40 <= 0, q >= 0,
                  Or(And(*motor_1057_1377_band),
                     And(*motor_1377_1457_band,
                         17680 * i2 + 55250 * p1 + 442 * q + 20000 * z - 483917 == 0),
                     And(*motor_1457_1697_band,
                         25550000 * i2 + 79843750 * p1 + 638750 * q - 2500000 * z - 448579359 == 0)),
                  td_cases, p1_motor_cases),
              And(20 * n - 50 * q - 2 * td - 500 * z - 6535 < 0, n - td == 0,
                  715 * n - 76 * td - 19000 * z - 248330 <= 0,
                  20 * n - 2 * td - 500 * z - 8535 <= 0, 20 * n - 2 * td - 500 * z - 6535 >= 0,
                  q - 40 <= 0, q >= 0, 50 * z - 449 == 0, *p1_motor_1377_td_region),
              And(4420 * n - 11050 * q - 442 * td + 81700 * z - 3170191 < 0, n - td == 0,
                  158015 * n - 16796 * td + 3104600 * z - 120467258 <= 0,
                  4420 * n - 442 * td + 81700 * z - 3612191 <= 0,
                  4420 * n - 442 * td + 81700 * z - 3170191 >= 0, q - 40 <= 0, q >= 0,
                  z_250_cases, td_cases,
                  p1_split(50 * z - 891 == 0,
                           17680 * i2 + 55250 * p1 + 442 * q + 20000 * z - 483917 == 0)),
              And(boundary_1 < 0, n - td == 0, 3575 * n - 380 * td - 95000 * z - 1387722 <= 0,
                  100 * n - 10 * td - 2500 * z - 46519 <= 0,
                  100 * n - 10 * td - 2500 * z - 36519 >= 0, q - 40 <= 0, q >= 0,
                  *p1_motor_1457_td_region),
              And(boundary_8 > 0, n - td == 0,
                  1460720 * i2 - 35750 * n + 4564750 * p1 + 36518 * q + 3800 * td + 950000 * z - 12726143 >= 0,
                  76880 * i2 - 2000 * n + 240250 * p1 + 1922 * q + 200 * td + 50000 * z - 469797 >= 0,
                  boundary_7 <= 0, q - 40 <= 0, q >= 0,
                  Or(And(*motor_1057_1377_band,
                         17680 * i2 + 55250 * p1 + 442 * q + 20000 * z - 483917 == 0),
                     And(*motor_1377_1457_band),
                     And(*motor_1457_1697_band,
                         29970000 * i2 + 93656250 * p1 + 749250 * q + 2500000 * z - 569558609 == 0)),
                  td_cases, p1_motor_cases),
              And(31937500 * n - 79843750 * q - 3193750 * td - 571562500 * z - 13629165033 < 0,
                  n - td == 0,
                  1141765625 * n - 121362500 * td - 21719375000 * z - 517908271254 <= 0,
                  31937500 * n - 3193750 * td - 571562500 * z - 16822915033 <= 0,
                  31937500 * n - 3193750 * td - 571562500 * z - 13629165033 >= 0, q - 40 <= 0,
                  q >= 0, z_plus_156250_cases, td_cases,
                  p1_split(156250 * z + 16518749 == 0,
                           25550000 * i2 + 79843750 * p1 + 638750 * q - 2500000 * z - 448579359 == 0)),
              And(187312500 * n - 468281250 * q - 18731250 * td - 4082187500 * z - 74105780531 < 0,
                  n - td == 0,
                  6696421875 * n - 711787500 * td - 155123125000 * z - 2816019660178 <= 0,
                  187312500 * n - 18731250 * td - 4082187500 * z - 92837030531 <= 0,
                  187312500 * n - 18731250 * td - 4082187500 * z - 74105780531 >= 0,
                  q - 40 <= 0, q >= 0, z_minus_156250_cases, td_cases,
                  p1_split(156250 * z - 22087499 == 0,
                           29970000 * i2 + 93656250 * p1 + 749250 * q + 2500000 * z - 569558609 == 0)),
              And(boundary_4 < 0, n - td == 0,
                  204250000 * i2 + 8937500 * n + 638281250 * p1 + 5106250 * q - 950000 * td - 190000000 * z - 7640077821 <= 0,
                  10750000 * i2 + 500000 * n + 33593750 * p1 + 268750 * q - 50000 * td - 10000000 * z - 452109359 <= 0,
                  boundary_3 >= 0, q - 40 <= 0, q >= 0,
                  Or(And(*motor_1057_1377_band,
                         25550000 * i2 + 79843750 * p1 + 638750 * q - 2500000 * z - 448579359 == 0),
                     And(*motor_1377_1457_band,
                         29970000 * i2 + 93656250 * p1 + 749250 * q + 2500000 * z - 569558609 == 0),
                     And(*motor_1457_1697_band)), td_cases, p1_motor_cases),
              And(*region_2, 20 * n - 2 * td - 300 * z - 8331 == 0, *p1_motor_1377_td_region),
              And(*region_2, n_td_z_300_cases, td_cases,
                  p1_split(20 * n - 2 * td - 300 * z - 2411 == 0, boundary_6 == 0)),
              And(*region_2, 100 * n - 10 * td - 2500 * z - 36519 == 0,
                  *p1_motor_1457_td_region),
              And(*region_2, n_td_z_500_cases, td_cases,
                  p1_split(20 * n - 2 * td - 500 * z + 1153 == 0, boundary_7 == 0)),
              And(*region_2, n_td_z_1250000_cases, td_cases,
                  p1_split(62500 * n - 6250 * td - 1250000 * z - 40571873 == 0, boundary_3 == 0)),
              And(*region_2,
                  Or(And(*motor_1057_1377_band, boundary_6 == 0),
                     And(*motor_1377_1457_band, boundary_7 == 0),
                     And(*motor_1457_1697_band, boundary_3 == 0)), td_cases, p1_motor_cases),
              And(*region_1, boundary_2 == 0, *p1_motor_1377_td_region),
              And(*region_1,
                  Or(And(20 * n - 50 * q - 2 * td - 300 * z - 5963 > 0, boundary_2 <= 0),
                     And(boundary_2 > 0, 20 * n - 50 * q - 2 * td - 300 * z - 8923 <= 0,
                         4420 * n - 11050 * q - 442 * td + 81700 * z - 3170191 == 0),
                     And(20 * n - 50 * q - 2 * td - 300 * z - 8923 > 0,
                         20 * n - 50 * q - 2 * td - 300 * z - 10699 < 0,
                         31937500 * n - 79843750 * q - 3193750 * td - 571562500 * z - 13629165033 == 0)),
                  td_cases,
                  p1_split(20 * n - 50 * q - 2 * td - 300 * z - 2411 == 0, boundary_5 == 0)),
              And(*region_1, boundary_1 == 0, *p1_motor_1457_td_region),
              And(*region_1,
                  Or(And(100 * n - 250 * q - 10 * td - 2500 * z - 17299 > 0,
                         20 * n - 50 * q - 2 * td - 500 * z - 6535 <= 0,
                         4420 * n - 11050 * q - 442 * td + 81700 * z - 3170191 == 0),
                     And(20 * n - 50 * q - 2 * td - 500 * z - 6535 > 0, boundary_1 <= 0),
                     And(boundary_1 > 0, 100 * n - 250 * q - 10 * td - 2500 * z - 48051 < 0,
                         187312500 * n - 468281250 * q - 18731250 * td - 4082187500 * z - 74105780531 == 0)),
                  td_cases,
                  p1_split(20 * n - 50 * q - 2 * td - 500 * z + 1153 == 0, boundary_8 == 0)),
              And(*region_1, n_q_td_z_1250000_cases, td_cases,
                  p1_split(62500 * n - 156250 * q - 6250 * td - 1250000 * z - 40571873 == 0,
                           boundary_4 == 0)),
              And(*region_1,
                  Or(And(*motor_1057_1377_band, boundary_5 == 0),
                     And(*motor_1377_1457_band, boundary_8 == 0),
                     And(*motor_1457_1697_band, boundary_4 == 0)), td_cases, p1_motor_cases))


series10 = make_series10()


def make_series11():
    """Construct the factorized series11 input formula."""
    region_1 = (q >= 0, q - 40 <= 0, n - td == 0, i2 == 0)

    return Or(And(*motor_1457_1697_band, *region_1, q_td_boundary > 0, p1_motor_cases, td_cases),
              And(*region_1, p1_boundary >= 0, motor_value - 1457 == 0, q_td_boundary > 0,
                  td_cases),
              And(*motor_1377_1457_band, *region_1, q_td_boundary > 0, p1_motor_cases, td_cases),
              And(*region_1, p1_boundary >= 0, motor_value - 1377 == 0, q_td_boundary > 0,
                  td_cases),
              And(*motor_1057_1377_band, *region_1, q_td_boundary > 0, p1_motor_cases, td_cases),
              And(*motor_1457_1697_band, *region_1, q_td_boundary <= 0, p1_motor_cases,
                  td_cases),
              And(*region_1, p1_boundary >= 0, motor_value - 1457 == 0, q_td_boundary <= 0,
                  td_cases),
              And(*motor_1377_1457_band, *region_1, q_td_boundary <= 0, p1_motor_cases,
                  td_cases),
              And(*region_1, p1_boundary >= 0, motor_value - 1377 == 0, q_td_boundary <= 0,
                  td_cases),
              And(*motor_1057_1377_band, *region_1, q_td_boundary <= 0, p1_motor_cases,
                  td_cases))


series11 = make_series11()


def make_series12():
    """Construct the factorized series12 input formula."""
    boundary_1 = 3 * td + 400 * z - 6130
    boundary_2 = 4 * td - 600 * z + 3245
    boundary_3 = 3 * td + 400 * z - 3730
    boundary_4 = 4 * td - 600 * z - 355
    boundary_5 = 200 * i2 - 10 * q - td + 385
    boundary_6 = 27 * n + 57 * td + 7600 * z - 116470
    boundary_7 = 81 * n - 152 * td + 22800 * z - 123310
    boundary_8 = 120 * n + 3 * td + 200 * z - 80636
    boundary_9 = 450 * n - 95 * td - 3750 * z - 204898
    boundary_10 = 600 * n + 15 * td - 5000 * z - 372364
    boundary_11 = 90 * n - 19 * td + 150 * z - 45602
    boundary_12 = 100 * n - 10 * td - 2500 * z - 36519
    boundary_13 = 20 * n - 2 * td - 300 * z - 8331
    boundary_14 = 120 * n + 3 * td + 200 * z - 84188
    boundary_15 = 120 * n + 3 * td - 1000 * z - 69860
    boundary_16 = 143718750 * n - 30340625 * td - 176718750 * z - 74285891086
    boundary_17 = 191625000 * n + 4790625 * td - 235625000 * z - 130719208948
    boundary_18 = 19890 * n - 4199 * td + 699150 * z - 16058722
    boundary_19 = 26520 * n + 663 * td + 932200 * z - 25794796
    boundary_20 = 374625000 * n + 9365625 * td - 1920625000 * z - 243897029812
    boundary_21 = 90 * n - 19 * td + 150 * z - 48266
    boundary_22 = 90 * n - 19 * td - 750 * z - 37520
    boundary_23 = 93656250 * n - 19771875 * td - 480156250 * z - 45494960578
    boundary_24 = 17680 * i2 + 55250 * p1 + 442 * q + 20000 * z - 483917
    boundary_25 = 25550000 * i2 + 79843750 * p1 + 638750 * q - 2500000 * z - 448579359
    boundary_26 = 29970000 * i2 + 93656250 * p1 + 749250 * q + 2500000 * z - 569558609
    boundary_27 = 10750000 * i2 + 500000 * n + 33593750 * p1 + 268750 * q - 50000 * td - 10000000 * z - 402109359
    boundary_28 = 1480 * i2 - 50 * n + 4625 * p1 + 37 * q + 5 * td + 750 * z - 4647
    boundary_29 = 76880 * i2 - 2000 * n + 240250 * p1 + 1922 * q + 200 * td + 50000 * z - 669797
    boundary_30 = 17760 * i2 - 600 * n + 55500 * p1 + 444 * q - 15 * td - 1000 * z + 97486
    boundary_31 = 230640 * i2 - 6000 * n + 720750 * p1 + 5766 * q - 150 * td + 50000 * z - 476891
    boundary_32 = 26640 * i2 - 900 * n + 83250 * p1 + 666 * q + 190 * td - 1500 * z - 2521
    boundary_33 = 32250000 * i2 + 1500000 * n + 100781250 * p1 + 806250 * q + 37500 * td - 5000000 * z - 1589453077
    boundary_34 = 691920 * i2 - 18000 * n + 2162250 * p1 + 17298 * q + 3800 * td + 150000 * z - 4405673
    boundary_35 = 96750000 * i2 + 4500000 * n + 302343750 * p1 + 2418750 * q - 950000 * td - 15000000 * z - 4024609231
    boundary_36 = 10750000 * i2 + 500000 * n + 33593750 * p1 + 268750 * q - 50000 * td - 10000000 * z - 452109359
    boundary_37 = 112480 * i2 - 3575 * n + 351500 * p1 + 2812 * q + 380 * td + 57000 * z - 353172
    boundary_38 = 1460720 * i2 - 35750 * n + 4564750 * p1 + 36518 * q + 3800 * td + 950000 * z - 12726143
    boundary_39 = 1480 * i2 - 50 * n + 4625 * p1 + 37 * q + 5 * td + 750 * z + 353
    boundary_40 = 204250000 * i2 + 8937500 * n + 638281250 * p1 + 5106250 * q - 950000 * td - 190000000 * z - 7640077821
    boundary_41 = 76880 * i2 - 2000 * n + 240250 * p1 + 1922 * q + 200 * td + 50000 * z - 469797
    region_1 = (n - td == 0, boundary_6 >= 0, boundary_3 >= 0, boundary_1 <= 0, q - 40 <= 0,
                q >= 0)
    region_2 = (n - td == 0, boundary_7 >= 0, boundary_4 <= 0, boundary_2 >= 0, q - 40 <= 0,
                q >= 0)
    region_3 = (n - td == 0,
                6696421875 * n - 711787500 * td - 155123125000 * z - 2816019660178 <= 0,
                187312500 * n - 18731250 * td - 4082187500 * z - 92837030531 <= 0,
                187312500 * n - 18731250 * td - 4082187500 * z - 74105780531 >= 0, q - 40 <= 0,
                q >= 0, z_minus_156250_cases)
    region_4 = (n - td == 0,
                1141765625 * n - 121362500 * td - 21719375000 * z - 517908271254 <= 0,
                31937500 * n - 3193750 * td - 571562500 * z - 16822915033 <= 0,
                31937500 * n - 3193750 * td - 571562500 * z - 13629165033 >= 0, q - 40 <= 0,
                q >= 0, z_plus_156250_cases)
    region_5 = (n - td == 0, n >= 0, q - 40 <= 0, q >= 0)
    region_6 = (n - td == 0, 158015 * n - 16796 * td + 3104600 * z - 120467258 <= 0,
                4420 * n - 442 * td + 81700 * z - 3612191 <= 0,
                4420 * n - 442 * td + 81700 * z - 3170191 >= 0, q - 40 <= 0, q >= 0,
                z_250_cases)
    region_7 = (n - td == 0, 715 * n - 76 * td - 11400 * z - 339074 <= 0,
                20 * n - 2 * td - 300 * z - 10923 <= 0, 20 * n - 2 * td - 300 * z - 8923 >= 0,
                q - 40 <= 0, q >= 0, 125 * z - 1012 == 0)
    region_8 = (n - td == 0, 715 * n - 76 * td - 19000 * z - 248330 <= 0,
                20 * n - 2 * td - 500 * z - 8535 <= 0, 20 * n - 2 * td - 500 * z - 6535 >= 0,
                q - 40 <= 0, q >= 0, 50 * z - 449 == 0)
    region_9 = (n - td == 0, 3575 * n - 380 * td - 95000 * z - 1387722 <= 0,
                100 * n - 10 * td - 2500 * z - 46519 <= 0, boundary_12 >= 0, q - 40 <= 0,
                q >= 0)
    region_10 = (n - td == 0, 715 * n - 76 * td - 11400 * z - 316578 <= 0,
                 20 * n - 2 * td - 300 * z - 10331 <= 0, boundary_13 >= 0, q - 40 <= 0, q >= 0)
    region_11 = (n - td == 0, boundary_37 >= 0, boundary_39 >= 0, boundary_28 <= 0, q - 40 <= 0,
                 q >= 0)
    region_12 = (n - td == 0, boundary_38 >= 0, boundary_41 >= 0, boundary_29 <= 0, q - 40 <= 0,
                 q >= 0)
    region_13 = (n - td == 0, boundary_40 <= 0, boundary_36 <= 0, boundary_27 >= 0, q - 40 <= 0,
                 q >= 0)
    region_14 = (td - 700 >= 0, td - 990 < 0, boundary_5 == 0)
    region_15 = (n_td_z_1250000_cases, td_split(boundary_1 == 0, boundary_2 == 0),
                 p1_split(62500 * n - 6250 * td - 1250000 * z - 40571873 == 0, boundary_27 == 0))
    region_16 = (n_td_z_300_cases, td_split(boundary_1 == 0, boundary_2 == 0),
                 p1_split(20 * n - 2 * td - 300 * z - 2411 == 0, boundary_28 == 0))
    region_17 = (n_td_z_500_cases, td_split(boundary_1 == 0, boundary_2 == 0),
                 p1_split(20 * n - 2 * td - 500 * z + 1153 == 0, boundary_29 == 0))
    region_18 = (n_td_z_1250000_cases, td_cases,
                 p1_split(62500 * n - 6250 * td - 1250000 * z - 40571873 == 0, boundary_27 == 0))
    cases_1 = Or(And(td - 400 >= 0, td - 700 < 0), And(td - 990 < 0, td - 700 == 0))
    case_1 = And(*motor_1057_1377_band, boundary_28 == 0)
    case_2 = And(*motor_1057_1377_band, boundary_30 == 0)
    case_3 = And(*motor_1057_1377_band, boundary_32 == 0)
    case_4 = And(*motor_1377_1457_band, boundary_31 == 0)
    case_5 = And(*motor_1377_1457_band, boundary_34 == 0)
    case_6 = And(*motor_1377_1457_band, boundary_29 == 0)
    case_7 = And(*motor_1457_1697_band, boundary_27 == 0)
    case_8 = And(*motor_1457_1697_band, boundary_33 == 0)
    case_9 = And(*motor_1457_1697_band, boundary_35 == 0)
    cases_2 = Or(And(*motor_1057_1377_band, boundary_24 == 0), And(*motor_1377_1457_band),
                 And(*motor_1457_1697_band, boundary_26 == 0))
    cases_3 = Or(And(*motor_1057_1377_band, boundary_25 == 0),
                 And(*motor_1377_1457_band, boundary_26 == 0), And(*motor_1457_1697_band))
    cases_4 = Or(And(*motor_1057_1377_band), And(*motor_1377_1457_band, boundary_24 == 0),
                 And(*motor_1457_1697_band, boundary_25 == 0))
    cases_5 = Or(case_1, case_6, case_7)
    cases_6 = Or(case_2, case_4, case_8)
    cases_7 = Or(case_3, case_5, case_9)
    cases_8 = Or(And(120 * n + 3 * td + 200 * z - 66428 > 0, boundary_8 <= 0),
                 And(boundary_8 > 0, boundary_14 <= 0, boundary_19 == 0),
                 And(boundary_14 > 0, 120 * n + 3 * td + 200 * z - 94844 < 0, boundary_17 == 0))
    cases_9 = Or(And(281250 * n - 59375 * td - 937500 * z - 171643741 < 0,
                     281250 * n - 59375 * td - 937500 * z - 147456241 >= 0, boundary_16 == 0),
                 And(281250 * n - 59375 * td - 937500 * z - 147456241 < 0,
                     281250 * n - 59375 * td - 937500 * z - 141409366 >= 0, boundary_23 == 0),
                 And(281250 * n - 59375 * td - 937500 * z - 141409366 < 0,
                     281250 * n - 59375 * td - 937500 * z - 123268741 > 0))
    cases_10 = Or(And(375000 * n + 9375 * td - 1250000 * z - 290837488 < 0,
                      375000 * n + 9375 * td - 1250000 * z - 258587488 >= 0, boundary_17 == 0),
                  And(375000 * n + 9375 * td - 1250000 * z - 258587488 < 0,
                      375000 * n + 9375 * td - 1250000 * z - 250524988 >= 0, boundary_20 == 0),
                  And(375000 * n + 9375 * td - 1250000 * z - 250524988 < 0,
                      375000 * n + 9375 * td - 1250000 * z - 226337488 > 0))
    cases_11 = Or(And(450 * n - 95 * td - 3750 * z - 118408 > 0, boundary_22 <= 0,
                      boundary_18 == 0), And(boundary_22 > 0, boundary_9 <= 0),
                  And(boundary_9 > 0, 450 * n - 95 * td - 3750 * z - 256792 < 0,
                      boundary_23 == 0))
    cases_12 = Or(And(600 * n + 15 * td - 5000 * z - 257044 > 0, boundary_15 <= 0,
                      boundary_19 == 0), And(boundary_15 > 0, boundary_10 <= 0),
                  And(boundary_10 > 0, 600 * n + 15 * td - 5000 * z - 441556 < 0,
                      boundary_20 == 0))
    cases_13 = Or(And(90 * n - 19 * td + 150 * z - 34946 > 0, boundary_11 <= 0),
                  And(boundary_11 > 0, boundary_21 <= 0, boundary_18 == 0),
                  And(boundary_21 > 0, 90 * n - 19 * td + 150 * z - 56258 < 0, boundary_16 == 0))

    return Or(And(q_td_boundary > 0, boundary_5 == 0, *region_5, boundary_13 == 0,
                  p1_boundary >= 0, motor_value - 1377 == 0,
                  td_split(boundary_1 == 0, boundary_2 == 0)),
              And(q_td_boundary > 0, boundary_5 == 0, *region_5, boundary_12 == 0,
                  p1_boundary >= 0, motor_value - 1457 == 0,
                  td_split(boundary_1 == 0, boundary_2 == 0)),
              And(q_td_boundary > 0, boundary_5 == 0, *region_5, cases_5,
                  td_split(boundary_1 == 0, boundary_2 == 0), p1_motor_cases),
              And(q_td_boundary > 0, i2 == 0, *region_5, boundary_13 == 0,
                  *p1_motor_1377_td_region),
              And(q_td_boundary > 0, i2 == 0, *region_5, boundary_12 == 0,
                  *p1_motor_1457_td_region),
              And(q_td_boundary > 0, i2 == 0, *region_5, cases_5, td_cases, p1_motor_cases),
              And(q_td_boundary <= 0, boundary_5 == 0, *region_5, boundary_13 == 0,
                  p1_boundary >= 0, motor_value - 1377 == 0,
                  td_split(boundary_1 == 0, boundary_2 == 0)),
              And(q_td_boundary <= 0, boundary_5 == 0, *region_5, boundary_12 == 0,
                  p1_boundary >= 0, motor_value - 1457 == 0,
                  td_split(boundary_1 == 0, boundary_2 == 0)),
              And(q_td_boundary <= 0, boundary_5 == 0, *region_5, cases_5,
                  td_split(boundary_1 == 0, boundary_2 == 0), p1_motor_cases),
              And(q_td_boundary <= 0, i2 == 0, *region_5, boundary_13 == 0,
                  *p1_motor_1377_td_region),
              And(q_td_boundary <= 0, i2 == 0, *region_5, boundary_12 == 0,
                  *p1_motor_1457_td_region),
              And(q_td_boundary <= 0, i2 == 0, *region_5, cases_5, td_cases, p1_motor_cases),
              And(q_td_boundary > 0, boundary_5 == 0, *region_1, boundary_8 == 0,
                  p1_boundary >= 0, motor_value - 1377 == 0, cases_1),
              And(q_td_boundary > 0, boundary_5 == 0, *region_1, boundary_10 == 0,
                  p1_boundary >= 0, motor_value - 1457 == 0, cases_1),
              And(q_td_boundary > 0, boundary_5 == 0, *region_1, cases_6, cases_1,
                  p1_motor_cases),
              And(q_td_boundary > 0, i2 == 0, *region_1, boundary_8 == 0,
                  *p1_motor_1377_td_region),
              And(q_td_boundary > 0, i2 == 0, *region_1, boundary_10 == 0,
                  *p1_motor_1457_td_region),
              And(q_td_boundary > 0, i2 == 0, *region_1, cases_6, td_cases, p1_motor_cases),
              And(q_td_boundary <= 0, boundary_5 == 0, *region_1, boundary_8 == 0,
                  p1_boundary >= 0, motor_value - 1377 == 0, cases_1),
              And(q_td_boundary <= 0, boundary_5 == 0, *region_1, boundary_10 == 0,
                  p1_boundary >= 0, motor_value - 1457 == 0, cases_1),
              And(q_td_boundary <= 0, boundary_5 == 0, *region_1, cases_6, cases_1,
                  p1_motor_cases),
              And(q_td_boundary <= 0, i2 == 0, *region_1, boundary_8 == 0,
                  *p1_motor_1377_td_region),
              And(q_td_boundary <= 0, i2 == 0, *region_1, boundary_10 == 0,
                  *p1_motor_1457_td_region),
              And(q_td_boundary <= 0, i2 == 0, *region_1, cases_6, td_cases, p1_motor_cases),
              And(q_td_boundary > 0, *region_14, *region_2, boundary_11 == 0, p1_boundary >= 0,
                  motor_value - 1377 == 0),
              And(q_td_boundary > 0, *region_14, *region_2, boundary_9 == 0, p1_boundary >= 0,
                  motor_value - 1457 == 0),
              And(q_td_boundary > 0, *region_14, *region_2, cases_7, p1_motor_cases),
              And(q_td_boundary > 0, i2 == 0, *region_2, boundary_11 == 0,
                  *p1_motor_1377_td_region),
              And(q_td_boundary > 0, i2 == 0, *region_2, boundary_9 == 0,
                  *p1_motor_1457_td_region),
              And(q_td_boundary > 0, i2 == 0, *region_2, cases_7, td_cases, p1_motor_cases),
              And(q_td_boundary <= 0, *region_14, *region_2, boundary_11 == 0, p1_boundary >= 0,
                  motor_value - 1377 == 0),
              And(q_td_boundary <= 0, *region_14, *region_2, boundary_9 == 0, p1_boundary >= 0,
                  motor_value - 1457 == 0),
              And(q_td_boundary <= 0, *region_14, *region_2, cases_7, p1_motor_cases),
              And(q_td_boundary <= 0, i2 == 0, *region_2, boundary_11 == 0,
                  *p1_motor_1377_td_region),
              And(q_td_boundary <= 0, i2 == 0, *region_2, boundary_9 == 0,
                  *p1_motor_1457_td_region),
              And(q_td_boundary <= 0, i2 == 0, *region_2, cases_7, td_cases, p1_motor_cases),
              And(q_td_boundary > 0, boundary_5 == 0, *region_5, *region_16),
              And(q_td_boundary > 0, boundary_5 == 0, *region_10, p1_boundary >= 0,
                  motor_value - 1377 == 0, td_split(boundary_8 == 0, boundary_11 == 0)),
              And(q_td_boundary > 0, boundary_5 == 0, *region_7, p1_boundary >= 0,
                  motor_value - 1457 == 0, td_split(boundary_14 == 0, boundary_21 == 0)),
              And(q_td_boundary > 0, boundary_5 == 0, *region_1, cases_8, cases_1,
                  p1_split(120 * n + 3 * td + 200 * z - 45116 == 0, boundary_30 == 0)),
              And(q_td_boundary > 0, boundary_5 == 0, *region_2, td - 700 >= 0, td - 990 < 0,
                  cases_13, p1_split(90 * n - 19 * td + 150 * z - 18962 == 0, boundary_32 == 0)),
              And(q_td_boundary > 0, boundary_5 == 0, *region_11, cases_4,
                  td_split(boundary_30 == 0, boundary_32 == 0), p1_motor_cases),
              And(q_td_boundary > 0, i2 == 0, *region_5, n_td_z_300_cases, td_cases,
                  p1_split(20 * n - 2 * td - 300 * z - 2411 == 0, boundary_28 == 0)),
              And(q_td_boundary > 0, i2 == 0, *region_10, *p1_motor_1377_td_region),
              And(q_td_boundary > 0, i2 == 0, *region_7, *p1_motor_1457_td_region),
              And(q_td_boundary > 0, i2 == 0, *region_11, cases_4, td_cases, p1_motor_cases),
              And(q_td_boundary <= 0, boundary_5 == 0, *region_5, *region_16),
              And(q_td_boundary <= 0, boundary_5 == 0, *region_10, p1_boundary >= 0,
                  motor_value - 1377 == 0, td_split(boundary_8 == 0, boundary_11 == 0)),
              And(q_td_boundary <= 0, boundary_5 == 0, *region_7, p1_boundary >= 0,
                  motor_value - 1457 == 0, td_split(boundary_14 == 0, boundary_21 == 0)),
              And(q_td_boundary <= 0, boundary_5 == 0, *region_1, cases_8, cases_1,
                  p1_split(120 * n + 3 * td + 200 * z - 45116 == 0, boundary_30 == 0)),
              And(q_td_boundary <= 0, boundary_5 == 0, *region_2, td - 700 >= 0, td - 990 < 0,
                  cases_13, p1_split(90 * n - 19 * td + 150 * z - 18962 == 0, boundary_32 == 0)),
              And(q_td_boundary <= 0, boundary_5 == 0, *region_11, cases_4,
                  td_split(boundary_30 == 0, boundary_32 == 0), p1_motor_cases),
              And(q_td_boundary <= 0, i2 == 0, *region_5, n_td_z_300_cases, td_cases,
                  p1_split(20 * n - 2 * td - 300 * z - 2411 == 0, boundary_28 == 0)),
              And(q_td_boundary <= 0, i2 == 0, *region_10, *p1_motor_1377_td_region),
              And(q_td_boundary <= 0, i2 == 0, *region_7, *p1_motor_1457_td_region),
              And(q_td_boundary <= 0, i2 == 0, *region_11, cases_4, td_cases, p1_motor_cases),
              And(q_td_boundary > 0, boundary_5 == 0, *region_5, *region_17),
              And(q_td_boundary > 0, boundary_5 == 0, *region_8, p1_boundary >= 0,
                  motor_value - 1377 == 0, td_split(boundary_15 == 0, boundary_22 == 0)),
              And(q_td_boundary > 0, boundary_5 == 0, *region_6,
                  td_split(boundary_19 == 0, boundary_18 == 0),
                  p1_split(50 * z - 891 == 0, boundary_24 == 0)),
              And(q_td_boundary > 0, boundary_5 == 0, *region_9, p1_boundary >= 0,
                  motor_value - 1457 == 0, td_split(boundary_10 == 0, boundary_9 == 0)),
              And(q_td_boundary > 0, boundary_5 == 0, *region_1, cases_12, cases_1,
                  p1_split(120 * n + 3 * td - 1000 * z - 23732 == 0, boundary_31 == 0)),
              And(q_td_boundary > 0, boundary_5 == 0, *region_2, td - 700 >= 0, td - 990 < 0,
                  cases_11, p1_split(90 * n - 19 * td - 750 * z - 2924 == 0, boundary_34 == 0)),
              And(q_td_boundary > 0, boundary_5 == 0, *region_12, cases_2,
                  td_split(boundary_31 == 0, boundary_34 == 0), p1_motor_cases),
              And(q_td_boundary > 0, i2 == 0, *region_5, n_td_z_500_cases, td_cases,
                  p1_split(20 * n - 2 * td - 500 * z + 1153 == 0, boundary_29 == 0)),
              And(q_td_boundary > 0, i2 == 0, *region_8, *p1_motor_1377_td_region),
              And(q_td_boundary > 0, i2 == 0, *region_6, td_cases,
                  p1_split(50 * z - 891 == 0, boundary_24 == 0)),
              And(q_td_boundary > 0, i2 == 0, *region_9, *p1_motor_1457_td_region),
              And(q_td_boundary > 0, i2 == 0, *region_12, cases_2, td_cases, p1_motor_cases),
              And(q_td_boundary <= 0, boundary_5 == 0, *region_5, *region_17),
              And(q_td_boundary <= 0, boundary_5 == 0, *region_8, p1_boundary >= 0,
                  motor_value - 1377 == 0, td_split(boundary_15 == 0, boundary_22 == 0)),
              And(q_td_boundary <= 0, boundary_5 == 0, *region_6,
                  td_split(boundary_19 == 0, boundary_18 == 0),
                  p1_split(50 * z - 891 == 0, boundary_24 == 0)),
              And(q_td_boundary <= 0, boundary_5 == 0, *region_9, p1_boundary >= 0,
                  motor_value - 1457 == 0, td_split(boundary_10 == 0, boundary_9 == 0)),
              And(q_td_boundary <= 0, boundary_5 == 0, *region_1, cases_12, cases_1,
                  p1_split(120 * n + 3 * td - 1000 * z - 23732 == 0, boundary_31 == 0)),
              And(q_td_boundary <= 0, boundary_5 == 0, *region_2, td - 700 >= 0, td - 990 < 0,
                  cases_11, p1_split(90 * n - 19 * td - 750 * z - 2924 == 0, boundary_34 == 0)),
              And(q_td_boundary <= 0, boundary_5 == 0, *region_12, cases_2,
                  td_split(boundary_31 == 0, boundary_34 == 0), p1_motor_cases),
              And(q_td_boundary <= 0, i2 == 0, *region_5, n_td_z_500_cases, td_cases,
                  p1_split(20 * n - 2 * td - 500 * z + 1153 == 0, boundary_29 == 0)),
              And(q_td_boundary <= 0, i2 == 0, *region_8, *p1_motor_1377_td_region),
              And(q_td_boundary <= 0, i2 == 0, *region_6, td_cases,
                  p1_split(50 * z - 891 == 0, boundary_24 == 0)),
              And(q_td_boundary <= 0, i2 == 0, *region_9, *p1_motor_1457_td_region),
              And(q_td_boundary <= 0, i2 == 0, *region_12, cases_2, td_cases, p1_motor_cases),
              And(q_td_boundary > 0, boundary_5 == 0, *region_5, *region_15),
              And(q_td_boundary > 0, boundary_5 == 0, *region_4,
                  td_split(boundary_17 == 0, boundary_16 == 0),
                  p1_split(156250 * z + 16518749 == 0, boundary_25 == 0)),
              And(q_td_boundary > 0, boundary_5 == 0, *region_3,
                  td_split(boundary_20 == 0, boundary_23 == 0),
                  p1_split(156250 * z - 22087499 == 0, boundary_26 == 0)),
              And(q_td_boundary > 0, boundary_5 == 0, *region_1, cases_10, cases_1,
                  p1_split(375000 * n + 9375 * td - 1250000 * z - 339212488 == 0,
                           boundary_33 == 0)),
              And(q_td_boundary > 0, boundary_5 == 0, *region_2, td - 700 >= 0, td - 990 < 0,
                  cases_9,
                  p1_split(281250 * n - 59375 * td - 937500 * z - 207924991 == 0,
                           boundary_35 == 0)),
              And(q_td_boundary > 0, boundary_5 == 0, *region_13, cases_3,
                  td_split(boundary_33 == 0, boundary_35 == 0), p1_motor_cases),
              And(q_td_boundary > 0, i2 == 0, *region_5, *region_18),
              And(q_td_boundary > 0, i2 == 0, *region_4, td_cases,
                  p1_split(156250 * z + 16518749 == 0, boundary_25 == 0)),
              And(q_td_boundary > 0, i2 == 0, *region_3, td_cases,
                  p1_split(156250 * z - 22087499 == 0, boundary_26 == 0)),
              And(q_td_boundary > 0, i2 == 0, *region_13, cases_3, td_cases, p1_motor_cases),
              And(q_td_boundary <= 0, boundary_5 == 0, *region_5, *region_15),
              And(q_td_boundary <= 0, boundary_5 == 0, *region_4,
                  td_split(boundary_17 == 0, boundary_16 == 0),
                  p1_split(156250 * z + 16518749 == 0, boundary_25 == 0)),
              And(q_td_boundary <= 0, boundary_5 == 0, *region_3,
                  td_split(boundary_20 == 0, boundary_23 == 0),
                  p1_split(156250 * z - 22087499 == 0, boundary_26 == 0)),
              And(q_td_boundary <= 0, boundary_5 == 0, *region_1, cases_10, cases_1,
                  p1_split(375000 * n + 9375 * td - 1250000 * z - 339212488 == 0,
                           boundary_33 == 0)),
              And(q_td_boundary <= 0, boundary_5 == 0, *region_2, td - 700 >= 0, td - 990 < 0,
                  cases_9,
                  p1_split(281250 * n - 59375 * td - 937500 * z - 207924991 == 0,
                           boundary_35 == 0)),
              And(q_td_boundary <= 0, boundary_5 == 0, *region_13, cases_3,
                  td_split(boundary_33 == 0, boundary_35 == 0), p1_motor_cases),
              And(q_td_boundary <= 0, i2 == 0, *region_5, *region_18),
              And(q_td_boundary <= 0, i2 == 0, *region_4, td_cases,
                  p1_split(156250 * z + 16518749 == 0, boundary_25 == 0)),
              And(q_td_boundary <= 0, i2 == 0, *region_3, td_cases,
                  p1_split(156250 * z - 22087499 == 0, boundary_26 == 0)),
              And(q_td_boundary <= 0, i2 == 0, *region_13, cases_3, td_cases, p1_motor_cases))


series12 = make_series12()


def make_series13():
    """Construct the factorized series13 input formula."""
    boundary_1 = 10750000 * i2 + 500000 * n + 33593750 * p1 + 268750 * q - 50000 * td - 10000000 * z - 402109359
    boundary_2 = 1480 * i2 - 50 * n + 4625 * p1 + 37 * q + 5 * td + 750 * z - 4647
    boundary_3 = 76880 * i2 - 2000 * n + 240250 * p1 + 1922 * q + 200 * td + 50000 * z - 669797

    return Or(And(*base_operating_region, p1_motor_cases, td_cases,
                  Or(And(*motor_1057_1377_band, boundary_2 == 0),
                     And(*motor_1377_1457_band, boundary_3 == 0),
                     And(*motor_1457_1697_band, boundary_1 == 0))),
              And(*base_operating_region, p1_boundary >= 0, motor_value - 1377 == 0,
                  20 * n - 2 * td - 300 * z - 8331 == 0, td_cases),
              And(*base_operating_region, p1_boundary >= 0, motor_value - 1457 == 0,
                  100 * n - 10 * td - 2500 * z - 36519 == 0, td_cases),
              And(*base_operating_region,
                  p1_split(20 * n - 2 * td - 300 * z - 2411 == 0, boundary_2 == 0), td_cases,
                  n_td_z_300_cases),
              And(q >= 0, q - 40 <= 0, boundary_2 <= 0,
                  1480 * i2 - 50 * n + 4625 * p1 + 37 * q + 5 * td + 750 * z + 353 >= 0,
                  112480 * i2 - 3575 * n + 351500 * p1 + 2812 * q + 380 * td + 57000 * z - 353172 >= 0,
                  n - td == 0, i2 == 0, p1_motor_cases, td_cases,
                  Or(And(*motor_1057_1377_band),
                     And(*motor_1377_1457_band,
                         17680 * i2 + 55250 * p1 + 442 * q + 20000 * z - 483917 == 0),
                     And(*motor_1457_1697_band,
                         25550000 * i2 + 79843750 * p1 + 638750 * q - 2500000 * z - 448579359 == 0))),
              And(q >= 0, q - 40 <= 0, 20 * n - 2 * td - 300 * z - 8331 >= 0,
                  20 * n - 2 * td - 300 * z - 10331 <= 0,
                  715 * n - 76 * td - 11400 * z - 316578 <= 0, n - td == 0, i2 == 0,
                  *p1_motor_1377_td_region),
              And(q >= 0, q - 40 <= 0, 20 * n - 2 * td - 300 * z - 8923 >= 0,
                  20 * n - 2 * td - 300 * z - 10923 <= 0,
                  715 * n - 76 * td - 11400 * z - 339074 <= 0, n - td == 0, i2 == 0,
                  p1_boundary >= 0, motor_value - 1457 == 0, 125 * z - 1012 == 0, td_cases),
              And(*base_operating_region,
                  p1_split(20 * n - 2 * td - 500 * z + 1153 == 0, boundary_3 == 0), td_cases,
                  n_td_z_500_cases),
              And(q >= 0, q - 40 <= 0, boundary_3 <= 0,
                  76880 * i2 - 2000 * n + 240250 * p1 + 1922 * q + 200 * td + 50000 * z - 469797 >= 0,
                  1460720 * i2 - 35750 * n + 4564750 * p1 + 36518 * q + 3800 * td + 950000 * z - 12726143 >= 0,
                  n - td == 0, i2 == 0, p1_motor_cases, td_cases,
                  Or(And(*motor_1057_1377_band,
                         17680 * i2 + 55250 * p1 + 442 * q + 20000 * z - 483917 == 0),
                     And(*motor_1377_1457_band),
                     And(*motor_1457_1697_band,
                         29970000 * i2 + 93656250 * p1 + 749250 * q + 2500000 * z - 569558609 == 0))),
              And(q >= 0, q - 40 <= 0, 20 * n - 2 * td - 500 * z - 6535 >= 0,
                  20 * n - 2 * td - 500 * z - 8535 <= 0,
                  715 * n - 76 * td - 19000 * z - 248330 <= 0, n - td == 0, i2 == 0,
                  p1_boundary >= 0, motor_value - 1377 == 0, 50 * z - 449 == 0, td_cases),
              And(q >= 0, q - 40 <= 0, 4420 * n - 442 * td + 81700 * z - 3170191 >= 0,
                  4420 * n - 442 * td + 81700 * z - 3612191 <= 0,
                  158015 * n - 16796 * td + 3104600 * z - 120467258 <= 0, n - td == 0, i2 == 0,
                  p1_split(50 * z - 891 == 0,
                           17680 * i2 + 55250 * p1 + 442 * q + 20000 * z - 483917 == 0),
                  td_cases, z_250_cases),
              And(q >= 0, q - 40 <= 0, 100 * n - 10 * td - 2500 * z - 36519 >= 0,
                  100 * n - 10 * td - 2500 * z - 46519 <= 0,
                  3575 * n - 380 * td - 95000 * z - 1387722 <= 0, n - td == 0, i2 == 0,
                  *p1_motor_1457_td_region),
              And(*base_operating_region,
                  p1_split(62500 * n - 6250 * td - 1250000 * z - 40571873 == 0, boundary_1 == 0),
                  td_cases, n_td_z_1250000_cases),
              And(q >= 0, q - 40 <= 0, boundary_1 >= 0,
                  10750000 * i2 + 500000 * n + 33593750 * p1 + 268750 * q - 50000 * td - 10000000 * z - 452109359 <= 0,
                  204250000 * i2 + 8937500 * n + 638281250 * p1 + 5106250 * q - 950000 * td - 190000000 * z - 7640077821 <= 0,
                  n - td == 0, i2 == 0, p1_motor_cases, td_cases,
                  Or(And(*motor_1057_1377_band,
                         25550000 * i2 + 79843750 * p1 + 638750 * q - 2500000 * z - 448579359 == 0),
                     And(*motor_1377_1457_band,
                         29970000 * i2 + 93656250 * p1 + 749250 * q + 2500000 * z - 569558609 == 0),
                     And(*motor_1457_1697_band))),
              And(q >= 0, q - 40 <= 0,
                  31937500 * n - 3193750 * td - 571562500 * z - 13629165033 >= 0,
                  31937500 * n - 3193750 * td - 571562500 * z - 16822915033 <= 0,
                  1141765625 * n - 121362500 * td - 21719375000 * z - 517908271254 <= 0,
                  n - td == 0, i2 == 0,
                  p1_split(156250 * z + 16518749 == 0,
                           25550000 * i2 + 79843750 * p1 + 638750 * q - 2500000 * z - 448579359 == 0),
                  td_cases, z_plus_156250_cases),
              And(q >= 0, q - 40 <= 0,
                  187312500 * n - 18731250 * td - 4082187500 * z - 74105780531 >= 0,
                  187312500 * n - 18731250 * td - 4082187500 * z - 92837030531 <= 0,
                  6696421875 * n - 711787500 * td - 155123125000 * z - 2816019660178 <= 0,
                  n - td == 0, i2 == 0,
                  p1_split(156250 * z - 22087499 == 0,
                           29970000 * i2 + 93656250 * p1 + 749250 * q + 2500000 * z - 569558609 == 0),
                  td_cases, z_minus_156250_cases))


series13 = make_series13()


MOTOR_SERIES = (("series1", series1, 710),
                ("series2", series2, 1420),
                ("series3", series3, 94),
                ("series4", series4, 292),
                ("series5", series5, 157),
                ("series6", series6, 994),
                ("series7", series7, 248),
                ("series8", series8, 473),
                ("series9", series9, 235),
                ("series10", series10, 478),
                ("series11", series11, 168),
                ("series12", series12, 2176),
                ("series13", series13, 358))
