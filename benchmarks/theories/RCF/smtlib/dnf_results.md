# SMT-LIB DNF benchmark results

These tables record atom counts, computation times, and outcomes from completed
benchmark runs. They are maintained from saved results; running pytest prints
new results without changing this file. Measurement dates and conditions are
recorded with the tables. Input provenance and selection history are in
[NOTES.md](NOTES.md).

## Reproducing the results

Run the commands from `benchmarks/theories/RCF/smtlib/`. The module docstrings
of [test_dnf_sizes.py](test_dnf_sizes.py) and
[test_dnf_times.py](test_dnf_times.py) contain the complete usage instructions.
Use the size tests for atom counts and approximate times, optionally in
parallel. Use the timing tests sequentially for controlled timings.

The standard comparison has 62 inputs and six configurations. `-m dnf_main`
selects this set; `-k` selects input names and configuration IDs within it:

| Table column | Configuration ID |
| ------------ | ---------------- |
| RL off       | `redlog-off`     |
| RL on        | `redlog-on`      |
| NONE         | `none`           |
| FINAL        | `final`          |
| AFTER_STEP   | `after-step`     |
| DURING_STEP  | `during-step`    |

For example, list the four cases for half-2 and levin-rhsc with FINAL and
AFTER_STEP, then compute their sizes using two concurrent workers:

```bash
pytest test_dnf_sizes.py -m dnf_main \
    -k '(half-2 or levin-rhsc) and (final or after-step)' --collect-only -q
pytest test_dnf_sizes.py -m dnf_main -n 2 --dnf-timeout=600 \
    -k '(half-2 or levin-rhsc) and (final or after-step)' > dnf-sizes.log 2>&1
```

Omit `-k` to select every standard input/configuration pair. Omit `-n` for
sequential execution. Redlog requires `redcsl` on PATH; record the executable
and version used. Set `--dnf-timeout=SECONDS` for each run. It applies separately
to each DNF computation and kills its worker process group, including Reduce,
at the deadline. Computation timeouts are expected outcomes: they remain
in the size table and terminal timeout count, without failure listings or
a failing exit status. `-x` does not stop on them; JUnit records them as
skipped tests. Errors and startup/validation timeouts remain failures.
The default is infinity: omitting the option imposes no
computation limit. Startup and validation retain separate 30-second guards.

The markers `dnf_under_60s` and `dnf_under_600s` select standard inputs whose
six configurations all completed below the threshold in these saved tables.
They select 46 inputs (276 computations) and 52 inputs (312 computations),
respectively. The groups overlap; the ten inputs with unresolved outcomes
belong to neither. Supplementary experiments have neither runtime marker.
These classifications do not impose limits or guarantee runtimes on a new run.

To retry unresolved computations, select their input names and configuration
IDs with `-m dnf_main -k ...` and supply the requested limit, for example
`--dnf-timeout=600`. Do not use a runtime marker for these retries: it would
exclude unresolved inputs. Reuse completed measurements unless a rerun is
explicitly requested, and do not retry recorded errors merely because the
limit has increased. Retain the raw log and the actual limit for every retry.

For controlled timings, use the same selections with the timing file and save
both timings and test outcomes:

```bash
pytest test_dnf_times.py -m dnf_main --dnf-timeout=600 \
    -k '(half-2 or levin-rhsc) and (final or after-step)' \
    --benchmark-json=dnf-times.json --junitxml=dnf-times.xml > dnf-times.log 2>&1
```

The supplementary tables have separate selections. Both levin-rhsc settings
are selected by name; the disabled setting alone has `dnf_implicit_ranges`:

```bash
pytest test_dnf_sizes.py -k levin-rhsc-during-step -n 2 --dnf-timeout=60
```

For sumcracker-14e, run the two positive-budget sweeps sequentially and
select the zero-budget controls separately. `-x` stops on errors, not on
expected computation timeouts:

```bash
pytest test_dnf_sizes.py -m dnf_pair_checks -k 'during-step and not 0-' -x --dnf-timeout=60
pytest test_dnf_sizes.py -m dnf_pair_checks -k 'after-step and not 0-' -x --dnf-timeout=60
pytest test_dnf_sizes.py -m dnf_pair_checks -k 0- --dnf-timeout=60
```

To reproduce the original stopping rule, run one budget at a time and
stop doubling after the first timeout. The commands above run all selected
budgets unless an error occurs.

The commands explicitly use the recorded 60-second limit. Both sweeps test
budgets 1 through 2048 by doubling. The table retains selected rows; the executable sweep also
includes 4, 8, 16, 32, 64, and 128. The PyEDA table is informational and has
no cases in either pytest file.

## Updating this record

Reuse saved results when changing presentation. When new computations are
requested, select the relevant cases and retain their logs before editing
the tables. For each new run, record the date, code revision and local changes,
RCF backend, Reduce version for Redlog calls, options, time limits, and worker count.
Distinguish approximate parallel timings from sequential timing measurements.
Input loading and output validation are outside the recorded computation time.

Keep table row names attached to the original SMT-LIB paths in
[smtlib_inputs.py](smtlib_inputs.py); these local names are aliases, not
necessarily original filenames. Preserve dates and settings of retained
measurements. The former configuration ID `cartesian` maps to `none` in saved
logs; `cartesian` remains the method used by all four native configurations.

Move an input to the completed group only when all six configurations return
validated results. A timeout has an unknown size and a time cell showing the
actual limit followed by `*`. Keep errors distinct from timeouts and report
which phase failed. An unknown size is not a measured zero. A successful test
checks input counts, DNF structure, and variables; it does not prove equivalence.

Before resuming investigation of the recorded Redlog errors, read the
[Reduce problem notes](../../../../doc/notes/RCF/reduce-problems/). Keep diagnostic
runs separate from benchmark measurements. Do not rerun PyEDA as part of the
standard comparison.


## Measurement conditions

Inputs are from the SMT-LIB 2025 non-incremental release,
https://doi.org/10.5281/zenodo.16740866. Original files, credits, licenses,
and selection criteria are recorded in [inputs](inputs/) and [NOTES.md](NOTES.md).

`RL off` uses Redlog with `bnfsac=False, bnfsm=False`; `RL on`
enables both. The other columns use `method="cartesian"` with
`Simplification.NONE`, `FINAL`, `AFTER_STEP`, or `DURING_STEP`.
The three simplifying schedules use `max_pair_checks=500`.

Each recorded main computation used a fresh spawned worker. The computation
limit was 600 seconds for the 16 inputs identified below, and 60 seconds
for the other main inputs. These are measurement conditions, not limits
assigned to future runs.
Startup, parsing, and output counting are outside timing; the timed call
includes a small pipe round trip. Atom counts include
repetitions across cubes. In timing tables, an asterisk marks a timeout at
the displayed limit: `600*` means a 600-second timeout. In size tables,
`t/o` denotes a timeout; `error` denotes a failed computation in either
table. Both have unknown result sizes. Completed measurements remain valid
when the computation limit changes.

The following tables cover 52 inputs for which all six configurations
returned results. Measurements used Sage, with one call per configuration,
on 2026-10-22, 2026-11-04, and 2026-11-05. [NOTES.md](NOTES.md) records their
provenance.

The 600-second rerun on 2026-11-05 covered ppda-schelling, bug-hoare,
cggmp2005, impulse, nested-c, and simple1 in the upper tables, and all ten
inputs in the lower tables. It used Sage, the corrected Redlog interface,
and the 7-Mar-2026 Reduce snapshot (CSL revision 7327). Two computations
ran concurrently; earlier runs were sequential. This difference can affect
timings and timeout outcomes.

For the other 46 upper-table inputs, Redlog measurements for beyer-figure4,
const1, and nutz-transform05 use the March 2026 snapshot; the rest use the
August 2024 snapshot. These older Redlog measurements predate the
result-conversion fix in `redlog.py`.

Sixteen retained timings in the upper tables measure the DNF call inside
the worker and exclude the pipe round trip. [NOTES.md](NOTES.md) identifies them.
All lower-table timings use the parent/worker protocol.

Both pairs of tables order rows by source family alphabetically, then by
natural alphanumeric case name within each family. Local names map to
original archive paths in [smtlib_inputs.py](smtlib_inputs.py).

## Results for completed inputs

Input and result atom counts:

| case              | input | RL off | RL on |  NONE | FINAL | AFTER_STEP | DURING_STEP |
| ----------------- | ----: | -----: | ----: | ----: | ----: | ---------: | ----------: |
| economics-0016c   |    16 |     42 |    42 |    42 |    42 |         42 |          42 |
| economics-0052c   |    23 |     22 |    22 |    21 |    21 |         21 |          21 |
| economics-0063c   |    11 |     11 |    11 |    11 |    11 |         11 |          11 |
| economics-0086c   |    39 |     38 |    38 |    38 |    38 |         38 |          38 |
| economics-1001c   |    40 |     40 |    40 |    40 |    40 |         40 |          40 |
| hendecagon        |    24 |     23 |    23 |    22 |    22 |         22 |          22 |
| kochanski         |    29 |     28 |    28 |    27 |    27 |         27 |          27 |
| pentagon-diagonal |    12 |     11 |    11 |    10 |    10 |         10 |          10 |
| right-triangle    |     9 |     14 |    14 |     8 |     8 |          8 |           8 |
| triangle-bottema  |    13 |     12 |    12 |    12 |    12 |         12 |          12 |
| triangle-sym321   |     9 |      8 |     8 |     8 |     8 |          8 |           8 |
| kissing-3-9       |    45 |     45 |    45 |    45 |    45 |         45 |          45 |
| kissing-4-14      |   105 |    105 |   105 |   105 |   105 |        105 |         105 |
| kissing-4-25      |   325 |    325 |   325 |   325 |   325 |        325 |         325 |
| mbo-e4e11         |     8 |      8 |     8 |     8 |     8 |          8 |           8 |
| mbo-e7e10         |     8 |      8 |     8 |     8 |     8 |          8 |           8 |
| mbo-e11e14        |     8 |      8 |     8 |     8 |     8 |          8 |           8 |
| mbo-e14           |     7 |      7 |     7 |     7 |     7 |          7 |           7 |
| polypaver-0326    |    19 |    120 |    14 |   120 |    14 |         14 |          14 |
| mgc-10            |    13 |     13 |    13 |    13 |    13 |         13 |          13 |
| pine-pendulum     |    16 |     60 |    60 |    60 |    60 |         60 |          60 |
| pine-quadratic    |    12 |     40 |    40 |    16 |    16 |         16 |          16 |
| ppda-escape25     |   545 |    496 |   496 |   496 |   496 |        496 |         496 |
| ppda-mod5         |   243 |    234 |   234 |   234 |   234 |        234 |         234 |
| ppda-mod7         |   655 |    638 |   638 |   638 |   638 |        638 |         638 |
| ppda-schelling    |   708 |    616 |   616 |   616 |   616 |        616 |         616 |
| beyer-figure4     |    88 |  19328 | 19328 |  9600 |  9600 |       9600 |        9600 |
| bug-hoare         |   100 |  33536 | 33536 | 11328 | 11328 |      11328 |       11328 |
| bugtest1          |    89 |   2528 |  2528 |  2528 |  2528 |       2528 |        2528 |
| cggmp2005         |   133 |   7680 |  7680 |  7424 |  7424 |       7424 |        7424 |
| const1            |   117 |   6656 |  6656 |   788 |   788 |        788 |         788 |
| half-2            |    81 |    288 |   288 |   280 |   280 |        280 |         280 |
| hhk2008           |   138 |    988 |   988 |   956 |   956 |        956 |         956 |
| impulse           |   108 |  23936 | 23936 | 23936 | 23936 |      23936 |       23936 |
| lex-index         |    46 |    616 |   616 |   304 |   304 |        304 |         304 |
| max-in-array      |   111 |    748 |   748 |   756 |   756 |        756 |         756 |
| monadic-invariant |    46 |    616 |   616 |   304 |   304 |        304 |         304 |
| nested-c          |   206 |  12096 | 12096 |  2816 |  2816 |       2816 |        2816 |
| nutz-transform05  |    87 |   2384 |  2384 |  2384 |  2384 |       2384 |        2384 |
| partial-init      |   125 |    804 |   804 |   876 |   876 |        876 |         876 |
| simple1           |   172 |  39936 | 39936 | 37888 | 37888 |      37888 |       37888 |
| two-index         |   126 |    860 |   860 |   796 |   796 |        796 |         796 |
| issac10-ex3e      |    27 |    368 |   368 |   368 |   368 |        368 |         368 |
| issac10-ex3f      |    32 |    864 |   864 |   864 |   864 |        864 |         864 |
| issac10-ex4h      |    39 |     10 |    10 |    10 |    10 |         10 |          10 |
| knopp-schur-b     |    29 |      9 |     9 |     9 |     9 |          9 |           9 |
| levin-rhsc        |    46 |     88 |    88 |    88 |    88 |         88 |          88 |
| mitrinovic-3-29   |    20 |      7 |     7 |     7 |     7 |          7 |           7 |
| mitrinovic-4-14   |    16 |      8 |     8 |     8 |     8 |          8 |           8 |
| sumcracker-14b    |    24 |    256 |   256 |   256 |   256 |        256 |         256 |
| sumcracker-14c    |    34 |   1408 |  1408 |  1408 |  1408 |       1408 |        1408 |
| sumcracker-14d    |    44 |   7168 |  7168 |  7168 |  7168 |       7168 |        7168 |

Corresponding computation times, in seconds:

| case              | RL off |   RL on |    NONE |   FINAL | AFTER_STEP | DURING_STEP |
| ----------------- | -----: | ------: | ------: | ------: | ---------: | ----------: |
| economics-0016c   |  0.119 |   0.120 |   0.058 |   0.092 |      0.096 |       0.092 |
| economics-0052c   |  0.115 |   0.115 |   0.079 |   0.080 |      0.083 |       0.079 |
| economics-0063c   |  0.102 |   0.103 |   0.010 |   0.010 |      0.010 |       0.010 |
| economics-0086c   |  0.158 |   0.152 |   0.061 |   0.059 |      0.061 |       0.063 |
| economics-1001c   |  0.309 |   0.310 |   0.094 |   0.094 |      0.096 |       0.105 |
| hendecagon        |  0.113 |   0.119 |   0.037 |   0.038 |      0.040 |       0.039 |
| kochanski         |  0.131 |   0.121 |   0.046 |   0.046 |      0.045 |       0.048 |
| pentagon-diagonal |  0.104 |   0.103 |   0.018 |   0.019 |      0.018 |       0.018 |
| right-triangle    |  0.108 |   0.101 |   0.013 |   0.012 |      0.012 |       0.012 |
| triangle-bottema  |  0.104 |   0.101 |   0.019 |   0.019 |      0.019 |       0.021 |
| triangle-sym321   |  0.104 |   0.107 |   0.011 |   0.012 |      0.014 |       0.012 |
| kissing-3-9       |  0.148 |   0.144 |   0.078 |   0.078 |      0.081 |       0.084 |
| kissing-4-14      |  0.342 |   0.336 |   0.244 |   0.249 |      0.250 |       0.291 |
| kissing-4-25      |  1.389 |   1.410 |   1.012 |   1.026 |      1.026 |       1.618 |
| mbo-e4e11         |  1.146 |   1.152 |   0.128 |   0.132 |      0.130 |       0.129 |
| mbo-e7e10         |  1.531 |   1.531 |   0.122 |   0.118 |      0.119 |       0.128 |
| mbo-e11e14        |  0.318 |   0.290 |   0.044 |   0.044 |      0.043 |       0.044 |
| mbo-e14           |  0.138 |   0.134 |   0.016 |   0.015 |      0.015 |       0.015 |
| polypaver-0326    |  0.143 |   0.112 |   0.070 |   0.044 |      0.040 |       0.040 |
| mgc-10            |  0.110 |   0.105 |   0.016 |   0.016 |      0.016 |       0.016 |
| pine-pendulum     |  0.120 |   0.118 |   0.034 |   0.119 |      0.122 |       0.118 |
| pine-quadratic    |  0.114 |   0.111 |   0.023 |   0.033 |      0.037 |       0.034 |
| ppda-escape25     |  0.764 |   0.762 |  30.832 |  30.137 |     30.419 |      32.458 |
| ppda-mod5         |  0.278 |   0.275 |   1.173 |   1.203 |      1.195 |       1.486 |
| ppda-mod7         |  1.216 |   1.194 |   9.164 |   9.343 |      9.247 |      12.694 |
| ppda-schelling    |  1.255 |   1.251 | 141.590 | 143.431 |    140.728 |     144.262 |
| beyer-figure4     | 16.491 |  54.134 |   3.465 |  34.770 |     35.902 |      35.374 |
| bug-hoare         | 29.802 | 135.107 |   4.912 |  46.734 |     45.498 |      46.987 |
| bugtest1          |  1.842 |   2.610 |   1.127 |  33.604 |     33.399 |      33.478 |
| cggmp2005         |  6.543 |  27.386 |   5.085 |  66.346 |     69.015 |      66.603 |
| const1            |  5.293 |  21.238 |   0.992 |   3.611 |      3.689 |       3.856 |
| half-2            |  0.221 |   0.239 |   0.439 |   0.742 |      0.760 |       0.753 |
| hhk2008           |  1.027 |   1.337 |   1.296 |   4.674 |      4.655 |       4.765 |
| impulse           | 18.813 |  77.269 |   9.509 |  56.189 |     55.741 |      57.251 |
| lex-index         |  0.442 |   0.452 |   0.201 |   0.935 |      0.927 |       0.924 |
| max-in-array      |  0.602 |   0.759 |   0.966 |   3.341 |      3.281 |       3.385 |
| monadic-invariant |  0.418 |   0.445 |   0.199 |   0.888 |      0.884 |       0.891 |
| nested-c          | 12.255 |  92.877 |  20.727 |  62.553 |     62.697 |      63.285 |
| nutz-transform05  |  1.775 |   3.139 |   1.609 |  39.167 |     38.266 |      38.460 |
| partial-init      |  0.669 |   0.863 |   1.137 |   3.905 |      3.984 |       4.023 |
| simple1           | 36.149 | 556.550 |  28.587 | 124.453 |    125.388 |     131.849 |
| two-index         |  0.740 |   0.990 |   1.521 |   4.686 |      4.782 |       4.885 |
| issac10-ex3e      |  0.281 |   0.251 |   0.191 |   4.120 |      4.126 |       4.152 |
| issac10-ex3f      |  0.437 |   0.512 |   0.447 |   5.500 |      5.571 |       5.546 |
| issac10-ex4h      |  0.110 |   0.109 |   0.046 |   0.046 |      0.048 |       0.046 |
| knopp-schur-b     |  0.122 |   0.123 |   0.049 |   0.049 |      0.044 |       0.044 |
| levin-rhsc        |  0.137 |   0.139 |   0.138 |   0.284 |      0.292 |       0.293 |
| mitrinovic-3-29   |  0.114 |   0.115 |   0.019 |   0.017 |      0.017 |       0.018 |
| mitrinovic-4-14   |  0.122 |   0.125 |   0.016 |   0.013 |      0.014 |       0.014 |
| sumcracker-14b    |  0.303 |   0.304 |   0.106 |   2.120 |      2.141 |       2.140 |
| sumcracker-14c    |  1.529 |   1.702 |   0.542 |   4.096 |      3.905 |       3.884 |
| sumcracker-14d    | 17.445 |  22.753 |   2.787 |   9.657 |      9.572 |       9.950 |

The inputs cover program invariants, polynomial proof obligations,
geometric constructions, and economic models. The Pine, GeoGebra, and
economics inputs were selected for their coupled nonlinear equations and
future Gröbner simplification benchmarks.

In polypaver-0326, the intermediate strict DNFs differ: FINAL and AFTER_STEP
retain seven cubes with 48 atoms after syntactic reduction, while DURING_STEP
retains three cubes with 20 atoms. Final semantic subsumption yields two
cubes with 14 atoms under each option. Selection criteria and relationships
are documented in [NOTES.md](NOTES.md).

The upper tables cover computation times from hundredths of a second to
556.550 seconds. Selection aims to cover a range of costs.

The three simplifying options agree on final atom counts for these inputs.
Their costs can differ: DURING_STEP takes 12.694 seconds on ppda-mod7,
compared with 9.343 for FINAL. The inputs include up to 708 input atoms;
the MBO and MGC cases instead contribute polynomials of degrees 20 and 23.

## Inputs with timeouts or errors

The remaining ten inputs have at least one timeout or error. Their 60
outcomes comprise 39 successful calls, 17 computation timeouts, and four
errors. All were recomputed in the 600-second run described above.

Input and result atom counts for inputs with timeouts or errors:

| case             | input | RL off | RL on |  NONE | FINAL | AFTER_STEP | DURING_STEP |
| ---------------- | ----: | -----: | ----: | ----: | ----: | ---------: | ----------: |
| all-diff2        |   594 |  error | error |   t/o |   t/o |        t/o |         t/o |
| compare-modified |   189 |  31644 |   t/o |  1204 |  1204 |       1204 |        1204 |
| ddlm2013         |   336 |  80896 |   t/o | 74752 |   t/o |        t/o |         t/o |
| gj2007           |   253 | 119040 |   t/o | 14144 | 14144 |      14144 |       14144 |
| gr2006           |   219 |    t/o | error | 12352 | 12352 |      12352 |       12352 |
| nested-i         |   245 | 374852 |   t/o | 14400 | 14400 |      14400 |       14400 |
| password         |   189 |  31644 |   t/o |  1204 |  1204 |       1204 |        1204 |
| running          |   211 |    t/o |   t/o | 97024 | 97024 |      97024 |       97024 |
| threadpooling    |   147 |    t/o |   t/o | 51584 | 51584 |      51584 |       51584 |
| sumcracker-14e   |    54 |  34816 | error | 34816 | 34816 |      34816 |       34816 |

Corresponding computation times, in seconds:

| case             |  RL off | RL on |    NONE |   FINAL | AFTER_STEP | DURING_STEP |
| ---------------- | ------: | ----: | ------: | ------: | ---------: | ----------: |
| all-diff2        |   error | error |    600* |    600* |       600* |        600* |
| compare-modified |  60.330 |  600* |   3.597 |  10.371 |     10.403 |      10.500 |
| ddlm2013         | 121.870 |  600* | 274.132 |    600* |       600* |        600* |
| gj2007           | 152.795 |  600* |  53.076 | 355.375 |    353.881 |     362.280 |
| gr2006           |    600* | error |  18.987 | 176.017 |    174.090 |     176.291 |
| nested-i         | 411.935 |  600* |  28.998 | 243.206 |    243.346 |     248.058 |
| password         |  60.205 |  600* |   3.612 |  10.638 |     10.610 |      10.512 |
| running          |    600* |  600* |  46.325 | 190.870 |    191.661 |     201.762 |
| threadpooling    |    600* |  600* |  25.064 | 117.545 |    119.754 |     118.839 |
| sumcracker-14e   |  44.411 | error |  15.041 |  50.492 |     50.250 |      51.529 |

All three simplifying configurations completed on eight of these ten inputs,
with equal atom counts for each input. Across the original 16-input rerun,
all 36 previously successful calls returned the same atom counts;
39 previous timeouts now have results.

The four errors occurred in Redlog: insufficient memory for all-diff2 with
both settings, stack overflow for gr2006 with Redlog on, and SIGSEGV for
sumcracker-14e with Redlog on. [NOTES.md](NOTES.md) preserves the previous
measurements, diagnostics, and computation protocol.
Case-specific investigations and notes for resuming them are collected in
[reduce-problems](../../../../doc/notes/RCF/reduce-problems/).

## Informational PyEDA results

Results from a separate run on 2026-10-21, using
`method="pyeda"` with default options and a 30-second computation limit.
These are not cases in either pytest file. PyEDA was not
benchmarked again. Each measurement excludes startup, parsing, and counting:

| case            | input | result atoms | seconds |
| --------------- | ----: | -----------: | ------: |
| mitrinovic-4-14 |    16 |            8 |   0.015 |
| mitrinovic-3-29 |    20 |            7 |   0.017 |
| knopp-schur-b   |    29 |            9 |   0.052 |
| half-2          |    81 |        28520 |  12.573 |
| issac10-ex4h    |    39 |           10 |   0.045 |
| levin-rhsc      |    46 |           88 |   0.137 |
| sumcracker-14d  |    44 |         7168 |   2.840 |
| sumcracker-14e  |    54 |        34816 |  15.338 |

## Implicit ranges on levin-rhsc

Additional levin-rhsc comparison using `Simplification.DURING_STEP`,
measured on 2026-10-22 with one call per setting and a 30-second limit.
Only initial `implicit_ranges` is changed; semantic implication checks
and final simplification retain their defaults. The current limit is
60 seconds. See [Reproducing the results](#reproducing-the-results) for the
command selecting both settings.

| initial implicit_ranges | result atoms | seconds |
| ----------------------- | -----------: | ------: |
| True (default)          |           88 |   0.293 |
| False                   |           16 |   0.203 |

In levin-rhsc, enabling implicit ranges reduces the initial formula but
collapses explicit variable sign cases. The current simplifier cannot
recover those signs from product constraints, leaving six contradictory
cubes in the final DNF. Disabling the option retains the cases and enables
their elimination, despite a larger intermediate DNF. This is the only
observed case so far where enabling implicit ranges increases the final
atom count.

Equivalence was checked by splitting each cube of the default result on
`V2 < 0` and `V2 > 0`. Every cube contains `V2 != 0`, so this split
is exhaustive. Simplifying the resulting disjunction produced a formula
structurally identical to the result with the option disabled.

## Pair-check budgets on sumcracker-14e

A separate sumcracker-14e benchmark doubles `max_pair_checks` from 1
for DURING_STEP and AFTER_STEP, with initial `implicit_ranges=True`.
Zero-budget controls disable only final semantic comparisons. The limit
is 60 seconds. Commands for independent sweeps and zero-budget controls
are given under [Reproducing the results](#reproducing-the-results).

Selected measurements from 2026-10-21, retained without recomputation.
The former labels early and subtree now mean DURING_STEP and AFTER_STEP:

| max_pair_checks | DURING_STEP atoms | DURING_STEP seconds | AFTER_STEP atoms | AFTER_STEP seconds |
| --------------: | ----------------: | ------------------: | ---------------: | -----------------: |
|               0 |             34816 |              44.241 |            34816 |             41.600 |
|               1 |             34816 |              43.890 |            34816 |             41.475 |
|               2 |             34816 |              43.357 |            34816 |             41.321 |
|             256 |             34816 |              45.836 |            34816 |             45.049 |
|             512 |             34816 |              49.177 |            34816 |             48.449 |
|            1024 |             34816 |              58.292 |            34816 |             56.887 |
|            2048 |           unknown |                 60* |          unknown |                60* |

Both doubling sweeps first timed out at budget 2048. Every completed run
returned 34,816 atom occurrences, matching the result with
`Simplification.NONE` and the zero-budget controls. Increasing the semantic
pair budget gave no reduction in the final atom count for this input.
