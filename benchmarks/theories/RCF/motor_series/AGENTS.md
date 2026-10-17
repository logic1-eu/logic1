# Motor Series Agent Instructions

These instructions apply to this directory. Read `NOTES.md` for provenance,
structural findings, atom counts, and completed verification.

## File responsibilities

- `motor_inputs.py` owns shared formula construction, case IDs, and input atom
  counts.
- `test_gsimplify_motor.py` owns only `gsimplify` options, expected output atom
  counts, and benchmark execution.
- `test_simplify_motor.py` owns only `simplify` options, expected output atom
  counts, and benchmark execution.
- `test_dnf_motor.py` owns the DNF configurations, measured output atom counts,
  and benchmark execution, including its per-computation timeout.
- Treat `motor_series_raw.py` as a raw reference; do not edit it unless the user
  explicitly requests that.
- Put benchmarks for other simplification functions in separate
  `test_<function>_motor.py` modules. Reuse `MOTOR_SERIES`, but keep each
  function's options and expected outputs in its own benchmark module.

## Benchmark matrices

- `test_gsimplify_motor.py` has two configurations, both with
  `use_redlog_cnf=True`: `radical=False` followed by `radical=True`.
- `test_simplify_motor.py` has the Cartesian product of `implicit_ranges` in
  `{False, True}` and `substitute` in `{0, 1, 2}`. `implicit_ranges` is the
  primary sort key, with `False` before `True`; `substitute` is secondary.
- `test_dnf_motor.py` has four configurations in this order: `redlog.dnf` with
  `bnfsac=False, bnfsm=False`; `redlog.dnf` with both options `True`;
  `RCF.bnf.dnf` with `method="cartesian"`; and `RCF.bnf.dnf` with
  `method="pyeda"`. Each runs on `series1` through `series13`.
- Give each configuration its own benchmark group so pytest-benchmark renders
  one 13-row table per configuration.
- Keep benchmark functions fully type-annotated. Use `BenchmarkFixture` for
  the fixture and `Formula` for inputs; do not disable mypy for the module.

## Structural invariants

- Preserve each original formula's exact tree, `repr`, argument order, and
  duplicates. Logical equivalence is insufficient.
- Original `testseries14` is AST-identical to `testseries1`; do not add a
  generated `series14`.
- Factor only exact repeated subtrees or contiguous argument sequences. Never
  distribute, reorder, deduplicate, or Boolean-simplify formulas.
- Keep `p1_split`, `q_td_split`, and `td_split` exact. Starred tuples must
  preserve flat argument sequences and their order.
- Construct formulas outside benchmark timing. All three benchmark modules use
  one round and one iteration, so each reported value represents one function
  call. The DNF benchmark also disables warmup and counts output atoms after
  the timed call.
- Keep each DNF computation in its own spawned worker with a 30-second wall
  time limit. A plain signal-based pytest timeout did not interrupt a
  long-running computation. The timed call includes a small pipe round trip;
  worker startup and result atom counting are outside it. Timeouts appear as
  pytest failures and have no output atom count.
- Keep every polynomial on one physical source line.
- For logical calls, do not break immediately after `And(` or `Or(`, and do not
  put a formula's closing parenthesis alone on a line.

## Intentional duplicates

Retain and locally comment the duplicate outer-`Or` operands found in the raw
formulas (positions are one-based):

- `testseries7`: 4/5 and 9/10
- `testseries8`: 1/3 and 6/10
- `testseries9`: 4/5 and 9/10

An exhaustive scan found no other duplicate sibling formulas, nested or
top-level, in `testseries1` through `testseries13`.

## Deferred Redlog equivalence validation

- A proposed explicit-request correctness check is documented in `NOTES.md`;
  do not implement it unless the user asks to revisit the design.
- If implemented, integrate it into the existing parameterized
  `test_simplify_motor` function and reuse the `result` returned by
  `benchmark.pedantic` for every option combination. Do not duplicate the
  matrix or recompute `simplify` in a separate test.
- Gate the Redlog check behind a pytest command-line option registered by a
  local `conftest.py`, and run it after the timed benchmark call. Normal pytest
  and benchmark invocations must not perform the check.

## Validation after structural changes

- Compare equality and exact `repr` with all 13 raw formulas.
- Verify the input atom counts recorded in `MOTOR_SERIES`.
- Compile the input and benchmark modules.
- Run mypy with `--disallow-untyped-defs` on all three benchmark modules.
- Confirm pytest collects exactly 26 `gsimplify` cases and 78 `simplify`
  cases, and 52 DNF cases.
- Run at least one complete six-configuration `simplify` series and both
  `gsimplify` radical settings with benchmarking disabled.
- Run all four DNF configurations on at least one series. The completed full
  DNF run and its seven observed 30-second timeouts are recorded in `NOTES.md`.
- After changing the motor formulas or DNF implementations, rerun the full DNF
  matrix and update both module-docstring tables and timeout observations.

Commands and the previously verified results are recorded in `NOTES.md`.
