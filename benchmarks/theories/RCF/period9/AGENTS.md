# Period-9 Benchmark Agent Instructions

These instructions apply to this directory. Read `NOTES.md` for provenance,
input details, and completed verification.

## File responsibilities

- `period9_inputs.py` owns formula construction and input metadata.
- `test_qe_period9.py` owns the worker configurations, benchmark execution,
  and expected result.
- Keep input construction separate from benchmark execution so it is never
  included in the measured time.

## Input invariants

- Preserve the formula from
  `logic1/theories/RCF/test_qe_parallel.txt`, including operand order and
  logical tree structure.
- Quantify all free variables in the order produced by
  `sorted(set(period9.fvars()), key=Term.sort_key)`, namely `x11` through
  `x1`.
- The quantified input contains 38 atomic formulas and eliminates to `T`.
- Keep formula construction at module import time, outside the benchmark call.

## Benchmark invariants

- Benchmark `qe` with `workers=0`, `1`, `2`, `4`, and `8`, in that order.
- Keep all five cases in the `qe-period9` benchmark group so they appear in
  one comparison table.
- Use `benchmark.pedantic` with one round and one iteration. Each reported
  value must represent one complete quantifier-elimination call.
- Keep the benchmark function fully type-annotated and use
  `BenchmarkFixture` for the fixture.
- Verify both the 38-atom input and the expected result `T` outside the timed
  call.

## Validation after changes

- Compile both Python modules.
- Run mypy with `--disallow-untyped-defs` on both modules.
- Confirm pytest collects exactly five cases in worker order `0`, `1`, `2`,
  `4`, `8`.
- Run all five cases with benchmarking disabled.
- Run one complete `--benchmark-only` comparison when multiprocessing is
  available.

The corresponding commands are recorded in `NOTES.md`.
