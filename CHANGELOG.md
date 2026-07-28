# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-06-30

### Added

#### new module `interactive`

- A loader for RCF for interactive use, which does the following:
    1. `from logic1.firstorder import *`
    2. Define python symbols `a`, ..., `z` as corresponding variables

#### new module `Complex`

- A theory `Complex` for ring arithmtic along with `I`, `Re`, `Im`, `Conj` over the complex numbers. Quantifier elimination uses reduction to real QE. See [Faross-Sturm](https://doi.org/10.48550/arXiv.2604.26400) for theoretical details.

#### module `RCF`

- Options `xopt` and `elimination_order` for method `qe`. This activates optimized quantifier elimination by virtual substitution for weakly parametric linear formulas.

- Option `implicit_ranges` for method `simplify`. This improves the use nonlocal information during simplification.

- Option `assumptions` for `Term.is_definite`.

- Support for substituting rational functions into terms within atoms

- Support for input of rational coefficients as Python floats in class `Term`. New methods `Term.primitive_part`, `Term.subs_linear_solution`, `Term.summands`

- New method `AtomicFormula.is_weakly_parametric_linear`

- An experimental Flint-based implemenation of `Term` as an alternative to the existing Sage-based implementation. So far, this can be acivated only by editing `term/__init__.py`.

### Fixed

#### module `abc`

- Method `Simplify._simpl_and_or` did not terminate in rare cases.

- Raise an exception with assumptions on bound variables in `qe.QuantifierElimination.quantifier_elimination`.

#### module `RCF`

- Method `qe` wrongly computed `F` for `Ex([a, b], a != 0)`, where the problem was the quantification of an unused variable.

- Method `qe` raised an error when applied to `Ex(x, T)` or, more generally, quantifications of formulas equivalent to truth values modulo application of the method `simplify`.

- Add a method `Term.__setstate__` to prevent
cached hashes of Terms from being sent to processes with a different hash seed in parallel quantifier elimination.

- Arguments `prefer_order=True` and `prefer_weak=True` were missing in the root node simplification in method `qe.VirtualSubstitution.create_root_nodes`.

- Methods `cnf` and `dnf` could return equivalent formulas that were not in the respective normal form, because final simplification split atoms.

### Changed

#### module `abc`

- Attribute `qe.NodeList.memory: Set` is generic now, supporting arbitrary Hashables as set members.

- Instances of class `qe.Assumptions` are hashable now.

#### module `RCF`

- Improve method `simplify._Knowledge._term_as_range` to obtain more precise ranges with the new simplifier option `implicit_ranges` mentiomed above. The key idea is propagating bounds on variables via interval arithmetic.

- Migrate class `simplify._Range` to Cython. Add mutable arithmetic. Remove depency on class `mpfr`, and method `is_finite` in favor of inline code.

- Refactor class `qe.Node` into new additional module `node`, changing the return type of method `process` in class `Node` from `list` to `Sequence`. Add sublasses `node.vs.Node` and `node.xopt.Node` with corresponding helper classes.

- Move and rename subclasses `qe.CLUSTERING`, `qe.GENERIC` to `node.base.Clustering`, `node.base.Generic`, respectively.

- Add `@lru_cache` to methods `constant_coefficient`, `content`, `lc`, `normalize` in class `term.term_sage.Term`. Caching for `lc` required adapting implementation details of `Term.__eq__`.

- Refactor module `atomic` into `atomic` and new module `term`. Likewise, `firstorder.atomic`.

- More systematic choice of log levels in method `qe`.

#### module `support`

- support printing of `__str__` representation in `support.tracing`

#### Infrastructure

- Renamed all modules `*.typing` to `*.types` in order to avoid name clashes with Python standard library modules.

- Thorough revision of the main `Makefile`.

- Trigger GitHub worflows on push to main.

- Bumped deployment target from Python 3.11/Sage 10.0 to Python 3.12/Sage 10.6

### Removed

- Exception `RCF.qe.Failed`


## [0.2.0] - 2025-02-11

### Added

- Stubs to support MyPy for `gmpy`and `sage`.

#### module `RCF`

- New submodule `redlog` supports programmatic access to [Redlog](redlog) functions via process communication.

### Fixed

#### module `RCF`

- Improve issues around the encapsulation of the Sage polynomial ring in the `Term` class.

- Negative definite terms are recognized more reliably during simplification.

### Changed

#### License

- Relax license from `GPL-3.0` to `GPL-2.0-or-later`.

#### module `firstorder``

- `VV.imp` raises an exception when used outside the top-level of module `__main__`. Previously this was only an assertion violation.

#### module `abc`

- Refactor Submodule `simplify` and adapt corresponding theory modules.

- Restart loop in `simpl_and_or` when new substitutions occur, which further improves simplification.

#### module `RCF`

- Keep track of methods using `@lru_cache` and provide methods `cache_info` and `cache_clear` that do not require arguments.

- Class `Term` uses rational coefficients instead of integers now. `mpq`is expected and used for input and output of the coefficients.

- Submodule `simplify` uses `mpq` instead of Sage `Rational`. Generally, the module does not depend on Sage anymore. It uses monic polynomials over $\mathbb{Q}$ instead of primitive polynomials over $\mathbb{Z}$ during simplification. Final lifting to monic is optional but default.

- `simplify` supports substitution of linear binomial equations, based on a fixed order of variables.

- The definiteness tests in `simpl_at` have been reimplemented, replacing class `TSQ(Enum)` with class `DEFINITE(Enum)`.

- Refactor module `simplify`, moving class `_Subsitution` to its own module. A slightly more efficient Cython variant exists but is not used at present.


## [0.1.0] - 2024-10-29

### Added

#### module `firstorder`

- Abstract class `Formula` implements recursive representations of and methods for first-order formulas built from the operators `F`, `T`, `Not`, `And`, `Or`, `Implies`, `Equivalent`, `Ex`, and `All`.

- Prenex normal form computation is available as a method of `Formula`, but is implemented in the external module `pnf`.

#### module `theories`

- Collects submodules implementing various logical theories based on the `firstorder` module. At present, these include `RCF` (real closed fields) and `Sets` (with unary relation symbols for cardinality constraints).

- `RCF` implements terms as Sage polynomials with integer coefficients. Atoms are equations, disequalities, and inequalities based on corresponding dunder methods, which support infix notation.

- `RCF` quantifier elimination is based on the generic implementation in `abc`. It implements [Košta (2016)](https://doi.org/10.22028/D291-26679), limited to quantified variables of total degree 2.

- `RCF` simplification uses the generic implementation in `abc`, supplemented by deduction and substitution of constant variable values during recursion.

- Another submodule `parser` contains experimental code for parsing `RCF` formulas from strings. The parser uses a liberal but not rigorously specified syntax based on Python operators and keywords such as `"&"`, `"and"`, `"="`, `"=="`, etc. Parts of the code are generic within module `abc`.

- `Sets` uses only variables as terms, which are implemented as strings.

- `Sets` quantifier elimination uses a classical reduction approach, which is not elementary recursive.

- `Sets` simplification uses the generic implementation in `abc`. At the implicit-theory level, cardinality constraints are contracted using a union-find data structure.

#### module `abc`

- Provides CNF and DNF computation based on [PyEDA](https://pyeda.readthedocs.io).

- Submodule `qe` provides a generic implementation of first-order quantifier elimination. It reduces the problem to the elimination of a single prenex block of existential quantifiers, which is implemented in `RCF` and `Sets`, respectively.

- `qe` supports optional parallel computation based on Python's `multiprocessing` library. The number of workers can be passed as an argument.

- Submodule `simplify` provides a generic implementation of simplification based on implicit theories in the style of [Dolzmann–Sturm (1997)](https://doi.org/10.1006/jsco.1997.0123).

#### module `support`

- Submodule `excepthook` provides a class `NoTraceException` for concise interactive error reporting.

- Submodule `logging` provides support classes `DeltaTimeFormatter`, `RateFilter`, `Timer` for logging computation progress.

- Submodule `tracing` provides as decorator `@trace()` for logging information on entering end exiting of decorated functions to a specified stream, which is `sys.stdout` by default.