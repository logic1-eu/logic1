# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-09-14

### Added

- Add `py.typed` marker and adapt `pyproject.toml` accordingly.

- Use `setuptools_scm` more systematically for versioning in code and documentation.

### Fixed

#### module `logic1`

- Runtime and documentation version metadata was stale and malformed in `__init__.py`.

#### module `abc`

- Parallel QE could wait forever after an unexpected worker exception.

- `qe` more systematically recognizes and handles contradictory `Assumptions` now.

- `qe.QuantifierElimination` classified an entire successor
   batch using only its first node.

- Reloading the `qe` module duplicated global logging handlers.

- The working node classes of `qe` explicitly require positive NNF now.

- In `qe.QuantifierElimination`, fix formatting and details in `nodes_as_str` and `timings`.

- The mutable `qe.Assumptions` is not hashable anymore.

- Normal form computation in `bnf` could fail on formulas containing `T` or `F`.

- `bnf` had a leak not resetting renaming tables for abstracting atoms.

- `parser` now implements `<<` as converse implications and catches more exceptions.

#### module `firstorder`

- `Formula.simplify` could deliver wrong results with `T` in `Implies`. It was non-idempotent for `Equivalent(F, F)`, could reintroduce duplicates while flattening, and never sorted although this was documented.

- `Formula.traverse` crashed on `Not` and `Implies`.

- `Formula.count_alternations` falsely returned `-1` alternations for quantifier-free formulas.

- `Formula._repr_latex_` could cut through a LaTeX control word.

- `Formula.subs` could unnecessarily rename variables.

- An assertion in `Formula.to_pnf` rejected non-positive NNF with `is_nnf=True`.

- Instances of `Formula` are not mutable anymore.

- In `QuantifiedFormula.__le__`, sorting crashed for both `Complex` and `Sets`.

- Prevent instantiation of `QuantifiedFormula`.

#### module `Complex`

- `Term.set_normal_form` mutated hashes of existing terms.

- `normalize.WeakNormalizer` had issues with normal form computation for signed factors and nested exponentiation.

- `Term.__repr__` and `AtomicFormula.__repr__` both had issues with constants.

- `simplify.min_weight_partial_edge_cover` did not consider multi-edges.

- `qe.qe` silently discarded options when calling `redlog.qe`.

- `VariableSet.__getitem__` was too liberal. We admit only Python identifiers as variables now.

#### module `RCF`

- Flaws in `range._Range._imul_core` affected the correctness of `simplify`.

- `term_sage.Term.factor` crashed on nonzero constants.

- `term_sage._PolynomialRing` had issues sorting variables.

- `term_sage._PolynomialRing.MPolynomialRing_factory` could cause an overflow in Singular when allocating too many variables.

- `term_sage.Variable.__init__` could be called directly, instead of using `VV`.

- `term_sage` had inconsistencies about admissible number types.

- `term_flint.Term.degree` could yield degree 0 instead of -1 for the zero polynomial.

- `term_flint.Term.derivative` accepted negative derivative orders. Furthermore, it could return 0 instead of the input for derivative order 0.

- `term_flint.Term.factor` did not reconstruct repeated non-monic factors.

- `term_flint.Term.reduce` consumed generators twice.

- `term_flint._caches` contained a non-existent class.

- `vs._TestPoint._translate` had misplaced assertions.

#### module `Sets`

- `simplify.InternalRepresentation` erased the finite-cardinality constraint `C_(oo)`. Furthermore, its `extract` method had duplicate code lines.

- `atomic` and `simplify` incorrectly assumed `float('inf')` to be a singleton.

- `C.__new__` and `C_.__new__` admitted Boolean values as arguments.

- The singleton `atomic.VariableSet` is implemented more robustly now.

- `qe.Node.copy` did not copy its mutable variable list.

#### module `support`

- Reloading `support.excepthook` made ordinary exception handling recurse forever.

- An empty `NoTraceException` caused a secondary `IndexError`.

- `logging.DeltaTimeFormatter` corrupted exact whole-second durations.

- The indentation in the `trace` decorator was flawed.

#### Development/Build

- Revise and simplify `Makefile`.

- Add `pytest` support for ignoring inactive `RCF` term backend.

- In `pyproject.toml`, declare runtime dependencies `Sage`, `NetworkX`, `typing-extensions`, add optional backend `python-flint`, and remove `more-itertools`.

- Add dependency `clang` to `logic1_dev.yaml`.

#### Documentation

- Fix numerous user-facing documentation typos and broken references.

- Fix various issues in doctests.

### Removed

- Remove file `cython.yaml`.

- Comment unused variable `qe.QuantifierElimination.time_import_success_nodes`.

## [0.3.0] - 2026-08-18

### Added

#### new module `interactive`

- Loaders for Complex, RCF and Sets for interactive use, which do the following:
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
