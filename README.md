# Logic1 &ndash; Interpreted First-order Logic in Python
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/logic1/badges/version.svg?)](https://anaconda.org/conda-forge/logic1?)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/logic1/badges/latest_release_date.svg?)](https://anaconda.org/conda-forge/logic1?)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/logic1/badges/platforms.svg?)](https://anaconda.org/conda-forge/logic1?)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/logic1/badges/downloads.svg?)](https://anaconda.org/conda-forge/logic1?)

Authors: Nicolas Faroß, Lorenz Leutgeb, Thomas Sturm

License: GPL-2.0-or-later. See the [LICENSE](LICENSE) file for details.

Documentation: [docs.logic1.eu](https://docs.logic1.eu)

## About

This software is still at an early development stage. Nevertheless, you are very
welcome to use it already now. Any feedback is highly appreciated!

Logic1 can be installed via Conda from the conda-forge conda channel. You will
need a working Conda installation: either Miniforge, Mambaforge, Miniconda,
or Anaconda. Miniforge and Mambaforge use conda-forge as the default channel.
If you are using Miniconda or Anaconda, set it up to use conda-forge as follows:

```shell
$ conda config --add channels conda-forge
$ conda config --set channel_priority strict
```

Create and activate a new conda environment containing Logic1, either with mamba
or conda:

```shell
$ conda create -n logic1 logic1
$ conda activate logic1
```

## Description

First-order logic recursively builds terms from variables and a specified set of
function symbols with specified arities, which includes constant symbols with
arity zero. Next, atomic formulas are built from terms and a specified set of
relation symbols with specified arities. Finally, first-order formulas are
recursively built from atomic formulas and a fixed set of logical operators.

Logic1 focuses on interpreted first-order logic, where the above-mentioned
function and relation symbols have implicit semantics, which is not explicitly
expressed via axioms within the logical framework. Typical applications include
algebraic decision procedures and, more generally, quantifier elimination
procedures, e.g., over the real numbers.

## Examples

Consider the real numbers with arithmetic, equations, and inequality. From a
formal perpective, this is the theory of real closed fields (RCF). Logic1 allows
to formalize the question for the existence of solutions of a parametric
quadratic equation:

``` python
>>> from logic1 import *                # import Logic1
>>> from logic1.theories.RCF import *   # import RCF
>>> VV.imp('a', 'b', 'c', 'x')          # declare variables
>>> phi = Ex(x, a*x**2 + b*x + c == 0)  # formalization with existential quantifier
>>> qe(phi)                             # quantifier elimination
Or(And(c == 0, b == 0, a == 0), And(b != 0, a == 0), And(a != 0, 4*a*c - b^2 <= 0))

```

Consider the infinite real sequence defined by $x_{i+2} = |x_{i+1}| - x_{i}$.
Logic1 can check that this sequence has period 9 for all possible choices of
$x_1$, $x_2$. The final output T is a constant logical operator representing
"True":

``` python
>>> from logic1 import *
>>> from logic1.theories.RCF import *
>>> VV.imp(*(f'x{i}' for i in range(1, 12)))
>>> phi = And(Or(x2 >= 0, x3 == x2 - x1, x2 < 0, x3 == - x2 - x1),
...           Or(x3 >= 0, x4 == x3 - x2, x3 < 0, x4 == - x3 - x2),
...           Or(x4 >= 0, x5 == x4 - x3, x4 < 0, x5 == - x4 - x3),
...           Or(x5 >= 0, x6 == x5 - x4, x5 < 0, x6 == - x5 - x4),
...           Or(x6 >= 0, x7 == x6 - x5, x6 < 0, x7 == - x6 - x5),
...           Or(x7 >= 0, x8 == x7 - x6, x7 < 0, x8 == - x7 - x6),
...           Or(x8 >= 0, x9 == x8 - x7, x8 < 0, x9 == - x8 - x7),
...           Or(x9 >= 0, x10 == x9 - x8, x9 < 0, x10 == - x9 - x8),
...           Or(x10 >= 0, x11 == x10 - x9, x10 < 0, x11 == - x10 - x9))
>>> p9 = Implies(phi, And(x1 == x10, x2 == x11)).all()  # universal quantifiers for all variables
>>> qe(p9, workers=4)                                   # use four processors in parallel
T

```
