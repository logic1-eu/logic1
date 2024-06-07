# Logic1 &ndash; A Python package for interpreted first-order logic

Authors: Nicolas Faroß, Thomas Sturm

License: GPL-2.0-or-later. See the [LICENSE](LICENSE) file for details.

## About

This software is still a reasearch prototype, but you are welcome to have a look and follow us.

## Description
First-order logic recursively builds terms from variables and a specified
set of *function* symbols with specified arities, which includes *constant*
symbols with arity zero. Next, atomic formulas are built from terms and a
specified set of *relation* symbols with specified arities. Finally,
first-order formulas formulas are recursively built from atomic formulas and a
fixed set of logical operators.

Logic1 focuses on interpreted first-order logic, where the above-mentioned
function and relation symbols have an implicit semantics, which is not
explicitly expressed via axioms within the logical framework.

Typical applications include algebraic decision procedures and, more generally,
quantifier elimination procedures, e.g., over the real numbers.