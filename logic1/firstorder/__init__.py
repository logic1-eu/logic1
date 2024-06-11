r"""Implementation of first-order formulas over arbitrary signatures
(languages) and theories.

Throughout this documentation we use the term *theory* to refer to a choice of
function and relation symbols along with their arities (signature) plus a
choice of a semantics of those symbols.

An abstract base class :class:`Formula` implements representations of and
methods on first-order formulas recursively built using first-order operators:

1. Boolean operators:

   a. Truth values :math:`\top` and :math:`\bot`

   b. Negation :math:`\lnot`

   c. Conjunction :math:`\land` and discjunction :math:`\lor`

   d. Implication :math:`\longrightarrow`

   e. Bi-implication (syntactic equivalence) :math:`\longleftrightarrow`

2. Quantifiers :math:`\exists x` and :math:`\forall x`, where :math:`x` is a
   variable.

As an abstract base class, :class:`Formula` cannot be instantiated.
Nevertheless, it implements a number of methods on first-order formulas. Those
methods are typically syntactic in the sense that they do not need to know the
semantics of the underlying theories.

Boolean operators are implemented as classes derived from another abstract
class :class:`BooleanFormula` which is, in turn, derived from :class:`Formula`.
Operators are mapped to classes as follows:

+--------------+--------------+---------------+---------------+--------------+-------------------------+-----------------------------+
| :math:`\top` | :math:`\bot` | :math:`\lnot` | :math:`\land` | :math:`\lor` | :math:`\longrightarrow` | :math:`\longleftrightarrow` |
+--------------+--------------+---------------+---------------+--------------+-------------------------+-----------------------------+
| :class:`_T`  | :class:`_F`  | :class:`Not`  | :class:`And`  | :class:`Or`  | :class:`Implies`        | :class:`Equivalent`         |
+--------------+--------------+---------------+---------------+--------------+-------------------------+-----------------------------+

The truth values :math:`\top` and :math:`\bot` are operators of arity 0. As
such, they are implemented as singleton classes :class:`_T` and :class:`_F`
with unique instances :data:`T` and :data:`F`, respectively.

>>> T is _T()
True

For details on the arities of the other Boolean operators see the documentation
of the corresponding classes. On these grounds we can construct formulas based
on truth values as follows:

>>> And(Implies(F, T), Or(T, Not(T)))
And(Implies(F, T), Or(T, Not(T)))

More interesting formulas require atomic formulas as another basis of the
recursive construction. This is implemented via another abstract subclass
:class:`AtomicFormula` of :class:`Formula`, which provides an interface to
various theories via further subclassing. For some of its methods,
:class:`AtomicFormula` already provides implementations, which delegate
theory-specific parts to an abstract class :class:`Term` for argument terms
of relations of atomic formulas.

We give an example using the theory RCF of Real Closed Fields, which is
implemented outside :mod:`logic1.firstorder`. RCF is the first-order theory of
the real numbers using the signature of ordered rings:

>>> from logic1.theories import RCF
>>> # Assign RCF variables to Python identifiers:
>>> a, b, x = RCF.VV.get('a', 'b', 'x')
>>> # Construct formula using first-order operators along with RCF variables,
>>> # functions and relations:
>>> And(x >= 0, a*x + b == 0)
And(x >= 0, a*x + b == 0)

After learning about the introduction of variables in theories, we can finally
discuss first-order quantifiers and corresponding quantified formulas.
Quantifiers are implemented as classes derived from another abstract class
:class:`QuantifiedFormula` which is, in turn, derived from :class:`Formula`.
Quantifiers are mapped to classes as follows:

+-----------------+-----------------+
| :math:`\exists` | :math:`\forall` |
+-----------------+-----------------+
| :class:`Ex`     | :class:`All`    |
+-----------------+-----------------+

For details on the arguments of quantifiers see the documentation of the
corresponding classes. Here is an example:

>>> from logic1.theories import RCF
>>> a, b, x = RCF.VV.get('a', 'b', 'x')
>>> f = Ex(x, And(x >= 0, a*x + b == 0))
>>> f
Ex(x, And(x >= 0, a*x + b == 0))

Implementations of theories provide semantically meaningful methods and
functions on first-order formulas. For instance, quantifier elimination (QE)
computes equivalent formulas that do not contain quantifiers anymore:

>>> RCF.qe(f)
Or(And(b == 0, a == 0), And(a != 0, a*b <= 0))
"""  # noqa

from .formula import Formula  # noqa

from .atomic import AtomicFormula, Term, Variable  # noqa

from .boolean import BooleanFormula, Equivalent, Implies, And, Or, Not, _T, T, _F, F  # noqa

from .quantified import QuantifiedFormula, Ex, All  # noqa

from .pnf import pnf  # noqa


__all__ = [
    'Ex', 'All',

    'Equivalent', 'Implies', 'And', 'Or', 'Not', 'T', 'F',

    'pnf'
]
