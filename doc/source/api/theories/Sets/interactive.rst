.. _api-Sets-interactive:

*Sets with Cardinality Constraints*

***************
Interactive Use
***************

.. module:: logic1.interactive.Sets
   :synopsis: Interactive use of the theory of Sets

The following computation illustrates the use of the concepts introduced so far.
It imports all necessary symbols from various modules, defines variables, and
then prepares and performs the intended computation itself.

>>> from logic1.firstorder import Ex
>>> from logic1.theories.Sets import VV, qe
>>> x, y = VV.get('x', 'y')
>>> qe(Ex([x, y], x != y))
C(2)

In programmatic use of Logic1 as a library, it is desirable to have such an
explicit interface, where the code exactly specifies the symbols used and their
origin. This makes the overall program and its dependencies on external modules
easier to understand. Furthermore, it keeps the namespace clean.

In interactive use, in contrast, it is often desirable to have a more convenient
interface, where the relevant symbols are implicitly available. The module
:mod:`logic1.interactive.Sets` provides such an interface. It imports all
necessary symbols from the various modules and pre-defines lowercase
single-letter variables. The above computation can then conveniently be
performed as follows:

>>> from logic1.interactive.Sets import *
>>> qe(Ex([x, y], x != y))
C(2)