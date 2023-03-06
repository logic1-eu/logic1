.. _api-index:

#############
API Reference
#############

.. toctree::
  :hidden:

  atomlib.rst
  firstorder.rst
  theories.rst
  abc.rst
  support.rst

:ref:`Package logic1.atomlib <api-atomlib>`
===========================================
A library of classes for atomic formulas with SymPy terms. Theories can select
a set classes from this collection, possibly modify via subclassing, and
implement additional atomic formulas on their own. In particular, subclassing
allows to restrict the set of admissible function symbols via argument checking
in the constructors and initializers of the derived classes.

:ref:`Package logic1.firstorder <api-firstorder>`
=================================================

:ref:`Package logic1.theories <api-theories>`
=============================================

:ref:`Package logic1.abc <api-abc>`
===================================

:ref:`Package logic1.support <api-support>`
===========================================
