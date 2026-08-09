.. _api-interactive:

***********
Interactive
***********

Module for automatically importing the first-order framework, theories, and
variables into the global namespace. This module is intended to be used in an
interactive Python session, e.g. in a Jupyter notebook.

>>> from logic1.interactive.Complex import *
>>> And(a * ~a >= 0, Re(z) == mpq(1, 2))
And(a * ~a >= 0, 1/2 * z + 1/2 * ~z == 1/2)

>>> from logic1.interactive.RCF import *
>>> And(a == mpq(1,2), y > 0)
And(a - 1/2 == 0, y > 0)

>>> from logic1.interactive.Sets import *
>>> And(a == b, z != y)
And(a == b, z != y)