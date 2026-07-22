.. _api-Complex:

*******
Complex
*******

This module implements a theory of complex numbers following
`[Faross-Sturm] <https://doi.org/10.48550/arXiv.2604.26400>`_.

In this theory, terms are polynomial expressions build from rational numbers,
the imaginary unit :math:`i`, complex variables, complex conjugation, and real
and imaginary parts.

To construct :class:`Terms <.Complex.term.Term>`, one can first obtain
:class:`Variables <.Complex.term.Variable>` from the global variable set :data:`VV <.Complex.term.VV>`.
These can then be combined with the usual arithmetic operations, the imaginary unit
:data:`I <.Complex.term.I>`, and the functions :func:`Re <.Complex.term.Re>`, :func:`Im <.Complex.term.Im>`,
and :func:`Conj <.Complex.term.Conj>` to build larger terms.

>>> a, b = VV.get('a', 'b')
>>> 2 * a**2 * b - 1
2 * a**2 * b - 1
>>> (a + I)**2
a**2 + 2 * I * a - 1
>>> a - Conj(b)
a - ~b
>>> Re(a)**2 + Im(a)**2
a * ~a

Note that terms are represented by default in *conjugate normal form*, i.e.
as polynomials in complex variables and their conjugates. This can
be changed to *cartesian normal form* using the function
:meth:`Term.set_normal_form <.Complex.term.Term.set_normal_form>`.

>>> z = VV['z']
>>> z * Re(z)
1/2 * z**2 + 1/2 * z * ~z
>>> Term.set_normal_form(cartesian_normal_form)
>>> z * Re(z)
Re(z)**2 + I * Re(z) * Im(z)

:class:`Atomic formulas <.Complex.atomic.AtomicFormula>` in this theory
are given by equalities and inequalies between terms and can be constructed using
the usual comparison operators. However, inequalities are restricted
to the case where both sides are real terms.

>>> z = VV['z']
>>> z == 0
z == 0
>>> Re(z) < 0
1/2 * z + 1/2 * ~z < 0
>>> z >= 0
Traceback (most recent call last):
...
ValueError: Cannot create atomic formula z >= 0 because it is not real

Given a :class:`Formula <.firstorder.formula.Formula>` in this theory of complex
numbers, one can use the functions :func:`simplify <.Complex.simplify.simplify>` and
:func:`qe <.Complex.qe.qe>` to for simplification and quantifier elimination, respectively.

>>> z = VV['z']
>>> simplify(z**2 * Conj(z) == 0)
z == 0
>>> qe(Exists(z, z**2 + 1 == 0))
T

.. toctree::
   :hidden:

   atomic.rst
   simplify.rst
   qe.rst
   ast.rst
   types.rst