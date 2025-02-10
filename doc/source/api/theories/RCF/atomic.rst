.. _api-RCF-atomic:

*Real Closed Fields*

***********************
Variables, Terms, Atoms
***********************

.. automodule:: logic1.theories.RCF.atomic

  .. autoclass:: VariableSet
    :special-members:

    .. automethod:: __getitem__

    .. automethod:: fresh

    .. method::
      pop() -> None
      push() -> None
      :abstractmethod:

      Implement abstract methods
      :meth:`.logic1.firstorder.atomic.VariableSet.pop` and
      :meth:`.logic1.firstorder.atomic.VariableSet.push`.


  .. data:: VV
    :value: VariableSet()

    The unique instance of :class:`.VariableSet`.


  .. autoclass:: DEFINITE
    :members:


  .. autoclass:: Term
    :special-members:

    .. method:: +, *, -, **, /
                __add__(other: object) -> Term
                __mul__(other: object) -> Term
                __neg__() -> Term
                __pow__(other: object) -> Term
                __radd__(other: object) -> Term
                __rmul__(other: object) -> Term
                __rsub__(other: object) -> Term
                __sub__(other: object) -> Term
                __truediv__(other: object) -> Term

      Arithmetic operations on Terms are available as overloaded operators.

    .. method:: ==, >=, >, <=, <, !=
                __eq__(other: Term | int) -> Eq
                __ge__(other: Term | int) -> Ge | Le
                __gt__(other: Term | int) -> Gt | Lt
                __le__(other: Term | int) -> Ge | Le
                __lt__(other: Term | int) -> Gt | Lt
                __ne__(other: Term | int) -> Ne

      Construction of instances of :class:`Eq`, :class:`Ge`, :class:`Gt`,
      :class:`Le`, :class:`Lt`, :class:`Ne` is available via overloaded
      operators.

    .. automethod:: __iter__

    .. automethod:: as_latex

    .. automethod:: coefficient

    .. automethod:: constant_coefficient

    .. automethod:: content

    .. automethod:: degree

    .. automethod:: derivative

    .. automethod:: factor

    .. automethod:: is_constant

    .. automethod:: is_definite

    .. automethod:: is_zero

    .. automethod:: lc

    .. automethod:: monomials

    .. automethod:: quo_rem

    .. automethod:: pseudo_quo_rem

    .. automethod:: sort_key

    .. automethod:: subs

    .. automethod:: vars


  .. autoclass:: Variable
    :special-members:

    .. automethod:: fresh


  .. autoclass:: AtomicFormula
    :special-members:

    .. property:: lhs
                  rhs
      :type: Term

      The left hand side term and the right hand side term of an atomic formula.

    .. automethod:: __bool__

    .. automethod:: __le__

    .. automethod:: __str__

    .. automethod:: as_latex

    .. automethod:: bvars

    .. method:: complement(cls) -> type[AtomicFormula]
                converse(cls) -> type[AtomicFormula]
                dual(cls) -> type[AtomicFormula]
      :classmethod:

      Complement relation, converse relation, and dual relation.
      :meth:`complement` implements the abstract method
      :meth:`.firstorder.atomic.AtomicFormula.complement`.

      +--------------------+-------------+-------------+-------------+-------------+-------------+-------------+
      |                    | :class:`Eq` | :class:`Ne` | :class:`Le` | :class:`Ge` | :class:`Lt` | :class:`Gt` |
      +====================+=============+=============+=============+=============+=============+=============+
      | :meth:`complement` | :class:`Ne` | :class:`Eq` | :class:`Gt` | :class:`Lt` | :class:`Ge` | :class:`Le` |
      +--------------------+-------------+-------------+-------------+-------------+-------------+-------------+
      | :meth:`converse`   | :class:`Eq` | :class:`Ne` | :class:`Ge` | :class:`Le` | :class:`Gt` | :class:`Lt` |
      +--------------------+-------------+-------------+-------------+-------------+-------------+-------------+
      | :meth:`dual`       | :class:`Ne` | :class:`Eq` | :class:`Lt` | :class:`Gt` | :class:`Le` | :class:`Ge` |
      +--------------------+-------------+-------------+-------------+-------------+-------------+-------------+

      .. admonition:: Mathematical definitions

        Let :math:`\varrho \subseteq A^n` be an  :math:`n`-ary relation. Then
        the *complement relation* is defined as

        .. math::
          \overline{\varrho} = A^n \setminus \varrho.

        It follows that :math:`\overline{\varrho}(a_1, \dots, a_n)` is equivalent to
        :math:`\lnot
        \varrho(a_1, \dots, a_n)`, which is an important property for Logic1.

        If :math:`\varrho` is binary, then the *converse relation* is defined as

        .. math::
          \varrho^{-1} = \{\,(y, x) \in A^2 \mid (x, y) \in \varrho\,\}.

        In other words, the converse swaps sides. It is the inverse with respect to
        composition, i.e., :math:`\varrho \circ \varrho^{-1} = \varrho^{-1}
        \circ \varrho = \Delta_A`. The diagonal :math:`\Delta_A = \{\,(x, y) \in A^2
        \mid x = y\,\}` is equality on :math:`A`.

        Finally, the *dual relation* is defined as

        .. math::
          \varrho^d = \overline{\varrho^{-1}},

        which generally equals :math:`(\overline{\varrho})^{-1}`. For our relations
        here, dualization amounts to turning strict relations into weak relations, and
        vice versa.

        Each of these transformations of relations is involutive in the sense that
        :math:`\overline{\overline{\varrho}} = (\varrho^{-1})^{-1} = (\varrho^d)^d =
        \varrho`.

      .. seealso::
        Inherited method :meth:`.firstorder.atomic.AtomicFormula.to_complement`

    .. automethod:: fvars

    .. automethod:: simplify

    .. automethod:: strict_part

      +---------------------+-------------+-------------+-------------+-------------+
      |                     | :class:`Le` | :class:`Ge` | :class:`Lt` | :class:`Gt` |
      +=====================+=============+=============+=============+=============+
      | :meth:`strict_part` | :class:`Lt` | :class:`Gt` | :class:`Lt` | :class:`Gt` |
      +---------------------+-------------+-------------+-------------+-------------+

    .. automethod:: subs


  .. class:: Eq
             Ge
             Gt
             Le
             Lt
             Ne

    Bases: :class:`.AtomicFormula`

