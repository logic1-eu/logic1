.. _api-RCF-bnf:

*Real Closed Fields*

********************
Boolean Normal Forms
********************

.. automodule:: logic1.theories.RCF.bnf

  .. autoclass:: Simplification
    :members:
    :exclude-members: __new__

  .. autofunction:: cnf

  .. autofunction:: dnf

  .. rubric:: Syntactic reduction followed by semantic subsumption

  The formula ``f`` below is a DNF with four cubes and 11 atom occurrences.
  Preprocessing leaves it unchanged:

  >>> from logic1.firstorder import And, Or
  >>> from logic1.theories.RCF import Simplification, VV, dnf
  >>> from logic1.theories.RCF.bnf import FinalCartesianBnf
  >>> x, y, z = VV.get('x', 'y', 'z')
  >>> f = Or(And(z <= 0, y - 1 > 0, x > 0),
  ...        And(z > 0, y - 1 > 0, x > 0),
  ...        And(z > 0, y > 0, x > 0),
  ...        And(y > 0, x > 0))
  >>> final_cartesian_bnf = FinalCartesianBnf(max_pair_checks=500)
  >>> final_cartesian_bnf.simplify(f) == f
  True

  With :attr:`Simplification.FINAL`, syntactic reduction applies both
  structural subsumption and cut. The cube for ``And(y > 0, x > 0)``
  structurally subsumes the one for ``And(z > 0, y > 0, x > 0)``. The other
  two cubes cut to ``And(y - 1 > 0, x > 0)`` because ``z <= 0`` and ``z > 0``
  are complementary. Two cubes with four atom occurrences remain.

  Semantic subsumption then uses an implication check to establish that
  ``And(y - 1 > 0, x > 0)`` implies ``And(y > 0, x > 0)``, since
  ``y - 1 > 0`` implies ``y > 0``. The former cube is therefore removed.
  One cube with two atoms remains:

  >>> dnf(f, simplification=Simplification.FINAL)
  And(y > 0, x > 0)

  .. rubric:: Illustration of different simplification schedules (1)

  Different values of ``simplification`` can produce different DNFs. The
  formula ``f`` defined below has seven outer disjuncts and 19 atoms. It is not
  in DNF. Preprocessing using the RCF simplifier leaves ``f`` unchanged:

  >>> from logic1.firstorder import And, Or
  >>> from logic1.theories.RCF import Simplification, VV, dnf
  >>> from logic1.theories.RCF.bnf import CartesianBnf
  >>> a, b, c, d = VV.get('a', 'b', 'c', 'd')
  >>> f = Or(And(d > 0, b <= 0),
  ...        And(d > 0, b > 0, a <= 0),
  ...        And(d > 0, b > 0, a > 0),
  ...        And(d > 0, a > 0),
  ...        And(c > 0, b <= 0),
  ...        And(c > 0, a > 0),
  ...        And(c > 0, Or(And(b > 0, a <= 0), And(b > 0, a > 0))))
  >>> cartesian_bnf = CartesianBnf(max_pair_checks=0)
  >>> cartesian_bnf.simplify(f) == f
  True

  With :attr:`Simplification.NONE` distributing the nested disjunction if ``f``
  with neither syntactic reduction nor semantic subsumption, we obtain eight
  cubes with 20 atoms:

  >>> dnf(f, simplification=Simplification.NONE)
  Or(And(d > 0, b <= 0),
     And(d > 0, b > 0, a <= 0),
     And(d > 0, b > 0, a > 0),
     And(d > 0, a > 0),
     And(c > 0, b <= 0),
     And(c > 0, b > 0, a <= 0),
     And(c > 0, b > 0, a > 0),
     And(c > 0, a > 0))

  With :attr:`Simplification.FINAL`, shorter cubes subsume
  ``And(c > 0, b > 0, a > 0)`` and ``And(d > 0, b > 0, a > 0)`` before
  either can merge with its counterpart containing ``a <= 0``. Six cubes
  remain, containing 14 atom occurrences:

  >>> dnf(f, simplification=Simplification.FINAL)
  Or(And(d > 0, b <= 0),
     And(d > 0, b > 0, a <= 0),
     And(d > 0, a > 0),
     And(c > 0, b <= 0),
     And(c > 0, b > 0, a <= 0),
     And(c > 0, a > 0))

  With :attr:`Simplification.AFTER_STEP`, the nested disjunction first reduces
  to ``b > 0``. The resulting ``And(c > 0, b > 0)`` then merges with
  ``And(c > 0, b <= 0)`` to give ``c > 0``. Four cubes remain, containing
  eight atom occurrences:

  >>> dnf(f, simplification=Simplification.AFTER_STEP)
  Or(c > 0,
     And(d > 0, b <= 0),
     And(d > 0, b > 0, a <= 0),
     And(d > 0, a > 0))

  With :attr:`Simplification.DURING_STEP`, the two three-atom cubes containing
  ``d > 0`` merge before the shorter ``And(d > 0, a > 0)`` is inserted.
  A further merge with ``And(d > 0, b <= 0)`` gives ``d > 0``. The result
  contains two cubes and two atom occurrences:

  >>> dnf(f, simplification=Simplification.DURING_STEP)
  Or(d > 0, c > 0)

  .. rubric:: Illustration of different simplification schedules (2)

  A higher simplification index need not give a smaller DNF. The formula
  ``f`` defined below has five outer disjuncts and 19 atoms. It is not in DNF.
  Preprocessing using the RCF simplifier leaves ``f`` unchanged:

  >>> from logic1.firstorder import And, Or
  >>> from logic1.theories.RCF import Simplification, VV, dnf
  >>> from logic1.theories.RCF.bnf import CartesianBnf
  >>> a, b, c, d, e = VV.get('a', 'b', 'c', 'd', 'e')
  >>> f = Or(And(e <= 0, d > 0, c <= 0, a <= 0),
  ...        And(e <= 0, d > 0, c <= 0, a > 0),
  ...        And(d > 0, a <= 0),
  ...        And(c <= 0, a > 0),
  ...        And(b <= 0, Or(And(d > 0, c <= 0, a <= 0),
  ...                       And(d > 0, c <= 0, a > 0))))
  >>> cartesian_bnf = CartesianBnf(max_pair_checks=0)
  >>> cartesian_bnf.simplify(f) == f
  True

  With :attr:`Simplification.NONE`, distribution expands the nested
  disjunction without syntactic reduction or semantic subsumption. The result
  contains six cubes and 20 atoms:

  >>> dnf(f, simplification=Simplification.NONE)
  Or(And(e <= 0, d > 0, c <= 0, a <= 0),
     And(e <= 0, d > 0, c <= 0, a > 0),
     And(d > 0, c <= 0, b <= 0, a <= 0),
     And(d > 0, c <= 0, b <= 0, a > 0),
     And(d > 0, a <= 0),
     And(c <= 0, a > 0))

  With :attr:`Simplification.FINAL`, the cubes for
  ``And(d > 0, a <= 0)`` and ``And(c <= 0, a > 0)`` subsume all four
  cubes with four atoms. Two cubes remain, containing four atom occurrences:

  >>> dnf(f, simplification=Simplification.FINAL)
  Or(And(d > 0, a <= 0), And(c <= 0, a > 0))

  With :attr:`Simplification.AFTER_STEP`, the nested disjunction first reduces
  to ``And(d > 0, c <= 0)``. The enclosing conjunction therefore produces
  ``And(d > 0, c <= 0, b <= 0)``. The two outer four-atom cubes are subsumed,
  but neither two-atom cube subsumes the merged cube. Three cubes remain,
  containing seven atom occurrences:

  >>> dnf(f, simplification=Simplification.AFTER_STEP)
  Or(And(d > 0, c <= 0, b <= 0),
     And(d > 0, a <= 0),
     And(c <= 0, a > 0))

  With :attr:`Simplification.DURING_STEP`, the first two outer disjuncts merge
  to ``And(e <= 0, d > 0, c <= 0)`` before the two-atom cubes are inserted.
  Neither two-atom cube subsumes this merged cube, which is retained in
  addition to the three cubes obtained with :attr:`Simplification.AFTER_STEP`.
  The result contains four cubes and ten atom occurrences:

  >>> dnf(f, simplification=Simplification.DURING_STEP)
  Or(And(e <= 0, d > 0, c <= 0),
     And(d > 0, c <= 0, b <= 0),
     And(d > 0, a <= 0),
     And(c <= 0, a > 0))


  .. rubric:: An example with quantifiers:

  >>> from logic1 import *
  >>> from logic1.theories.RCF import *
  >>> a, b, c = VV.get('a', 'b', 'c')
  >>> f = Equivalent(Ex([a, b], And(a == 0, b == 0)),
  ...                All(c, And(a == 0, c == 0)))
  >>> dnf(f)
  Ex(G0001_c, Ex(G0002_a, Ex(G0001_b, All(G0001_a, All(b, All(c,
     Or(And(c == 0, a == 0, G0002_a == 0, G0001_b == 0),
        And(c == 0, a == 0, G0001_c != 0),
        And(b != 0, a != 0),
        And(b != 0, G0002_a == 0, G0001_b == 0),
        And(b != 0, G0001_c != 0),
        And(a != 0, G0001_a != 0),
        And(G0002_a == 0, G0001_b == 0, G0001_a != 0),
        And(G0001_c != 0, G0001_a != 0))))))))
  >>> cnf(f)
  All(G0003_a, All(b, All(c, Ex(G0002_c, Ex(G0004_a, Ex(G0002_b,
      And(Or(c == 0, b != 0, G0003_a != 0),
          Or(b != 0, a == 0, G0003_a != 0),
          Or(a != 0, G0004_a == 0, G0002_c != 0),
          Or(a != 0, G0002_c != 0, G0002_b == 0))))))))


  Details
  *******

  .. attention::
    The material below addresses implementers rather than users.

  The Cartesian implementation has the following parts:

  - Class :class:`Cube`
    represents a conjunction of atoms and provides operations on such
    conjunctions, including simplification.
  - Class :class:`StrictDnf`
    represents a disjunction of cubes and provides operations on such
    disjunctions, including simplification.
  - Class :class:`CartesianBnf`
    controls recursive DNF construction.
    Extension points :meth:`~CartesianBnf.conjoin_dnfs`,
    :meth:`~CartesianBnf.disjoin_dnfs`, and
    :meth:`~CartesianBnf.finish_strict_dnf` allow customization of the DNF
    construction process, including DNF-specific simplification. The method
    :meth:`~CartesianBnf.finish_strict_dnf` is set up to perform semantic
    subsumption; the attribute :attr:`~CartesianBnf.max_pair_checks` controls
    the maximum number of cube pairs to check.
  - Subclasses :class:`FinalCartesianBnf`,
    :class:`AfterStepCartesianBnf`, and
    :class:`DuringStepCartesianBnf` of :class:`CartesianBnf` add
    syntactic reduction by overriding one or more of the extension points
    mentioned above.
  - Functions :func:`cnf` and :func:`dnf` with ``method="cartesian"`` create and
    delegate to an instance of :class:`CartesianBnf` or one of its subclasses.

    |
    | ``Cube``
    | ``StrictDnf``
    | ``CartesianBnf``
    | ``├── FinalCartesianBnf``
    | ``├── AfterStepCartesianBnf``
    | ``└── DuringStepCartesianBnf``
    | ``cnf``
    | ``dnf``
    |


  .. autodata:: AndOr
    :annotation: = NewType("AndOr", Formula)

  .. autofunction:: is_and_or

  .. autoclass:: Cube
    :members:
    :exclude-members: _atoms
    :special-members: __hash__, __init__, __iter__, __le__, __len__

    .. autoattribute:: Cube._atoms

  .. autoclass:: StrictDnf
    :members:
    :exclude-members: __slots__, _cubes
    :special-members: __contains__, __eq__, __init__, __iter__, __len__

    .. autoattribute:: StrictDnf.__slots__

    .. autoattribute:: StrictDnf._cubes

  .. autoclass:: CartesianBnf
    :members:
    :private-members: _dnf_from_and_or, _dnf_from_formula
    :exclude-members: max_pair_checks
    :special-members: __init__

    .. autoattribute:: CartesianBnf.max_pair_checks

  .. autoclass:: FinalCartesianBnf
    :members:
    :exclude-members: __init__

  .. autoclass:: AfterStepCartesianBnf
    :members:
    :exclude-members: __init__

  .. autoclass:: DuringStepCartesianBnf
    :members:
    :exclude-members: __init__

  .. autoclass:: PyedaBnf
    :members:
    :exclude-members: __init__
