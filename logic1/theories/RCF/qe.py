"""Real quantifier elimination by virtual substitution [Sturm-2018]_.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from logic1 import abc
from logic1.firstorder import _T
from logic1.theories.RCF import term
from logic1.theories.RCF.term import Term, Variable
from logic1.theories.RCF.atomic import AtomicFormula
from logic1.theories.RCF.node import Assumptions, Clustering, Generic, Node
from logic1.theories.RCF.simplify import simplify
from logic1.theories.RCF.types import Formula


@dataclass
class Options(abc.qe.Options):
    """Required by :class:`.VirtualSubstitution` for instantiating the type
    variable :data:`.abc.qe.ω` of :class:`.abc.qe.QuantifierElimination`.

    The options specified here, as well as the options ``log_level``,
    ``log_rate``, ``workers`` inherited from :class:`.abc.qe.Options`, can be
    passed to :func:`.qe` as keyword arguments.
    """

    clustering: Clustering
    """The clustering strategy to be used by :func:`.qe`. The default is
    :attr:`.Clustering.FULL`. For theoretical details on clustering see
    [Kosta-2016]_.

    >>> from logic1.firstorder import *
    >>> from logic1.theories.RCF import *
    >>> a, b, x = VV.get('a', 'b', 'x')

    >>> phi_6 = Ex(x, And(a * x + b <= 0, x <= b))
    >>> qe(phi_6, clustering=Clustering.NONE)
    Or(a > 0, And(b <= 0, a == 0), And(a < 0, a*b + b <= 0))
    >>> qe(phi_6, clustering=Clustering.FULL)
    Or(a > 0, And(b <= 0, a == 0), And(a < 0, a**2*b + a*b >= 0))

    >>> phi_7 = Ex(x, a * x**2 + b * x + c == 0)
    >>> qe(phi_7, clustering=Clustering.NONE)
    Or(And(c == 0, b == 0, a == 0),
       And(b < 0, a == 0), And(b > 0, a == 0),
       And(a < 0, 4*a*c - b**2 == 0), And(a < 0, 4*a*c - b**2 < 0),
       And(a > 0, 4*a*c - b**2 == 0), And(a > 0, 4*a*c - b**2 < 0))
    >>> qe(phi_7, clustering=Clustering.FULL)
    Or(And(c == 0, b == 0, a == 0),
       And(b != 0, a == 0),
       And(a != 0, 4*a*c - b**2 <= 0))
    """

    elimination_order: int
    """Strategy for determining the variable elimination order. This option
    affects only Xopt nodes. With ``elimination_order=0``, variables in each
    quantifier block are eliminated from the inside out, i.e., the last variable
    is eliminated first. With ``elimination_order=1`` (default), dynamic
    heuristics are used to select the next variable to eliminate.
    """

    generic: Generic
    r"""The degree of genericity used by the quantifier elimination. The
    default is :attr:`.Generic.NONE`. The principal idea of generic quantifier
    elimination is to assume certain disequalities on the parameters of the
    input formula in order to avoid case distinctions during quantifier
    elimination. Technically, these disequalities are added to
    :attr:`qe.assumptions <.abc.qe.QuantifierElimination.assumptions>`, which is
    initialized with the  ``assume`` argument of :func:`.qe`.
    The following options are available:

    :attr:`.Generic.NONE`
      uses regular quantifier elimination without making any assumptions.

    :attr:`.Generic.MONOMIAL`
      admits assumptions of the form :math:`m \neq 0` where :math:`m` is a
      monomial in the parameters of the input formula.

    :attr:`.Generic.FULL`
      admits assumptions of the form :math:`p \neq 0` where :math:`p` is a
      polynomial in the parameters of the input formula.

    For theoretical details on generic quantifier elimination see
    [DolzmannSturmWeispfenning-1998]_, [Sturm-1999]_.

    >>> from logic1.firstorder import *
    >>> from logic1.theories.RCF import *
    >>> a, b, c, x = VV.get('a', 'b', 'c', 'x')

    >>> qe(Ex(x, (a**2 - 2) * x**2 + b * x + c == 0),
    ...    assume=[c > 0])
    Or(And(b != 0, a**2 - 2 == 0),
       And(a**2 - 2 != 0, 4*a**2*c - b**2 - 8*c <= 0))
    >>> qe.assumptions
    [c > 0]

    >>> qe(Ex(x, (a**2 - 2) * x**2 + b * x + c == 0),
    ...    assume=[c > 0], generic=Generic.MONOMIAL)
    Or(a**2 - 2 == 0, 4*a**2*c - b**2 - 8*c <= 0)
    >>> qe.assumptions
    [c > 0, b != 0]

    >>> qe(Ex(x, (a**2 - 2) * x**2 + b * x + c == 0),
    ...    assume=[c > 0], generic=Generic.FULL)
    4*a**2*c - b**2 - 8*c <= 0
    >>> qe.assumptions
    [c > 0, a**2 - 2 != 0]
    """

    traditional_guards: bool
    """The default is ``traditional_guards=True``. Setting
    ``traditional_guards=False`` strictly follows the construction of guards as
    described in [Kosta-2016]_.

    >>> from logic1.firstorder import *
    >>> from logic1.theories.RCF import *
    >>> a, b, c, x = VV.get('a', 'b', 'c', 'x')

    >>> qe(Ex(x, a * x**2 + b * x + c == 0))
    Or(And(c == 0, b == 0, a == 0),
       And(b != 0, a == 0),
       And(a != 0, 4*a*c - b**2 <= 0))

    >>> qe(Ex(x, a * x**2 + b * x + c == 0), traditional_guards=False)
    Or(And(c == 0, b == 0, a == 0),
       And(b != 0, Or(c == 0, a == 0)),
       And(a != 0, 4*a*c - b**2 <= 0))
    """

    xopt: bool
    """The default ``xopt=True`` admits Xopt for subproblems in which all terms
    are weakly parametric linear.

    .. seealso::

      :func:`.qe`
        for more information on Xopt and the notion of subproblems.

      :meth:`.is_weakly_parametric_linear`
        for the definition of weakly parametric linear terms.
    """

    def __init__(self, /, clustering: Clustering = Clustering.FULL,
                 generic: Generic = Generic.NONE, traditional_guards: bool = True,
                 xopt: bool = True, elimination_order: int = 1, **kwargs) \
            -> None:
        super().__init__(**kwargs)
        self.clustering = clustering
        self.generic = generic
        self.traditional_guards = traditional_guards
        self.xopt = xopt
        self.elimination_order = elimination_order


@dataclass
class VirtualSubstitution(abc.qe.QuantifierElimination[Node, tuple[Formula, frozenset[Term]],
      Assumptions, list[str], Options, AtomicFormula, Term, Variable, int]):
    """Real quantifier elimination by virtual substitution.

    Implements the abstract methods
    :meth:`create_options() <.abc.qe.QuantifierElimination.create_options>`,
    :meth:`create_root_nodes() <.abc.qe.QuantifierElimination.create_root_nodes>`,
    :meth:`create_assumptions() <.abc.qe.QuantifierElimination.create_assumptions>`,
    :meth:`create_true_node() <.abc.qe.QuantifierElimination.create_true_node>`,
    :meth:`final_simplify() <.abc.qe.QuantifierElimination.final_simplify>`,
    :meth:`init_env() <.abc.qe.QuantifierElimination.init_env>`,
    :meth:`init_env_arg() <.abc.qe.QuantifierElimination.init_env_arg>` of its
    super class :class:`.abc.qe.QuantifierElimination`.
    """

    def create_options(self, **kwargs) -> Options:
        """Implements the abstract method :meth:`.abc.qe.QuantifierElimination.create_options`.
        """
        return Options(**kwargs)

    def create_root_nodes(self, variables: Iterable[Variable], matrix: Formula) -> list[Node]:
        """Implements the abstract method
        :meth:`.abc.qe.QuantifierElimination.create_root_nodes`.
        """
        assert self.options is not None
        assert self._assumptions is not None
        return [Node(variables=list(variables),
                     formula=simplify(matrix, assume=self._assumptions.atoms,
                                              prefer_order=True,
                                              prefer_weak=True),
                     answer=[],
                     outermost_block=not self.blocks,
                     options=self.options,
                     passive_list=set())]

    def create_assumptions(self, assume: Iterable[AtomicFormula]) -> Assumptions:
        """Implements the abstract method
        :meth:`.abc.qe.QuantifierElimination.create_assumptions`.
        """
        return Assumptions(assume)

    def create_true_node(self) -> Node:
        """Implements the abstract method :meth:`.abc.qe.QuantifierElimination.create_true_node`.
        """
        assert self.options is not None
        return Node(variables=[],
                    formula=_T(),
                    answer=[],
                    outermost_block=False,
                    options=self.options,
                    passive_list=set())

    def final_simplify(self, formula: Formula, assume: Iterable[AtomicFormula] = []) -> Formula:
        """Implements the abstract method
        :meth:`.abc.qe.QuantifierElimination.final_simplify`.
        """
        return simplify(formula, assume)

    @classmethod
    def init_env(cls, ring_vars: list[str]) -> None:
        """Implements the abstract method :meth:`.abc.qe.QuantifierElimination.init_env`.
        """
        term.init_env(ring_vars)

    def init_env_arg(self) -> list[str]:
        """Implements the abstract method :meth:`.abc.qe.QuantifierElimination.init_env_arg`.
        """
        return term.init_env_arg()


qe = virtual_substitution = VirtualSubstitution()
r""" Real quantifier elimination. Returns a quantifier-free equivalent
``f'`` of ``f`` modulo the assumptions provided by the attribute
:attr:`qe.assumptions <.abc.qe.QuantifierElimination.assumptions>`.

.. math::
  \textsf{RCF} \models \bigwedge \mathtt{qe.assumptions} \longrightarrow
                       (\mathtt{f} \longleftrightarrow \mathtt{f'}).

With regular quantifier elimination, :attr:`qe.assumptions
<.abc.qe.QuantifierElimination.assumptions>` contains the assumptions passed as the
``assume`` parameter, modulo simplification. In particular, we obtain
:math:`\mathbb{R} \models \mathtt{f} \longleftrightarrow \mathtt{f'}` with the
default ``assume=[]``. With generic quantifier elimination
[DolzmannSturmWeispfenning-1998]_, [Sturm-1999]_, disequalities in the
parameters may be added in the course of the elimination.

Technically, :obj:`logic1.theories.RCF.qe.qe` is an instance of the callable
class :class:`.VirtualSubstitution`. Its attributes are reset and reused with
each call of :func:`.qe`. Additional, independent instances of quantifier
elimination can be created and used as follows:

>>> from logic1.firstorder import *
>>> from logic1.theories.RCF import *
>>> from logic1.theories.RCF.qe import VirtualSubstitution
>>> another_qe = VirtualSubstitution()
>>> a, b, c, x = VV.get('a', 'b', 'c', 'x')
>>> qe(Ex(x, (a + 1) * x**2 + b * x + c == 0), generic=Generic.FULL)
4*a*c - b**2 + 4*c <= 0
>>> another_qe(Ex(x, a * x + b == 0), generic=Generic.FULL)
T
>>> qe.assumptions
[a + 1 != 0]
>>> another_qe.assumptions
[a != 0]

In general, our implementation essentially follows [Kosta-2016]_ up to degree
two. For subproblems in which all terms are weakly parametric linear, we use a
specialized approach, which we call *Xopt*, based on [Weispfenning-1997]_.

.. seealso::

  :class:`.Options`
    for the options that can be passed to this function. The documentation of
    the options also contains some more quantifier elimination examples.

  :attr:`Options.generic`
    explains generic quantifier elimination in more detail.

  :attr:`Options.workers`
    explains how to specify parallel computation of subproblems.

  :class:`logic1.theories.RCF.node.Node`
    The subproblems referred to above correspond to instances of this class.

  :meth:`.is_weakly_parametric_linear`
    for the definition of "weakly parametric linear terms".

  :class:`logic1.theories.RCF.qe.VirtualSubstitution`
    :obj:`qe` is an instance of this callable class.
"""