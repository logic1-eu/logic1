"""Real quantifier elimination by virtual substitution [Sturm-2018]_.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from logic1 import abc
from logic1.firstorder import _T
from logic1.theories.RCF.atomic import AtomicFormula, polynomial_ring, Term, Variable
from logic1.theories.RCF.node import Assumptions, Clustering, Generic, Node
from logic1.theories.RCF.simplify import simplify
from logic1.theories.RCF.typing import Formula


@dataclass
class Options(abc.qe.Options):
    """The options specified here, as well as the options inherited from
    :class:`.abc.qe.Options`, can be passed to the callable class
    :class:`.VirtualSubstitution` as keyword arguments.

    Required by :class:`.VirtualSubstitution` for instantiating the type
    variable :data:`.abc.qe.ω` of :class:`.abc.qe.QuantifierElimination`.
    """

    clustering: Clustering
    """The clustering strategy used by :class:`.VirtualSubstitution`. See
    [Kosta-2016]_ for details on clustering.
    """

    generic: Generic
    """The degree of genericity used by :class:`.VirtualSubstitution`. See
    [DolzmannSturmWeispfenning-1998]_, [Sturm-1999]_ for details on generic
    quantifier elimination.

    >>> from logic1.firstorder import *
    >>> from logic1.theories.RCF import *
    >>> a, b, c, x = VV.get('a', 'b', 'c', 'x')

    >>> qe(Ex(x, (a**2 - 2) * x**2 + b * x + c == 0),
    ...    assume=[c > 0])
    Or(And(b != 0, a^2 - 2 == 0),
       And(a^2 - 2 != 0, 4*a^2*c - b^2 - 8*c <= 0))
    >>> qe.assumptions
    [c > 0]

    >>> qe(Ex(x, (a**2 - 2) * x**2 + b * x + c == 0),
    ...    assume=[c > 0], generic=GENERIC.FULL)
    4*a^2*c - b^2 - 8*c <= 0
    >>> qe.assumptions
    [c > 0, a^2 - 2 != 0]

    >>> qe(Ex(x, (a**2 - 2) * x**2 + b * x + c == 0),
    ...    assume=[c > 0], generic=GENERIC.MONOMIAL)
    Or(a^2 - 2 == 0, 4*a^2*c - b^2 - 8*c <= 0)
    >>> qe.assumptions
    [c > 0, b != 0]
    """

    traditional_guards: bool
    """`traditional_guards=False` strictly follows the construction of guards
    as described in [Kosta-2016]_.

    >>> from logic1.firstorder import *
    >>> from logic1.theories.RCF import *
    >>> a, b, c, x = VV.get('a', 'b', 'c', 'x')

    >>> qe(Ex(x, a * x**2 + b * x + c == 0))
    Or(And(c == 0, b == 0, a == 0),
       And(b != 0, a == 0),
       And(a != 0, 4*a*c - b^2 <= 0))

    >>> qe(Ex(x, a * x**2 + b * x + c == 0), traditional_guards=False)
    Or(And(c == 0, b == 0, a == 0),
       And(b != 0, Or(c == 0, a == 0)),
       And(a != 0, 4*a*c - b^2 <= 0))
    """

    xopt: bool

    def __init__(self, /, clustering: Clustering = Clustering.FULL,
                 generic: Generic = Generic.NONE, traditional_guards: bool = True,
                 xopt: bool = True, **kwargs) \
            -> None:
        super().__init__(**kwargs)
        self.clustering = clustering
        self.generic = generic
        self.traditional_guards = traditional_guards
        self.xopt = xopt


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
                     formula=simplify(matrix, assume=self._assumptions.atoms),
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
    def init_env(cls, ring_vars: list[str]):
        """Implements the abstract method :meth:`.abc.qe.QuantifierElimination.init_env`.
        """
        polynomial_ring.add_vars(ring_vars)

    def init_env_arg(self) -> list[str]:
        """Implements the abstract method :meth:`.abc.qe.QuantifierElimination.init_env_arg`.
        """
        # We pass the ring variables to the workers. The workers
        # reconstruct the ring.
        return [str(v) for v in polynomial_ring.get_vars()]


qe = virtual_substitution = VirtualSubstitution()
"""
Real quantifier elimination by virtual substitution. The implementation
essentially follows [Kosta-2016]_ up to degree two. It also offers generic
quantifier elimination [DolzmannSturmWeispfenning-1998]_, [Sturm-1999]_.

Technically, :func:`.qe` is an instance of the callable class
:class:`.VirtualSubstitution`.

:param f:
  The input formula to which quantifier elimination will be applied.

:param assume:
  A list of atomic formulas that are assumed to hold. The return value
  is equivalent modulo those assumptions.

:param workers:
  Specifies the number of processes to be used in parallel:

  * The default value `workers=0` uses a sequential implementation,
    which avoids overhead when input problems are small. For all other
    values, there are additional processes started.

  * A positive value `workers=n > 0` uses `n + 2` processes: the master
    process, `n` worker processes, and a proxy processes that manages
    shared data.

    .. note::
      `workers=1` uses the parallel implementation with only one
      worker. Algorithmically this is similar to the sequential version
      with `workers=0` but comes at the cost of 2 additional processes.

  * A negative value `workers=-n < 0` specifies ``os.num_cpu() - n``
    many workers.  It follows that `workers=-2` exactly allocates all
    of CPUs of the machine, and workers=-3 is an interesting choice,
    which leaves one CPU free for smooth interaction with the machine.

:param `**options`:
  Keyword arguments with keywords corresponding to attributes of
  :class:`.Options`. Those are :attr:`.clustering`, :attr:`.generic`,
  :attr:`.log_level`, :attr:`.log_rate`, :attr:`.traditional_guards`.

:returns:
  A quantifier-free equivalent of `f` modulo assumptions that are available in
  :attr:`qe.assumptions <.abc.qe.QuantifierElimination.assumptions>` at the end of
  the computation. With regular quantifier elimination, the assumptions are
  those passed as the `assume` parameter, modulo simplification. With
  *generic quantifier elimination*, inequations in the parameters can be
  added in the course of the elimination. See :attr:`.Options.generic` for
  examples.
"""
