from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import auto, Enum
from logging import Logger
from typing import Final, Optional, TYPE_CHECKING

from logic1 import abc
from logic1.firstorder import And, Or
from logic1.theories.RCF.term import Term, Variable
from logic1.theories.RCF.atomic import AtomicFormula, Eq, Ge, Le
from logic1.theories.RCF.simplify import simplify
from logic1.theories.RCF.types import Formula

if TYPE_CHECKING:
    from logic1.theories.RCF.qe import Options


_trace: bool = False


def trprint(*args):
    if _trace:
        return print(*args)


CACHE_SIZE: Final[Optional[int]] = 2**16


class FoundF(Exception):
    pass


class Statistics:
    passive_list_hits: int = 0
    nodes_processed: int = 0
    nodes_false = 0
    gauss_instances: int = 0
    gauss_and: int = 0
    gauss_or: int = 0


class Assumptions(abc.qe.Assumptions[AtomicFormula, Term, Variable, int]):
    """Implements the abstract method :meth:`simplify()
    <.abc.qe.Assumptions.simplify>` of its super class
    :class:`.abc.qe.Assumptions`. Required by :class:`.Node` and
    :class:`.VirtualSubstitution` for instantiating the type variable
    :data:`.abc.qe.λ` of :class:`.abc.qe.Node` and
    :class:`.abc.qe.QuantifierElimination`, respectively.
    """

    def simplify(self, f: Formula) -> Formula:
        """Implements the abstract method :meth:`.abc.qe.Assumptions.simplify`.
        """
        return simplify(f, explode_always=False, prefer_order=False, prefer_weak=True)


class Clustering(Enum):
    """Clustering strategies available.

    Required by :class:`.RCF.qe.Options`.
    """
    NONE = auto()
    """No clustering at all
    """

    FULL = auto()
    """Full clustering
    """


class DegreeViolation(abc.qe.NodeProcessFailure):
    pass


class Generic(Enum):
    """Available degrees of genericity. For details on generic quantifier
    elimination see

    Required by :class:`.RCF.qe.Options`.
    """
    NONE = auto()
    """Regular quantifier elimination, not making any assumptions.
    """

    MONOMIAL = auto()
    """Admit assumptions on parameters by adding atomic formulas to
    :attr:`.abc.qe.QuantifierElimination.assumptions`, where the left hand side of those
    atomic formulas is a monomial (and the right hand side is zero).
    """

    FULL = auto()
    """Admit assumptions on parameters by adding atomic formulas to
    :attr:`.abc.qe.QuantifierElimination.assumptions`.
    """


@dataclass
class Node(abc.qe.Node[
        AtomicFormula, Term, Variable, int, Assumptions,
        tuple[tuple[Variable, ...], Formula, frozenset[Term]]]):
    """Implements the abstract methods :meth:`copy() <.abc.qe.Node.copy>` and
    :meth:`process() <.abc.qe.Node.process>` of its super class
    :class:`.abc.qe.Node`. Required by :class:`.VirtualSubstitution` for
    instantiating the type variable :data:`.abc.qe.ν` of
    :class:`.abc.qe.QuantifierElimination`.
    """
    answer: list
    outermost_block: bool
    options: Options
    passive_list: set[Term]

    def __str__(self):
        s = f'Node({self.variables}, {self.formula}, ...'
        if isinstance(self, xopt.Node):
            s += f', {self.passive_list}'
        s += ')'
        return s

    def admits_xopt(self) -> bool:
        """Check whether this node can be processed using xopt.
        """

        def recurse(formula: Formula) -> bool:
            if isinstance(formula, (Eq, Le, Ge)):
                return formula.lhs.is_weakly_parametric_linear(self.variables)
            elif isinstance(formula, AtomicFormula):
                return False
            else:
                assert isinstance(formula, (And, Or))
                return all(recurse(arg) for arg in formula.args)

        result = recurse(self.formula)
        return result

    def as_vs_node(self):
        return vs.Node(variables=self.variables,
                      formula=self.formula,
                      answer=self.answer,
                      outermost_block=self.outermost_block,
                      options=self.options,
                      passive_list=set())

    def as_xo_node(self):
        return xopt.Node(variables=self.variables,
                         formula=self.formula,
                         answer=self.answer,
                         outermost_block=self.outermost_block,
                         options=self.options,
                         passive_list=set())

    def copy(self) -> Node:
        """Implements the abstract method :meth:`.abc.qe.Node.copy`.
        """
        return Node(variables=self.variables,
                    formula=self.formula,
                    answer=self.answer,
                    outermost_block=self.outermost_block,
                    options=self.options,
                    passive_list=self.passive_list)

    def logger(self) -> Logger:
        if self.options.workers == 0:
            return abc.qe.logger
        else:
            return abc.qe.multiprocessing_logger

    def memorize(self) -> tuple[tuple[Variable, ...], Formula, frozenset[Term]]:
        return (tuple(self.variables),
                self.formula,
                frozenset(self.passive_list))

    def process(self, assumptions: Assumptions) -> Sequence[Node]:
        """Implements the abstract method :meth:`.abc.qe.Node.process`.
        """
        self.logger().debug(f'Entering process')
        if isinstance(self, xopt.Node):
            return self.process(assumptions=assumptions)
        elif isinstance(self, vs.Node):
            if self.options.xopt and self.admits_xopt():
                return self.as_xo_node().process(assumptions=assumptions)
            else:
                return self.process(assumptions=assumptions)
        else:
            if self.options.xopt and self.admits_xopt():
                return self.as_xo_node().process(assumptions=assumptions)
            else:
                return self.as_vs_node().process(assumptions=assumptions)


from logic1.theories.RCF.node import vs, xopt