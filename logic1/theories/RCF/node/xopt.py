from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import auto, Enum
from functools import lru_cache
from typing import Iterable, Iterator, Optional, Self

from gmpy2 import mpq

from logic1 import abc
from logic1.firstorder import And, _F, Or, _T
from logic1.theories.RCF.term import Term, Variable
from logic1.theories.RCF.atomic import AtomicFormula, Eq, Ne, Ge, Le
from logic1.theories.RCF.simplify import simplify
from logic1.theories.RCF.types import Formula
from logic1.theories.RCF.node.base import Assumptions, CACHE_SIZE, FoundF, Node, Statistics


class _XoBoundType(Enum):
    EQUATION = auto()
    LOWER_BOUND = auto()
    UPPER_BOUND = auto()
    NONE = auto()


@dataclass
class _XoCandidateSet:
    equations: set[Term] = field(default_factory=set)
    lower_bounds: set[Term] = field(default_factory=set)
    upper_bounds: set[Term] = field(default_factory=set)
    has_finite_solution_set: Optional[bool] = field(default=None, init=False)

    def __iand__(self, other: Self) -> Self:
        if other.has_finite_solution_set is None:
            return self
        self.equations |= other.equations
        self.lower_bounds |= other.lower_bounds
        self.upper_bounds |= other.upper_bounds
        assert self.has_finite_solution_set is not True
        assert other.has_finite_solution_set is False
        self.has_finite_solution_set = False
        return self

    def __ior__(self, other: Self) -> Self:
        if other.has_finite_solution_set is None:
            return self
        self.equations |= other.equations
        self.lower_bounds |= other.lower_bounds
        self.upper_bounds |= other.upper_bounds
        if self.has_finite_solution_set is None:
            self.has_finite_solution_set = other.has_finite_solution_set
        else:
            self.has_finite_solution_set &= other.has_finite_solution_set
        return self

    def __len__(self) -> int:
        return len(self.equations) + len(self.lower_bounds) + len(self.upper_bounds)

    def __post_init__(self):
        if len(self) == 0:
            self.has_finite_solution_set = None
        elif len(self) == 1:
            if self.equations:
                self.has_finite_solution_set = True
            else:
                self.has_finite_solution_set = False
        else:
            raise ValueError("illegal arguments in XoptCandidateSet()")

    def apply_passive_list(self, passive_list: Iterable[Term]) -> Self:
        for subset in (self.equations, self.lower_bounds, self.upper_bounds):
            # Do not mutate a collection while iterating over it
            to_remove = []
            for candidate in subset:
                if candidate in passive_list:
                    to_remove.append(candidate)
                    Statistics.passive_list_hits += 1
                else:
                    assert not isinstance(simplify(candidate == 0, (Ne(t, 0) for t in passive_list)), _F)
            for candidate in to_remove:
                subset.remove(candidate)
        return self

    def elimination_set(self) -> _XoEliminationSet:
        if len(self.upper_bounds) == 0 and len(self.lower_bounds) == 0:
            assert len(self.equations) > 0
            standard_terms = self.equations
            infinity = None
        elif len(self.upper_bounds) <= len(self.lower_bounds):
            standard_terms = self.upper_bounds | self.equations
            infinity = _XoInfinity.PLUS
        else:
            standard_terms = self.lower_bounds | self.equations
            infinity = _XoInfinity.MINUS
        return _XoEliminationSet(standard_terms=sorted(standard_terms), infinity=infinity)


@dataclass
class _XoEliminationSet:
    standard_terms: list[Term]
    infinity: Optional[_XoInfinity]

    def __iter__(self) -> Iterator[Term | _XoInfinity]:
        if self.infinity is not None:
            yield self.infinity
        yield from self.standard_terms

    def __len__(self) -> int:
        if self.infinity is None:
            return len(self.standard_terms)
        else:
            return len(self.standard_terms) + 1

    def _choice(self) -> str:
        if self.infinity == _XoInfinity.PLUS:
            return f'{len(self)} upper bounds and equations'
        elif self.infinity == _XoInfinity.MINUS:
            return f'{len(self)} lower bounds and equations'
        else:
            return f'{len(self)} equations'


class _XoInfinity(Enum):
    PLUS = auto()
    MINUS = auto()

    def __mul__(self, other: mpq) -> _XoInfinity:
        if other > 0:
            return self
        elif other < 0:
            if self is _XoInfinity.PLUS:
                return _XoInfinity.MINUS
            else:
                assert self is _XoInfinity.MINUS
                return _XoInfinity.PLUS
        else:
            raise ValueError(f'{other}')


@dataclass
class XoNode(Node):

    def best_elimination_set(self, X: list[Variable], assumptions: Assumptions) -> tuple[_XoEliminationSet, Variable]:
        best_length = None
        if self.options.elimination_order == 0:
            x = X[-1]
            candidate_set = XoNode.candidate_set(self.formula, x)
            assert len(candidate_set) > 0
            candidate_set.apply_passive_list(self.passive_list)
            if len(candidate_set) == 0:
                raise FoundF()
            elimination_set = candidate_set.elimination_set()
            return elimination_set, x
        elif self.options.elimination_order == 1:
            for x in reversed(X):
                candidate_set = XoNode.candidate_set(self.formula, x)
                assert len(candidate_set) > 0
                candidate_set.apply_passive_list(self.passive_list)
                if len(candidate_set) == 0:
                    raise FoundF()
                elimination_set = candidate_set.elimination_set()
                length = len(elimination_set)
                if best_length is None or length < best_length:
                    # <, in contrast to <=, prefers the innermost quantifier.
                    best_length = length
                    best_choice = (elimination_set, x)
            return best_choice
        elif self.options.elimination_order == 2:
            raise NotImplementedError('elimination_order > 1 is not supported')
            # Some experimental code has been removed here. The general idea is
            # to consider all smallest elimination sets and find a good one, at
            # the price of substituting.
            #
            # On the one hand, the number of nodes computed in SC50A decreases
            # by about 20 percent, while the computation time increases. On the
            # other hand the number of nodes increases by about 20 percent in
            # SC50A-r. There is no effect on the number of nodes computed in
            # MTP3. We need more benchmarks and fresh ideas.
        assert False

    @staticmethod
    def bound_type(atom: AtomicFormula, x: Variable) -> _XoBoundType:
        c = atom.lhs.monomial_coefficient(x)
        if c == 0:
            return _XoBoundType.NONE
        elif c > 0 and isinstance(atom, Le) or c < 0 and isinstance(atom, Ge):
            return _XoBoundType.UPPER_BOUND
        elif c > 0 and isinstance(atom, Ge) or c < 0 and isinstance(atom, Le):
            return _XoBoundType.LOWER_BOUND
        else:
            assert isinstance(atom, Eq)
            return _XoBoundType.EQUATION

    @staticmethod
    def candidate_set(formula: Formula, x: Variable) -> _XoCandidateSet:
        """Compute a candidate set using structural elimination, which covers
        generalizations of Gaussian elimination.
        """
        if isinstance(formula, AtomicFormula):
            bound_type = XoNode.bound_type(formula, x)
            if bound_type is _XoBoundType.NONE:
                return _XoCandidateSet()
            elif bound_type is _XoBoundType.UPPER_BOUND:
                return _XoCandidateSet(upper_bounds={formula.lhs})
            elif bound_type is _XoBoundType.LOWER_BOUND:
                return _XoCandidateSet(lower_bounds={formula.lhs})
            else:
                assert bound_type is _XoBoundType.EQUATION
                return _XoCandidateSet(equations={formula.lhs})
        elif isinstance(formula, And):
            result = _XoCandidateSet()
            for arg in formula.args:
                candidate_set_ = XoNode.candidate_set(arg, x)
                if candidate_set_.has_finite_solution_set:
                    Statistics.gauss_and += 1
                    return candidate_set_
                else:
                    result &= candidate_set_
            return result
        else:
            assert isinstance(formula, Or)
            result = _XoCandidateSet()
            for arg in formula.args:
                result |= XoNode.candidate_set(arg, x)
            if result.has_finite_solution_set:
                Statistics.gauss_or += 1
            return result

    def process(self, assumptions: Assumptions) -> Sequence[XoNode]:
        Statistics.nodes_processed += 1
        formula = self.formula
        variables = []
        seen: set[Variable] = set()
        occurring_variables = set(formula.fvars())
        for x in reversed(self.variables):
            if x in occurring_variables and x not in seen:
                seen.add(x)
                variables.append(x)
        variables.reverse()
        passive_list = self.passive_list
        if not variables:
            # Return one success node.
            return [XoNode(variables=[],
                           formula=formula,
                           answer=[],
                           outermost_block=self.outermost_block,
                           options=self.options,
                           passive_list=passive_list.copy())]
        try:
            elimination_set, x = self.best_elimination_set(variables, assumptions)
        except FoundF:
            return []
        variables.remove(x)
        successors: deque[XoNode] = deque()
        for testpoint in sorted(elimination_set, key=self.sort_key):
            substituted_formula = XoNode.subs_into_formula(formula, x, testpoint, assumptions)
            if isinstance(substituted_formula, _T):
                raise abc.qe.FoundT()
            elif isinstance(substituted_formula, _F):
                Statistics.nodes_false += 1
                # It would be correct to leave this to the else-case, but
                # we want to avoid computing the substituted_passive_list_copy.
            else:
                substituted_passive_list = XoNode.subs_into_passive_list(passive_list, x, testpoint)
                node = XoNode(variables=variables.copy(),
                              formula=substituted_formula,
                              answer=[],
                              outermost_block=self.outermost_block,
                              options=self.options,
                              passive_list=substituted_passive_list)
                # nodes with many variables come first in order to have better
                # passive lists. Nevetheless, we want to continue DFS with few
                # bound variables first. Therefore, we implicitly reverse here:
                successors.appendleft(node)
            if isinstance(testpoint, Term):
                passive_list.add(testpoint)
        return successors

    def sort_key(self, testpoint: Term | _XoInfinity) -> tuple[int, int]:
        """Sort test points of an elimination set before substitution. Smaller
        test points enter the passive lists of all greater test points. The
        general idea is that test points with many variables are small, because
        elements of passive list without no parameters and few quantified
        variables die early by becoming constant via substitution.

        Based on AFIRO, SC50A, and MTP3, the lexicographic order used here
        seems to be slightly better than simply considering all_variables. This
        requires validation with more benchmarks, or the could should be
        simplified.
        """
        if isinstance(testpoint, Term):
            all_variables = frozenset(testpoint.vars())
            bound_variables = all_variables & frozenset(self.variables)
            free_variables = all_variables - bound_variables
            return (-len(bound_variables), -len(free_variables))
            # -(-len(all_variables)) performed slightly better on a small set of
            # benchmarks comprising AFIRO, SC50A, SC50B, MPT-2, and MTP3.
            # Our choice above is essentially as good and more inutitive.
        else:
            return (0, 0)

    @staticmethod
    @lru_cache(maxsize=CACHE_SIZE)
    def subs_into_formula(formula: Formula, x: Variable, testpoint: Term | _XoInfinity,
                          assumptions: Assumptions) -> Formula:
        """Substitute ``testpoint`` for ``x`` in ``formula``, and apply
        simplification modulo ``assumptions``
        """

        def subs_infinity_at(atom: AtomicFormula, x: Variable, inf: _XoInfinity) -> Formula:
            c = atom.lhs.monomial_coefficient(x)
            if c == 0:
                return atom
            if isinstance(atom, Eq):
                return _F()
            inf = inf * c
            if isinstance(atom, Ge):
                return _T() if inf is _XoInfinity.PLUS else _F()
            else:
                assert isinstance(atom, Le)
                return _T() if inf is _XoInfinity.MINUS else _F()

        if isinstance(testpoint, Term):
            new_formula = formula.traverse(
                map_atoms=lambda atom: atom.op(atom.lhs.subs_linear_solution(x, testpoint), 0))
        else:
            assert isinstance(testpoint, _XoInfinity)
            new_formula = formula.traverse(
                map_atoms=lambda atom: subs_infinity_at(atom, x, testpoint))
        return simplify(new_formula, assume=assumptions.atoms,
                                     explode_always=False,
                                     implicit_ranges=False,
                                     prefer_order=True,
                                     prefer_weak=True,
                                     substitute=1)

    @staticmethod
    def subs_into_passive_list(passive_list: set[Term], x: Variable,
                               testpoint: Term | _XoInfinity) -> set[Term]:
        """Returns a copy of ``passive_list`` with ``testpoint`` substituted for
        ``x``.
        """
        result = set()
        if isinstance(testpoint, Term):
            for passive_term in passive_list:
                new_term = passive_term.subs_linear_solution(x, testpoint)
                if new_term.is_constant():
                    # We cannot obtain zero, because such situations are
                    # filtered via the application of the passive list.
                    assert not new_term.is_zero(), f'{passive_term=}, {x=}, {testpoint=}'
                else:
                    result.add(new_term.primitive_part(positive=True))
        else:
            assert isinstance(testpoint, _XoInfinity)
            for passive_term in passive_list:
                if passive_term.monomial_coefficient(x) == 0:
                    result.add(passive_term)
        return result
