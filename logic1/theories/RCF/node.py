from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import auto, Enum
from logging import Logger
from typing import ClassVar, Iterable, Iterator, Literal, Optional, Self, TypeAlias, TYPE_CHECKING

from gmpy2 import mpq

from logic1 import abc
from logic1.firstorder import And, _F, Not, Or, _T
from logic1.theories.RCF.atomic import AtomicFormula, Eq, Ne, Ge, Le, Gt, Lt, Term, Variable
from logic1.theories.RCF.simplify import is_valid, simplify
from logic1.theories.RCF.typing import Formula

if TYPE_CHECKING:
    from logic1.theories.RCF.qe import Options


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

    Required by :class:`.qe.Options`.
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

    Required by :class:`.qe.Options`.
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
class Node(abc.qe.Node[Formula, Variable, Assumptions]):
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
        if isinstance(self, XoNode):
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
        return VsNode(variables=self.variables,
                      formula=self.formula,
                      answer=self.answer,
                      outermost_block=self.outermost_block,
                      options=self.options,
                      passive_list=set())

    def as_xo_node(self):
        return XoNode(variables=self.variables,
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

    def process(self, assumptions: Assumptions) -> Sequence[Node]:
        """Implements the abstract method :meth:`.abc.qe.Node.process`.
        """
        self.logger().debug(f'Entering process')
        if isinstance(self, XoNode):
            return self.process(assumptions=assumptions)
        elif isinstance(self, VsNode):
            if self.options.xopt and self.admits_xopt():
                return self.as_xo_node().process(assumptions=assumptions)
            else:
                return self.process(assumptions=assumptions)
        else:
            if self.options.xopt and self.admits_xopt():
                return self.as_xo_node().process(assumptions=assumptions)
            else:
                return self.as_vs_node().process(assumptions=assumptions)


class _VsNsp(Enum):
    """Non-Standard Part
    """
    NONE = auto()
    PLUS_EPSILON = auto()
    MINUS_EPSILON = auto()
    PLUS_INFINITY = auto()
    MINUS_INFINITY = auto()


class _VsTag(Enum):
    """Describes the bound type - upper, lower, or any.
    """
    XLB = auto()
    XUB = auto()
    ANY = auto()


_VsSignSequence: TypeAlias = tuple[Literal[-1, 0, 1], ...]


@dataclass(frozen=True)
class _VsRootSpec:
    """Root Specification
    """

    signs: _VsSignSequence
    index: int

    def __neg__(self) -> _VsRootSpec:
        return _VsRootSpec(signs=tuple(-i for i in self.signs), index=self.index)

    def bound_type(self, atom: AtomicFormula) -> tuple[bool, Optional[_VsTag]]:
        """Return value None means that atom has a constant truth value
        """
        zero_index = 2 * self.index - 1
        assert self.signs[zero_index] == 0, (self, atom)
        left = self.signs[zero_index - 1]
        right = self.signs[zero_index + 1]
        assert left != 0 and right != 0, (self, atom)
        match (atom, left, right):
            case (Eq(), _, _):
                return (False, _VsTag.ANY)

            case (Ne(), _, _):
                return (True, _VsTag.ANY)

            case (Lt(), -1, -1) | (Gt(), 1, 1):
                return (True, _VsTag.ANY)
            case (Lt(), -1, 1) | (Gt(), 1, -1):
                return (True, _VsTag.XUB)
            case (Lt(), 1, -1) | (Gt(), -1, 1):
                return (True, _VsTag.XLB)
            case (Lt(), 1, 1) | (Gt(), -1, -1):
                return (True, None)

            case (Le(), -1, -1) | (Ge(), 1, 1):
                return (False, None)
            case (Le(), -1, 1) | (Ge(), 1, -1):
                return (False, _VsTag.XUB)
            case (Le(), 1, -1) | (Ge(), -1, 1):
                return (False, _VsTag.XLB)
            case (Le(), 1, 1) | (Ge(), -1, -1):
                return (False, _VsTag.ANY)

            case _:
                assert False, (atom, left, right)

    def guard(self, term: Term, x: Variable) -> Formula:
        match term.degree(x):
            case -1 | 0:
                assert False, (self, term, x)
            case 1:
                a = term.coefficient({x: 1})
                match self.signs:
                    case (-1, 0, 1):
                        return a > 0
                    case (1, 0, -1):
                        return a < 0
                    case _:
                        assert False, (self, term, x)
            case 2:
                a = term.coefficient({x: 2})
                b = term.coefficient({x: 1})
                c = term.coefficient({x: 0})
                d2 = b**2 - 4 * a * c
                match self.signs:
                    case (1, 0, -1, 0, 1):
                        return And(a > 0, d2 > 0)
                    case (-1, 0, 1, 0, -1):
                        return And(a < 0, d2 > 0)
                    case (1, 0, 1):
                        return And(a > 0, d2 == 0)
                    case (-1, 0, -1):
                        return And(a < 0, d2 == 0)
                    case _:
                        assert False, (self, term, x)
            case _:
                raise DegreeViolation(self, term, x)

    def kosta_code(self, d: int) -> int:
        D: dict[tuple[int, _VsSignSequence], int] = {
            (1, (-1, 0, 1)): 1,
            (1, (1, 0, -1)): -1,
            (2, (1, 0, -1, 0, 1)): 1,
            (2, (1, 0, 1)): 2,
            (2, (1,)): 3,
            (2, (-1, 0, 1, 0, -1)): -1,
            (2, (-1, 0, -1)): -2,
            (2, (-1,)): -3,
            (3, (-1, 0, 1)): 1,
            (3, (-1, 0, -1, 0, 1)): 2,
            (3, (-1, 0, 1, 0, 1)): 3,
            (3, (-1, 0, 1, 0, -1, 0, 1)): 4,
            (3, (1, 0, -1)): -1,
            (3, (1, 0, 1, 0, -1)): -2,
            (3, (1, 0, -1, 0, -1)): -3,
            (3, (1, 0, -1, 0, 1, 0, -1)): -4}
        return D[d, self.signs]


@dataclass(frozen=True)
class _VsCluster:
    """A Cluster wraps a tuple of Root Specifications.
    """
    root_specs: tuple[_VsRootSpec, ...]

    def __neg__(self) -> _VsCluster:
        return _VsCluster(tuple(- root_spec for root_spec in self.root_specs))

    def __iter__(self) -> Iterator[_VsRootSpec]:
        return iter(self.root_specs)

    def bound_type(self, atom: AtomicFormula, x: Variable, assumptions: Assumptions)\
            -> tuple[bool, Optional[_VsTag]]:
        epsilons = set()
        tags = set()
        for root_spec in self.root_specs:
            if simplify(root_spec.guard(atom.lhs, x), assume=assumptions.atoms) is _F():
                continue
            with_epsilon, tag = root_spec.bound_type(atom)
            if tag is not None:
                epsilons.add(with_epsilon)
                tags.add(tag)
        assert len(epsilons) <= 1, (self, atom, x)
        try:
            epsilon = next(iter(epsilons))
        except StopIteration:
            epsilon = False
        if len(tags) == 0:
            tag = None
        elif tags == {_VsTag.XLB} or tags == {_VsTag.XLB, _VsTag.ANY}:
            tag = _VsTag.XLB
        elif tags == {_VsTag.XUB} or tags == {_VsTag.XUB, _VsTag.ANY}:
            tag = _VsTag.XUB
        else:
            tag = _VsTag.ANY
        return (epsilon, tag)

    def guard(self, term: Term, x: Variable) -> Formula:
        d = term.degree(x)
        match d, self:
            case 1, _VsCluster((_VsRootSpec(signs=(-1, 0, 1), index=1),
                                _VsRootSpec(signs=(1, 0, -1), index=1))):
                a = term.coefficient({x: 1})
                return a != 0
            case 2, _VsCluster((_VsRootSpec(signs=(1, 0, -1, 0, 1), index=1),
                                _VsRootSpec(signs=(-1, 0, 1, 0, -1), index=2),
                                _VsRootSpec(signs=(1, 0, 1), index=1),
                                _VsRootSpec(signs=(-1, 0, -1), index=1))):
                a = term.coefficient({x: 2})
                b = term.coefficient({x: 1})
                c = term.coefficient({x: 0})
                d2 = b**2 - 4 * a * c
                return And(a != 0, d2 >= 0)
            case _:
                return Or(*(root_spec.guard(term, x) for root_spec in self.root_specs))


@dataclass(frozen=True)
class _VsPRD:
    """Parametric Root Description
    """
    term: Term
    variable: Variable
    cluster: _VsCluster
    xguard: Formula = field(default_factory=_T)

    def guard(self, assumptions: Assumptions) -> Formula:
        guard = self.cluster.guard(self.term, self.variable)
        return simplify(And(self.xguard, guard), assume=assumptions.atoms)

    def vsubs(self, atom: AtomicFormula) -> Formula:
        """Virtually substitute self into atom yielding a quantifier-free
        formula
        """
        match atom:
            case Ne() | Gt() | Ge():
                return Not(self._vsubs(atom.to_complement())).to_nnf()
            case Eq() | Lt() | Le():
                return self._vsubs(atom)
            case _:
                assert False, (self, atom)

    def _vsubs(self, atom: AtomicFormula) -> Formula:
        """Virtual substitution of PRD into atom.
        """
        x = self.variable
        deg_g = atom.lhs.degree(x)
        match deg_g:
            case -1 | 0:
                return atom
            case 1:
                aa = atom.lhs.coefficient({x: 1})
                bb = atom.lhs.coefficient({x: 0})
            case _:
                raise NotImplementedError(deg_g)
        deg_f = self.term.degree(x)
        assert deg_g < deg_f, f'{self=}, {atom=}'  # Pseudo-division has been applied
        # f into g
        match deg_f:
            case -1 | 0 | 1:
                assert False
            case 2:
                a = self.term.coefficient({x: 2})
                b = self.term.coefficient({x: 1})
                c = self.term.coefficient({x: 0})
                A = 2 * a * aa * bb - aa**2 * b
                B = a * bb**2 + aa**2 * c - aa * b * bb
                C = 2 * a * bb - aa * b
                match (deg_g, atom, self.cluster):
                    # Kosta Appendix A.1: Without clustering
                    case (1, Eq(), _VsCluster((_VsRootSpec(signs=(1, 0, -1, 0, 1), index=1),))):
                        return And(A >= 0, B == 0)
                    case (1, Eq(), _VsCluster((_VsRootSpec(signs=(1, 0, -1, 0, 1), index=2),))):
                        return And(A <= 0, B == 0)
                    case (1, Eq(), _VsCluster((_VsRootSpec(signs=(1, 0, 1), index=1),))):
                        return C == 0
                    case (1, Lt(), _VsCluster((_VsRootSpec(signs=(1, 0, -1, 0, 1), index=1),))):
                        return Or(And(C < 0, B > 0), And(aa >= 0, Or(C < 0, B < 0)))
                    case (1, Lt(), _VsCluster((_VsRootSpec(signs=(1, 0, -1, 0, 1), index=2),))):
                        return Or(And(C < 0, B > 0), And(aa <= 0, Or(C < 0, B < 0)))
                    case (1, Lt(), _VsCluster((_VsRootSpec(signs=(1, 0, 1), index=1),))):
                        return C < 0
                    case (1, Le(), _VsCluster((_VsRootSpec(signs=(1, 0, -1, 0, 1), index=1),))):
                        return Or(And(C <= 0, B >= 0), And(aa >= 0, B <= 0))
                    case (1, Le(), _VsCluster((_VsRootSpec(signs=(1, 0, -1, 0, 1), index=2),))):
                        return Or(And(C <= 0, B >= 0), And(aa <= 0, B <= 0))
                    case (1, Le(), _VsCluster((_VsRootSpec(signs=(1, 0, 1), index=1),))):
                        return C <= 0
                    # Kosta Appendix A.3: With clustering
                    case (1, Eq(), _VsCluster((_VsRootSpec(signs=(1, 0, -1, 0, 1), index=1),
                                               _VsRootSpec(signs=(-1, 0, 1, 0, -1), index=2),
                                               _VsRootSpec(signs=(1, 0, 1), index=1),
                                               _VsRootSpec(signs=(-1, 0, -1), index=1)))):
                        return And(A >= 0, B == 0)
                    case (1, Lt(), _VsCluster((_VsRootSpec(signs=(1, 0, -1, 0, 1), index=1),
                                               _VsRootSpec(signs=(-1, 0, 1, 0, -1), index=2),
                                               _VsRootSpec(signs=(1, 0, 1), index=1),
                                               _VsRootSpec(signs=(-1, 0, -1), index=1)))):
                        return Or(And(a * C < 0, a * B > 0),
                                  And(a * aa >= 0, Or(a * C < 0, a * B < 0)))
                    case (1, Le(), _VsCluster((_VsRootSpec(signs=(1, 0, -1, 0, 1), index=1),
                                               _VsRootSpec(signs=(-1, 0, 1, 0, -1), index=2),
                                               _VsRootSpec(signs=(1, 0, 1), index=1),
                                               _VsRootSpec(signs=(-1, 0, -1), index=1)))):
                        return Or(And(a * C <= 0, a * B >= 0), And(a * aa >= 0, a * B <= 0))
                    case _:
                        assert False, f'{self=}, {atom=}'
            case _:
                raise NotImplementedError(f'{self=}, {atom=}')

    def _translate(self) -> str:
        x = self.variable
        deg_f = self.term.degree(x)
        a = self.term.coefficient({x: 2})
        b = self.term.coefficient({x: 1})
        c = self.term.coefficient({x: 0})
        match deg_f:
            case 1:
                match self.cluster:
                    # CLUSTERING.NONE
                    case _VsCluster((_VsRootSpec(signs=(-1, 0, 1), index=1),)):
                        return f'({-c}) / ({b})'
                    # CLUSTERING.FULL
                    case _VsCluster((_VsRootSpec(signs=(-1, 0, 1), index=1),
                                     _VsRootSpec(signs=(1, 0, -1), index=1))):
                        return f'({-c}) / ({b})'
                    case _:
                        assert False, self
            case 2:
                match self.cluster:
                    # CLUSTERING.NONE
                    case _VsCluster((_VsRootSpec(signs=(1, 0, -1, 0, 1), index=1),)):
                        return f'({-b} - sqrt({b**2- 4*a*c})) / ({2*a})'
                    case _VsCluster((_VsRootSpec(signs=(1, 0, -1, 0, 1), index=2),)):
                        return f'({-b} + sqrt({b**2- 4*a*c})) / ({2*a})'
                    case _VsCluster((_VsRootSpec(signs=(1, 0, 1), index=1),)):
                        return f'({-b} ± sqrt({0})) / ({2*a})'
                    # CLUSTERING.FULL
                    case _VsCluster((_VsRootSpec(signs=(1, 0, -1, 0, 1), index=1),
                                     _VsRootSpec(signs=(-1, 0, 1, 0, -1), index=2),
                                     _VsRootSpec(signs=(1, 0, 1), index=1),
                                     _VsRootSpec(signs=(-1, 0, -1), index=1))):
                        return f'({-b} - sqrt({b**2- 4*a*c})) / ({2*a})'
                    case _:
                        assert False, self
            case _:
                assert False, self


@dataclass(frozen=True)
class _VsCandidateSolution:
    """A candidate solution combines a parametric root description with a
    flag indicating that epsilon will be needed and a bound type.

    CandidateSolutions are used as elements of sets. In order to become
    hashable, the dataclass is frozen, along with RootSpec, PRD, and RealType.
    """
    prd: _VsPRD
    with_epsilon: bool
    tag: _VsTag


@dataclass
class _VsTestPoint:
    """A test point combinines a parametric root description with an optional
    non-standard part.
    """
    prd: Optional[_VsPRD] = None
    nsp: _VsNsp = _VsNsp.NONE

    def guard(self, assumptions: Assumptions):
        if self.prd is None:
            return _T()
        else:
            guard = self.prd.guard(assumptions)
            assert guard is not _F(), self
            return guard

    def _translate(self) -> str:
        assert self.prd is not None
        match self.nsp:
            case _VsNsp.NONE:
                return self.prd._translate()
            case _VsNsp.PLUS_EPSILON:
                return self.prd._translate() + ' + epsilon'
            case _VsNsp.MINUS_EPSILON:
                return self.prd._translate() + ' - epsilon'
            case _VsNsp.PLUS_INFINITY:
                return '+inf'
            case _VsNsp.MINUS_INFINITY:
                return '-inf'
            case _:
                assert False, self


@dataclass
class _VsEliminationSet:

    variable: Variable
    test_points: list[_VsTestPoint]
    method: str

    def _translate(self, assumptions: Assumptions):
        return (self.method,
                self.variable,
                [(tp.guard(assumptions), tp._translate()) for tp in self.test_points])


@dataclass
class VsNode(Node):
    """Real linear and quadratic quantifier elimination based on [Kosta-2016]_
    """

    real_type_selection: ClassVar[dict[Clustering,
                                       dict[int, list[_VsCluster]]]] = {
        # W.l.o.g. the last sign in the first SignSequence of each tuple is always +1.
        Clustering.NONE: {
            1: [_VsCluster((_VsRootSpec(signs=(-1, 0, 1), index=1),))],
            2: [_VsCluster((_VsRootSpec(signs=(1, 0, -1, 0, 1), index=1),)),
                _VsCluster((_VsRootSpec(signs=(1, 0, -1, 0, 1), index=2),)),
                _VsCluster((_VsRootSpec(signs=(1, 0, 1), index=1),))]
        },
        Clustering.FULL: {
            1: [_VsCluster((_VsRootSpec(signs=(-1, 0, 1), index=1),
                            _VsRootSpec(signs=(1, 0, -1), index=1)))],
            2: [_VsCluster((_VsRootSpec(signs=(1, 0, -1, 0, 1), index=1),
                            _VsRootSpec(signs=(-1, 0, 1, 0, -1), index=2),
                            _VsRootSpec(signs=(1, 0, 1), index=1),
                            _VsRootSpec(signs=(-1, 0, -1), index=1)))]
        }
    }

    def eset(self, assumptions: Assumptions) -> _VsEliminationSet:
        return self.gauss_eset(assumptions) or self.regular_eset(assumptions)

    def gauss_eset(self, assumptions: Assumptions) -> Optional[_VsEliminationSet]:
        if not isinstance(self.formula, And):
            return None
        for degree in (1, 2):
            # Look for degree-Gauss with a non-zero coefficient modulo assumptions
            for round_ in (Generic.NONE, Generic.MONOMIAL, Generic.FULL):
                if round_ == Generic.MONOMIAL and not self.outermost_block:
                    break
                if round_ == Generic.MONOMIAL and self.options.generic == Generic.NONE:
                    break
                if round_ == Generic.FULL and self.options.generic == Generic.MONOMIAL:
                    break
                for x in self.variables:
                    for arg in self.formula.args:
                        if not isinstance(arg, Eq):
                            continue
                        lhs = arg.lhs
                        if lhs.degree(x) != degree:
                            # Possibly lhs.degree(x) < 0 when x does not occur
                            continue
                        a = lhs.coefficient({x: degree})
                        match round_:
                            case Generic.NONE:
                                if not is_valid(a != 0, assumptions.atoms):
                                    continue
                                self.logger().debug(f'{degree}-Gauss')
                            case Generic.MONOMIAL:
                                if len(a.monomials()) > 1:
                                    continue
                                if not set(a.vars()).isdisjoint(self.variables):
                                    continue
                                assumptions.append(a != 0)
                                self.logger().debug(f'{degree}-Gauss assuming {a != 0}')
                            case Generic.FULL:
                                if not set(a.vars()).isdisjoint(self.variables):
                                    continue
                                assumptions.append(a != 0)
                                self.logger().debug(f'{degree}-Gauss assuming {a != 0}')
                        self.variables.remove(x)
                        test_points = []
                        for cluster in self.real_type_selection[self.options.clustering][degree]:
                            for sign in (1, -1):
                                prd = _VsPRD(sign * lhs, x, cluster)
                                if prd.guard(assumptions) is not _F():
                                    test_points.append(_VsTestPoint(prd))
                        eset = _VsEliminationSet(variable=x, test_points=test_points, method='g')
                        return eset
        return None

    def is_admissible_assumption(self, atom: Ne) -> bool:
        match self.options.generic:
            case Generic.NONE:
                return False
            case Generic.MONOMIAL:
                if len(atom.lhs.monomials()) > 1:
                    return False
                if not set(atom.fvars()).isdisjoint(self.variables):
                    return False
                return True
            case Generic.FULL:
                if not set(atom.fvars()).isdisjoint(self.variables):
                    return False
                return True
            case _:
                assert False, self.options.generic

    def regular_eset(self, assumptions: Assumptions) -> _VsEliminationSet:

        def red(f: Term, x: Variable, d: int) -> Term:
            return f - f.coefficient({x: d}) * x ** d

        def at_cs(atom: AtomicFormula, x: Variable) -> set[_VsCandidateSolution]:
            """Produce the set of candidate solutions of an atomic formula.
            """
            candidate_solutions = set()
            xguard: Formula = _T()
            while (d := atom.lhs.degree(x)) > 0:
                clusters = VsNode.real_type_selection[self.options.clustering][d]
                for cluster in clusters:
                    prd = _VsPRD(atom.lhs, x, cluster, xguard)
                    (with_epsilon, tag) = cluster.bound_type(atom, x, assumptions)
                    if tag is not None:
                        cs = _VsCandidateSolution(prd, with_epsilon, tag)
                        candidate_solutions.add(cs)
                    if set(cluster) != set(- cluster):
                        prd = _VsPRD(- atom.lhs, x, cluster, xguard)
                        (with_epsilon, tag) = (- cluster).bound_type(atom, x, assumptions)
                        if tag is not None:
                            cs = _VsCandidateSolution(prd, with_epsilon, tag)
                            candidate_solutions.add(cs)
                lc = atom.lhs.coefficient({x: d})
                if self.is_admissible_assumption(lc != 0):
                    assumptions.append(lc != 0)
                    break
                atom = atom.op(red(atom.lhs, x, d), 0)
                if self.options.traditional_guards:
                    xguard = And(xguard, lc == 0)
            return candidate_solutions

        smallest_eset_size = None
        assert self.variables
        for x in self.variables:
            # We can use (with_epsilon, TAG) as a key in the future.
            candidates: dict[_VsTag, set[_VsCandidateSolution]] = {tag: set() for tag in _VsTag}
            for atom in sorted(set(self.formula.atoms())):
                assert isinstance(atom, AtomicFormula)
                assert atom.rhs == Term(0)
                match atom.lhs.degree(x):
                    case -1:
                        assert False, atom
                    case 0 | 1 | 2:
                        for candidate in at_cs(atom, x):
                            if candidate.prd.guard(assumptions) is not _F():
                                candidates[candidate.tag].add(candidate)
                    case _:
                        raise DegreeViolation(atom, x, atom.lhs.degree(x))
            num_xub = len(candidates[_VsTag.XUB])
            num_xlb = len(candidates[_VsTag.XLB])
            num_any = len(candidates[_VsTag.ANY])
            eset_size = min(num_xub, num_xlb) + num_any
            if smallest_eset_size is None or eset_size < smallest_eset_size:
                smallest_eset_size = eset_size
                best_variable = x
                best_candidates = candidates
                if num_xub < num_xlb:
                    best_inf = _VsNsp.PLUS_INFINITY
                    best_eps = _VsNsp.MINUS_EPSILON
                    best_xb = _VsTag.XUB
                else:
                    best_inf = _VsNsp.MINUS_INFINITY
                    best_eps = _VsNsp.PLUS_EPSILON
                    best_xb = _VsTag.XLB
        self.variables.remove(best_variable)
        test_points = [_VsTestPoint(nsp=best_inf)]
        for tag in (_VsTag.ANY, best_xb):
            for candidate in best_candidates[tag]:
                if candidate.with_epsilon:
                    test_points.append(_VsTestPoint(candidate.prd, best_eps))
                else:
                    test_points.append(_VsTestPoint(candidate.prd))
        eset = _VsEliminationSet(variable=best_variable, test_points=test_points, method='e')
        return eset

    def vsubs(self, eset: _VsEliminationSet, assumptions: Assumptions) -> list[VsNode]:

        def vs_at(atom: AtomicFormula, tp: _VsTestPoint, x: Variable) -> Formula:
            """Virtually substitute a test point into an atom.
            """
            match tp.nsp:
                case _VsNsp.NONE:
                    assert tp.prd is not None
                    h = pseudo_sgn_rem(atom.lhs, tp.prd, x)
                    return vs_prd_at(atom.op(h, 0), tp.prd, x)
                case _VsNsp.PLUS_EPSILON | _VsNsp.MINUS_EPSILON:
                    phi = expand_eps_at(atom, tp.nsp, x)
                    recurse = lambda atom: vs_at(atom, _VsTestPoint(tp.prd, _VsNsp.NONE), x)  # noqa E731
                    return phi.traverse(map_atoms=recurse)
                case _VsNsp.PLUS_INFINITY | _VsNsp.MINUS_INFINITY:
                    return vs_inf_at(atom, tp.nsp, x)
                case _:
                    assert False, tp.nsp

        def pseudo_sgn_rem(g: Term, prd: _VsPRD, x: Variable) -> Term:
            """Sign-corrected pseudo-remainder
            """
            f = prd.term
            if g.degree(x) < f.degree(x):
                return g
            _, h = g.pseudo_quo_rem(f, x)
            delta = g.degree(x) - f.degree(x) + 1
            if delta % 2 == 1:
                lc_signs = set(root_spec.signs[-1] for root_spec in prd.cluster)
                if len(lc_signs) == 1:
                    lc_sign = next(iter(lc_signs))
                    assert lc_sign in (-1, 1)
                    if lc_sign == -1:
                        h = - h
                else:
                    # Since there are no assumptions, we need not worry about
                    # f_lc == 0. We currently believe that otherwise the guard
                    # takes care that parametric f_lc cannot vanish.
                    f_lc = f.coefficient({x: f.degree(x)})
                    if is_valid(f_lc >= 0):
                        pass
                    elif is_valid(f_lc <= 0):
                        h = - h
                    else:
                        h *= f_lc
            # One could check for even powers of f_lc in h. Currently, the
            # simplifier takes care of this.
            return h

        def vs_prd_at(atom: AtomicFormula, prd: _VsPRD, x: Variable) -> Formula:
            """Virtually substitute a parametric root description into an atom.
            """
            return prd.vsubs(atom)

        def vs_inf_at(atom: AtomicFormula, nsp: _VsNsp, x: Variable) -> Formula:
            """Virtually substitute ±∞ into an atom
            """
            assert nsp in (_VsNsp.PLUS_INFINITY, _VsNsp.MINUS_INFINITY), nsp
            match atom:
                case Eq() | Ne():
                    return tau(atom, x)
                case Le() | Lt() | Ge() | Gt():
                    c = atom.lhs.coefficient({x: 0})
                    mu: Formula = atom.op(c, 0)
                    for e in range(1, atom.lhs.degree(x) + 1):
                        c = atom.lhs.coefficient({x: e})
                        if nsp == _VsNsp.MINUS_INFINITY and e % 2 == 1:
                            c = - c
                        mu = Or(atom.op.strict_part()(c, 0), And(Eq(c, 0), mu))
                    return mu
                case _:
                    assert False, atom

        def expand_eps_at(atom: AtomicFormula, nsp: _VsNsp, x: Variable) -> Formula:
            """Reduce virtual substitution of a parametric root description ±ε
            to virtual substitution of a parametric root description.
            """
            assert nsp in (_VsNsp.PLUS_EPSILON, _VsNsp.MINUS_EPSILON), nsp
            match atom:
                case Eq() | Ne():
                    return tau(atom, x)
                case Le() | Lt() | Ge() | Gt():
                    return nu(atom, nsp, x)
                case _:
                    assert False, atom

        def nu(atom: AtomicFormula, nsp: _VsNsp, x: Variable) -> Formula:
            """Recursion on the vanishing of derivatives
            """
            if atom.lhs.degree(x) <= 0:
                return atom
            lhs_prime = atom.lhs.derivative(x)
            if nsp == _VsNsp.MINUS_EPSILON:
                lhs_prime = - lhs_prime
            atom_strict = atom.op.strict_part()(atom.lhs, 0)
            atom_prime = atom.op(lhs_prime, 0)
            return Or(atom_strict, And(Eq(atom.lhs, 0), nu(atom_prime, nsp, x)))

        def tau(atom: AtomicFormula, x: Variable) -> Formula:
            """Virtually substitute a transcendental element into an equation
            or inequation.
            """
            args: list[AtomicFormula] = []
            match atom:
                case Eq():
                    for e in range(atom.lhs.degree(x) + 1):
                        c = atom.lhs.coefficient({x: e})
                        if c.is_zero():
                            continue
                        if c.is_constant():
                            return _F()
                        args.append(Eq(c, 0))
                    return And(*args)
                case Ne():
                    for e in range(atom.lhs.degree(x) + 1):
                        c = atom.lhs.coefficient({x: e})
                        if c.is_zero():
                            continue
                        if c.is_constant():
                            return _T()
                        args.append(Ne(c, 0))
                    return Or(*args)
                case _:
                    assert False, atom

        variables = self.variables
        x = eset.variable
        new_nodes = []
        for tp in eset.test_points:
            new_formula = self.formula.traverse(map_atoms=lambda atom: vs_at(atom, tp, x))
            # requires discussion: guard will be simplified twice
            new_formula = simplify(And(tp.guard(assumptions), new_formula),
                                   assume=assumptions.atoms)
            if new_formula is _T():
                raise abc.qe.FoundT()
            new_nodes.append(VsNode(variables=variables.copy(),
                                    formula=new_formula,
                                    answer=[],
                                    outermost_block=self.outermost_block,
                                    options=self.options,
                                    passive_list=set()))
        return new_nodes

    def process(self, assumptions: Assumptions) -> Sequence[VsNode]:
        eset = self.eset(assumptions)
        nodes = self.vsubs(eset, assumptions)
        return nodes


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
    finite_solution_set: Optional[bool] = field(default=None, init=False)

    def __ior__(self, other: Self) -> Self:
        self.equations |= other.equations
        self.lower_bounds |= other.lower_bounds
        self.upper_bounds |= other.upper_bounds
        self.finite_solution_set = (self.finite_solution_set and other.finite_solution_set)
        return self

    def __len__(self) -> int:
        return len(self.equations) + len(self.lower_bounds) + len(self.upper_bounds)

    def __post_init__(self):
        if  len(self) > 1:
            raise ValueError("illegal arguments in XoptCandidateSet()")
        if self.equations:
            self.finite_solution_set = True
        else:
            self.finite_solution_set = False

    def apply_passive_list(self, passive_list: Iterable[Term]) -> Self:
        for subset in (self.equations, self.lower_bounds, self.upper_bounds):
            # Do not mutate a collection while iterating over it
            to_remove = []
            for candidate in subset:
                if candidate in passive_list:
                    to_remove.append(candidate)
                    # TODO: count passive list hits
            for candidate in to_remove:
                subset.remove(candidate)
        return self

    def elimination_set(self) -> _XoEliminationSet:
        if len(self.upper_bounds) == 0 and len(self.lower_bounds) == 0:
            assert len(self.equations) > 0
            standard_terms = self.equations
            infinity = None
        if len(self.upper_bounds) <= len(self.lower_bounds):
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

    def candidate_set(self, x: Variable) -> _XoCandidateSet:
        """Compute a candidate set using structural elimination, which covers
        generalizations of Gaussian elimination.
        """
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

        def recurse(formula: Formula, x: Variable) -> _XoCandidateSet:
            if isinstance(formula, AtomicFormula):
                if bound_type(formula, x) is _XoBoundType.NONE:
                    return _XoCandidateSet()
                elif bound_type(formula, x) is _XoBoundType.UPPER_BOUND:
                    return _XoCandidateSet(upper_bounds={formula.lhs})
                elif bound_type(formula, x) is _XoBoundType.LOWER_BOUND:
                    return _XoCandidateSet(lower_bounds={formula.lhs})
                else:
                    assert bound_type(formula, x) is _XoBoundType.EQUATION
                    return _XoCandidateSet(equations={formula.lhs})
            elif isinstance(formula, And):
                result = _XoCandidateSet()
                for arg in formula.args:
                    candidate_set_ = recurse(arg, x)
                    if candidate_set_.finite_solution_set:
                        return candidate_set_
                    else:
                        result |= candidate_set_
                return result
            else:
                assert isinstance(formula, Or)
                result = _XoCandidateSet()
                for arg in formula.args:
                    result |= recurse(arg, x)
                return result

        return recurse(self.formula, x)

    def process(self, assumptions: Assumptions) -> Sequence[XoNode]:
        x = self.variables[0]  # for now
        variables = self.variables[1:]
        passive_list = self.passive_list
        candidate_set = self.candidate_set(x)
        if len(candidate_set) == 0:
            return [XoNode(variables=variables.copy(),
                           formula=self.formula,
                           answer=[],
                           outermost_block=self.outermost_block,
                           options=self.options,
                           passive_list=passive_list.copy())]
        self.logger().debug(f'Applying {passive_list=} to {candidate_set=}')
        candidate_set.apply_passive_list(passive_list)
        self.logger().debug(f'yields {candidate_set=}')
        if len(candidate_set) == 0:
            # Redlog returns a single Node representing F here.
            return []
        elimination_set = candidate_set.elimination_set()
        self.logger().debug(f'eliminating {x} with {elimination_set}')
        passive_list = self.passive_list
        successors = []
        for testpoint in elimination_set:
            formula = XoNode.subs_into_formula(self.formula, x, testpoint, assumptions)
            if isinstance(formula, _T):
                raise abc.qe.FoundT()
            if isinstance(formula, _F):
                continue
            substituted_passive_list_copy = XoNode.subs_into_passive_list(passive_list, x, testpoint)
            node = XoNode(variables=variables.copy(),
                          formula=formula,
                          answer=[],
                          outermost_block=self.outermost_block,
                          options=self.options,
                          passive_list=substituted_passive_list_copy)
            successors.append(node)
            if isinstance(testpoint, Term):
                passive_list.add(testpoint)
        return successors

    @staticmethod
    def subs_into_formula(formula: Formula, x: Variable, pt: Term | _XoInfinity,
                            assumptions: Assumptions) -> Formula:

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

        if isinstance(pt, Term):
            new_formula = formula.traverse(
                map_atoms=lambda atom: atom.op(atom.lhs.subs_linear_solution(x, pt), 0))
        else:
            assert isinstance(pt, _XoInfinity)
            new_formula = formula.traverse(
                map_atoms=lambda atom: subs_infinity_at(atom, x, pt))
        return simplify(new_formula, assume=assumptions.atoms)

    @staticmethod
    def subs_into_passive_list(passive_list: set[Term], x: Variable,
                               pt: Term | _XoInfinity) -> set[Term]:
        result = set()
        if isinstance(pt, Term):
            for passive_term in passive_list:
                new_term = passive_term.subs_linear_solution(x, pt)
                if new_term.is_constant():
                    # We cannot obtain zero, because such situations are
                    # filtered via the application of the passive list.
                    assert not new_term.is_zero(), f'{passive_term=}, {x=}, {pt=}'
                else:
                    result.add(new_term.primitive_part(positive=True))
        else:
            assert isinstance(pt, _XoInfinity)
            for term in passive_list:
                if term.monomial_coefficient(x) == 0:
                    result.add(term)
        return result
