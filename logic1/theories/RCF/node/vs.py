from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import ClassVar, Iterator, Literal, Optional, TypeAlias

from logic1 import abc
from logic1.firstorder import And, _F, Not, Or, _T
from logic1.theories.RCF.term import Term, Variable
from logic1.theories.RCF.atomic import AtomicFormula, Eq, Ne, Ge, Le, Gt, Lt
from logic1.theories.RCF.simplify import is_valid, simplify
from logic1.theories.RCF.types import Formula
from logic1.theories.RCF.node.base import Assumptions, Clustering, DegreeViolation, Generic, Node as BaseNode


class Nsp(Enum):
    """Non-Standard Part
    """
    NONE = auto()
    PLUS_EPSILON = auto()
    MINUS_EPSILON = auto()
    PLUS_INFINITY = auto()
    MINUS_INFINITY = auto()


class Tag(Enum):
    """Describes the bound type - upper, lower, or any.
    """
    XLB = auto()
    XUB = auto()
    ANY = auto()


SignSequence: TypeAlias = tuple[Literal[-1, 0, 1], ...]


@dataclass(frozen=True)
class RootSpec:
    """Root Specification
    """

    signs: SignSequence
    index: int

    def __neg__(self) -> RootSpec:
        return RootSpec(signs=tuple(-i for i in self.signs), index=self.index)

    def bound_type(self, atom: AtomicFormula) -> tuple[bool, Optional[Tag]]:
        """Return value None means that atom has a constant truth value
        """
        zero_index = 2 * self.index - 1
        assert self.signs[zero_index] == 0, (self, atom)
        left = self.signs[zero_index - 1]
        right = self.signs[zero_index + 1]
        assert left != 0 and right != 0, (self, atom)
        match (atom, left, right):
            case (Eq(), _, _):
                return (False, Tag.ANY)

            case (Ne(), _, _):
                return (True, Tag.ANY)

            case (Lt(), -1, -1) | (Gt(), 1, 1):
                return (True, Tag.ANY)
            case (Lt(), -1, 1) | (Gt(), 1, -1):
                return (True, Tag.XUB)
            case (Lt(), 1, -1) | (Gt(), -1, 1):
                return (True, Tag.XLB)
            case (Lt(), 1, 1) | (Gt(), -1, -1):
                return (True, None)

            case (Le(), -1, -1) | (Ge(), 1, 1):
                return (False, None)
            case (Le(), -1, 1) | (Ge(), 1, -1):
                return (False, Tag.XUB)
            case (Le(), 1, -1) | (Ge(), -1, 1):
                return (False, Tag.XLB)
            case (Le(), 1, 1) | (Ge(), -1, -1):
                return (False, Tag.ANY)

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
        D: dict[tuple[int, SignSequence], int] = {
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
class Cluster:
    """A Cluster wraps a tuple of Root Specifications.
    """
    root_specs: tuple[RootSpec, ...]

    def __neg__(self) -> Cluster:
        return Cluster(tuple(- root_spec for root_spec in self.root_specs))

    def __iter__(self) -> Iterator[RootSpec]:
        return iter(self.root_specs)

    def bound_type(self, atom: AtomicFormula, x: Variable, assumptions: Assumptions)\
            -> tuple[bool, Optional[Tag]]:
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
        elif tags == {Tag.XLB} or tags == {Tag.XLB, Tag.ANY}:
            tag = Tag.XLB
        elif tags == {Tag.XUB} or tags == {Tag.XUB, Tag.ANY}:
            tag = Tag.XUB
        else:
            tag = Tag.ANY
        return (epsilon, tag)

    def guard(self, term: Term, x: Variable) -> Formula:
        d = term.degree(x)
        match d, self:
            case 1, Cluster((RootSpec(signs=(-1, 0, 1), index=1),
                                RootSpec(signs=(1, 0, -1), index=1))):
                a = term.coefficient({x: 1})
                return a != 0
            case 2, Cluster((RootSpec(signs=(1, 0, -1, 0, 1), index=1),
                                RootSpec(signs=(-1, 0, 1, 0, -1), index=2),
                                RootSpec(signs=(1, 0, 1), index=1),
                                RootSpec(signs=(-1, 0, -1), index=1))):
                a = term.coefficient({x: 2})
                b = term.coefficient({x: 1})
                c = term.coefficient({x: 0})
                d2 = b**2 - 4 * a * c
                return And(a != 0, d2 >= 0)
            case _:
                return Or(*(root_spec.guard(term, x) for root_spec in self.root_specs))


@dataclass(frozen=True)
class PRD:
    """Parametric Root Description
    """
    term: Term
    variable: Variable
    cluster: Cluster
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
                return Not(self.ubs(atom.to_complement())).to_nnf()
            case Eq() | Lt() | Le():
                return self.ubs(atom)
            case _:
                assert False, (self, atom)

    def ubs(self, atom: AtomicFormula) -> Formula:
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
                    case (1, Eq(), Cluster((RootSpec(signs=(1, 0, -1, 0, 1), index=1),))):
                        return And(A >= 0, B == 0)
                    case (1, Eq(), Cluster((RootSpec(signs=(1, 0, -1, 0, 1), index=2),))):
                        return And(A <= 0, B == 0)
                    case (1, Eq(), Cluster((RootSpec(signs=(1, 0, 1), index=1),))):
                        return C == 0
                    case (1, Lt(), Cluster((RootSpec(signs=(1, 0, -1, 0, 1), index=1),))):
                        return Or(And(C < 0, B > 0), And(aa >= 0, Or(C < 0, B < 0)))
                    case (1, Lt(), Cluster((RootSpec(signs=(1, 0, -1, 0, 1), index=2),))):
                        return Or(And(C < 0, B > 0), And(aa <= 0, Or(C < 0, B < 0)))
                    case (1, Lt(), Cluster((RootSpec(signs=(1, 0, 1), index=1),))):
                        return C < 0
                    case (1, Le(), Cluster((RootSpec(signs=(1, 0, -1, 0, 1), index=1),))):
                        return Or(And(C <= 0, B >= 0), And(aa >= 0, B <= 0))
                    case (1, Le(), Cluster((RootSpec(signs=(1, 0, -1, 0, 1), index=2),))):
                        return Or(And(C <= 0, B >= 0), And(aa <= 0, B <= 0))
                    case (1, Le(), Cluster((RootSpec(signs=(1, 0, 1), index=1),))):
                        return C <= 0
                    # Kosta Appendix A.3: With clustering
                    case (1, Eq(), Cluster((RootSpec(signs=(1, 0, -1, 0, 1), index=1),
                                               RootSpec(signs=(-1, 0, 1, 0, -1), index=2),
                                               RootSpec(signs=(1, 0, 1), index=1),
                                               RootSpec(signs=(-1, 0, -1), index=1)))):
                        return And(A >= 0, B == 0)
                    case (1, Lt(), Cluster((RootSpec(signs=(1, 0, -1, 0, 1), index=1),
                                               RootSpec(signs=(-1, 0, 1, 0, -1), index=2),
                                               RootSpec(signs=(1, 0, 1), index=1),
                                               RootSpec(signs=(-1, 0, -1), index=1)))):
                        return Or(And(a * C < 0, a * B > 0),
                                  And(a * aa >= 0, Or(a * C < 0, a * B < 0)))
                    case (1, Le(), Cluster((RootSpec(signs=(1, 0, -1, 0, 1), index=1),
                                               RootSpec(signs=(-1, 0, 1, 0, -1), index=2),
                                               RootSpec(signs=(1, 0, 1), index=1),
                                               RootSpec(signs=(-1, 0, -1), index=1)))):
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
                    case Cluster((RootSpec(signs=(-1, 0, 1), index=1),)):
                        return f'({-c}) / ({b})'
                    # CLUSTERING.FULL
                    case Cluster((RootSpec(signs=(-1, 0, 1), index=1),
                                     RootSpec(signs=(1, 0, -1), index=1))):
                        return f'({-c}) / ({b})'
                    case _:
                        assert False, self
            case 2:
                match self.cluster:
                    # CLUSTERING.NONE
                    case Cluster((RootSpec(signs=(1, 0, -1, 0, 1), index=1),)):
                        return f'({-b} - sqrt({b**2- 4*a*c})) / ({2*a})'
                    case Cluster((RootSpec(signs=(1, 0, -1, 0, 1), index=2),)):
                        return f'({-b} + sqrt({b**2- 4*a*c})) / ({2*a})'
                    case Cluster((RootSpec(signs=(1, 0, 1), index=1),)):
                        return f'({-b} ± sqrt({0})) / ({2*a})'
                    # CLUSTERING.FULL
                    case Cluster((RootSpec(signs=(1, 0, -1, 0, 1), index=1),
                                     RootSpec(signs=(-1, 0, 1, 0, -1), index=2),
                                     RootSpec(signs=(1, 0, 1), index=1),
                                     RootSpec(signs=(-1, 0, -1), index=1))):
                        return f'({-b} - sqrt({b**2- 4*a*c})) / ({2*a})'
                    case _:
                        assert False, self
            case _:
                assert False, self


@dataclass(frozen=True)
class CandidateSolution:
    """A candidate solution combines a parametric root description with a
    flag indicating that epsilon will be needed and a bound type.

    CandidateSolutions are used as elements of sets. In order to become
    hashable, the dataclass is frozen, along with RootSpec, PRD, and RealType.
    """
    prd: PRD
    with_epsilon: bool
    tag: Tag


@dataclass
class TestPoint:
    """A test point combinines a parametric root description with an optional
    non-standard part.
    """
    prd: Optional[PRD] = None
    nsp: Nsp = Nsp.NONE

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
            case Nsp.NONE:
                return self.prd._translate()
            case Nsp.PLUS_EPSILON:
                return self.prd._translate() + ' + epsilon'
            case Nsp.MINUS_EPSILON:
                return self.prd._translate() + ' - epsilon'
            case Nsp.PLUS_INFINITY:
                return '+inf'
            case Nsp.MINUS_INFINITY:
                return '-inf'
            case _:
                assert False, self


@dataclass
class EliminationSet:

    variable: Variable
    test_points: list[TestPoint]
    method: str

    def _translate(self, assumptions: Assumptions):
        return (self.method,
                self.variable,
                [(tp.guard(assumptions), tp._translate()) for tp in self.test_points])


@dataclass
class Node(BaseNode):
    """Real linear and quadratic quantifier elimination based on [Kosta-2016]_
    """

    real_type_selection: ClassVar[dict[Clustering,
                                       dict[int, list[Cluster]]]] = {
        # W.l.o.g. the last sign in the first SignSequence of each tuple is always +1.
        Clustering.NONE: {
            1: [Cluster((RootSpec(signs=(-1, 0, 1), index=1),))],
            2: [Cluster((RootSpec(signs=(1, 0, -1, 0, 1), index=1),)),
                Cluster((RootSpec(signs=(1, 0, -1, 0, 1), index=2),)),
                Cluster((RootSpec(signs=(1, 0, 1), index=1),))]
        },
        Clustering.FULL: {
            1: [Cluster((RootSpec(signs=(-1, 0, 1), index=1),
                            RootSpec(signs=(1, 0, -1), index=1)))],
            2: [Cluster((RootSpec(signs=(1, 0, -1, 0, 1), index=1),
                            RootSpec(signs=(-1, 0, 1, 0, -1), index=2),
                            RootSpec(signs=(1, 0, 1), index=1),
                            RootSpec(signs=(-1, 0, -1), index=1)))]
        }
    }

    def eset(self, assumptions: Assumptions) -> EliminationSet:
        return self.gauss_eset(assumptions) or self.regular_eset(assumptions)

    def gauss_eset(self, assumptions: Assumptions) -> Optional[EliminationSet]:
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
                                prd = PRD(sign * lhs, x, cluster)
                                if prd.guard(assumptions) is not _F():
                                    test_points.append(TestPoint(prd))
                        eset = EliminationSet(variable=x, test_points=test_points, method='g')
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

    def regular_eset(self, assumptions: Assumptions) -> EliminationSet:

        def red(f: Term, x: Variable, d: int) -> Term:
            return f - f.coefficient({x: d}) * x ** d

        def at_cs(atom: AtomicFormula, x: Variable) -> set[CandidateSolution]:
            """Produce the set of candidate solutions of an atomic formula.
            """
            candidate_solutions = set()
            xguard: Formula = _T()
            while (d := atom.lhs.degree(x)) > 0:
                clusters = Node.real_type_selection[self.options.clustering][d]
                for cluster in clusters:
                    prd = PRD(atom.lhs, x, cluster, xguard)
                    (with_epsilon, tag) = cluster.bound_type(atom, x, assumptions)
                    if tag is not None:
                        cs = CandidateSolution(prd, with_epsilon, tag)
                        candidate_solutions.add(cs)
                    if set(cluster) != set(- cluster):
                        prd = PRD(- atom.lhs, x, cluster, xguard)
                        (with_epsilon, tag) = (- cluster).bound_type(atom, x, assumptions)
                        if tag is not None:
                            cs = CandidateSolution(prd, with_epsilon, tag)
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
            candidates: dict[Tag, set[CandidateSolution]] = {tag: set() for tag in Tag}
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
            num_xub = len(candidates[Tag.XUB])
            num_xlb = len(candidates[Tag.XLB])
            num_any = len(candidates[Tag.ANY])
            eset_size = min(num_xub, num_xlb) + num_any
            if smallest_eset_size is None or eset_size < smallest_eset_size:
                smallest_eset_size = eset_size
                best_variable = x
                best_candidates = candidates
                if num_xub < num_xlb:
                    best_inf = Nsp.PLUS_INFINITY
                    best_eps = Nsp.MINUS_EPSILON
                    best_xb = Tag.XUB
                else:
                    best_inf = Nsp.MINUS_INFINITY
                    best_eps = Nsp.PLUS_EPSILON
                    best_xb = Tag.XLB
        self.variables.remove(best_variable)
        test_points = [TestPoint(nsp=best_inf)]
        for tag in (Tag.ANY, best_xb):
            for candidate in best_candidates[tag]:
                if candidate.with_epsilon:
                    test_points.append(TestPoint(candidate.prd, best_eps))
                else:
                    test_points.append(TestPoint(candidate.prd))
        eset = EliminationSet(variable=best_variable, test_points=test_points, method='e')
        return eset

    def vsubs(self, eset: EliminationSet, assumptions: Assumptions) -> list[Node]:

        def vs_at(atom: AtomicFormula, tp: TestPoint, x: Variable) -> Formula:
            """Virtually substitute a test point into an atom.
            """
            match tp.nsp:
                case Nsp.NONE:
                    assert tp.prd is not None
                    h = pseudo_sgn_rem(atom.lhs, tp.prd, x)
                    return vs_prd_at(atom.op(h, 0), tp.prd, x)
                case Nsp.PLUS_EPSILON | Nsp.MINUS_EPSILON:
                    phi = expand_eps_at(atom, tp.nsp, x)
                    recurse = lambda atom: vs_at(atom, TestPoint(tp.prd, Nsp.NONE), x)  # noqa E731
                    return phi.traverse(map_atoms=recurse)
                case Nsp.PLUS_INFINITY | Nsp.MINUS_INFINITY:
                    return vs_inf_at(atom, tp.nsp, x)
                case _:
                    assert False, tp.nsp

        def pseudo_sgn_rem(g: Term, prd: PRD, x: Variable) -> Term:
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

        def vs_prd_at(atom: AtomicFormula, prd: PRD, x: Variable) -> Formula:
            """Virtually substitute a parametric root description into an atom.
            """
            return prd.vsubs(atom)

        def vs_inf_at(atom: AtomicFormula, nsp: Nsp, x: Variable) -> Formula:
            """Virtually substitute ±∞ into an atom
            """
            assert nsp in (Nsp.PLUS_INFINITY, Nsp.MINUS_INFINITY), nsp
            match atom:
                case Eq() | Ne():
                    return tau(atom, x)
                case Le() | Lt() | Ge() | Gt():
                    c = atom.lhs.coefficient({x: 0})
                    mu: Formula = atom.op(c, 0)
                    for e in range(1, atom.lhs.degree(x) + 1):
                        c = atom.lhs.coefficient({x: e})
                        if nsp == Nsp.MINUS_INFINITY and e % 2 == 1:
                            c = - c
                        mu = Or(atom.op.strict_part()(c, 0), And(Eq(c, 0), mu))
                    return mu
                case _:
                    assert False, atom

        def expand_eps_at(atom: AtomicFormula, nsp: Nsp, x: Variable) -> Formula:
            """Reduce virtual substitution of a parametric root description ±ε
            to virtual substitution of a parametric root description.
            """
            assert nsp in (Nsp.PLUS_EPSILON, Nsp.MINUS_EPSILON), nsp
            match atom:
                case Eq() | Ne():
                    return tau(atom, x)
                case Le() | Lt() | Ge() | Gt():
                    return nu(atom, nsp, x)
                case _:
                    assert False, atom

        def nu(atom: AtomicFormula, nsp: Nsp, x: Variable) -> Formula:
            """Recursion on the vanishing of derivatives
            """
            if atom.lhs.degree(x) <= 0:
                return atom
            lhs_prime = atom.lhs.derivative(x)
            if nsp == Nsp.MINUS_EPSILON:
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
                                   assume=assumptions.atoms,
                                   prefer_order=True,
                                   prefer_weak=True)
            if new_formula is _T():
                raise abc.qe.FoundT()
            new_nodes.append(Node(variables=variables.copy(),
                                    formula=new_formula,
                                    answer=[],
                                    outermost_block=self.outermost_block,
                                    options=self.options,
                                    passive_list=set()))
        return new_nodes

    def process(self, assumptions: Assumptions) -> Sequence[Node]:
        eset = self.eset(assumptions)
        nodes = self.vsubs(eset, assumptions)
        return nodes