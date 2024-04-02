# mypy: strict_optional = False

"""Real quantifier elimination by virtual substitution.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import auto, Enum
import logging
import multiprocessing as mp
import multiprocessing.managers
import multiprocessing.queues
import os
import queue
import sys
import threading
import time
from typing import (Callable, ClassVar, Collection, Iterable, Literal,
                    Optional, TypeAlias)

from logic1.firstorder import (
    All, And, F, _F, Formula, Not, Or, pnf, QuantifiedFormula, T)
from logic1.support.logging import DeltaTimeFormatter
from logic1.support.tracing import trace  # noqa
from logic1.theories.RCF.simplify import simplify
from logic1.theories.RCF.rcf import (
    AtomicFormula, Eq, Ne, Ge, Le, Gt, Lt, ring, Term, Variable)

# Create logger
delta_time_formatter = DeltaTimeFormatter(
    f'%(asctime)s - %(name)s - %(levelname)-5s - %(delta)s: %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(delta_time_formatter)

logger = logging.getLogger(__name__)
logger.propagate = False
logger.addHandler(stream_handler)
logger.addFilter(lambda record: record.msg.strip() != '')
logger.setLevel(logging.WARNING)

# Create multiprocessing logger
multiprocessing_formatter = DeltaTimeFormatter(
    f'%(asctime)s - %(name)s/%(process)-6d - %(levelname)-5s - %(delta)s: %(message)s')

multiprocessing_handler = logging.StreamHandler()
multiprocessing_handler.setFormatter(multiprocessing_formatter)
multiprocessing_logger = logging.getLogger('multiprocessing')
multiprocessing_logger.propagate = False
multiprocessing_logger.addHandler(multiprocessing_handler)


class DegreeViolation(Exception):
    pass


class Failed(Exception):
    pass


class FoundT(Exception):
    pass


class Timer:

    def __init__(self):
        self.reset()

    def get(self) -> float:
        return time.time() - self._reference_time

    def reset(self) -> None:
        self._reference_time = time.time()


class QuantifierBlocks(list[tuple[type[QuantifiedFormula], list[Variable]]]):

    def __str__(self) -> str:
        return '  '.join(q.__qualname__ + ' ' + str(v) for q, v in self)


class NSP(Enum):
    NONE = auto()
    PLUS_EPSILON = auto()
    MINUS_EPSILON = auto()
    PLUS_INFINITY = auto()
    MINUS_INFINITY = auto()


class TAG(Enum):
    IP = auto()
    EP = auto()
    SLB = auto()
    WLB = auto()
    SUB = auto()
    WUB = auto()


SignSequence: TypeAlias = tuple[Literal[-1, 0, 1], ...]


@dataclass(frozen=True)
class RealType:

    guards: ClassVar[dict[tuple[int, SignSequence],
                          Callable[..., Formula]]] = {
        (1, (-1, 0, 1)):
            lambda a, b: a > 0,
        (1, (1, 0, -1)):
            lambda a, b: a < 0,
        (2, (1, 0, -1, 0, 1)):
            lambda a, b, c: And(a > 0, RealType.D2(a, b, c) > 0),
        (2, (-1, 0, 1, 0, -1)):
            lambda a, b, c: And(a < 0, RealType.D2(a, b, c) > 0),
        (2, (1, 0, 1)):
            lambda a, b, c: And(a > 0, RealType.D2(a, b, c) == 0),
        (2, (-1, 0, -1)):
            lambda a, b, c: And(a < 0, RealType.D2(a, b, c) == 0)
    }

    degree: int
    signs: SignSequence

    @property
    def kosta_code(self) -> int:
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
        return D[self.degree, self.signs]

    @staticmethod
    def D2(a: Term, b: Term, c: Term) -> Term:
        return b**2 - 4 * a * c

    def guard(self, f: Term, x: Variable) -> Formula:
        d = f._degree(x)
        assert d == self.degree, (self, f, x)
        coefficients = tuple(f._coefficient({x: e}) for e in range(d, -1, -1))
        try:
            return RealType.guards[self.degree, self.signs](*coefficients)
        except KeyError:
            raise NotImplementedError(f'{self}')


@dataclass(frozen=True)
class RootSpec:

    real_type: RealType
    index: int


@dataclass(unsafe_hash=True)
class PRD:
    """Parametric Root Description"""

    term: Term
    variable: Variable
    roots: tuple[RootSpec]
    guard: Formula

    def __init__(self, term: Term, variable: Variable, roots: tuple[RootSpec]):
        self.term = term
        self.variable = variable
        self.roots = roots
        assert len(self.roots) == 1  # no clustering
        self.guard = simplify(self.roots[0].real_type.guard(term, variable))


@dataclass(frozen=True)
class CandidateSolution:
    # CandidateSolutions are used as elements of sets. In order to become
    # hashable, the dataclass is frozen, along with RootSpec, PRD, and
    # RealType.

    prd: PRD
    tag: TAG


@dataclass
class TestPoint:

    prd: Optional[PRD] = None
    nsp: NSP = NSP.NONE

    @property
    def guard(self):
        if self.prd is None:
            return T
        else:
            assert self.prd.guard is not F, self
            return self.prd.guard


@dataclass
class EliminationSet:

    variable: Variable
    test_points: list[TestPoint]
    method: str


@dataclass
class Node:

    variables: list[Variable]
    formula: Formula
    answer: list

    def __str__(self):
        return f'Node({self.variables}, {self.formula}, {self.answer})'

    def eset(self) -> EliminationSet:
        return self.gauss_eset() or self.regular_eset()

    def gauss_eset(self) -> Optional[EliminationSet]:
        if isinstance(self.formula, And):
            for x in self.variables:
                for arg in self.formula.args:
                    if isinstance(arg, Eq):
                        lhs = arg.lhs
                        if lhs._degree(x) == 1:
                            a = lhs._coefficient({x: 1})
                            if a._is_constant():
                                self.variables.remove(x)
                                if a.poly > 0:
                                    root_spec = RootSpec(RealType(1, (-1, 0, 1)), 1)
                                else:
                                    assert a.poly < 0
                                    root_spec = RootSpec(RealType(1, (1, 0, -1)), 1)
                                tp = TestPoint(PRD(lhs, x, (root_spec,)))
                                return EliminationSet(variable=x, test_points=[tp], method='g')
        return None

    def regular_eset(self) -> EliminationSet:

        def at_cs_1(atom: AtomicFormula, x: Variable) -> set[CandidateSolution]:
            """Produce the set of candidate solutions of an atomic formula of
            degree one.
            """
            prd_1 = PRD(atom.lhs, x, (RootSpec(RealType(1, (-1, 0, 1)), 1),))
            prd_2 = PRD(atom.lhs, x, (RootSpec(RealType(1, (1, 0, -1)), 1),))
            match atom:
                case Eq():
                    cs1 = CandidateSolution(prd_1, TAG.IP)
                    cs2 = CandidateSolution(prd_2, TAG.IP)
                    return {cs1, cs2}
                case Ne():
                    cs1 = CandidateSolution(prd_1, TAG.EP)
                    cs2 = CandidateSolution(prd_2, TAG.EP)
                    return {cs1, cs2}
                case Le():
                    cs1 = CandidateSolution(prd_1, TAG.WUB)
                    cs2 = CandidateSolution(prd_2, TAG.WLB)
                    return {cs1, cs2}
                case Ge():
                    cs1 = CandidateSolution(prd_1, TAG.WLB)
                    cs2 = CandidateSolution(prd_2, TAG.WUB)
                    return {cs1, cs2}
                case Lt():
                    cs1 = CandidateSolution(prd_1, TAG.SUB)
                    cs2 = CandidateSolution(prd_2, TAG.SLB)
                    return {cs1, cs2}
                case Gt():
                    cs1 = CandidateSolution(prd_1, TAG.SLB)
                    cs2 = CandidateSolution(prd_2, TAG.SUB)
                    return {cs1, cs2}
                case _:
                    assert False, atom

        smallest_eset_size = None
        assert self.variables
        for x in self.variables:
            candidates: dict[TAG, set[CandidateSolution]] = {tag: set() for tag in TAG}
            for atom in sorted(set(self.formula.atoms())):
                assert isinstance(atom, AtomicFormula)
                assert atom.rhs == Term(0)
                match atom.lhs._degree(x):
                    case -1:
                        assert False, atom
                    case 0:
                        # lhs can still be a polynomial in other variables.
                        continue
                    case 1:
                        for candidate in at_cs_1(atom, x):
                            if candidate.prd.guard is not F:
                                candidates[candidate.tag].add(candidate)
                    case _:
                        raise DegreeViolation(atom, x, atom.lhs._degree(x))
            num_xub = len(candidates[TAG.WUB]) + len(candidates[TAG.SUB])
            num_xlb = len(candidates[TAG.WLB]) + len(candidates[TAG.SLB])
            num_xp = len(candidates[TAG.IP]) + len(candidates[TAG.EP])
            eset_size = min(num_xub, num_xlb) + num_xp
            if smallest_eset_size is None or eset_size < smallest_eset_size:
                smallest_eset_size = eset_size
                best_variable = x
                best_candidates = candidates
                if num_xub < num_xlb:
                    best_inf, best_eps, best_wb, best_sb = (
                        NSP.PLUS_INFINITY, NSP.MINUS_EPSILON, TAG.WUB, TAG.SUB)
                else:
                    best_inf, best_eps, best_wb, best_sb = (
                        NSP.MINUS_INFINITY, NSP.PLUS_EPSILON, TAG.WLB, TAG.SLB)
        self.variables.remove(best_variable)
        test_points = [TestPoint(nsp=best_inf)]
        for tag in (TAG.IP, best_wb):
            for candidate in best_candidates[tag]:
                test_points.append(TestPoint(candidate.prd))
        for tag in (TAG.EP, best_sb):
            for candidate in best_candidates[tag]:
                test_points.append(TestPoint(candidate.prd, best_eps))
        return EliminationSet(variable=best_variable, test_points=test_points, method='e')

    def vsubs(self, eset: EliminationSet) -> list[Node]:

        def vs_at(atom: AtomicFormula, tp: TestPoint, x: Variable) -> Formula:
            """Virtually substitute a test point into an atom.
            """
            match tp.nsp:
                case NSP.NONE:
                    h = pseudo_sgn_rem(atom.lhs, tp.prd, x)
                    return vs_prd_at(atom.func(h, 0), tp.prd, x)
                case NSP.PLUS_EPSILON | NSP.MINUS_EPSILON:
                    phi = expand_eps_at(atom, tp.nsp, x)
                    recurse = lambda atom: vs_at(atom, TestPoint(tp.prd, NSP.NONE), x)  # noqa E731
                    return phi.transform_atoms(recurse)
                case NSP.PLUS_INFINITY | NSP.MINUS_INFINITY:
                    return vs_inf_at(atom, tp.nsp, x)
                case _:
                    assert False, tp.nsp

        def pseudo_sgn_rem(g: Term, prd: PRD, x: Variable) -> Term:
            """Sign-corrected pseudo-remainder
            """
            g1 = g.poly.polynomial(x.poly)
            f1 = prd.term.poly.polynomial(x.poly)
            _, h = g1.pseudo_quo_rem(f1)
            delta = g1.degree() - f1.degree() + 1
            if delta % 2 == 1:
                first_signs = set(root.real_type.signs[0] for root in prd.roots)
                if len(first_signs) == 1:
                    first_sign = next(iter(first_signs))
                    assert first_sign in (-1, 1)
                    if first_sign == 1:
                        h = - h
                else:
                    h *= f1.lc()
            # One could check for even powers of f1.lc() in h. Currently the
            # simplifier takes care of this.
            return Term(h)

        def vs_prd_at(atom: AtomicFormula, prd: PRD, x: Variable) -> Formula:
            """Virtually substitute a parametric root description into an atom.
            """
            match prd.term._degree(x):
                case -1 | 0:
                    assert False, prd
                case 1:
                    return atom
                case _:
                    raise NotImplementedError()

        def vs_inf_at(atom: AtomicFormula, nsp: NSP, x: Variable) -> Formula:
            """Virtually substitute ±∞ into an atom
            """
            assert nsp in (NSP.PLUS_INFINITY, NSP.MINUS_INFINITY), nsp
            match atom:
                case Eq() | Ne():
                    return tau(atom, x)
                case Le() | Lt() | Ge() | Gt():
                    c = atom.lhs._coefficient({x: 0})
                    mu: Formula = atom.func(c, 0)
                    for e in range(1, atom.lhs._degree(x) + 1):
                        c = atom.lhs._coefficient({x: e})
                        if nsp == NSP.MINUS_INFINITY and e % 2 == 1:
                            c = - c
                        strict_func = atom.func.strict_part
                        mu = Or(strict_func(c, 0), And(Eq(c, 0), mu))
                    return mu
                case _:
                    assert False, atom

        def expand_eps_at(atom: AtomicFormula, nsp: NSP, x: Variable) -> Formula:
            """Reduce virtual substitution of a parametric root description ±ε
            to virtual substituion of a parametric root description.
            """
            assert nsp in (NSP.PLUS_EPSILON, NSP.MINUS_EPSILON), nsp
            match atom:
                case Eq() | Ne():
                    return tau(atom, x)
                case Le() | Lt() | Ge() | Gt():
                    return nu(atom, nsp, x)
                case _:
                    assert False, atom

        def nu(atom: AtomicFormula, nsp: NSP, x: Variable) -> Formula:
            """Recursion on the vanishing of derivatives
            """
            if atom.lhs._degree(x) <= 0:
                return atom
            lhs_prime = atom.lhs._derivative(x)
            if nsp == NSP.MINUS_EPSILON:
                lhs_prime = - lhs_prime
            atom_strict = atom.func.strict_part(atom.lhs, 0)
            atom_prime = atom.func(lhs_prime, 0)
            return Or(atom_strict, And(Eq(atom.lhs, 0), nu(atom_prime, nsp, x)))

        def tau(atom: AtomicFormula, x: Variable) -> Formula:
            """Virtually substitute a transcendental element into an equation
            or inequation.
            """
            args: list[AtomicFormula] = []
            match atom:
                case Eq():
                    for e in range(atom.lhs._degree(x) + 1):
                        c = atom.lhs._coefficient({x: e})
                        if c._is_zero():
                            continue
                        if c._is_constant():
                            return F
                        args.append(Eq(c, 0))
                    return And(*args)
                case Ne():
                    for e in range(atom.lhs._degree(x) + 1):
                        c = atom.lhs._coefficient({x: e})
                        if c._is_zero():
                            continue
                        if c._is_constant():
                            return T
                        args.append(Ne(c, 0))
                    return Or(*args)
                case _:
                    assert False, atom

        variables = self.variables
        x = eset.variable
        new_nodes = []
        for tp in eset.test_points:
            new_formula = self.formula.transform_atoms(lambda atom: vs_at(atom, tp, x))
            if tp.guard is not T:
                new_formula = And(tp.guard, new_formula)
            new_formula = simplify(new_formula)
            if new_formula is T:
                raise FoundT()
            new_nodes.append(Node(variables.copy(), new_formula, []))
        return new_nodes


@dataclass
class NodeList(Collection):

    nodes: list[Node] = field(default_factory=list)
    memory: set[Formula] = field(default_factory=set)
    hits: int = 0
    candidates: int = 0

    def __contains__(self, obj: object) -> bool:
        return obj in self.nodes

    def __iter__(self):
        yield from self.nodes

    def __len__(self) -> int:
        return len(self.nodes)

    def append(self, node: Node) -> bool:
        is_new = node.formula not in self.memory
        if is_new:
            self.nodes.append(node)
            self.memory.add(node.formula)
        else:
            self.hits += 1
        self.candidates += 1
        return is_new

    def extend(self, nodes: Iterable[Node]) -> None:
        for node in nodes:
            self.append(node)

    def final_statistics(self, key: str) -> str:
        hits = self.hits
        candidates = self.candidates
        num_nodes = candidates - hits
        if num_nodes == 0:
            return ''
        ratio = self.hit_ratio()
        return (f'produced {num_nodes} {key} nodes, '
                f'dropped {hits}/{candidates} = {ratio:.0%}')

    def hit_ratio(self) -> float:
        try:
            return float(self.hits) / self.candidates
        except ZeroDivisionError:
            return float('nan')

    def periodic_statistics(self, key: str) -> str:
        num_nodes = self.candidates - self.hits
        if num_nodes == 0:
            return ''
        ratio = self.hit_ratio()
        return f'{key}={num_nodes}, H={ratio:.0%}'


@dataclass
class WorkingNodeList(NodeList):

    node_counter: Counter[int] = field(default_factory=Counter)

    def append(self, node: Node) -> bool:
        is_new = super().append(node)
        if is_new:
            n = len(node.variables)
            self.node_counter[n] += 1
        return is_new

    def final_statistics(self, key: Optional[str] = None) -> str:
        if key:
            return super().final_statistics(key)
        hits = self.hits
        candidates = self.candidates
        num_nodes = candidates - hits
        ratio = self.hit_ratio()
        return (f'performed {num_nodes} elimination steps, '
                f'skipped {hits}/{candidates} = {ratio:.0%}')

    def periodic_statistics(self, key: str = 'W') -> str:
        node_counter = self.node_counter
        ratio = self.hit_ratio()
        try:
            num_variables = max(k for k, v in node_counter.items() if v != 0)
            v = f'V={num_variables}'
            nc = '.'.join(f'{node_counter[n]}'
                          for n in reversed(range(1, num_variables + 1)))
            w = f'{key}={nc}'
            vw = f'{v}, {w}'
        except ValueError:
            vw = 'V=0'
        return f'{vw}, H={ratio:.0%}'

    def pop(self) -> Node:
        node = self.nodes.pop()
        n = len(node.variables)
        self.node_counter[n] -= 1
        return node

    def extend(self, nodes: Iterable[Node]) -> None:
        for node in nodes:
            match node.formula:
                case _F():
                    continue
                case Or(args=args):
                    for arg in args:
                        subnode = Node(node.variables.copy(), arg, node.answer)
                        self.append(subnode)
                case And() | AtomicFormula():
                    self.append(node)
                case _:
                    assert False, node


@dataclass
class NodeListManager:

    nodes: list[Node] = field(default_factory=list)
    nodes_lock: threading.Lock = field(default_factory=threading.Lock)
    memory: set[Formula] = field(default_factory=set)
    memory_lock: threading.Lock = field(default_factory=threading.Lock)
    hits: int = 0
    hits_candidates_lock: threading.Lock = field(default_factory=threading.Lock)
    candidates: int = 0

    def get_nodes(self) -> list[Node]:
        with self.nodes_lock:
            return self.nodes.copy()

    def get_memory(self) -> set[Formula]:
        with self.memory_lock:
            return self.memory.copy()

    def get_candidates(self) -> int:
        return self.candidates

    def get_hits(self) -> int:
        return self.hits

    def __len__(self):
        with self.nodes_lock:
            return len(self.nodes)

    def append(self, node: Node) -> bool:
        with self.memory_lock:
            is_new = node.formula not in self.memory
            if is_new:
                self.memory.add(node.formula)
        if is_new:
            with self.nodes_lock:
                self.nodes.append(node)
        with self.hits_candidates_lock:
            if not is_new:
                self.hits += 1
            self.candidates += 1
        return is_new

    def extend(self, nodes: Iterable[Node]) -> None:
        for node in nodes:
            self.append(node)

    def statistics(self) -> tuple:
        with self.hits_candidates_lock:
            return (self.hits, self.candidates)


class NodeListProxy(mp.managers.BaseProxy, Collection):

    @property
    def nodes(self):
        return self._callmethod('get_nodes')

    @property
    def memory(self):
        return self._callmethod('get_memory')

    @property
    def candidates(self):
        return self._callmethod('get_candidates')

    @property
    def hits(self):
        return self._callmethod('get_hits')

    def __contains__(self, obj: object) -> bool:
        match obj:
            case Node():
                return obj in self._callmethod('get_nodes')  # type: ignore
            case _:
                return False

    def __iter__(self):
        yield from self._callmethod('get_nodes')

    def __len__(self):
        return self._callmethod('__len__')

    def append(self, node: Node) -> bool:
        return self._callmethod('append', (node,))  # type: ignore

    def extend(self, nodes: list[Node]) -> None:
        self._callmethod('extend', (nodes,))

    def final_statistics(self, key: str) -> str:
        hits, candidates = self._callmethod('statistics')  # type: ignore
        num_nodes = candidates - hits
        if num_nodes == 0:
            return ''
        ratio = self.hit_ratio(hits, candidates)
        return (f'produced {num_nodes} {key} nodes, '
                f'dropped {hits}/{candidates} = {ratio:.0%}')

    def hit_ratio(self, hits, candidates) -> float:
        try:
            return float(hits) / candidates
        except ZeroDivisionError:
            return float('nan')

    def periodic_statistics(self, key: str) -> str:
        hits, candidates = self._callmethod('statistics')  # type: ignore
        num_nodes = candidates - hits
        if num_nodes == 0:
            return ''
        ratio = self.hit_ratio(hits, candidates)
        return f'{key}={num_nodes}, H={ratio:.0%}'


@dataclass
class WorkingNodeListManager(NodeListManager):

    busy: int = 0
    busy_lock: threading.Lock = field(default_factory=threading.Lock)
    node_counter: Counter[int] = field(default_factory=Counter)
    node_counter_lock: threading.Lock = field(default_factory=threading.Lock)

    def get_node_counter(self):
        with self.node_counter_lock:
            return self.node_counter.copy()

    def append(self, node: Node) -> bool:
        is_new = super().append(node)
        if is_new:
            n = len(node.variables)
            with self.node_counter_lock:
                self.node_counter[n] += 1
        return is_new

    def is_finished(self) -> bool:
        with self.nodes_lock, self.busy_lock:
            return len(self.nodes) == 0 and self.busy == 0

    def statistics(self) -> tuple:
        # hits and candidates are always consistent. hits/candidates, busy,
        # node_counter are three snapshots at different times, each of which is
        # consistent.
        with self.hits_candidates_lock, self.busy_lock, self.node_counter_lock:
            return (self.hits, self.candidates, self.busy, self.node_counter)

    def pop(self) -> Node:
        with self.nodes_lock:
            node = self.nodes.pop()
        n = len(node.variables)
        with self.node_counter_lock:
            self.node_counter[n] -= 1
        with self.busy_lock:
            self.busy += 1
        return node

    def task_done(self) -> None:
        with self.busy_lock:
            self.busy -= 1


class WorkingNodeListProxy(NodeListProxy):

    @property
    def node_counter(self):
        return self._callmethod('get_node_counter')

    def extend(self, nodes: Iterable[Node]) -> None:
        newnodes = []
        for node in nodes:
            match node.formula:
                case _F():
                    continue
                case Or(args=args):
                    for arg in args:
                        subnode = Node(node.variables.copy(), arg, node.answer)
                        if subnode not in newnodes:
                            newnodes.append(subnode)
                case And() | AtomicFormula():
                    if node not in newnodes:
                        newnodes.append(node)
                case _:
                    assert False, node
        self._callmethod('extend', (newnodes,))

    def final_statistics(self, key: Optional[str] = None) -> str:
        if key:
            return super().final_statistics(key)
        hits, candidates, _, _ = self._callmethod('statistics')  # type: ignore
        num_nodes = candidates - hits
        ratio = self.hit_ratio(hits, candidates)
        return (f'performed {num_nodes} elimination steps, '
                f'skipped {hits}/{candidates} = {ratio:.0%}')

    def is_finished(self):
        return self._callmethod('is_finished')

    def periodic_statistics(self, key: str = 'W') -> str:
        hits, candidates, busy, node_counter = self._callmethod('statistics')  # type: ignore
        ratio = self.hit_ratio(hits, candidates)
        try:
            num_variables = max(k for k, v in node_counter.items() if v != 0)
            v = f'V={num_variables}'
            nc = '.'.join(f'{node_counter[n]}'
                          for n in reversed(range(1, num_variables + 1)))
            w = f'{key}={nc}'
            vw = f'{v}, {w}'
        except ValueError:
            vw = 'V=0'
        return f'{vw}, B={busy}, H={ratio:.0%}'

    def pop(self) -> Node:
        return self._callmethod('pop')  # type: ignore

    def task_done(self) -> None:
        self._callmethod('task_done')


class SyncManager(mp.managers.SyncManager):
    pass


SyncManager.register("NodeList", NodeListManager, NodeListProxy,
                     ['get_nodes', 'get_memory', 'get_candidates', 'get_hits',
                      '__len__', 'append', 'extend', 'statistics'])

SyncManager.register("WorkingNodeList", WorkingNodeListManager, WorkingNodeListProxy,
                     ['get_nodes', 'get_memory', 'get_candidates', 'get_hits',
                      'get_node_counter', '__len__', 'append', 'extend',
                      'is_finished', 'pop', 'statistics', 'task_done'])


@dataclass
class VirtualSubstitution:
    """Quantifier elimination by virtual substitution.
    """

    blocks: Optional[QuantifierBlocks] = None
    matrix: Optional[Formula] = None
    negated: Optional[bool] = None
    root_node: Optional[Node] = None
    working_nodes: Optional[WorkingNodeList] = None
    success_nodes: Optional[NodeList] = None
    failure_nodes: Optional[NodeList] = None
    result: Optional[Formula] = None

    workers: int = 0
    log: int = logging.NOTSET
    log_rate: float = 0.5

    time_final_simplification: Optional[float] = None
    time_import_failure_nodes: Optional[float] = None
    time_import_success_nodes: Optional[float] = None
    time_import_working_nodes: Optional[float] = None
    time_multiprocessing: Optional[float] = None
    time_start_first_worker: Optional[float] = None
    time_start_all_workers: Optional[float] = None
    time_syncmanager_enter: Optional[float] = None
    time_syncmanager_exit: Optional[float] = None
    time_total: Optional[float] = None

    def __call__(self, f: Formula, workers: int = 0, log: int = logging.NOTSET,
                 log_rate: float = 0.5) -> Optional[Formula]:
        """Virtual substitution entry point.

        workers is the number of processes used for virtual substitution. The
        default value workers=0 uses a sequential implementation, avoiding
        overhead when input problems are small. For all other values of
        workers, there are additional processes started. In particular,
        workers=1 uses the parallel implementation with one worker process,
        which is interesting mostly for testing and debugging. A negative value
        workers = z < 0 selects os.num_cpu() - abs(z) many workers. When there
        are n > 0 many workers selected, then the overall number of processes
        running will be n + 2, i.e., the workers plus the master process plus a
        manager processes providing proxy objects for shared data. It follows
        that workers=-2 matches the number of CPUs of the machine. For hard
        computations, workers=-3 is an interesting choice, which leaves one CPU
        free for smooth interaction with the machine.
        """
        timer = Timer()
        VirtualSubstitution.__init__(self)
        try:
            self.log_level = log
            self.log_rate = log_rate
            save_level = logger.getEffectiveLevel()
            logger.setLevel(self.log_level)
            delta_time_formatter.set_reference_time(time.time())
            result = self.virtual_substitution(f, workers)
        except KeyboardInterrupt:
            print('KeyboardInterrupt', file=sys.stderr, flush=True)
            return None
        finally:
            logger.info('finished')
            logger.setLevel(save_level)
        self.time_total = timer.get()
        return result

    def collect_success_nodes(self) -> None:
        logger.debug(f'entering {self.collect_success_nodes.__name__}')
        self.matrix = Or(*(node.formula for node in self.success_nodes))
        if self.negated:
            self.matrix = Not(self.matrix)
        self.negated = None
        self.working_nodes = None
        self.success_nodes = None

    def final_simplification(self):
        logger.debug(f'entering {self.final_simplification.__name__}')
        if logger.isEnabledFor(logging.DEBUG):
            num_atoms = sum(1 for _ in self.matrix.atoms())
            logger.debug(f'found {num_atoms} atoms')
        logger.info('final simplification')
        timer = Timer()
        self.result = simplify(self.matrix)
        self.time_final_simplification = timer.get()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'{self.time_final_simplification=:.3f}')
            num_atoms = sum(1 for _ in self.result.atoms())
            logger.debug(f'produced {num_atoms} atoms')

    def parallel_process_block(self) -> None:

        def wait_for_processes_to_finish():
            still_running = sentinels.copy()
            while still_running:
                for sentinel in mp.connection.wait(still_running):
                    still_running.remove(sentinel)
                num_finished = self.workers - len(still_running)
                pl = 'es' if num_finished > 1 else ''
                logger.debug(f'{num_finished} worker process{pl} finished, '
                             f'{len(still_running)} running')
            # The following call joins all finished processes as a side effect.
            # Otherwise they would remain in the process table as zombies.
            mp.active_children()

        logger.debug('entering sync manager context')
        timer = Timer()
        with SyncManager() as manager:
            self.time_syncmanager_enter = timer.get()
            logger.debug(f'{self.time_syncmanager_enter=:.3f}')
            m_lock = manager.Lock()
            working_nodes = manager.WorkingNodeList()  # type: ignore
            working_nodes.append(self.root_node)
            self.root_node = None
            success_nodes: multiprocessing.Queue[Optional[list[Node]]] = multiprocessing.Queue()
            self.success_nodes = NodeList()
            failure_nodes = manager.NodeList()  # type: ignore
            found_t = manager.Value('i', 0)
            processes: list[Optional[mp.Process]] = [None] * self.workers
            sentinels: list[Optional[int]] = [None] * self.workers
            # We pass the ring variables to the workers. The workers
            # reconstruct the ring. This is not really necessary at the moment.
            # However, future introduction of variables by workers will cause
            # problems, and expect  reconstruction of the ring to be part of
            # the solution.
            ring_vars = tuple(str(v) for v in ring.get_vars())
            log_level = logger.getEffectiveLevel()
            reference_time = delta_time_formatter.get_reference_time()
            logger.debug(f'starting worker processes in {range(self.workers)}')
            born_processes = manager.Value('i', 0)
            timer.reset()
            for i in range(self.workers):
                processes[i] = mp.Process(
                    target=self.parallel_process_block_worker,
                    args=(working_nodes, success_nodes, failure_nodes, m_lock,
                          found_t, ring_vars, i, log_level, reference_time,
                          born_processes))
                processes[i].start()
                sentinels[i] = processes[i].sentinel
            try:
                born = 0
                while born < 1:
                    time.sleep(0.0001)
                    with m_lock:
                        born = born_processes.value
                self.time_start_first_worker = timer.get()
                logger.debug(f'{self.time_start_first_worker=:.3f}')
                if self.workers > 1:
                    while born < self.workers:
                        time.sleep(0.0001)
                        with m_lock:
                            born = born_processes.value
                    self.time_start_all_workers = timer.get()
                else:
                    self.time_start_all_workers = self.time_start_first_worker
                logger.debug(f'{self.time_start_all_workers=:.3f}')
                workers_running = self.workers
                log_timer = Timer()
                while workers_running > 0:
                    if logger.isEnabledFor(logging.INFO):
                        t = log_timer.get()
                        if t >= self.log_rate:
                            logger.info(working_nodes.periodic_statistics())
                            logger.info(self.success_nodes.periodic_statistics('S'))
                            logger.info(failure_nodes.periodic_statistics('F'))
                            log_timer.reset()
                    try:
                        nodes = success_nodes.get(timeout=0.001)
                    except queue.Empty:
                        continue
                    if nodes is not None:
                        self.success_nodes.extend(nodes)
                    else:
                        workers_running -= 1
            except KeyboardInterrupt:
                logger.debug('KeyboardInterrupt, waiting for processes to finish')
                wait_for_processes_to_finish()
                raise
            wait_for_processes_to_finish()
            self.time_multiprocessing = timer.get() - self.time_start_first_worker
            logger.debug(f'{self.time_multiprocessing=:.3f}')
            if found_t.value > 0:
                pl = 's' if found_t.value > 1 else ''
                logger.debug(f'{found_t.value} worker{pl} found T')
                # The exception handler for FoundT in virtual_substitution will
                # log final statistics. We do not retrieve nodes and memory,
                # which would cost significant time and space. We neither
                # retreive the node_counter, which would be not consistent with
                # our empty nodes.
                self.working_nodes = WorkingNodeList(
                    hits=working_nodes.hits,
                    candidates=working_nodes.candidates)
                # TODO: wipe self.success_nodes and self.failure_nodes
                raise FoundT()
            logger.info(working_nodes.final_statistics())
            logger.info(self.success_nodes.final_statistics('success'))
            logger.info(failure_nodes.final_statistics('failure'))
            logger.info('importing results from mananager')
            logger.debug('importing working nodes from mananager')
            # We do not retrieve the memory, which would cost significant time
            # and space. Same for success nodes and failure nodes below.
            timer.reset()
            self.working_nodes = WorkingNodeList(
                nodes=working_nodes.nodes,
                hits=working_nodes.hits,
                candidates=working_nodes.candidates,
                node_counter=working_nodes.node_counter)
            self.time_import_working_nodes = timer.get()
            logger.debug(f'{self.time_import_working_nodes=:.3f}')
            assert self.working_nodes.nodes == []
            assert self.working_nodes.node_counter.total() == 0
            logger.debug('importing failure nodes from mananager')
            timer.reset()
            self.failure_nodes = NodeList(nodes=failure_nodes.nodes,
                                          hits=failure_nodes.hits,
                                          candidates=failure_nodes.candidates)
            self.time_import_failure_nodes = timer.get()
            logger.debug(f'{self.time_import_failure_nodes=:.3f}')
            logger.debug('leaving sync manager context')
            timer.reset()
        self.time_syncmanager_exit = timer.get()
        logger.debug(f'{self.time_syncmanager_exit=:.3f}')

    @staticmethod
    def parallel_process_block_worker(working_nodes: WorkingNodeListProxy,
                                      success_nodes: multiprocessing.Queue[Optional[list[Node]]],
                                      failure_nodes: NodeListProxy,
                                      m_lock: threading.Lock,
                                      found_t: mp.sharedctypes.Synchronized,
                                      ring_vars: list[str],
                                      i: int,
                                      log_level: int,
                                      reference_time: float,
                                      born_processes: mp.sharedctypes.Synchronized) -> None:
        try:
            with m_lock:
                born_processes.value += 1
            multiprocessing_logger.setLevel(log_level)
            multiprocessing_formatter.set_reference_time(reference_time)
            multiprocessing_logger.debug(f'worker process {i} is running')
            ring.ensure_vars(ring_vars)
            while found_t.value == 0 and not working_nodes.is_finished():
                try:
                    node = working_nodes.pop()
                except IndexError:
                    time.sleep(0.001)
                    continue
                try:
                    eset = node.eset()
                except DegreeViolation:
                    failure_nodes.append(node)
                    working_nodes.task_done()
                    continue
                try:
                    nodes = node.vsubs(eset)
                except FoundT:
                    with m_lock:
                        found_t.value += 1
                    break
                if nodes[0].variables:
                    working_nodes.extend(nodes)
                else:
                    success_nodes.put(nodes)
                working_nodes.task_done()
        except KeyboardInterrupt:
            multiprocessing_logger.debug(f'worker process {i} caught KeyboardInterrupt')
        success_nodes.put(None)
        multiprocessing_logger.debug(f'worker process {i} exiting')

    def pop_block(self) -> None:
        logger.debug(f'entering {self.pop_block.__name__}')
        assert self.matrix is not None, self.matrix  # discuss
        logger.info(str(self.blocks))
        if logger.isEnabledFor(logging.DEBUG):
            s = str(self.matrix)
            logger.debug(s[:50] + '...' if len(s) > 53 else s)
        quantifier, vars_ = self.blocks.pop()
        matrix = self.matrix
        self.matrix = None
        if quantifier is All:
            self.negated = True
            matrix = Not(matrix)
        else:
            self.negated = False
        self.root_node = Node(vars_, simplify(matrix), [])

    def process_block(self) -> None:
        logger.debug(f'entering {self.process_block.__name__}')
        if self.workers > 0:
            return self.parallel_process_block()
        return self.sequential_process_block()

    def sequential_process_block(self) -> None:
        self.working_nodes = WorkingNodeList()
        self.working_nodes.append(self.root_node)
        self.root_node = None
        self.success_nodes = NodeList()
        self.failure_nodes = NodeList()
        if logger.isEnabledFor(logging.INFO):
            last_log = time.time()
        while self.working_nodes.nodes:
            if logger.isEnabledFor(logging.INFO):
                t = time.time()
                if t - last_log >= self.log_rate:
                    logger.info(self.working_nodes.periodic_statistics())
                    logger.info(self.success_nodes.periodic_statistics('S'))
                    logger.info(self.failure_nodes.periodic_statistics('F'))
                    last_log = t
            node = self.working_nodes.pop()
            try:
                eset = node.eset()
            except DegreeViolation:
                self.failure_nodes.append(node)
                continue
            nodes = node.vsubs(eset)
            if nodes[0].variables:
                self.working_nodes.extend(nodes)
            else:
                self.success_nodes.extend(nodes)
        logger.info(self.working_nodes.final_statistics())
        logger.info(self.success_nodes.final_statistics('success'))
        logger.info(self.failure_nodes.final_statistics('failure'))

    def setup(self, f: Formula, workers: int) -> None:
        logger.debug(f'entering {self.setup.__name__}')
        if workers >= 0:
            self.workers = workers
        else:
            self.workers = os.cpu_count() + workers
        f = pnf(f)
        self.matrix, blocks = f.matrix()
        self.blocks = QuantifierBlocks(blocks)

    def status(self, dump_nodes: bool = False) -> str:

        def negated_as_str() -> str:
            match self.negated:
                case None:
                    read_as = ''
                case False:
                    read_as = '  # read as Ex'
                case True:
                    read_as = '  # read as Not All'
                case _:
                    assert False, self.negated
            return f'{self.negated},{read_as}'

        def nodes_as_str(nodes: Optional[Collection[Node]]) -> Optional[str]:
            if nodes is None:
                return None
            match dump_nodes:
                case True:
                    h = ',\n                '.join(f'{node}' for node in nodes)
                    return f'[{h}]'
                case False:
                    return f'{len(nodes)}'

        return (f'{self.__class__.__qualname__}(\n'
                f'    blocks        = {self.blocks},\n'
                f'    matrix        = {self.matrix},\n'
                f'    negated       = {negated_as_str()}\n'
                f'    root_node     = {self.root_node}\n'
                f'    working_nodes = {nodes_as_str(self.working_nodes)},\n'
                f'    success_nodes = {nodes_as_str(self.success_nodes)},\n'
                f'    failure_nodes = {nodes_as_str(self.failure_nodes)},\n'
                f'    result        = {self.result}'
                f')')

    def timings(self, precision: int = 7) -> None:
        match self.workers:
            case 0:
                print(f'{self.workers=}')
                print(f'{self.time_syncmanager_enter=}')
                print(f'{self.time_start_first_worker=}')
                print(f'{self.time_start_all_workers=}')
                print(f'{self.time_multiprocessing=}')
                print(f'{self.time_import_working_nodes=}')
                print(f'{self.time_import_success_nodes=}')
                print(f'{self.time_import_failure_nodes=}')
                print(f'{self.time_final_simplification=:.{precision}f}')
                print(f'{self.time_syncmanager_exit=}')
                print(f'{self.time_total=:.{precision}}')
            case _:
                print(f'{self.workers=}')
                print(f'{self.time_syncmanager_enter=:.{precision}f}')
                print(f'{self.time_start_first_worker=:.{precision}f}')
                print(f'{self.time_start_all_workers=:.{precision}f}')
                print(f'{self.time_multiprocessing=:.{precision}f}')
                print(f'{self.time_import_working_nodes=:.{precision}f}')
                print(f'{self.time_import_failure_nodes=:.{precision}f}')
                print(f'{self.time_final_simplification=:.{precision}f}')
                print(f'{self.time_syncmanager_exit=:.{precision}f}')
                print(f'{self.time_total=:.{precision}f}')

    def virtual_substitution(self, f: Formula, workers: int):
        """Virtual substitution main loop.
        """
        logger.debug(f'entering {self.virtual_substitution.__name__}')
        self.setup(f, workers)
        while self.blocks:
            try:
                self.pop_block()
                self.process_block()
            except FoundT:
                logger.info('found T')
                logger.info(self.working_nodes.final_statistics())
                self.working_nodes = None
                self.success_nodes = NodeList(nodes=[Node(variables=[], formula=T, answer=[])])
            self.collect_success_nodes()
            if self.failure_nodes:
                raise NotImplementedError('failure_nodes = {self.failure_nodes}')
        self.final_simplification()
        logger.debug(f'leaving {self.virtual_substitution.__name__}')
        return self.result


qe = virtual_substitution = VirtualSubstitution()
