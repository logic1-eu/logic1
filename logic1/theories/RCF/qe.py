# mypy: strict_optional = False

"""Real quantifier elimination by virtual substitution.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import auto, Enum
import logging
import multiprocessing
import pickle
from sage.rings.fraction_field import FractionField  # type: ignore
from sage.rings.integer_ring import ZZ  # type: ignore
import time
from typing import Collection, Optional, TypeAlias

from ...firstorder import (
    All, And, AtomicFormula, F, _F, Formula, Not, Or, QuantifiedFormula, T)
from ...support.logging import DeltaTimeFormatter, TimePeriodFilter
from ...support.tracing import trace  # noqa
from . import rcf
from .pnf import pnf as _pnf
from .rcf import (
    RcfAtomicFormula, RcfAtomicFormulas, ring, Term, Variable, Eq, Ne, Ge, Le,
    Gt, Lt)
from .simplify import simplify as _simplify


Quantifier: TypeAlias = type[QuantifiedFormula]
QuantifierBlock: TypeAlias = tuple[Quantifier, list[Variable]]


# Create logger
formatter = DeltaTimeFormatter(
    f'%(asctime)s - %(name)s - %(levelname)s - %(delta)s: %(message)s')

handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.WARNING)

# Create flogger
filter_ = TimePeriodFilter()

flogger = logging.getLogger(__name__ + '/F')
flogger.addHandler(handler)
flogger.addFilter(filter_)
flogger.setLevel(logging.WARNING)


class DegreeViolation(Exception):
    pass


class Failed(Exception):
    pass


class FoundT(Exception):
    pass


class NSP(Enum):
    NONE = auto()
    PLUS_EPSILON = auto()
    MINUS_EPSILON = auto()
    PLUS_INFINITY = auto()
    MINUS_INFINITY = auto()


@dataclass
class TestPoint:

    guard: Optional[Formula] = None
    num: Term = 0
    den: Term = 1
    nsp: NSP = NSP.NONE


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
                        if lhs.degree(x) == 1:
                            a = lhs.coefficient({x: 1})
                            if a.is_constant():
                                self.variables.remove(x)
                                b = lhs.coefficient({x: 0})
                                tp = TestPoint(num=-b, den=a)
                                return EliminationSet(variable=x, test_points=[tp], method='g')
        return None

    def regular_eset(self) -> EliminationSet:

        def guard():
            return Ne(a, 0) if not a.is_constant() else None

        x = self.variables.pop()
        test_points = [TestPoint(nsp=NSP.MINUS_INFINITY)]
        for atom in self.formula.atoms():
            assert isinstance(atom, RcfAtomicFormulas)
            match atom.lhs.degree(x):
                case -1 | 0:
                    continue
                case 1:
                    a = atom.lhs.coefficient({x: 1})
                    b = atom.lhs.coefficient({x: 0})
                    match atom:
                        case Eq():
                            tp = TestPoint(guard=guard(), num=-b, den=a)
                        case Ne():
                            tp = TestPoint(guard=guard(), num=-b, den=a, nsp=NSP.PLUS_EPSILON)
                        case Le():
                            if a.is_constant() and a > 0:
                                continue
                            tp = TestPoint(guard=guard(), num=-b, den=a)
                        case Ge():
                            if a.is_constant() and a < 0:
                                continue
                            tp = TestPoint(guard=guard(), num=-b, den=a)
                        case Lt():
                            if a.is_constant() and a > 0:
                                continue
                            tp = TestPoint(guard=guard(), num=-b, den=a, nsp=NSP.PLUS_EPSILON)
                        case Gt():
                            if a.is_constant() and a < 0:
                                continue
                            tp = TestPoint(guard=guard(), num=-b, den=a, nsp=NSP.PLUS_EPSILON)
                        case _:
                            assert False, atom
                    test_points.append(tp)
                case _:
                    raise DegreeViolation(atom, x, atom.lhs.degree(x))
        return EliminationSet(variable=x, test_points=test_points, method='e')

    def simplify(self, *args, **kwargs) -> Formula:
        return _simplify(*args, **kwargs)

    def vsubs(self, eset: EliminationSet) -> list[Node]:
        variables = self.variables
        x = eset.variable
        new_nodes = []
        for tp in eset.test_points:
            new_formula = self.formula.transform_atoms(lambda atom: self.vsubs_atom(atom, x, tp))
            if tp.guard is not None:
                new_formula = And(tp.guard, new_formula)
            new_formula = self.simplify(new_formula)
            if new_formula is T:
                raise FoundT()
            new_nodes.append(Node(variables.copy(), new_formula, []))
        return new_nodes

    def vsubs_atom(self, atom: RcfAtomicFormula, x: Variable, tp: TestPoint) -> Formula:

        def mu() -> Formula:
            """Substitute ±oo into ordering constraint.
            """
            c = lhs.coefficient({x: 0})
            mu: Formula = func(c, 0)
            for e in range(1, lhs.degree(x) + 1):
                c = lhs.coefficient({x: e})
                if tp.nsp == NSP.MINUS_INFINITY and e % 2 == 1:
                    c = - c
                mu = Or(Gt(c, 0), And(Eq(c, 0), mu))
            return mu

        def nu(lhs: Term) -> Formula:
            """Substitute ±ε into any constraint.
            """
            if lhs.degree(x) <= 0:
                return func(lhs, 0)
            lhs_prime = lhs.derivative(x)
            if tp.nsp == NSP.MINUS_EPSILON:
                lhs_prime = - lhs_prime
            return Or(Gt(lhs, 0), And(Eq(lhs, 0), nu(lhs_prime)))

        def sigma() -> Formula:
            """Substitute quotient into any constraint.
            """
            lhs = atom.lhs
            func = atom.func
            fraction_field = FractionField(ring.sage_ring)
            lhs = lhs.subs(**{str(x): fraction_field(tp.num, tp.den)})
            match func:
                case rcf.Eq | rcf.Ne:
                    lhs = lhs.numerator()
                case rcf.Ge | rcf.Le | rcf.Gt | rcf.Lt:
                    lhs = (lhs * lhs.denominator() ** 2).numerator()
                case _:
                    assert False, func
            assert lhs.parent() in (ring.sage_ring, ZZ), lhs.parent()
            return func(lhs, 0)

        def tau():
            """Substitute transcendental element into equality.
            """
            args = []
            for e in range(lhs.degree(x) + 1):
                c = lhs.coefficient({x: e})
                if c.is_zero():
                    continue
                if c.is_constant():
                    return F
                args.append(func(c, 0))
            return And(*args) if func is Eq else Or(*args)

        if tp.nsp is NSP.NONE:
            # Substitute quotient into any constraint.
            return sigma()
        if atom.func in (Le, Lt):
            atom = atom.converse_func(- atom.lhs, atom.rhs)
        match atom:
            case Eq(func=func, lhs=lhs) | Ne(func=func, lhs=lhs):
                # Substitute transcendental element in to equality.
                result = tau()
            case Ge(func=func, lhs=lhs) | Gt(func=func, lhs=lhs):
                match tp.nsp:
                    case NSP.PLUS_EPSILON | NSP.MINUS_EPSILON:
                        std_tp = TestPoint(num=tp.num, den=tp.den, nsp=NSP.NONE)
                        result = nu(lhs).transform_atoms(lambda at: self.vsubs_atom(at, x, std_tp))
                    case NSP.PLUS_INFINITY | NSP.MINUS_INFINITY:
                        # Substitute ±oo into ordering constraint.
                        result = mu()
            case _:
                assert False, (atom, tp)
        if atom.func in (Le, Lt):
            result = Not(result)
        return result


class QeManager:

    def __init__(self):
        self.manager = multiprocessing.Manager()

    def dict(self, *args, **kwargs):
        return self.manager.dict(*args, **kwargs)

    def list(self, nodes: list[Node]) -> QeListProxy:
        L = [pickle.dumps(node) for node in nodes]
        return QeListProxy(self.manager.list(L))

    def Lock(self):
        return self.manager.Lock()


class QeListProxy:

    def __init__(self, list_proxy: multiprocessing.managers.ListProxy):
        self.list_proxy = list_proxy

    def __iter__(self):
        for pickled_node in self.list_proxy:
            yield pickle.loads(pickled_node)

    def __len__(self):
        return len(self.list_proxy)

    def append(self, node: Node):
        self.list_proxy.append(pickle.dumps(node))

    def extend(self, nodes: list[Node]):
        self.list_proxy.extend([pickle.dumps(node) for node in nodes])

    def pop(self):
        return pickle.loads(self.list_proxy.pop())


@dataclass
class VirtualSubstitution:
    """Quantifier elimination by virtual substitution.
    """

    blocks: Optional[list[QuantifierBlock]] = None
    matrix: Optional[Formula] = None
    negated: Optional[bool] = None
    working_nodes: Optional[list[Node]] = None
    success_nodes: Optional[list[Node]] = None
    failure_nodes: Optional[list[Node]] = None

    max_len: int = 0

    def __call__(self, f: Formula, nprocs: int = 0,
                 log: int = logging.NOTSET, log_rate: float = 0.5) -> Formula:
        """Virtual substitution entry point.

        nprocs is the number of processors to use. Tge default nprocs=0 uses
        the sequential implementation. In contrast, nprocs=1 uses the parallel
        implementation with one worker process, which is interesting for
        testing and debugging.
        """
        try:
            save_level = logger.getEffectiveLevel()
            save_flevel = flogger.getEffectiveLevel()
            logger.setLevel(log)
            flogger.setLevel(log)
            filter_.set_rate(log_rate)
            formatter.reset_clock()
            result = self.virtual_substitution(f, nprocs)
        finally:
            logger.setLevel(save_level)
            flogger.setLevel(save_flevel)
        return result

    def __str__(self):

        def format_negated():
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

        return (f'{self.__class__.__qualname__} [\n'
                f'    blocks        = {self.blocks_as_str()},\n'
                f'    matrix        = {self.matrix},\n'
                f'    negated       = {format_negated()}\n'
                f'    working_nodes = {self.nodes_as_str(self.working_nodes)},\n'
                f'    success_nodes = {self.nodes_as_str(self.success_nodes)}\n'
                f'    failure_nodes = {self.nodes_as_str(self.failure_nodes)}\n'
                f']')

    def collect_success_nodes(self) -> None:
        logger.debug(f'entering {self.collect_success_nodes.__name__}')
        self.matrix = Or(*(node.formula for node in self.success_nodes))
        if self.negated:
            self.matrix = Not(self.matrix)
        self.negated = None
        self.working_nodes = None
        self.success_nodes = None

    def blocks_as_str(self):
        if self.blocks is None:
            return None
        h = '  '.join(q.__qualname__ + ' ' + str(v) for q, v in self.blocks)
        return f'{h}'

    def nodes_as_str(self, nodes):
        if nodes is None:
            return None
        h = ',\n                '.join(f'{node}' for node in nodes)
        return f'[{h}]'

    def pnf(self, *args, **kwargs) -> Formula:
        return _pnf(*args, **kwargs)

    def pop_block(self) -> None:
        logger.debug(f'entering {self.pop_block.__name__}')
        assert self.matrix, self.matrix
        logger.info(self.blocks_as_str())
        if logger.isEnabledFor(logging.DEBUG):
            s = str(self.matrix)
            if len(s) > 80:
                s = s[:76] + ' ...'
            logger.debug(s)
        quantifier, vars_ = self.blocks.pop()
        matrix = self.matrix
        self.matrix = None
        if quantifier is All:
            self.negated = True
            matrix = Not(matrix)
        else:
            self.negated = False
        self.working_nodes = []
        self.push_to(self.working_nodes, [Node(vars_, self.simplify(matrix), [])])
        self.success_nodes = []
        self.failure_nodes = []

    def process_block(self, nprocs: int) -> None:
        logger.debug(f'entering {self.process_block.__name__}')
        assert nprocs >= 0, nprocs
        match nprocs:
            case 0:
                return self.process_block_sequential()
            case _:
                return self.process_block_parallel(nprocs)

    def process_block_sequential(self) -> None:
        while self.working_nodes:
            if flogger.isEnabledFor(logging.INFO):
                flogger.info(self.statistics(self.working_nodes,
                                             self.success_nodes,
                                             self.failure_nodes))
            node = self.working_nodes.pop()
            try:
                eset = node.eset()
            except DegreeViolation:
                self.failure_nodes.append(node)
                continue
            nodes = node.vsubs(eset)
            if nodes[0].variables:
                self.push_to(self.working_nodes, nodes)
            else:
                self.success_nodes.extend(nodes)

    def process_block_parallel(self, nprocs: int) -> None:
        manager = QeManager()
        working_nodes = manager.list(self.working_nodes)
        success_nodes = manager.list(self.success_nodes)
        failure_nodes = manager.list(self.failure_nodes)
        dictionary = manager.dict({'busy': 0, 'found_t': 0})
        working_lock = manager.Lock()
        success_lock = manager.Lock()
        failure_lock = manager.Lock()
        dictionary_lock = manager.Lock()
        p: list[Optional[multiprocessing.Process]] = [None] * nprocs
        s: list[Optional[int]] = [None] * nprocs
        for i in range(nprocs):
            p[i] = multiprocessing.Process(
                target=self.process_block_parallel_worker,
                args=(working_nodes, success_nodes, failure_nodes, dictionary,
                      working_lock, success_lock, failure_lock, dictionary_lock,
                      tuple(str(v) for v in ring.get_vars())))
            p[i].start()
            s[i] = p[i].sentinel
        still_running = s.copy()
        while still_running:
            with working_lock:
                wn = list(working_nodes)
            with success_lock:
                num_sn = len(success_nodes)
            with failure_lock:
                num_fn = len(failure_nodes)
            if flogger.isEnabledFor(logging.INFO):
                flogger.info(self.statistics_(wn, num_sn, num_fn))
            if multiprocessing.connection.wait(still_running, timeout=0):
                # All processes are going to finish now.
                while still_running:
                    finished = multiprocessing.connection.wait(still_running,
                                                               timeout=0.001)
                    if finished:
                        for sentinel in finished:
                            assert isinstance(sentinel, int)
                            still_running.remove(sentinel)
                        num_finished = nprocs - len(still_running)
                        logger.debug(f'{num_finished} worker(s) finished, '
                                     f'{len(still_running)} running')
        with dictionary_lock:
            found_t = dictionary['found_t']
        if found_t > 0:
            logger.debug(f'{found_t} worker found T')
            raise FoundT()
        self.working_nodes = list(working_nodes)
        self.success_nodes = list(success_nodes)
        self.failure_nodes = list(failure_nodes)

    @staticmethod
    def process_block_parallel_worker(working_nodes, success_nodes, failure_nodes,
                                      dictionary, working_lock, success_lock,
                                      failure_lock, dictionary_lock, ring_vars) -> None:
        def work_left():
            with working_lock:
                return list(working_nodes) or dictionary['busy'] > 0

        ring.set_vars(*ring_vars)
        while True:
            with working_lock, dictionary_lock:
                if dictionary['found_t'] > 0:
                    break
                if len(working_nodes) == 0 and dictionary['busy'] == 0:
                    break
                try:
                    node = working_nodes.pop()
                except IndexError:
                    node = None
                else:
                    dictionary['busy'] += 1
            if node is None:
                time.sleep(0.001)
                continue
            try:
                eset = node.eset()
            except DegreeViolation:
                with failure_lock, dictionary_lock:
                    failure_nodes.append(node)
                    dictionary['busy'] -= 1
                continue
            try:
                nodes = node.vsubs(eset)
            except FoundT:
                with dictionary_lock:
                    dictionary['found_t'] += 1
                break
            if nodes[0].variables:
                with working_lock, dictionary_lock:
                    VirtualSubstitution.push_to(working_nodes, nodes)
                    dictionary['busy'] -= 1
            else:
                with success_lock, dictionary_lock:
                    success_nodes.extend(nodes)
                    dictionary['busy'] -= 1

    @staticmethod
    def push_to(working_nodes, nodes: list[Node]) -> None:
        for node in nodes:
            match node.formula:
                case _F():
                    continue
                case Or(args=args):
                    for arg in args:
                        working_nodes.append(Node(node.variables.copy(), arg, node.answer))
                case And() | AtomicFormula():
                    working_nodes.append(node)
                case _:
                    assert False, node

    def setup(self, f: Formula) -> None:
        logger.debug(f'entering {self.setup.__name__}')
        f = self.pnf(f)
        self.matrix, self.blocks = f.matrix()

    def simplify(self, *args, **kwargs) -> Formula:
        return _simplify(*args, **kwargs)

    def virtual_substitution(self, f, nprocs):
        """Virtual substitution main loop.
        """
        self.setup(f)
        while self.blocks:
            try:
                self.pop_block()
                self.process_block(nprocs)
            except FoundT:
                self.working_nodes = None
                self.success_nodes = [Node(variables=[], formula=T, answer=[])]
            self.collect_success_nodes()
            if self.failure_nodes:
                raise NotImplemented('failure_nodes = {self.failure_nodes}')
        return self.simplify(self.matrix)

    def statistics(self, working_nodes: Collection[Node], success_nodes: Collection[Node],
                   failure_nodes: Collection[Node]) -> str:
        return self.statistics_(working_nodes, len(success_nodes), len(failure_nodes))

    def statistics_(self, working_nodes: Collection[Node],
                    num_success_nodes: int, num_failure_nodes: int) -> str:
        counter = Counter(len(node.variables) for node in working_nodes)
        if counter.keys():
            m = max(counter.keys())
            string = f'V={m}, {counter[m]}'
            for n in reversed(range(1, m)):
                string += f'.{counter[n]}'
            string += ', '
        else:
            string = ''
        if self.max_len > len(string):
            string += (self.max_len - len(string)) * ' '
        else:
            self.max_len = len(string)
        string += (f'W={len(working_nodes)}, '
                   f'S={num_success_nodes}, '
                   f'F={num_failure_nodes}')
        return string


qe = virtual_substitution = VirtualSubstitution()
