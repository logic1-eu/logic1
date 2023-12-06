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
import os
import pickle
from queue import Queue
from sage.rings.fraction_field import FractionField  # type: ignore
from sage.rings.integer_ring import ZZ  # type: ignore
import sys
import threading
import time
from typing import Optional, TypeAlias

from logic1.firstorder import (All, And, AtomicFormula, F, _F, Formula, Not,
                               Or, QuantifiedFormula, T)
from logic1.support.logging import DeltaTimeFormatter, RateFilter
from logic1.support.tracing import trace  # noqa
from logic1.theories.RCF import rcf
from logic1.theories.RCF.pnf import pnf
from logic1.theories.RCF.simplify import simplify
from logic1.theories.RCF.rcf import (Eq, Ne, Ge, Le, Gt, Lt, RcfAtomicFormula,
                                     RcfAtomicFormulas, ring, Term, Variable)

Quantifier: TypeAlias = type[QuantifiedFormula]
QuantifierBlock: TypeAlias = tuple[Quantifier, list[Variable]]


# Create logger
delta_time_formatter = DeltaTimeFormatter(
    f'%(asctime)s - %(name)s - %(levelname)-5s - %(delta)s: %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(delta_time_formatter)

logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.WARNING)

# Create rlogger (rate logger)
rate_filter = RateFilter()

rlogger = logging.getLogger(__name__ + '/r')
rlogger.addHandler(stream_handler)
rlogger.addFilter(rate_filter)
rlogger.setLevel(logging.WARNING)

# Create multiprocessing logger
multiprocessing_formatter = DeltaTimeFormatter(
    f'%(asctime)s - %(name)s/%(process)-6d - %(levelname)-5s - %(delta)s: %(message)s')

multiprocessing_handler = logging.StreamHandler()
multiprocessing_handler.setFormatter(multiprocessing_formatter)
multiprocessing_logger = logging.getLogger('multiprocessing')
multiprocessing_logger.addHandler(multiprocessing_handler)


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

    def vsubs(self, eset: EliminationSet) -> list[Node]:
        variables = self.variables
        x = eset.variable
        new_nodes = []
        for tp in eset.test_points:
            new_formula = self.formula.transform_atoms(lambda atom: self.vsubs_atom(atom, x, tp))
            if tp.guard is not None:
                new_formula = And(tp.guard, new_formula)
            new_formula = simplify(new_formula)
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


@dataclass
class WorkingNodeList(list[Node]):

    memory: set[Formula]
    node_counter: Counter[int]
    hits: int
    candidates: int

    @property
    def computed(self) -> int:
        return self.candidates - self.hits

    @property
    def hit_ratio(self) -> float:
        try:
            return float(self.hits) / self.candidates
        except ZeroDivisionError:
            return float('nan')

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.memory: set[Formula] = set()
        self.node_counter: Counter[int] = Counter()
        self.hits = 0
        self.candidates = 0

    def append_if_unknown(self, node: Node) -> None:
        if node.formula not in self.memory:
            self.append(node)
            self.memory.update({node.formula})
            n = len(node.variables)
            self.node_counter[n] += 1
        else:
            self.hits += 1
        self.candidates += 1

    def final_statistics(self) -> str:
        t = self.computed
        h = self.hits
        c = self.candidates
        p = self.hit_ratio
        return (f'computed {t} nodes, skipped {h}/{c} = {p:.0%}')

    def pop(self, *args, **kwargs) -> Node:
        node = super().pop(*args, **kwargs)
        n = len(node.variables)
        self.node_counter[n] -= 1
        return node

    def push(self, nodes: list[Node]) -> None:
        for node in nodes:
            match node.formula:
                case _F():
                    continue
                case Or(args=args):
                    for arg in args:
                        subnode = Node(node.variables.copy(), arg, node.answer)
                        self.append_if_unknown(subnode)
                case And() | AtomicFormula():
                    self.append_if_unknown(node)
                case _:
                    assert False, node


@dataclass
class NodeListParallel:

    nodes: list[bytes] = field(default_factory=list)
    busy: mp.sharedctypes.Synchronized = None

    def __init__(self, busy: mp.sharedctypes.Synchronized, nodes: list[Node]) -> None:
        self.nodes = [pickle.dumps(node) for node in nodes]
        self.busy = busy

    def __len__(self):
        return len(self.nodes)

    def append(self, pickled_node: bytes) -> None:
        self.nodes.append(pickled_node)
        self.busy.value -= 1

    def get_nodes(self) -> list[bytes]:
        return self.nodes

    def extend(self, pickled_nodes: list[bytes]) -> None:
        self.nodes.extend(pickled_nodes)
        self.busy.value -= 1


class NodeListProxy(mp.managers.BaseProxy):

    def __iter__(self):
        for pickled_node in self._callmethod('get_nodes'):
            yield pickle.loads(pickled_node)

    def __len__(self):
        return self._callmethod('__len__')

    def append(self, node: Node) -> None:
        pickled_node = pickle.dumps(node)
        self._callmethod('append', (pickled_node,))

    def extend(self, nodes: list[Node]) -> None:
        pickled_nodes = [pickle.dumps(node) for node in nodes]
        self._callmethod('extend', (pickled_nodes,))


@dataclass
class WorkingNodeListParallel:

    nodes: list[Node] = field(default_factory=list)
    busy: mp.sharedctypes.Synchronized = None
    memory: set[Formula] = field(default_factory=set)
    node_counter: Counter[int] = field(default_factory=Counter)
    hits: int = 0
    candidates: int = 0

    def __init__(self, busy) -> None:
        self.nodes = []
        self.busy = busy
        self.memory = set()
        self.node_counter = Counter()
        self.hits = 0
        self.candidates = 0

    def __len__(self):
        return len(self.nodes)

    def append_if_unknown(self, node: Node) -> None:
        if node.formula not in self.memory:
            self.nodes.append(node)
            self.memory.update({node.formula})
            n = len(node.variables)
            self.node_counter[n] += 1
        else:
            self.hits += 1
        self.candidates += 1

    def get_candidates(self) -> int:
        return self.candidates

    def get_hits(self) -> int:
        return self.hits

    def get_nodes(self) -> list[bytes]:
        return [pickle.dumps(node) for node in self.nodes]

    def get_node_counter(self) -> dict[int, int]:
        return self.node_counter

    def is_finished(self) -> bool:
        return len(self.nodes) == 0 and self.busy.value == 0

    def pop(self) -> bytes:
        node = self.nodes.pop()
        n = len(node.variables)
        self.node_counter[n] -= 1
        self.busy.value += 1
        return pickle.dumps(node)

    def push(self, pickled_nodes: list[bytes], track_busy: bool = True) -> None:
        for pickled_node in pickled_nodes:
            node = pickle.loads(pickled_node)
            match node.formula:
                case _F():
                    continue
                case Or(args=args):
                    for arg in args:
                        subnode = Node(node.variables.copy(), arg, node.answer)
                        self.append_if_unknown(subnode)
                case And() | AtomicFormula():
                    self.append_if_unknown(node)
                case _:
                    assert False, node
        if track_busy:
            self.busy.value -= 1


class WorkingNodeListProxy(mp.managers.BaseProxy):

    @property
    def busy(self):
        return self._callmethod('get_busy')

    @property
    def candidates(self):
        return self._callmethod('get_candidates')

    @property
    def computed(self) -> int:
        return self.candidates - self.hits

    @property
    def hits(self):
        return self._callmethod('get_hits')

    @property
    def node_counter(self):
        return self._callmethod('get_node_counter')

    @property
    def hit_ratio(self) -> float:
        try:
            return float(self.hits) / self.candidates
        except ZeroDivisionError:
            return float('nan')

    def __iter__(self):
        for pickled_node in self._callmethod('get_nodes'):
            yield pickle.loads(pickled_node)

    def __len__(self):
        return self._callmethod('__len__')

    def decrement_busy(self) -> None:
        self._callmethod('decrement_busy')

    def is_finished(self):
        return self._callmethod('is_finished')

    def pop(self) -> Node:
        return pickle.loads(self._callmethod('pop'))  # type: ignore

    def push(self, nodes: list[Node], track_busy: bool = True) -> None:
        pickled_nodes = [pickle.dumps(node) for node in nodes]
        self._callmethod('push', (pickled_nodes, track_busy))


class SyncManager(mp.managers.SyncManager):
    pass


SyncManager.register("nodeList", NodeListParallel, NodeListProxy,
                     ['append', 'get_nodes', 'extend', '__len__'])

SyncManager.register("workingNodeList", WorkingNodeListParallel,
                     WorkingNodeListProxy,
                     ['decrement_busy', 'get_busy', 'get_candidates',
                      'get_hits', 'get_nodes', 'get_node_counter',
                      'is_finished', '__len__', 'pop', 'push'])


@dataclass
class VirtualSubstitution:
    """Quantifier elimination by virtual substitution.
    """

    workers: int = 0
    blocks: Optional[list[QuantifierBlock]] = None
    matrix: Optional[Formula] = None
    negated: Optional[bool] = None
    working_nodes: Optional[WorkingNodeList] = None
    success_nodes: Optional[list[Node]] = None
    failure_nodes: Optional[list[Node]] = None

    log: int = logging.NOTSET
    log_rate: float = 0.5

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
        try:
            self.log_level = log
            self.log_rate = log_rate
            save_level = logger.getEffectiveLevel()
            save_rlevel = rlogger.getEffectiveLevel()
            logger.setLevel(self.log_level)
            rlogger.setLevel(self.log_level)
            rate_filter.set_rate(self.log_rate)
            delta_time_formatter.set_reference_time(time.time())
            result = self.virtual_substitution(f, workers)
        except KeyboardInterrupt:
            print('KeyboardInterrupt', file=sys.stderr, flush=True)
            return None
        finally:
            logger.setLevel(save_level)
            rlogger.setLevel(save_rlevel)
        return result

    def __str__(self) -> str:

        def format_negated() -> str:
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

    def blocks_as_str(self) -> str:
        if self.blocks is None:
            return None
        h = '  '.join(q.__qualname__ + ' ' + str(v) for q, v in self.blocks)
        return f'{h}'

    def collect_success_nodes(self) -> None:
        logger.debug(f'entering {self.collect_success_nodes.__name__}')
        self.matrix = Or(*(node.formula for node in self.success_nodes))
        if self.negated:
            self.matrix = Not(self.matrix)
        self.negated = None
        self.working_nodes = None
        self.success_nodes = None

    def nodes_as_str(self, nodes: list[Node]) -> str:
        if nodes is None:
            return None
        h = ',\n                '.join(f'{node}' for node in nodes)
        return f'[{h}]'

    def parallel_process_block(self) -> None:

        statistics_max_len = 0

        def periodic_statistics() -> str:
            nonlocal statistics_max_len
            with m_lock:
                nc = working_nodes.node_counter
                r = working_nodes.hit_ratio
                b = busy.value
                s = len(success_nodes)
                f = len(failure_nodes)
            try:
                m = max(k for k, v in nc.items() if v != 0)
                v = f'V={m}'
                tup = '.'.join(f'{nc[n]}' for n in reversed(range(1, m + 1)))
                w = f'W={tup}'
                vw = f'{v}, {w}'
            except ValueError:
                vw = 'V=0'
            padding = (statistics_max_len - len(vw)) * ' '
            statistics_max_len = max(statistics_max_len, len(vw))
            return f'{vw}, {padding}B={b}, S={s}, F={f}, H={r:.0%}'

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

        with SyncManager() as manager:
            busy = manager.Value('i', 0)
            found_t = manager.Value('i', 0)
            working_nodes = manager.workingNodeList(busy)  # type: ignore
            working_nodes.push(self.working_nodes, track_busy=False)
            success_nodes = manager.nodeList(busy, self.success_nodes)  # type: ignore
            failure_nodes = manager.nodeList(busy, self.failure_nodes)  # type: ignore
            m_lock = manager.Lock()
            processes: list[Optional[mp.Process]] = [None] * self.workers
            sentinels: list[Optional[int]] = [None] * self.workers
            ring_vars = tuple(str(v) for v in ring.get_vars())
            log_level = logger.getEffectiveLevel()
            reference_time = delta_time_formatter.get_reference_time()
            logger.debug(f'starting worker processes in {range(self.workers)}')
            for i in range(self.workers):
                processes[i] = mp.Process(
                    target=self.parallel_process_block_worker,
                    args=(working_nodes, success_nodes, failure_nodes, m_lock, busy,
                          found_t, ring_vars, i, log_level, reference_time))
                processes[i].start()
                sentinels[i] = processes[i].sentinel
            try:
                if logger.isEnabledFor(logging.INFO):
                    while not mp.connection.wait(sentinels, timeout=self.log_rate):
                        logger.info(periodic_statistics())
                else:
                    mp.connection.wait(sentinels)
            except KeyboardInterrupt:
                logger.debug('KeyboardInterrupt, waiting for processes to finish')
                wait_for_processes_to_finish()
                raise
            wait_for_processes_to_finish()
            if found_t.value > 0:
                pl = 's' if found_t.value > 1 else ''
                logger.debug(f'{found_t.value} worker{pl} found T')
                # The exception handler for FoundT in virtual_substitution in will
                # log final statistics. Therefore we copy over candidates and hits.
                # The nodes themselves have become meaningless.
                self.working_nodes.candidates = working_nodes.candidates
                self.working_nodes.hits = working_nodes.hits
                raise FoundT()
            self.working_nodes = WorkingNodeList(working_nodes)
            self.working_nodes.candidates = working_nodes.candidates
            self.working_nodes.hits = working_nodes.hits
            logger.info(self.working_nodes.final_statistics())
            self.success_nodes = list(success_nodes)
            self.failure_nodes = list(failure_nodes)

    @staticmethod
    def parallel_process_block_worker(working_nodes: WorkingNodeListProxy,
                                      success_nodes: NodeListProxy,
                                      failure_nodes: NodeListProxy,
                                      m_lock: threading.Lock,
                                      busy: mp.sharedctypes.Synchronized,
                                      found_t: mp.sharedctypes.Synchronized,
                                      ring_vars: list[str],
                                      i: int,
                                      log_level: int,
                                      reference_time: float) -> None:
        try:
            multiprocessing_logger.setLevel(log_level)
            multiprocessing_formatter.set_reference_time(reference_time)
            multiprocessing_logger.debug(f'worker process {i} is running')
            working_nodes_buffer: Queue[Optional[list[Node]]] = Queue()
            t_lock = threading.Lock()
            thread1 = threading.Thread(
                target=VirtualSubstitution.parallel_process_block_worker1,
                args=(working_nodes, working_nodes_buffer, m_lock, t_lock))
            thread1.start()
            ring.set_vars(*ring_vars)
            while True:
                with m_lock:
                    if found_t.value > 0 or working_nodes.is_finished():
                        working_nodes_buffer.put(None)
                        break
                    try:
                        node = working_nodes.pop()
                    except IndexError:
                        node = None
                if node is None:
                    time.sleep(0.001)
                    continue
                try:
                    eset = node.eset()
                except DegreeViolation:
                    with m_lock:
                        failure_nodes.append(node)
                    continue
                try:
                    nodes = node.vsubs(eset)
                except FoundT:
                    with m_lock:
                        found_t.value += 1
                    break
                if nodes[0].variables:
                    working_nodes_buffer.put(nodes)
                    with m_lock:
                        working_nodes.push(nodes)
                else:
                    with m_lock:
                        success_nodes.extend(nodes)
        except KeyboardInterrupt:
            multiprocessing_logger.debug(f'worker process {i} caught KeyboardInterrupt')
        thread1.join()
        multiprocessing_logger.debug(f'worker process {i} finished')

    @staticmethod
    def parallel_process_block_worker1(working_nodes: WorkingNodeListProxy,
                                       working_nodes_buffer: Queue[Optional[list[Node]]],
                                       m_lock: threading.Lock,
                                       t_lock: threading.Lock):
        while True:
            nodes = working_nodes_buffer.get()
            if nodes is None:
                break
            # with m_lock:
            #     working_nodes.push(nodes)

    def pop_block(self) -> None:
        logger.debug(f'entering {self.pop_block.__name__}')
        assert self.matrix, self.matrix
        logger.info(self.blocks_as_str())
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
        self.working_nodes = WorkingNodeList()
        self.working_nodes.push([Node(vars_, simplify(matrix), [])])
        self.success_nodes = []
        self.failure_nodes = []

    def process_block(self) -> None:
        logger.debug(f'entering {self.process_block.__name__}')
        if self.workers > 0:
            return self.parallel_process_block()
        return self.sequential_process_block()

    def sequential_process_block(self) -> None:

        statistics_max_len = 0

        def periodic_statistics() -> str:
            nonlocal statistics_max_len
            nc = self.working_nodes.node_counter
            r = self.working_nodes.hit_ratio
            s = len(self.success_nodes)
            f = len(self.failure_nodes)
            try:
                m = max(k for k, v in nc.items() if v != 0)
                v = f'V={m}'
                tup = '.'.join(f'{nc[n]}' for n in reversed(range(1, m + 1)))
                w = f'W={tup}'
                vw = f'{v}, {w}'
            except ValueError:
                vw = 'V=0'
            padding = (statistics_max_len - len(vw)) * ' '
            statistics_max_len = max(statistics_max_len, len(vw))
            return f'{vw}, {padding}S={s}, F={f}, H={r:.0%}'

        if logger.isEnabledFor(logging.INFO):
            last_log = time.time()
        while self.working_nodes:
            if logger.isEnabledFor(logging.INFO):
                t = time.time()
                if t - last_log >= self.log_rate:
                    logger.info(periodic_statistics())
                    last_log = t
            node = self.working_nodes.pop()
            try:
                eset = node.eset()
            except DegreeViolation:
                self.failure_nodes.append(node)
                continue
            nodes = node.vsubs(eset)
            if nodes[0].variables:
                self.working_nodes.push(nodes)
            else:
                self.success_nodes.extend(nodes)
        logger.info(self.working_nodes.final_statistics())

    def setup(self, f: Formula, workers: int) -> None:
        logger.debug(f'entering {self.setup.__name__}')
        if workers >= 0:
            self.workers = workers
        else:
            self.workers = os.cpu_count() + workers
        f = pnf(f)
        self.matrix, self.blocks = f.matrix()

    def virtual_substitution(self, f: Formula, workers: int):
        """Virtual substitution main loop.
        """
        self.setup(f, workers)
        while self.blocks:
            try:
                self.pop_block()
                self.process_block()
            except FoundT:
                logger.info('found T')
                logger.info(self.working_nodes.final_statistics())
                self.working_nodes = None
                self.success_nodes = [Node(variables=[], formula=T, answer=[])]
            self.collect_success_nodes()
            if self.failure_nodes:
                raise NotImplementedError('failure_nodes = {self.failure_nodes}')
        return simplify(self.matrix)


qe = virtual_substitution = VirtualSubstitution()
