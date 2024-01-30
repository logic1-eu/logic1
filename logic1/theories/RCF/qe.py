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
from sage.rings.fraction_field import FractionField  # type: ignore[import-untyped]
from sage.rings.integer_ring import ZZ  # type: ignore[import-untyped]
import sys
import threading
import time
from typing import Collection, Iterable, Optional

from logic1.firstorder import (
    All, And, F, _F, Formula, Not, Or, QuantifiedFormula, T)
from logic1.support.logging import DeltaTimeFormatter
from logic1.support.tracing import trace  # noqa
from logic1.theories.RCF import rcf
from logic1.theories.RCF.pnf import pnf
from logic1.theories.RCF.simplify import simplify
from logic1.theories.RCF.rcf import (
    AtomicFormula, Eq, Ne, Ge, Le, Gt, Lt, ring, Term, Variable)

# Create logger
delta_time_formatter = DeltaTimeFormatter(
    f'%(asctime)s - %(name)s - %(levelname)-5s - %(delta)s: %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(delta_time_formatter)

logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.addFilter(lambda record: record.msg.strip() != '')
logger.setLevel(logging.WARNING)

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


@dataclass
class TestPoint:

    guard: Optional[Formula] = None
    num: Term = field(default_factory=lambda: Term(0))
    den: Term = field(default_factory=lambda: Term(1))
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
                        if lhs._degree(x) == 1:
                            a = lhs._coefficient({x: 1})
                            if a._is_constant():
                                self.variables.remove(x)
                                b = lhs._coefficient({x: 0})
                                tp = TestPoint(num=-b, den=a)
                                return EliminationSet(variable=x, test_points=[tp], method='g')
        return None

    def regular_eset(self) -> EliminationSet:

        def guard():
            return Ne(a, 0) if not a._is_constant() else None

        x = self.variables.pop()
        test_points = [TestPoint(nsp=NSP.MINUS_INFINITY)]
        for atom in self.formula.atoms():
            assert isinstance(atom, AtomicFormula)
            match atom.lhs._degree(x):
                case -1 | 0:
                    continue
                case 1:
                    a = atom.lhs._coefficient({x: 1})
                    b = atom.lhs._coefficient({x: 0})
                    match atom:
                        case Eq():
                            tp = TestPoint(guard=guard(), num=-b, den=a)
                        case Ne():
                            tp = TestPoint(guard=guard(), num=-b, den=a, nsp=NSP.PLUS_EPSILON)
                        case Le():
                            if a._is_constant() and a.poly > 0:
                                continue
                            tp = TestPoint(guard=guard(), num=-b, den=a)
                        case Ge():
                            if a._is_constant() and a.poly < 0:
                                continue
                            tp = TestPoint(guard=guard(), num=-b, den=a)
                        case Lt():
                            if a._is_constant() and a.poly > 0:
                                continue
                            tp = TestPoint(guard=guard(), num=-b, den=a, nsp=NSP.PLUS_EPSILON)
                        case Gt():
                            if a._is_constant() and a.poly < 0:
                                continue
                            tp = TestPoint(guard=guard(), num=-b, den=a, nsp=NSP.PLUS_EPSILON)
                        case _:
                            assert False, atom
                    test_points.append(tp)
                case _:
                    raise DegreeViolation(atom, x, atom.lhs._degree(x))
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

    def vsubs_atom(self, atom: AtomicFormula, x: Variable, tp: TestPoint) -> Formula:

        def mu() -> Formula:
            """Substitute ±oo into ordering constraint.
            """
            c = lhs._coefficient({x: 0})
            mu: Formula = func(c, 0)
            for e in range(1, lhs._degree(x) + 1):
                c = lhs._coefficient({x: e})
                if tp.nsp == NSP.MINUS_INFINITY and e % 2 == 1:
                    c = - c
                mu = Or(Gt(c, 0), And(Eq(c, 0), mu))
            return mu

        def nu(lhs: Term) -> Formula:
            """Substitute ±ε into any constraint.
            """
            if lhs._degree(x) <= 0:
                return func(lhs, 0)
            lhs_prime = lhs._derivative(x)
            if tp.nsp == NSP.MINUS_EPSILON:
                lhs_prime = - lhs_prime
            return Or(Gt(lhs, 0), And(Eq(lhs, 0), nu(lhs_prime)))

        def sigma() -> Formula:
            """Substitute quotient into any constraint.
            """
            func = atom.func
            FF = FractionField(ring.sage_ring)  # discuss
            lhq = ring(atom.lhs.poly).subs(**{str(x.poly): FF(tp.num.poly, tp.den.poly)})
            match func:
                case rcf.Eq | rcf.Ne:
                    lhp = lhq.numerator()
                case rcf.Ge | rcf.Le | rcf.Gt | rcf.Lt:
                    lhp = (lhq * lhq.denominator() ** 2).numerator()
                case _:
                    assert False, func
            assert lhp.parent() in (ring.sage_ring, ZZ), lhp.parent()
            return func(Term(lhp), 0)

        def tau():
            """Substitute transcendental element into equality.
            """
            args = []
            for e in range(lhs._degree(x) + 1):
                c = lhs._coefficient({x: e})
                if c._is_zero():
                    continue
                if c._is_constant():
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
                # Substitute transcendental element into equality.
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
                    logger.info(self.success_nodes.periodic_statistics('F'))
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
        logger.info(self.success_nodes.final_statistics('failure'))

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
