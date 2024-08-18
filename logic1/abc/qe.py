from __future__ import annotations

from abc import abstractmethod
from collections import Counter
from dataclasses import dataclass, field
import logging
import multiprocessing as mp
import multiprocessing.managers
import multiprocessing.queues
import queue
import os
import threading
import time
from typing import (Any, Collection, Generic, Iterable, Iterator, Optional,
                    Self, TypeVar)
from typing import reveal_type  # noqa

from logic1.support.excepthook import NoTraceException
from logic1.firstorder import (All, And, AtomicFormula, _F, Formula, Not, Or,
                               Prefix, Term, Variable)
from logic1.support.logging import DeltaTimeFormatter

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

α = TypeVar('α', bound=AtomicFormula)
φ = TypeVar('φ', bound=Formula)
ν = TypeVar('ν', bound='Node')
ι = TypeVar('ι')
σ = TypeVar('σ')
τ = TypeVar('τ', bound='Term')
χ = TypeVar('χ', bound=Variable)
θ = TypeVar('θ', bound='Theory')
ω = TypeVar('ω', bound='Options')


class FoundT(Exception):
    pass


class NodeProcessFailure(Exception):
    pass


class Timer:

    def __init__(self):
        self.reset()

    def get(self) -> float:
        return time.time() - self._reference_time

    def reset(self) -> None:
        self._reference_time = time.time()


@dataclass
class Node(Generic[φ, χ, θ]):
    # sequential and parallel

    variables: list[χ]
    formula: φ

    @abstractmethod
    def copy(self) -> Self:
        ...

    @abstractmethod
    def process(self, theory: θ) -> list[Self]:
        ...


@dataclass
class NodeList(Collection[ν], Generic[φ, ν]):
    # Sequential only

    nodes: list[ν] = field(default_factory=list)
    memory: set[φ] = field(default_factory=set)
    hits: int = 0
    candidates: int = 0

    def __contains__(self, obj: object) -> bool:
        return obj in self.nodes

    def __iter__(self) -> Iterator[ν]:
        yield from self.nodes

    def __len__(self) -> int:
        return len(self.nodes)

    def append(self, node: ν) -> bool:
        is_new = node.formula not in self.memory
        if is_new:
            self.nodes.append(node)
            self.memory.add(node.formula)
        else:
            self.hits += 1
        self.candidates += 1
        return is_new

    def extend(self, nodes: Iterable[ν]) -> None:
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
class WorkingNodeList(NodeList[φ, ν]):
    # Sequential only

    node_counter: Counter[int] = field(default_factory=Counter)

    def append(self, node: ν) -> bool:
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

    def pop(self) -> ν:
        node = self.nodes.pop()
        n = len(node.variables)
        self.node_counter[n] -= 1
        return node

    def extend(self, nodes: Iterable[ν]) -> None:
        for node in nodes:
            match node.formula:
                case _F():
                    continue
                case Or(args=args):
                    for arg in args:
                        sub_node = node.copy()
                        sub_node.variables = sub_node.variables.copy()
                        sub_node.formula = arg
                        self.append(sub_node)
                case And() | AtomicFormula():
                    self.append(node)
                case _:
                    assert False, node


@dataclass
class NodeListManager(Generic[φ, ν]):

    nodes: list[ν] = field(default_factory=list)
    nodes_lock: threading.Lock = field(default_factory=threading.Lock)
    memory: set[φ] = field(default_factory=set)
    memory_lock: threading.Lock = field(default_factory=threading.Lock)
    hits: int = 0
    hits_candidates_lock: threading.Lock = field(default_factory=threading.Lock)
    candidates: int = 0

    def get_nodes(self) -> list[ν]:
        with self.nodes_lock:
            return self.nodes.copy()

    def get_memory(self) -> set[φ]:
        with self.memory_lock:
            return self.memory.copy()

    def get_candidates(self) -> int:
        return self.candidates

    def get_hits(self) -> int:
        return self.hits

    def __len__(self) -> int:
        with self.nodes_lock:
            return len(self.nodes)

    def append(self, node: ν) -> bool:
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

    def extend(self, nodes: Iterable[ν]) -> None:
        for node in nodes:
            self.append(node)

    def statistics(self) -> tuple[Any, ...]:
        with self.hits_candidates_lock:
            return (self.hits, self.candidates)


class NodeListProxy(Collection[ν], Generic[φ, ν]):
    # parallel

    def __init__(self, proxy: _NodeListProxy) -> None:
        self._proxy = proxy

    @property
    def nodes(self) -> list[ν]:
        return self._proxy.get_nodes()

    @property
    def memory(self) -> set[φ]:
        return self._proxy.get_memory()

    @property
    def candidates(self) -> int:
        return self._proxy.get_candidates()

    @property
    def hits(self) -> int:
        return self._proxy.get_hits()

    def __contains__(self, obj: object) -> bool:
        match obj:
            case Node():
                return obj in self._proxy.get_nodes()
            case _:
                return False

    def __iter__(self) -> Iterator[ν]:
        yield from self._proxy.get_nodes()

    def __len__(self) -> int:
        return len(self._proxy)

    def append(self, node: ν) -> bool:
        return self._proxy.append(node)

    def extend(self, nodes: list[ν]) -> None:
        self._proxy.extend(nodes)

    def final_statistics(self, key: str) -> str:
        hits, candidates = self._proxy.statistics()
        num_nodes = candidates - hits
        if num_nodes == 0:
            return ''
        ratio = self.hit_ratio(hits, candidates)
        return (f'produced {num_nodes} {key} nodes, '
                f'dropped {hits}/{candidates} = {ratio:.0%}')

    def hit_ratio(self, hits: int, candidates: int) -> float:
        try:
            return float(hits) / candidates
        except ZeroDivisionError:
            return float('nan')

    def periodic_statistics(self, key: str) -> str:
        hits, candidates = self._proxy.statistics()
        num_nodes = candidates - hits
        if num_nodes == 0:
            return ''
        ratio = self.hit_ratio(hits, candidates)
        return f'{key}={num_nodes}, H={ratio:.0%}'


class _NodeListProxy(mp.managers.BaseProxy, Generic[φ, ν]):

    def get_nodes(self) -> list[ν]:
        return self._callmethod('get_nodes')  # type: ignore[func-returns-value]

    def get_memory(self) -> set[φ]:
        return self._callmethod('get_memory')  # type: ignore[func-returns-value]

    def get_candidates(self) -> int:
        return self._callmethod('get_candidates')  # type: ignore[func-returns-value]

    def get_hits(self) -> int:
        return self._callmethod('get_hits')  # type: ignore[func-returns-value]

    def __len__(self) -> int:
        return self._callmethod('__len__')  # type: ignore[func-returns-value]

    def append(self, node: ν) -> bool:
        return self._callmethod('append', (node,))  # type: ignore[func-returns-value]

    def extend(self, nodes: Iterable[ν]) -> None:
        return self._callmethod('extend', (nodes,))  # type: ignore[func-returns-value]

    def statistics(self) -> tuple[Any, ...]:
        return self._callmethod('statistics')  # type: ignore[func-returns-value]


@dataclass
class WorkingNodeListManager(NodeListManager[φ, ν]):

    busy: int = 0
    busy_lock: threading.Lock = field(default_factory=threading.Lock)
    node_counter: Counter[int] = field(default_factory=Counter)
    node_counter_lock: threading.Lock = field(default_factory=threading.Lock)

    def get_node_counter(self) -> Counter[int]:
        with self.node_counter_lock:
            return self.node_counter.copy()

    def append(self, node: ν) -> bool:
        is_new = super().append(node)
        if is_new:
            n = len(node.variables)
            with self.node_counter_lock:
                self.node_counter[n] += 1
        return is_new

    def is_finished(self) -> bool:
        with self.nodes_lock, self.busy_lock:
            return len(self.nodes) == 0 and self.busy == 0

    def statistics(self) -> tuple[Any, ...]:
        # hits and candidates are always consistent. hits/candidates, busy,
        # node_counter are three snapshots at different times, each of which is
        # consistent.
        with self.hits_candidates_lock, self.busy_lock, self.node_counter_lock:
            return (self.hits, self.candidates, self.busy, self.node_counter)

    def pop(self) -> ν:
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


class WorkingNodeListProxy(NodeListProxy[φ, ν]):

    def __init__(self, proxy: _WorkingNodeListProxy) -> None:
        self._proxy: _WorkingNodeListProxy = proxy

    @property
    def node_counter(self) -> Counter[int]:
        return self._proxy.get_node_counter()

    def extend(self, nodes: Iterable[ν]) -> None:
        new_nodes = []
        for node in nodes:
            match node.formula:
                case _F():
                    continue
                case Or(args=args):
                    for arg in args:
                        sub_node = node.copy()
                        sub_node.variables = node.variables.copy()
                        sub_node.formula = arg
                        if sub_node not in new_nodes:
                            new_nodes.append(sub_node)
                case And() | AtomicFormula():
                    if node not in new_nodes:
                        new_nodes.append(node)
                case _:
                    assert False, node
        self._proxy.extend(new_nodes)

    def final_statistics(self, key: Optional[str] = None) -> str:
        if key:
            return super().final_statistics(key)
        hits, candidates, _, _ = self._proxy.statistics()
        num_nodes = candidates - hits
        ratio = self.hit_ratio(hits, candidates)
        return (f'performed {num_nodes} elimination steps, '
                f'skipped {hits}/{candidates} = {ratio:.0%}')

    def is_finished(self) -> bool:
        return self._proxy.is_finished()

    def periodic_statistics(self, key: str = 'W') -> str:
        hits, candidates, busy, node_counter = self._proxy.statistics()
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

    def pop(self) -> ν:
        return self._proxy.pop()

    def task_done(self) -> None:
        self._proxy.task_done()


class _WorkingNodeListProxy(_NodeListProxy[φ, ν]):

    def get_node_counter(self) -> Counter[int]:
        return self._callmethod('get_node_counter')  # type: ignore[func-returns-value]

    def is_finished(self) -> bool:
        return self._callmethod('is_finished')  # type: ignore[func-returns-value]

    def pop(self) -> ν:
        return self._callmethod('pop')  # type: ignore[func-returns-value]

    def task_done(self) -> None:
        return self._callmethod('task_done')  # type: ignore[func-returns-value]


class SyncManager(mp.managers.SyncManager, Generic[φ, ν]):

    def NodeList(self) -> NodeListProxy[φ, ν]:
        proxy = self._NodeListProxy()  # type: ignore[attr-defined]
        return NodeListProxy(proxy)

    def WorkingNodeList(self) -> WorkingNodeListProxy[φ, ν]:
        proxy = self._WorkingNodeListProxy()  # type: ignore[attr-defined]
        return WorkingNodeListProxy(proxy)


SyncManager.register('_NodeListProxy', NodeListManager, _NodeListProxy,
                     ['get_nodes', 'get_memory', 'get_candidates', 'get_hits',
                      '__len__', 'append', 'extend', 'statistics'])

SyncManager.register('_WorkingNodeListProxy', WorkingNodeListManager, _WorkingNodeListProxy,
                     ['get_nodes', 'get_memory', 'get_candidates', 'get_hits',
                      'get_node_counter', '__len__', 'append', 'extend',
                      'is_finished', 'pop', 'statistics', 'task_done'])


@dataclass
class Theory(Generic[α, τ, χ, σ]):

    class Inconsistent(Exception):
        pass

    atoms: list[α]

    def __init__(self, atoms: Iterable[α]) -> None:
        self.atoms = list(atoms)

    def append(self, new_atom: α) -> None:
        self.extend([new_atom])

    def extend(self, new_atoms: Iterable[α]) -> None:
        self.atoms.extend(new_atoms)
        # NF nörgelt
        theta = self.simplify(And(*self.atoms))
        if Formula.is_atomic(theta):
            self.atoms = [theta]
        elif Formula.is_and(theta):
            self.atoms = [*theta.args]
        elif Formula.is_true(theta):
            self.atoms = []
        elif Formula.is_false(theta):
            raise self.Inconsistent(f'{self=}, {new_atoms=}, {theta=}')
        else:
            assert False, (self, new_atoms, theta)

    @abstractmethod
    def simplify(self, f: Formula[α, τ, χ, σ]) -> Formula[α, τ, χ, σ]:
        ...


@dataclass
class Options:

    log_level: int = logging.NOTSET
    log_rate: float = 0.5


@dataclass
class QuantifierElimination(Generic[ν, θ, ι, ω, α, τ, χ, σ]):

    blocks: Optional[Prefix[χ]] = None
    matrix: Optional[Formula[α, τ, χ, σ]] = None
    negated: Optional[bool] = None
    root_nodes: Optional[list[ν]] = None
    working_nodes: Optional[WorkingNodeList[Formula[α, τ, χ, σ], ν]] = None
    success_nodes: Optional[NodeList[Formula[α, τ, χ, σ], ν]] = None
    failure_nodes: Optional[NodeList[Formula[α, τ, χ, σ], ν]] = None
    theory: Optional[θ] = None
    result: Optional[Formula[α, τ, χ, σ]] = None

    workers: int = 0
    options: Optional[ω] = None

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

    def __call__(self, f: Formula[α, τ, χ, σ], assume: Iterable[α] = [],
                 workers: int = 0, **options) -> Optional[Formula[α, τ, χ, σ]]:
        """Quantifier elimination entry point.

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
        delta_time_formatter.set_reference_time(time.time())
        # We call __init__ in order to reset all attributes of the data class
        # also within __call__. This is not really nice, but it does the job
        # and saves some code.
        QuantifierElimination.__init__(self)  # dicuss: NF is :-(
        self.theory = self.create_theory(assume)
        if workers >= 0:
            self.workers = workers
        else:
            cpu_count = os.cpu_count()
            if cpu_count is None:
                raise ValueError(f'{os.cpu_count()=}, i.e. undetermined')
            if cpu_count + workers < 1:
                raise ValueError(f'negative number of workers')
            self.workers = cpu_count + workers
        self.options = self.create_options(**options)
        save_level = logger.getEffectiveLevel()
        try:
            logger.setLevel(self.options.log_level)
            logger.info(f'{self.options}')
            result = self.quantifier_eliminiation(f, workers)
            logger.info('finished')
        except KeyboardInterrupt:
            logger.info('keyboard interrupt')
            raise NoTraceException('KeyboardInterrupt')
        finally:
            logger.setLevel(save_level)
        self.time_total = timer.get()
        return result

    @property
    def assume(self) -> list[α]:
        assert self.theory is not None
        return self.theory.atoms

    def collect_success_nodes(self) -> None:
        assert self.success_nodes is not None
        logger.debug(f'entering {self.collect_success_nodes.__name__}')
        self.matrix = Or(*(node.formula for node in self.success_nodes))
        if self.negated:
            self.matrix = Not(self.matrix)
        self.negated = None
        self.working_nodes = None
        self.success_nodes = None

    @abstractmethod
    def create_options(self, **kwargs) -> ω:
        ...

    @abstractmethod
    def create_root_nodes(self, variables: Iterable[χ], matrix: Formula[α, τ, χ, σ]) -> list[ν]:
        ...

    @abstractmethod
    def create_theory(self, assume: Iterable[α]) -> θ:
        ...

    @abstractmethod
    def create_true_node(self) -> ν:
        ...

    def pop_block(self) -> None:
        assert self.blocks is not None
        logger.debug(f'entering {self.pop_block.__name__}')
        assert self.matrix is not None, self.matrix
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
        self.root_nodes = self.create_root_nodes(vars_, matrix)

    def final_simplification(self):
        logger.debug(f'entering {self.final_simplification.__name__}')
        if logger.isEnabledFor(logging.DEBUG):
            num_atoms = sum(1 for _ in self.matrix.atoms())
            logger.debug(f'found {num_atoms} atoms')
        logger.info('final simplification')
        timer = Timer()
        self.result = self.final_simplify(self.matrix, assume=self.theory.atoms)
        self.time_final_simplification = timer.get()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'{self.time_final_simplification=:.3f}')
            num_atoms = sum(1 for _ in self.result.atoms())
            logger.debug(f'produced {num_atoms} atoms')

    @abstractmethod
    def final_simplify(self, formula: Formula[α, τ, χ, σ], assume: Iterable[α] = []) \
            -> Formula[α, τ, χ, σ]:
        ...

    @classmethod
    @abstractmethod
    def init_env(cls, arg: ι) -> None:
        ...

    @abstractmethod
    def init_env_arg(self) -> ι:
        ...

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
            # Otherwise, they would remain in the process table as zombies.
            mp.active_children()

        assert self.options is not None
        assert self.root_nodes is not None
        assert self.theory is not None
        logger.debug('entering sync manager context')
        timer = Timer()
        manager: SyncManager[Formula[α, τ, χ, σ], ν]
        with SyncManager() as manager:
            self.time_syncmanager_enter = timer.get()
            logger.debug(f'{self.time_syncmanager_enter=:.3f}')
            m_lock = manager.Lock()
            working_nodes: WorkingNodeListProxy[Formula[α, τ, χ, σ], ν] = manager.WorkingNodeList()
            working_nodes.extend(self.root_nodes)
            self.root_nodes = None
            success_nodes: multiprocessing.Queue[Optional[list[ν]]] = multiprocessing.Queue()
            self.success_nodes = NodeList()
            failure_nodes = manager.NodeList()  # type: ignore
            final_theories: multiprocessing.Queue[θ] = multiprocessing.Queue()
            found_t = manager.Value('i', 0)
            processes: list[mp.Process] = []
            sentinels: list[int] = []
            log_level = logger.getEffectiveLevel()
            reference_time = delta_time_formatter.get_reference_time()
            logger.debug(f'starting worker processes in {range(self.workers)}')
            born_processes = manager.Value('i', 0)
            timer.reset()
            for i in range(self.workers):
                process = mp.Process(
                    target=self.parallel_process_block_worker,
                    args=(working_nodes, success_nodes, failure_nodes,
                          self.theory, final_theories, m_lock, found_t,
                          i, log_level, reference_time,
                          born_processes, self.init_env_arg()))
                process.start()
                processes.append(process)
                sentinels.append(process.sentinel)
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
                        if t >= self.options.log_rate:
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
            new_assumptions = []
            for i in range(self.workers):
                new_assumptions.extend(final_theories.get().atoms)
            self.theory.extend(new_assumptions)
            if found_t.value > 0:
                pl = 's' if found_t.value > 1 else ''
                logger.debug(f'{found_t.value} worker{pl} found T')
                # The exception handler for FoundT in virtual_substitution will
                # log final statistics. We do not retrieve nodes and memory,
                # which would cost significant time and space. We neither
                # retrieve the node_counter, which would be not consistent with
                # our empty nodes.
                self.working_nodes = WorkingNodeList(
                    hits=working_nodes.hits,
                    candidates=working_nodes.candidates)
                # TODO: wipe self.success_nodes and self.failure_nodes
                raise FoundT()
            logger.info(working_nodes.final_statistics())
            logger.info(self.success_nodes.final_statistics('success'))
            logger.info(failure_nodes.final_statistics('failure'))
            logger.info('importing results from manager')
            logger.debug('importing working nodes from manager')
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
            logger.debug('importing failure nodes from manager')
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

    @classmethod
    def parallel_process_block_worker(cls,
                                      working_nodes: WorkingNodeListProxy,
                                      success_nodes: multiprocessing.Queue[Optional[list[Node]]],
                                      failure_nodes: NodeListProxy,
                                      theory: Theory,
                                      final_theories: multiprocessing.Queue[Theory],
                                      m_lock: threading.Lock,
                                      found_t: mp.sharedctypes.Synchronized,
                                      i: int,
                                      log_level: int,
                                      reference_time: float,
                                      born_processes: mp.sharedctypes.Synchronized,
                                      init_env_arg: ι) -> None:
        try:
            with m_lock:
                born_processes.value += 1
            multiprocessing_logger.setLevel(log_level)
            multiprocessing_formatter.set_reference_time(reference_time)
            multiprocessing_logger.debug(f'worker process {i} is running')
            cls.init_env(init_env_arg)
            while found_t.value == 0 and not working_nodes.is_finished():
                try:
                    node = working_nodes.pop()
                except IndexError:
                    time.sleep(0.001)
                    continue
                try:
                    nodes = node.process(theory)
                except NodeProcessFailure:
                    failure_nodes.append(node)
                    working_nodes.task_done()
                    continue
                except FoundT:
                    with m_lock:
                        found_t.value += 1
                    break
                if nodes:
                    if nodes[0].variables:
                        working_nodes.extend(nodes)
                    else:
                        success_nodes.put(nodes)
                working_nodes.task_done()
        except KeyboardInterrupt:
            multiprocessing_logger.debug(f'worker process {i} caught KeyboardInterrupt')
        success_nodes.put(None)
        multiprocessing_logger.debug(f'sending {theory=}')
        final_theories.put(theory)
        multiprocessing_logger.debug(f'worker process {i} exiting')

    def process_block(self) -> None:
        logger.debug(f'entering {self.process_block.__name__}')
        if self.workers > 0:
            return self.parallel_process_block()
        return self.sequential_process_block()

    def sequential_process_block(self) -> None:
        assert self.options is not None
        assert self.root_nodes is not None
        self.working_nodes = WorkingNodeList()
        self.working_nodes.extend(self.root_nodes)
        self.root_nodes = None
        self.success_nodes = NodeList()
        self.failure_nodes = NodeList()
        if logger.isEnabledFor(logging.INFO):
            last_log = time.time()
        while self.working_nodes.nodes:
            if logger.isEnabledFor(logging.INFO):
                t = time.time()
                if t - last_log >= self.options.log_rate:
                    logger.info(self.working_nodes.periodic_statistics())
                    logger.info(self.success_nodes.periodic_statistics('S'))
                    logger.info(self.failure_nodes.periodic_statistics('F'))
                    last_log = t
            node = self.working_nodes.pop()
            try:
                nodes = node.process(self.theory)
            except NodeProcessFailure:
                self.failure_nodes.append(node)
                continue
            if nodes:
                if nodes[0].variables:
                    self.working_nodes.extend(nodes)
                else:
                    self.success_nodes.extend(nodes)
        logger.info(self.working_nodes.final_statistics())
        logger.info(self.success_nodes.final_statistics('success'))
        logger.info(self.failure_nodes.final_statistics('failure'))

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
                f'    root_nodes    = {self.root_nodes}\n'
                f'    working_nodes = {nodes_as_str(self.working_nodes)},\n'
                f'    success_nodes = {nodes_as_str(self.success_nodes)},\n'
                f'    failure_nodes = {nodes_as_str(self.failure_nodes)},\n'
                f'    result        = {self.result}'
                f')')

    # discuss: We probably do not want to print
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

    def quantifier_eliminiation(self, f: Formula[α, τ, χ, σ], workers: int):
        """Quantifier elimination main loop.
        """
        logger.debug(f'entering {self.quantifier_eliminiation.__name__}')
        f = f.to_pnf()
        self.matrix, self.blocks = f.matrix()
        while self.blocks:
            try:
                self.pop_block()
                self.process_block()
            except FoundT:
                logger.info('found T')
                if self.working_nodes is None:
                    logger.info(WorkingNodeList().final_statistics())
                else:
                    logger.info(self.working_nodes.final_statistics())
                    self.working_nodes = None
                self.success_nodes = NodeList(nodes=[self.create_true_node()])
            else:
                if self.failure_nodes:
                    n = len(self.failure_nodes.nodes)
                    raise NoTraceException(f'Failed - {n} failure nodes')
            self.collect_success_nodes()
        self.final_simplification()
        logger.debug(f'leaving {self.quantifier_eliminiation.__name__}')
        return self.result
