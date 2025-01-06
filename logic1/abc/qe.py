"""This module :mod:`logic1.abc.qe` provides generic classes for effective
quantifier elimination, which can used by various theories via subclassing.
"""

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
from logic1.support.logging import DeltaTimeFormatter, Timer

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
τ = TypeVar('τ', bound='Term')
χ = TypeVar('χ', bound=Variable)
σ = TypeVar('σ')

φ = TypeVar('φ', bound=Formula)
"""A type variable denoting a formula with upper bound
:class:`logic1.firstorder.formula.Formula`.
"""

ν = TypeVar('ν', bound='Node')
"""A type variable denoting a node with upper bound
:class:`logic1.abc.qe.Node`.
"""

ι = TypeVar('ι')
"""A type variable denoting the type of the principal argument of the
abstract method :meth:`.QuantifierElimination.init_env`."""

λ = TypeVar('λ', bound='Assumptions')
"""A type variable denoting a assumptions with upper bound :class:`.Assumptions`.
"""

ω = TypeVar('ω', bound='Options')
"""A type variable denoting a options for
:meth:`.QuantifierElimination.__call__` with upper bound :class:`.Options`.
"""


class FoundT(Exception):
    pass


class NodeProcessFailure(Exception):
    pass


@dataclass
class Node(Generic[φ, χ, λ]):
    """Holds a subproblem for existential quantifier elimination. Theories
    implementing the interface can put restrictions on the existing fields and
    add further fields.
    """

    # This is used in both the sequential and the parallel code.

    variables: list[χ]
    """A list of variables.
    """

    formula: φ
    """A quantifier-free formula.
    """

    @abstractmethod
    def copy(self) -> Self:
        """Create a copy of this node.
        """
        ...

    @abstractmethod
    def process(self, assumptions: λ) -> list[Self]:
        """This `node` describes a formula ``Ex(node.variables,
        node.formula)``. Select a `variable` from ``node.variables`` and
        compute a list `S` of successor nodes such that:

        1. `variable` is not in ``successor.variables`` for `successor` in `S`;

        2. `variable` does not occur in ``successor.formula`` for `successor`
           in `S`;

        3. ``Or(*(Ex(successor.variables, successor.formula) for s in S))`` is
           logically equivalent to ``Ex(node.variables, node.formula)``.
        """
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
class Assumptions(Generic[α, τ, χ, σ]):
    """Holds the currently valid assumptions. This starts with user assumptions
    explicitly provided by the user. Certain variants of quantified elimination
    may add further assumptions in the course of the elimination.

    .. seealso::
        * The argument `assume` of :meth:`.QuantifierElimination.__call__`.
        * Generic quantifier elimination in :mod:`.RCF.qe`.

    This is an upper bound for the type variable :data:`.λ`.
    """

    class Inconsistent(Exception):
        """Raised when the assumptions made become inconsistent.
        """
        pass

    atoms: list[α]
    """A list of atoms holding the current set of assumptions.
    """

    def __init__(self, atoms: Iterable[α]) -> None:
        self.atoms = list(atoms)

    def append(self, new_atom: α) -> None:
        """Add `new_atom` as another assumption and simplify.
        """
        self.extend([new_atom])

    def extend(self, new_atoms: Iterable[α]) -> None:
        """Add `new_atoms` as further assumptions and simplify.
        """
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
        """`f` is a (possibly unary or trivial) conjunction of atoms. Simplifes
        `f` in such a way that the result is again a (possibly unary or
        trivial) conjunction of atoms. Raises :class:`.Inconsistent` if `f` is
        simplified to :data:`.F`.
        """
        ...


@dataclass
class Options:
    """This class holds options that can be provided to
    :meth:`.QuantifierElimination.__call__`. Theories subclassing
    :class:`.QuantifierElimination` can add further options by subclassing
    :class:`.Options`.

    This is an upper bound for the type variable :data:`.ω`.
    """

    log_level: int
    """The `log_level` of the logger used by :class:`.QuantifierElimination`.
    """

    log_rate: float
    """The minimal timespan (in s) between to log outputs in certain loops.
    """

    workers: int
    """The number of worker processes used. For more information see the
    documentation of the parameter `workers` of :meth:`.__call__`.

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
    """

    def __init__(self, log_level: int = logging.NOTSET, log_rate: float = 0.5,
                 workers: int = 0) -> None:
        self.log_level = log_level
        self.log_rate = log_rate
        if workers >= 0:
            self.workers = workers
        else:
            cpu_count = os.cpu_count()
            if cpu_count is None:
                raise ValueError(f'{os.cpu_count()=}, i.e. undetermined')
            if cpu_count + workers < 1:
                raise ValueError(f'negative number of workers')
            self.workers = cpu_count + workers


@dataclass
class QuantifierElimination(Generic[ν, λ, ι, ω, α, τ, χ, σ]):
    """A generic callable class that implements quantifier elimination.
    """

    # Attribute group 1 - arguments of :meth:`.__call__`:

    options: Optional[ω] = None
    """The options that have been passed to :meth:`.__call__`.
    """

    # Attribute group 2 - arguments state of the computation:
    _assumptions: Optional[λ] = None
    """Wraps a list of atoms, which serve as external assumptions. This
    includes the assumptions passed via the `assume` parameter of
    :meth:`__call__`. Some theories have an option for *generic quantifier
    elimination*, which adds additional assumptions on parameters in the course
    of the elimination.
    """

    blocks: Optional[Prefix[χ]] = None
    """Remaining quantifier blocks, to be processed after the current block.
    """

    matrix: Optional[Formula[α, τ, χ, σ]] = None
    """The quantifier-free formula associated with :attr:`.blocks`. This is
    :obj:`None` while there is a block being processed.
    """

    negated: Optional[bool] = None
    """Indicates whether or not the block currently processed has been
    logically negated in order to equivalently transform universal quantifiers
    into existential quanitifers.
    """

    root_nodes: Optional[list[ν]] = None
    """The root nodes of the next block to be processed. Logically, the list
    describes a disjunction, and each `node` in `root_nodes` describes a
    quantifier elimination subproblem ``Ex(node.variables, node.formula)``.
    This is an intermediate object for moving the innermost block with and the
    matrix into the :attr:`.working_nodes`.
    """

    working_nodes: Optional[WorkingNodeList[Formula[α, τ, χ, σ], ν]] = None
    """Subproblems left for the current block. Element nodes of
    :attr:`.working_nodes` have the same shape as element nodes of
    :attr:`.root_nodes`
    """

    success_nodes: Optional[NodeList[Formula[α, τ, χ, σ], ν]] = None
    """Finished subproblems of the current block. For each `node` in
    `success_nodes` we have ``node.variables == []``.
    """

    failure_nodes: Optional[NodeList[Formula[α, τ, χ, σ], ν]] = None
    """Failed subproblems of the current block, which can occur with incomplete
    quantifier elimination procedures. An element nodes of
    :attr:`.failure_nodes` have the same shape as an element node of
    :attr:`.working_nodes`, but quantifier elimination procedure could not
    eliminate any variable from the node.
    """

    result: Optional[Formula[α, τ, χ, σ]] = None
    """The final result as returned by :meth:`.__call__`.
    """

    # Attribute group 3 - timings; all times are wall times in seconds:
    time_final_simplification: Optional[float] = None
    """The time spent for finally simplifying the disjunction over all
    :attr:`.success_nodes` imported from the workers. This yields the final
    :attr:`.result`, which is also the return value of :meth:`.__call__`.
    """

    time_import_failure_nodes: Optional[float] = None
    """The time spent for importing all :attr:`.failure_nodes` from the
    :class:`SyncManager <multiprocessing.managers.SyncManager>` into the master
    process after all workers have terminated.
    """

    time_import_success_nodes: Optional[float] = None
    """The time spent for importing all :attr:`.success_nodes` from the
    :class:`SyncManager <multiprocessing.managers.SyncManager>` into the master
    process after all workers have terminated.
    """

    time_import_working_nodes: Optional[float] = None
    """The time spent for importing all :attr:`.working_nodes` from the
    :class:`SyncManager <multiprocessing.managers.SyncManager>` into the master
    process after all workers have terminated.
    """

    time_multiprocessing: Optional[float] = None
    """The time spent in :mod:`multiprocessing` after the first worker process
    has been started and until the last worker process has terminated.
    """

    time_start_first_worker: Optional[float] = None
    """The time spent for starting the first worker process in
    :mod:`multiprocessing`.
    """

    time_start_all_workers: Optional[float] = None
    """The time spent for starting all worker processes in
    :mod:`multiprocessing`.
    """

    time_syncmanager_enter: Optional[float] = None
    """The time spent for starting the :class:`SyncManager
    <multiprocessing.managers.SyncManager>`, which is a proxy process that
    manages shared data in :mod:`multiprocessing`.
    """

    time_syncmanager_exit: Optional[float] = None
    """The time spent for exiting the :class:`SyncManager
    <multiprocessing.managers.SyncManager>`.
    """

    time_total: Optional[float] = None
    """The total time spent in :meth:`.__call__`.
    """

    def __call__(self, f: Formula[α, τ, χ, σ], assume: Iterable[α] = [], **options) \
            -> Optional[Formula[α, τ, χ, σ]]:
        """The entry point of the callable class
        :class:`.QuantifierElimination`.

        :param f:
          The input formula to which quantifier elimination will be applied.

        :param assume:
          A list of atomic formulas that are assumed to hold. The return value
          is equivalent modulo those assumptions.

        :param `**options`:
          Keyword arguments with keywords corresponding to attributes of the
          generic type :data:`.ω`, which extends :class:`.Options`.

        :returns:
          A quantifier-free equivalent of `f` modulo certain assumptions. A
          simplified equivalent of all relevant assumptions are available as
          :attr:`.assumptions`.

          * Regularly, the assumptions are exactly those passed as the `assume`
            parameter.

          * Some theories have an option for *generic quantifier elimination*,
            which adds additional assumptions in the course of the elimination.
        """
        timer = Timer()
        delta_time_formatter.set_reference_time(time.time())
        # We call __init__ in order to reset all attributes of the data class
        # also within __call__. This is not really nice, but it does the job
        # and saves some code.
        QuantifierElimination.__init__(self)  # dicuss: NF is :-(
        self._assumptions = self.create_assumptions(assume)
        self.options = self.create_options(**options)
        save_level = logger.getEffectiveLevel()
        try:
            logger.setLevel(self.options.log_level)
            logger.info(f'{self.options}')
            result = self.quantifier_eliminiation(f)
            logger.info('finished')
        except KeyboardInterrupt:
            logger.info('keyboard interrupt')
            raise NoTraceException('KeyboardInterrupt')
        finally:
            logger.setLevel(save_level)
        self.time_total = timer.get()
        return result

    @property
    def assumptions(self) -> list[α]:
        """A list of atoms, which serve as external assumptions. This includes
        the assumptions passed via the `assume` parameter of
        :meth:`__call__`. Some theories have an option for *generic quantifier
        elimination*, which adds additional assumptions on parameters in
        the course of the elimination.
        """
        assert self._assumptions is not None
        return self._assumptions.atoms.copy()

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
        """Create an instance of :data:`.ω` that holds `**kwargs`. The
        `**kwargs` arriving here are the `**options` that have been passed to
        :meth:`__call__`.
        """
        ...

    @abstractmethod
    def create_root_nodes(self, variables: Iterable[χ], matrix: Formula[α, τ, χ, σ]) -> list[ν]:
        """If `matrix` is not a disjunction, create a list containing one
        instance `node` of :data:`.ν` with ``node.variables == variables``
        and ``node.formula == matrix``. If `matrix` is a disjunction
        ``Or(*args)``, create a list containing one such node for each `arg` in
        `args`.
        """
        ...

    @abstractmethod
    def create_assumptions(self, assume: Iterable[α]) -> λ:
        """Create in instance of :data:`.λ` that holds `assume`. Those
        assumptions `assume` are the corresponding parameter of
        :meth:`.__call__`.
        """
        ...

    @abstractmethod
    def create_true_node(self) -> ν:
        """Create an instance `node` of :data:`.ν` with ``node.variables ==
        []`` and ``node.formula == _T()``.
        """
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
        self.result = self.final_simplify(self.matrix, assume=self._assumptions.atoms)
        self.time_final_simplification = timer.get()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'{self.time_final_simplification=:.3f}')
            num_atoms = sum(1 for _ in self.result.atoms())
            logger.debug(f'produced {num_atoms} atoms')

    @abstractmethod
    def final_simplify(self, formula: Formula[α, τ, χ, σ], assume: Iterable[α] = []) \
            -> Formula[α, τ, χ, σ]:
        """Used for simplifying the disjunction of all :attr:`.success_nodes`.
        The return value yields :attr:`result`, which is then used as the
        return value of :meth:`.__call__`.
        """
        ...

    @classmethod
    @abstractmethod
    def init_env(cls, arg: ι) -> None:
        """A hook for initialization of worker process. This is used, e.g., in
        :ref:`Real Closed Fields <api-RCF-qe>` for reconstructing within the
        worker the Sage polynomial ring of the master.
        """
        ...

    @abstractmethod
    def init_env_arg(self) -> ι:
        """Create an instance of :data:`.ι` to be used as an argument for a
        subsequent call of :meth:`.init_env()`.
        """
        ...

    def parallel_process_block(self) -> None:

        def wait_for_processes_to_finish():
            still_running = sentinels.copy()
            while still_running:
                for sentinel in mp.connection.wait(still_running):
                    still_running.remove(sentinel)
                num_finished = self.options.workers - len(still_running)
                pl = 'es' if num_finished > 1 else ''
                logger.debug(f'{num_finished} worker process{pl} finished, '
                             f'{len(still_running)} running')
            # The following call joins all finished processes as a side effect.
            # Otherwise, they would remain in the process table as zombies.
            mp.active_children()

        assert self.options is not None
        assert self.root_nodes is not None
        assert self._assumptions is not None
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
            final_assumptions: multiprocessing.Queue[λ] = multiprocessing.Queue()
            found_t = manager.Value('i', 0)
            processes: list[mp.Process] = []
            sentinels: list[int] = []
            log_level = logger.getEffectiveLevel()
            reference_time = delta_time_formatter.get_reference_time()
            logger.debug(f'starting worker processes in {range(self.options.workers)}')
            born_processes = manager.Value('i', 0)
            timer.reset()
            for i in range(self.options.workers):
                process = mp.Process(
                    target=self.parallel_process_block_worker,
                    args=(working_nodes, success_nodes, failure_nodes,
                          self._assumptions, final_assumptions, m_lock, found_t,
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
                if self.options.workers > 1:
                    while born < self.options.workers:
                        time.sleep(0.0001)
                        with m_lock:
                            born = born_processes.value
                    self.time_start_all_workers = timer.get()
                else:
                    self.time_start_all_workers = self.time_start_first_worker
                logger.debug(f'{self.time_start_all_workers=:.3f}')
                workers_running = self.options.workers
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
            for i in range(self.options.workers):
                new_assumptions.extend(final_assumptions.get().atoms)
            self._assumptions.extend(new_assumptions)
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
                                      assumptions: Assumptions,
                                      final_assumptions: multiprocessing.Queue[Assumptions],
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
                    nodes = node.process(assumptions)
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
        multiprocessing_logger.debug(f'sending {assumptions=}')
        final_assumptions.put(assumptions)
        multiprocessing_logger.debug(f'worker process {i} exiting')

    def process_block(self) -> None:
        assert self.options is not None
        logger.debug(f'entering {self.process_block.__name__}')
        if self.options.workers > 0:
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
                nodes = node.process(self._assumptions)
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
        assert self.options is not None
        match self.options.workers:
            case 0:
                print(f'{self.options.workers=}')
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
                print(f'{self.options.workers=}')
                print(f'{self.time_syncmanager_enter=:.{precision}f}')
                print(f'{self.time_start_first_worker=:.{precision}f}')
                print(f'{self.time_start_all_workers=:.{precision}f}')
                print(f'{self.time_multiprocessing=:.{precision}f}')
                print(f'{self.time_import_working_nodes=:.{precision}f}')
                print(f'{self.time_import_failure_nodes=:.{precision}f}')
                print(f'{self.time_final_simplification=:.{precision}f}')
                print(f'{self.time_syncmanager_exit=:.{precision}f}')
                print(f'{self.time_total=:.{precision}f}')

    def quantifier_eliminiation(self, f: Formula[α, τ, χ, σ]):
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
