from __future__ import annotations

import multiprocessing as mp
import multiprocessing.managers
import threading
from abc import abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Collection, Generic, Iterable, Iterator, Optional, Self, TypeVar

from ..firstorder import And, AtomicFormula, _F, Formula, Or, Variable

φ = TypeVar('φ', bound=Formula)
ν = TypeVar('ν', bound='Node')
χ = TypeVar('χ', bound=Variable)
θ = TypeVar('θ')


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


class QuantifierElimination:
    pass
