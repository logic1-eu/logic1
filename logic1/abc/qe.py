from abc import abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import (Collection, Generic, Iterable, Iterator, Optional, Self,
                    TypeVar)

from ..firstorder import And, AtomicFormula, _F, Formula, Or, Variable

φ = TypeVar('φ', bound='Formula')
ν = TypeVar('ν', bound='Node')
χ = TypeVar('χ', bound='Variable')
θ = TypeVar('θ')


@dataclass
class Node(Generic[φ, χ, θ]):
    # sequantial and parallel

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
    # Sequantial only

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
    # Sequantial only

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
                        subnode = node.copy()
                        subnode.variables = subnode.variables.copy()
                        subnode.formula = arg
                        self.append(subnode)
                case And() | AtomicFormula():
                    self.append(node)
                case _:
                    assert False, node


class QuantifierElimination:
    pass
