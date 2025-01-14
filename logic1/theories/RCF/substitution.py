from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from gmpy2 import mpq
from typing import Final, Iterator, Optional, Self

from .atomic import CACHE_SIZE, Eq, SortKey, Term, Variable


@dataclass(frozen=True)
class _SubstValue:
    coefficient: mpq
    variable: Optional[Variable]

    def __eq__(self, other: Self) -> bool:  # type: ignore[override]
        if self.coefficient != other.coefficient:
            return False
        if self.variable is None and other.variable is None:
            return True
        if self.variable is None or other.variable is None:
            return False
        return self.variable.sort_key() == other.variable.sort_key()

    def __post_init__(self) -> None:
        assert self.coefficient != 0 or self.variable is None

    @lru_cache(maxsize=CACHE_SIZE)
    def as_term(self) -> Term:
        if self.variable is None:
            return Term(self.coefficient)
        else:
            return self.coefficient * self.variable


@dataclass
class _Node:
    coefficient: mpq
    variable: Optional[Variable]
    parent: Optional[_Node]


@dataclass
class _Root:
    coefficient: mpq
    node: _Node


@dataclass
class _Substitution:
    nodes: dict[SortKey[Variable], _Node] = field(default_factory=dict)
    constant_node: Final[_Node] = field(default_factory=lambda: _Node(mpq(1), None, None))

    def __iter__(self) -> Iterator[tuple[Variable, _SubstValue]]:
        for key, node in self.nodes.items():
            if node.parent is None:
                continue
            root = self.find2(node)
            yield key.term, _SubstValue(root.coefficient, root.node.variable)

    def as_gb(self, ignore: Optional[Variable] = None) -> list[Term]:
        """Convert this :class:._Substitution` into a Gröbner basis that can be
        used as an argument to reduction methods.
        """
        G = []
        for var, val in self:
            if ignore is None or var.sort_key() != ignore.sort_key():
                G.append(var - val.as_term())
        return G

    def copy(self) -> Self:
        result = self.__class__()
        for var, val in self:
            result.union(_SubstValue(mpq(1), var), val)
        return result

    def find(self, v: Variable) -> _SubstValue:
        root = self.find1(v)
        return _SubstValue(root.coefficient, root.node.variable)

    def find1(self, v: Optional[Variable]) -> _Root:
        if v is None:
            return _Root(mpq(1), self.constant_node)
        key = v.sort_key()
        node: _Node
        try:
            node = self.nodes[key]
        except KeyError:
            node = self.nodes[key] = _Node(mpq(1), v, None)
        return self.find2(node)

    def find2(self, node: _Node) -> _Root:
        if node.variable is None:
            return _Root(mpq(1), node)
        parent = node.parent
        if parent is None:
            return _Root(mpq(1), node)
        root = self.find2(parent)
        node.coefficient *= root.coefficient
        node.parent = root.node
        return _Root(node.coefficient, root.node)

    def union(self, val1: _SubstValue, val2: _SubstValue) -> None:
        root1 = self.find1(val1.variable)
        root2 = self.find1(val2.variable)
        c1 = val1.coefficient * root1.coefficient
        c2 = val2.coefficient * root2.coefficient
        if root1.node.variable is not None and root2.node.variable is not None:
            sort_key1 = root1.node.variable.sort_key()
            sort_key2 = root2.node.variable.sort_key()
            if sort_key1 == sort_key2:
                if c1 != c2:
                    root1.node.coefficient = mpq(0)
                    root1.node.parent = self.constant_node
            elif sort_key1 < sort_key2:
                root2.node.coefficient = c1 / c2
                root2.node.parent = root1.node
            else:
                root1.node.coefficient = c2 / c1
                root1.node.parent = root2.node
        elif root1.node.variable is None and root2.node.variable is not None:
            root2.node.coefficient = c1 / c2
            root2.node.parent = self.constant_node
        elif root1.node.variable is not None and root2.node.variable is None:
            root1.node.coefficient = c2 / c1
            root1.node.parent = self.constant_node
        else:
            if c1 != c2:
                from .simplify import InternalRepresentation
                raise InternalRepresentation.Inconsistent()

    def is_redundant(self, val1: _SubstValue, val2: _SubstValue) -> Optional[bool]:
        """Check if the equation ``val1 == val2`` is redundant modulo self.
        """
        root1 = self.find1(val1.variable)
        root2 = self.find1(val2.variable)
        c1 = val1.coefficient * root1.coefficient
        c2 = val2.coefficient * root2.coefficient
        if root1.node.variable is None and root2.node.variable is None:
            return c1 == c2
        if root1.node.variable is None or root2.node.variable is None:
            return None
        if root1.node.variable == root2.node.variable and c1 == c2:
            return True
        else:
            return None

    def equations(self) -> Iterator[Eq]:
        raise NotImplementedError()
