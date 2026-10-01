
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Protocol, Self

from logic1.firstorder.term import Term, Variable
from logic1.theories.RCF.types import _NUMBER_TYPES, Number


class RingElement(Protocol):

    def __add__(self, other: Self) -> Self:
        ...

    def __sub__(self, other: Self) -> Self:
        ...

    def __mul__(self, other: Self) -> Self:
        ...

    def __neg__(self) -> Self:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __pow__(self, n: int) -> Self:
        ...

    def __hash__(self) -> int:
        ...

    def __str__(self) -> str:
        ...


class Ring[α: RingElement, β](ABC):

    _ring: β

    @abstractmethod
    def __init__(self, vars_: Iterable[str] = ()) -> None:
        ...

    def __or__(self, other: Ring[α, β]) -> Self:
        if self._ring is other._ring:
            return self
        else:
            names = set(self.get_names()) | set(other.get_names())
            return self.__class__(names)

    @abstractmethod
    def coerce_poly(self, x: Number | α) -> α:
        ...

    def get_names(self) -> tuple[str, ...]:
        return tuple(str(var) for var in self.get_gens())

    @abstractmethod
    def get_gens(self) -> tuple[α, ...]:
        ...


class AbstractTerm[τ: Term, χ: Variable, κ, α: RingElement, β](Term[τ, χ, Number, κ]):

    _hash: Optional[int]
    _poly: α

    @property
    @abstractmethod
    def _parent(self) -> Ring[α, β]:
        ...

    @abstractmethod
    def __init__(self, num: Number) -> None:
        ...

    def __add__(self, other: Number | Self) -> Self:
        if isinstance(other, _NUMBER_TYPES):
            other = self.__class__(other)
        parent = self._parent | other._parent
        self_poly = parent.coerce_poly(self._poly)
        other_poly = parent.coerce_poly(other._poly)
        return self.__class__._from_raw(self_poly + other_poly)

    def __getstate__(self):
        return {"_poly": self._poly}

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(self._poly)
        return self._hash

    def __mul__(self, other: Number | Self) -> Self:
        if isinstance(other, _NUMBER_TYPES):
            other = self.__class__(other)
        parent = self._parent | other._parent
        self_poly = parent.coerce_poly(self._poly)
        other_poly = parent.coerce_poly(other._poly)
        return self.__class__._from_raw(self_poly * other_poly)

    def __radd__(self, other: Number | Self) -> Self:
        assert not isinstance(other, Term)
        return self.__class__(other) + self

    def __rmul__(self, other: Number | Self) -> Self:
        assert not isinstance(other, Term)
        return self.__class__(other) * self

    def __rsub__(self, other: Number | Self) -> Self:
        assert not isinstance(other, Term)
        return self.__class__(other) - self

    def __setstate__(self, state):
        self._hash = None
        self._poly = state["_poly"]

    def __str__(self) -> str:
        return str(self._poly)

    def __sub__(self, other: Number | Self) -> Self:
        if isinstance(other, _NUMBER_TYPES):
            other = self.__class__(other)
        parent = self._parent | other._parent
        self_poly = parent.coerce_poly(self._poly)
        other_poly = parent.coerce_poly(other._poly)
        return self.__class__._from_raw(self_poly - other_poly)

    @classmethod
    @abstractmethod
    def _from_raw(cls, value: α) -> Self:
        ...
