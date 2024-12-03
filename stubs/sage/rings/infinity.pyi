from __future__ import annotations
from typing import Self


class MinusInfinity:

    # The following do not exist in sage. Compare sage.rings.rational.pyi.
    #
    ####################################################################
    def __gt__(self, other) -> bool:
        ...

    def __lt__(self, other) -> bool:
        ...
    ####################################################################

    def __add__(self, other: object) -> Self:
        # Addition of oo and -oo raises a exception.
        ...

    def __mul__(self, other: object) -> PlusInfinity | MinusInfinity:
        ...

    def __radd__(self, other: object) -> Self:
        # Addition of oo and -oo raises a exception.
        ...

    def __rmul__(self, other: object) -> PlusInfinity | MinusInfinity:
        ...

    def __neg__(self) -> PlusInfinity:
        ...


class PlusInfinity:

    # The following do not exist in sage. Compare sage.rings.rational.pyi.
    #
    ####################################################################
    def __gt__(self, other) -> bool:
        ...

    def __lt__(self, other) -> bool:
        ...
    ####################################################################

    def __add__(self, other: object) -> Self:
        # Addition of oo and -oo raises a exception.
        ...

    def __mul__(self, other: object) -> PlusInfinity | MinusInfinity:
        ...

    def __radd__(self, other: object) -> Self:
        # Addition of oo and -oo raises a exception.
        ...

    def __rmul__(self, other: object) -> PlusInfinity | MinusInfinity:
        ...

    def __neg__(self) -> MinusInfinity:
        ...
