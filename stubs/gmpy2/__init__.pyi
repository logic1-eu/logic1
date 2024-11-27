from sage.all import Rational


class mpq:

    def __init__(self, arg: Rational) -> None:
        ...

    def __gt__(self, other: object) -> bool:
        ...

    def __lt__(self, other: object) -> bool:
        ...

    def __neg__(self) -> Self:
        ...
