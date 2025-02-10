from typing import Self


class Polynomial:

    def pseudo_quo_rem(self, other: Self) -> tuple[Self, Self]:
        ...


class Polynomial_generic_dense(Polynomial):
    ...
