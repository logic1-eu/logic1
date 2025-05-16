from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from gmpy2 import mpq, sign

from .atomic import Term, Variable


Sign: TypeAlias = Literal[-1, 0, 1]


@dataclass
class Matrix:

    _data: list[list[Sign]] = field(default_factory=list)

    @property
    def cols(self) -> int:
        if self.rows == 0:
            return 0
        return len(self._data[0])

    @property
    def rows(self) -> int:
        return len(self._data)

    def __getitem__(self, indices: tuple[int, int]) -> Sign:
        i, j = indices
        return self._data[i][j]

    def append(self, row: list[Sign]) -> None:
        assert len(row) > 0
        assert self.cols == 0 or self.cols == len(row)
        self._data.append(row)

    def condense(self) -> None:
        self.transpose()
        new_data: list[list[Sign]] = []
        for row in self._data:
            if not new_data or row != new_data[-1]:
                new_data.append(row)
        self._data = new_data
        self.transpose()

    def delete(self, col: int) -> None:
        for row in self._data:
            del row[col]

    def duplicate(self, col: int) -> None:
        for row in self._data:
            row.insert(col, row[col])

    def transpose(self) -> None:
        self._data = [[self[i, j] for i in range(self.rows)] for j in range(self.cols)]


def chm(terms: list[Term], x: Variable) -> Matrix:
    if not terms:
        return Matrix()
    terms = list(sorted(terms, key=lambda t: t.degree(x)))
    f = terms.pop()
    terms.append(f.derivative(x))
    remainders = []
    for g in terms:
        h = real_pseudo_rem(f, g, x)
        assert h.degree(x) < f.degree(x)
        remainders.append(h)
    terms.extend(remainders)
    constants = list({t for t in terms if t.degree(x) == 0})
    terms = list({t for t in terms if t.degree(x) > 0})
    print(f'{len(constants)=}, {constants=}')
    M = chm(terms, x)
    # n = max(1, M.cols)
    # for constant in constants:
    #     row = n * [sign(constant.as_fraction())]
    #     M.append(row)
    return lemma923(term, M)


def lemma923(f: Term, G: list[Term], x: Variable) -> Matrix:
    f_prime = f.derivative(x)
    f0 = real_pseudo_rem(f, f_prime, x)
    F = [real_pseudo_rem(f, g, x) for g in G]
    M = chm([f_prime, *G, f0, *F], x)
    n = len(G) + 1
    j = 1
    stars = []
    while j < M.cols:
        for i in range(n):
            if M[i, 2 * j] == 0:
                stars.append(M[n + i, 2 * j])
                j += 2
                break
        else:
            M.delete(j)
            M.delete(j)
    return


def real_pseudo_rem(f: Term, g: Term, x: Variable) -> Term:
    """Sign-corrected pseudo-remainder
    """
    if f.degree(x) < g.degree(x):
        return f
    _, h = f.pseudo_quo_rem(g, x)
    delta = f.degree(x) - g.degree(x) + 1
    if delta % 2 == 1:
        lc_g_x = g.coefficient({x: g.degree(x)})
        h *= lc_g_x
    return h
