from logic1.theories.RCF import Term, Variable, VV
from gmpy2 import sign
from dataclasses import dataclass, field
from typing import Literal, TypeAlias


Sign: TypeAlias = Literal[-1, 0, 1]


@dataclass
class Matrix:

    rows: dict[Term, list[Sign]] = field(default_factory=dict)

    @property
    def num_rows(self) -> int:
        return len(self.rows)

    @property
    def num_columns(self) -> int:
        return len(list(self.rows.values())[0]) if self.rows else 0

    def __getitem__(self, f: Term) -> list[Sign]:
        return list(self.row(f))

    def row(self, f: Term) -> list[Sign]:
        return list(self.rows[f])

    def column(self, F: set[Term], i: int) -> dict[Term, sign]:
        return {f: self.rows[f][i] for f in F}

    def insert_constant_row(self, f: Term, s: Sign) -> None:
        assert f not in self.rows
        self.rows[f] = [s] * max(1, self.num_columns)

    def extend(self, column: dict[Term, Sign]) -> None:
        if self.num_rows == 0:
            self.rows = {f: [s] for f, s in column.items()}
        else:
            for f, s in column.items():
                self.rows[f].append(s)


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


def lemma923(f: Term, G: set[Term], x: Variable) -> Matrix:
    assert f not in G
    assert all(1 <= g.degree(x) <= f.degree(x) for g in G)

    print("call lemma923", f, G)

    g0 = f.derivative(x)
    h0 = real_pseudo_rem(f, g0, x)
    H = {g: real_pseudo_rem(f, g, x) for g in G}

    M = chm({g0, *G, h0, *H.values()}, x)

    result = Matrix({f: []} | {g: [] for g in G})
    lvalue = -M[g0][0]

    for j in range(1, M.num_columns, 2):
        for g in G:
            if M[g][j] == 0:
                rvalue = M[H[g]][j]
                if lvalue == 0:
                    result.extend({f: rvalue} | M.column(G, j-1))
                elif rvalue == 0:
                    result.extend({f: lvalue} | M.column(G, j-1))
                elif rvalue == lvalue:
                    result.extend({f: lvalue} | M.column(G, j-1))
                else:
                    result.extend({f: lvalue} | M.column(G, j-1))
                    result.extend({f: 0}      | M.column(G, j-1))
                    result.extend({f: rvalue} | M.column(G, j-1))
                result.extend({f: rvalue} | M.column(G, j))
                lvalue = rvalue
                break
        else:
            if M[g0][j] == 0 and M[h0][j] == 0:
                result.extend({f: lvalue} | M.column(G, j-1))
                result.extend({f: 0}      | M.column(G, j))
                lvalue = 0

    rvalue = M[g0][-1]
    if lvalue == 0:
        result.extend({f: rvalue} | M.column(G, -1))
    elif rvalue == 0:
        result.extend({f: lvalue} | M.column(G, -1))
    elif rvalue == lvalue:
        result.extend({f: lvalue} | M.column(G, -1))
    else:
        result.extend({f: lvalue} | M.column(G, -1))
        result.extend({f: 0}      | M.column(G, -1))
        result.extend({f: rvalue} | M.column(G, -1))


    print("return lemma923", result)
    return result


def chm(F: set[Term], x: Variable) -> Matrix:
    print("call chm", F)

    constants = {f for f in F if f.degree(x) <= 0}
    F = set(F) - constants

    if not F:
        M = Matrix({})
    else:
        largest = max(F, key=lambda f: f.degree(x))
        M = lemma923(largest, F - {largest}, x)

    for constant in constants:
        M.insert_constant_row(constant, sign(constant.lc()))

    print("return chm", M)
    return M

if __name__ == "__main__":
    x, = VV.get("x")
    F = {x + 1, 2*x + 1, x**2 - 1}
    M = chm(F, x)
    for f in F:
        print(f, M.row(f))


