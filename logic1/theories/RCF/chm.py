from logic1.firstorder import _T, _F, Formula, And, Or
from .simplify import is_valid, simplify
from .atomic import AtomicFormula, Term, Variable, VV
from gmpy2 import sign
from dataclasses import dataclass, field
from typing import Iterator, Literal, Tuple, Self, TypeAlias


Sign: TypeAlias = Literal[-1, 0, 1]


@dataclass
class Matrix:
    """
    A combined sign matrix of a set of polynomials.
    """

    rows: dict[Term, list[Sign]] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Tuple[Term, Tuple[Sign, ...]]]:
        for t in sorted(self.rows):
            yield t, tuple(self.rows[t])

    def __eq__(self, other) -> bool:
        return tuple(self) == tuple(other)
    
    def __hash__(self) -> int:
        return hash(tuple(self))
    
    def __str__(self) -> str:
        rows = '\n'.join(f'  {key}: {value}' for key, value in sorted(self.rows.items()))
        return f'Matrix:\n{rows}'    
            
    @property
    def num_rows(self) -> int:
        return len(self.rows)

    @property
    def num_columns(self) -> int:
        return len(list(self.rows.values())[0]) if self.rows else 0

    def __getitem__(self, f: Term) -> Tuple[Sign]:
        return tuple(self.row(f))

    def row(self, f: Term) -> Tuple[Sign]:
        return tuple(self.rows[f])

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

    def condense(self) -> Self:
        result = Matrix({k: [] for k in self.rows})
        for i in range(self.num_columns):
            if result.num_columns == 0 or result.column(self.rows, -1) != self.column(self.rows, i):
                result.extend(self.column(self.rows, i))
        return result 


def real_pseudo_rem(f: Term, g: Term, x: Variable, assume: set[AtomicFormula]) -> Term:
    """Sign-corrected pseudo-remainder
    """
    if f.degree(x) < g.degree(x):
        return f
    _, h = f.pseudo_quo_rem(g, x)
    delta = f.degree(x) - g.degree(x) + 1
    if delta % 2 == 1:
        lc_g_x = g.coefficient({x: g.degree(x)})
        if is_valid(lc_g_x >= 0, assume):
            pass
        elif is_valid(lc_g_x <= 0, assume):
            h = - h
        else:
            h *= lc_g_x
    return h


def lemma923(f, g0: Term, G: set[Term], h0: Term, H: dict[Term, Term], M: Matrix) -> Matrix:
    """
    Implements Lemma 9.23 in the lecture notes.
    Input:
        * non-constant polynomials f and G, f has maximal degree
        * g0 is the derivative of f
        * h0 is the remainder of f modulo g0
        * H contains the the remainder of f modulo g for each g in G
        * M is the combined sign matrix of h0, H, g0, G
    Output: 
        The combined sign matrix of f and G
    """
    #print()
    #print("call", f, g0, G, h0, H, M)
    result = Matrix({f: []} | {g: [] for g in G})
    lvalue = -M[g0][0]

    for j in range(1, M.num_columns + 1, 2):
        rvalue = None
        if j >= M.num_columns:
            rvalue = M[g0][-1]
        elif M[g0][j] == 0:
            rvalue = M[h0][j]
        else:
            for g in G:
                if M[g][j] == 0:
                    rvalue = M[H[g]][j]
                    break
        if rvalue is None:
            continue
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
        if j < M.num_columns:
            result.extend({f: rvalue} | M.column(G, j))
        lvalue = rvalue
        
    return result.condense()


def chm(F: set[Term], x: Variable, assume: set[AtomicFormula], depth: int = 0) -> Iterator[Tuple[Formula, Matrix]]:
    """
    Computes all possible combined sign matrices of the set of polynomials F with respect to the variable x and given assumptions. 
    The depth is used for debugging and records the current recusion depth.

    Outputs an iterator over pairs of precondtions and the corresponding concrete sign matrix under this precondition. 
    A sign matrix can appear multiple times.
    """
    assume = set(assume)
    condition = simplify(simplify(And(*assume)))
    # print("DEBUG:", And(*assume), "->", condition)
    if condition == _F():
        return
    # try to reduce the size of the assumptions
    if condition == _T():
        assume = set()
    elif isinstance(condition, AtomicFormula):
        assume = {condition}
    elif all(isinstance(a, AtomicFormula) for a in condition.args):
        assume = {*condition.args}

    # base case
    if not F:
        yield (condition, Matrix({}))
        return
    
    #print(f"DEBUG ({depth}):", f"call {F}, {assume}")

    # make sure that we don't have any constants in F
    for f in F:
        d = f.degree(x)
        lc = f.coefficient({x: d})
        if d < 1:
            #print(f"DEBUG ({depth}):", f"case dinstiction on constant {f}")
            # we found a constant and do a case distinction on its sign
            for assume2, M in chm(F - {f}, x, assume | {lc > 0}, depth+1):
                M.insert_constant_row(f, 1)
                yield assume2, M
            for assume2, M in chm(F - {f}, x, assume | {lc == 0}, depth+1):
                M.insert_constant_row(f, 0)
                yield assume2, M
            for assume2, M in chm(F - {f}, x, assume | {lc < 0}, depth+1):
                M.insert_constant_row(f, -1)
                yield assume2, M
            return
        
    # make sure that we only have non-zero leading coeffcients
    for f in F:
        d = f.degree(x)
        lc = f.coefficient({x: d})
        if not is_valid(lc != 0, assume):
            #print(f"DEBUG ({depth}):", f"case dinstiction on lc {f}")
            # we found a leading coefficient that could be zero. Consider the case lc = 0 and then assume lc != 0 in the following
            res = f - lc * x**d
            for assume2, M in chm((F - {f}) | {res}, x, assume | {lc == 0}, depth+1):
                M.rows[f] = M.rows[res]
                if res not in F:
                    del M.rows[res]
                yield assume2, M
            assume = assume | {lc != 0}

    # prepare the call to lemma 9.23
    f = max(F, key=lambda f: f.degree(x))
    G = F - {f}
    g0 = f.derivative(x)
    h0 = real_pseudo_rem(f, g0, x, assume)
    H = {g: real_pseudo_rem(f, g, x, assume) for g in G}
    # we have multiple options for the sign matrix M
    for assume2, M in chm({g0, *G, h0, *H.values()}, x, assume, depth+1):
        M = lemma923(f, g0, G, h0, H, M)
        yield assume2, M
        # to constants here at the end

def chm_collect(F: set[Term], x: Variable, assume: set[AtomicFormula]) -> Iterator[Tuple[Formula, Matrix]]:
    """
    Same as chm but groups duplicate matrices together.
    """
    result = dict()
    for assume2, M in chm(F, x, assume):
        result[M] = simplify(simplify(Or(assume2, result.get(M, _F()))))
    for key, val in result.items():
        yield val, key

""""
def chm2(F: set[Term], x: Variable, constants: set[Term], depth: int = 0):
    constants2 = {f for f in F if f.degree(x) < 1}
    F = F - constants2
    constants = constants | constants2

    if not F:
        yield (constants, Matrix({}))
        return
    
    for f in F:
        d = f.degree(x)
        lc = f.coefficient({x: d})
        res = f - lc * x**d
        yield from chm2((F - {f}) | {res}, x, constants, depth+1)
    
    f = max(F, key=lambda f: f.degree(x))
    G = F - {f}
    g0 = f.derivative(x)
    h0 = real_pseudo_rem(f, g0, x, set())
    H = {g: real_pseudo_rem(f, g, x, set()) for g in G}
    yield from chm2({g0, *G, h0, *H.values()}, x, constants, depth+1)

x, a, b, c, d, e, f, g = VV.get("x", "a", "b", "c", "d", "e", "f", "g")
F = {a * x**4 + b*x**3 + c*x**2 + d*x + e} # , e*x**2 + f*x + g}
for i in chm2(F, x, set()):
    out = set()
    for j in i[0]:
        #if j == 0: continue
        #print(j.factor())
        if j == 0: continue
        for k in j.factor()[1].keys():
            out.add(k)
    for o in sorted(out, key=lambda x: len(str(x))):
        print(o)
    print()
"""

def test_example_920():
    x, = VV.get("x")
    F = {x + 1, 2*x + 1, x**2 - 1}
    for assume, M in sorted(chm(F, x, set())): 
        print(assume, "\n", M)


def test_single_generic_linear():
    x, a, b = VV.get("x", "a", "b")
    F = {a * x + b}
    for assume, M in chm(F, x, set()):
        print(assume)
        print(M)
        print()

def test_single_generic_quadratic():
    x, a, b, c = VV.get("x", "a", "b", "c")
    F = {a*x**2 + b*x + c}
    for assume, M in chm(F, x, set([a != 0])):
        print(assume)
        print(M)
        print()

def test_single_generic_cubic():
    x, a, b, c, d = VV.get("x", "a", "b", "c", "d")
    F = {a * x**3 + b*x**2 + c*x + d}
    for assume, M in chm(F, x, set([a != 0])):
        print(simplify(assume, explode_always=True))
        print(M)
        print()
    print(len(list(chm(F, x, set([a == 1, b == 0])))))

def test_single_generic_cubic2():
    x, a, b, c, d = VV.get("x", "a", "b", "c", "d")
    F = {a * x**3 + b*x**2 + c*x + d}
    c = list(chm(F, x, set([a != 0])))
    print("Before", len(c))
    for assume, M in c:
        others = set(a for a, _ in c if a != assume)
        if simplify(assume, assume=others) == _T():
            continue
    # TODO remove implied conditions/nachdenken
    
    for assume, M in c:
        print(assume)
        print(M)
        print()
    

def test_marek_degree_two():
    x, a, b, c, d, e = VV.get("x", "a", "b", "c", "d", "e")
    F = {a * x**2 + b*x + c, d*x + e}
    for assume, M in chm(F, x, set()):
        print(assume)
        print(M)
        print()




def chm_test():
    print("running ...")
    # test_example_920()
    # test_single_generic_linear()
    # test_single_generic_quadratic()
    test_single_generic_cubic()
    # test_marek_degree_two()

if __name__ == "__main__":
    chm_test()


