from logic1.firstorder import *
from logic1.theories.Complex import *


def test_dump():
    a, b, x, y, z = VV.get("a", "b", "x", "y", "z")

    phi = z == Re(z) + I * Im(z)
    # assert phi._dump() == "Eq(Variable('z'), Add(Re(Variable('z')), Mul(_I(), Im(Variable('z')))))"

    phi = z**2 == -1
    # assert phi._dump() == "Eq(Pow(Variable('z'), 2), Neg(Rational(mpq(1,1))))"

    phi = x*y**2 + Re(a - b) - 3j
    # assert phi._dump() == "Add(Mul(Variable('x'), Pow(Variable('y'), 2)), Re(Add(Variable('a'), Neg(Variable('b')))), Neg(Mul(Rational(mpq(3,1)), _I())))"


# Symbolic Analysis Methods and Applications for Analog Circuits: A Tutorial Overview (1994)

def test_example1():
    s, G1, G2, G3, G4, G5, G6, G7, G8, G9, C1, C2 = VV.get("s", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "C1", "C2")
    p0 = -G4*G8*(G1*G2*G9 + G1*G3*G9 + G1*G9*G1*1 + G2*G6*G9 + G2*G6*G1*0)
    p1 = G7*C2*(G1*G3*G9 + G1*G3*G1*0 - G2*G5*G9 - G2*G5*G1*0)
    p2 = (-1)*G2*G7*C1*C2 * (G9 + G1*0)
    q0 = G1*1*(G9 + G1*0)*G4*G6*G8
    q1 = G1*1*(G9 + G1*0)*G5*G7*G2
    q2 = G1*1*(G9 + G1*0)*G7*G1*G2
    H = (p0 + p1*s + p2*s**2, q0 + q1*s + q2*s**2)


def test_example2():
    z, A, B, C, D, E, F, G, H, I, J, K, L = VV.get("z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L")
    p0 = D*K + D*J - A*L - A*H
    p1 = -2*D*K + A*L + A*G - D*J - D*I
    p2 = D * (K + I)
    q0 = A*E - D*B
    q1 = -A*E + 2*D*B - A*C + D*F
    q2 = (-1)*D*(F + B)
    H = (p0 + p1*z + p2*z**2, q0 + q1*z + q2*z**2)


def test_example3():
    s, gm1, gm2, gm3, go1, go2, gL, CC1, CC2, CL = VV.get("s", "gm1", "gm2", "gm3", "go1", "go2", "gL", "CC1", "CC2", "CL")
    H = (
        -gm1 * (gm2*gm3 + s*gm2*CC1 + s**2*CC1*CC2),
        (go1*go2*gL + s*gm2*gm3*CC2 + s**2*(gm3 + gL - gm2)*CC1*CC2 + s**3*CC1*CC2*CL)
    )

def test_repr():
    w, x, y, z = VV.get('w', 'x', 'y', 'z')

    term1 = Re(Im((x + y) * (z**2 - I / 2)))
    # assert repr(term1) == 'Im((z**2 + -1/2 * I) * (x + y))'

    term2 = -((x + 1j * y) * (z + w / I - 0.5)**3)
    # assert repr(term2) == '-((x + I * y) * (z + w * (-I) - 1/2)**3)'

    term3 = -(x + (y + z)) / I**3 * Re(Im(z * w - I)) + -1 * (-z**2)**0
    # assert repr(term3) == '-(x + y + z) * I * Re(Im(z * w - I)) + -1 * (-z**2)**0'

    formula1 = (x + I * y) * (z - w) == Re(Im(x**3 + y**2))
    # assert repr(formula1) == '(x + I * y) * (z - w) == Re(Im(x**3 + y**2))'

    formula2 = -(x * y + z) != I * w + Re(x**2)
    # assert repr(formula2) == '-(x * y + z) != I * w + Re(x**2)'

    formula3 = Re(x + I * y) >= Im(z**3 - w)
    # assert repr(formula3) == 'Re(x + I * y) >= Im(z**3 - w)'

    formula4 = Im((x + y)**2) <= Re(z * (I + w))
    # assert repr(formula4) == 'Im((x + y)**2) <= Re(z) * (I + w)'

    formula5 = Re(x * I) + Im(y + z) > -Re(w**2)
    # assert repr(formula5) == 'Re(x * I) + Im(y + z) > -w**2'

    formula6 = -Im(x + I * y) < Im(-1 * w - I**2)
    # assert repr(formula6) == '-(x + I * y) < x * Im(-1 * w) - I**2'

    # ((z**2 - I / 2)).normalize()
    # (Im(z) * Re(z**2)).normalize()


def test_str():
    x, y, z = VV.get('x', 'y', 'z')
    assert str(x**3) == 'x^3'
    # assert str(I**3) == '-i'


def test_latex():
    ...  # TODO