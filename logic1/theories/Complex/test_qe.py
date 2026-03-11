from logic1.firstorder import *
from logic1.theories.Complex import *

def test_qe():
    a, b, c = VV.get('a', 'b', 'c')
    phi = Ex(c, All([a, b], And( a * c + b == 0, c**2 - 1 == 0)))
    # qe(phi)  # assertion error

    # Ex. 7.4
    a, b, c, d = VV.get('a', 'b', 'c', 'd')
    phi = Ex(c, All([b, a], Implies(Or(And(a == d, b ==c), And(a == c, b == 1)), a**2 == b))) 
    qe(phi)

    # some random formula
    a, w, z = VV.get('a', 'w', 'z')
    phi = Ex([w, z], And( a * z + b == 0, a**2 * w + b == 0, w != z))
    qe(phi)


def test_colinear_points():
    # real version modeled in our framework
    A, B, C, D, E, F, G, H = VV.get('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H')
    colin = lambda a, b, c: Im((c - a) * Conj(b - a)) == 0
    phi = Implies(
        And(colin(A, B, D), colin(B, C, E), colin(C, D, F), colin(D, E, G), colin(E, F, H), colin(F, G, A), colin(G, H, B), colin(H, A, C)),
        And(colin(A, B, C), colin(A, B, D), colin(A, B, E), colin(A, B, F), colin(A, B, G), colin(A, B, H))
    )
    qe(phi)

    # REMIS: Real Geometry Proving - Example by MacLane
    xb, xc, yc, xd, yd, xe, ye, xf, yf, xg, yg, xh, yh = VV.get('xb', 'xc', 'yc', 'xd', 'yd', 'xe', 'ye', 'xf', 'yf', 'xg', 'yg', 'xh', 'yh')
    phi = All([yh, xe, ye, xf, yf, xg, yg], 
              Implies(
                  And(
                      xh * yc - xc * yh == 0, 
                      xg * yf - xf * yg == 0,
                      xb * ye - xc * ye + xe * yc - xb * yc == 0,
                      xb * yh - xg * yh + xh * yg - xb * yg == 0,
                      xc * yf - xd * yf - xf * yc + xd * yc == 0,
                      xd * yg - xe * yg + xg * ye - xd * ye == 0,
                      xe * yh - xf * yh + xh * yf - xe * yf - xh * ye + xf * ye == 0
                  ), xb * yc == 0))
    # qe(phi)  # fails because of recusion limit (?)

def test_counterexamples():
    # sqrt(-1) exists in C
    z = VV['z']
    phi = Ex(z, z**2 + 1 == 0)
    assert qe(phi) == T

    # every quadratic equation has a solution in C
    a, b, c, z = VV.get('a', 'b', 'c', 'z')
    phi = All([a, b, c], Ex(z, a * z**2 + b * z + c == 0))
    # assert qe(phi) == T  # failure nodes

    # REMIS: Counterexample for Geometry Provers over C
    x1, x2 = VV.get('x1', 'x2')
    phi = All([x1, x2], Implies(And(x1**2 + x2**2 == 1, x1 == 2), (x2 == 1)))
    assert qe(phi) == F

def test_circuits():
    # Symbolic Analysis Methods and Applications for Analog Circuits: A Tutorial Overview (1994)

    def cont_stability(Hp, Hq, s):
        return All(s, Implies(Hq == 0, Re(s) < 0))
    
    def disc_stability(Hp, Hq, z):
        return All(z, Implies(Hq == 0, z * Conj(z) < 1))

    # Example 1
    s, G1, G2, G3, G4, G5, G6, G7, G8, G9, C1, C2 = VV.get('s', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'C1', 'C2')
    assume = [G1 > 0, G2 > 0, G3 > 0, G4 > 0, G5 > 0, G6 > 0, G7 > 0, G8 > 0, G9 > 0, C1 > 0, C2 > 0]

    p0 = -G4*G8*(G1*G2*G9 + G1*G3*G9 + G1*G9*G1*1 + G2*G6*G9 + G2*G6*G1*0)
    p1 = G7*C2*(G1*G3*G9 + G1*G3*G1*0 - G2*G5*G9 - G2*G5*G1*0)
    p2 = (-1)*G2*G7*C1*C2 * (G9 + G1*0)
    q0 = G1*1*(G9 + G1*0)*G4*G6*G8
    q1 = G1*1*(G9 + G1*0)*G5*G7*G2
    q2 = G1*1*(G9 + G1*0)*G7*G1*G2

    Hp = p0 + p1*s + p2*s**2 
    Hq = q0 + q1*s + q2*s**2
    phi = And(*assume, cont_stability(Hp, Hq, s))
    qe(phi)

    # Example 2
    z, A, B, C, D, E, F, G, H, I, J, K, L = VV.get('z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L')
    assume = [A > 0, B > 0, C > 0, D > 0, E > 0, F > 0, G > 0, H > 0, I > 0, J > 0, K > 0, L > 0]

    p0 = D*K + D*J - A*L - A*H
    p1 = -2*D*K + A*L + A*G - D*J - D*I
    p2 = D * (K + I)
    q0 = A*E - D*B
    q1 = -A*E + 2*D*B - A*C + D*F
    q2 = (-1)*D*(F + B)

    Hp = p0 + p1*z + p2*z**2
    Hq = q0 + q1*z + q2*z**2
    phi = And(*assume, disc_stability(Hp, Hq, z))
    qe(phi)

    # Example 3
    s, gm1, gm2, gm3, go1, go2, gL, CC1, CC2, CL = VV.get('s', 'gm1', 'gm2', 'gm3', 'go1', 'go2', 'gL', 'CC1', 'CC2', 'CL')
    assume = [gm1 > 0, gm2 > 0, gm3 > 0, go1 > 0, go2 > 0, gL > 0, CC1 > 0, CC2 > 0, CL > 0]

    Hp = -gm1 * (gm2*gm3 + s*gm2*CC1 + s**2*CC1*CC2)
    Hq = go1*go2*gL + s*gm2*gm3*CC2 + s**2*(gm3 + gL - gm2)*CC1*CC2 + s**3*CC1*CC2*CL
    phi = And(*assume, cont_stability(Hp, Hq, s))
    # qe(phi)  # failure node because of third power of s in Hq