from logic1.firstorder import *
from logic1.theories.Complex import *


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

    formula4 = Im((x + y)**2) <= Re(z) * (I + w)
    # assert repr(formula4) == 'Im((x + y)**2) <= Re(z) * (I + w)'

    formula5 = Re(x * I) + Im(y + z) > -(w**2)
    # assert repr(formula5) == 'Re(x * I) + Im(y + z) > -w**2'

    formula6 = -(x + I * y) < x * Im(-1 * w) - I**2
    # assert repr(formula6) == '-(x + I * y) < x * Im(-1 * w) - I**2'

    # ((z**2 - I / 2)).normalize()
    # (Im(z) * Re(z**2)).normalize()


def test_str():
    x, y, z = VV.get('x', 'y', 'z')
    assert str(x**3) == 'x^3'
    # assert str(I**3) == '-i'


def test_latex():
    ...  # TODO

