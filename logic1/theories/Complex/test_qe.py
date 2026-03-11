from logic1.firstorder import *
from logic1.theories.Complex import *

def test_qe():
    a, b, c, x, y, z, w = VV.get('a', 'b', 'c', 'x', 'y', 'z', 'w')
    phi = Ex([w, z], And( a * z + b == 0, a**2 * w + b == 0, w != z))
    qe(phi)