from logic1.theories.Complex import Term, VV
from logic1.theories.Complex.ast import Rat

from gmpy2 import mpq



def test_regression_repr_numbers():
    z = VV['z']
    t = z + mpq(1, 3)
    u = eval(repr(t), {'z': z, 'mpq': mpq})
    assert t.sort_key() == u.sort_key()

    t = Term(mpq(1, 3))
    u = eval(repr(t), {'mpq': mpq, 'Term': Term})
    assert isinstance(u, Term)
    assert t.sort_key() == u.sort_key()

    s = Rat(mpq(1, 3))
    v = eval(repr(s), {'mpq': mpq, 'Rat': Rat})
    assert isinstance(v, Rat)
    assert s.sort_key() == v.sort_key()

    q = mpq(1, 3) * z
    assert repr(q) == "mpq(1,3) * z"