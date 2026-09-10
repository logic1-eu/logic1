from gmpy2 import mpq
import pytest

from logic1.firstorder import *
from logic1.theories.Complex import *


def test_eval():

    assert Term(5).eval() == (mpq(5), mpq(0))

    assert (-I).eval() == (mpq(0), mpq(-1))

    assert (2 + 7 * I).eval() == (mpq(2), mpq(7))

    assert (1 - I / 2).eval() == (mpq(1), mpq(-1, 2))

    assert (I ** 10).eval() == (mpq(-1), mpq(0))

    assert ((3 + I) * (1 - I)).eval() == (mpq(4), mpq(-2))

    assert ((2 + 0 * I) ** 3).eval() == (mpq(8), mpq(0))

    assert Re(9 + 5 * I).eval() == (mpq(9), mpq(0))

    assert Im(9 + 5 * I).eval() == (mpq(5), mpq(0))

    x = VV['x']
    with pytest.raises(ValueError):
        ((1 - x) ** 2).eval()
