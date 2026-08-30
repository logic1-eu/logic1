from gmpy2 import mpq
import pytest

from logic1.firstorder import *
from logic1.theories.Complex import *


def test_regression_mutable_hash():
    """Regression test to ensure that the hash of a variable is not affected
    by changing the global normal form of terms
    """
    z = VV['z']
    old = Term.set_normal_form(conjugate_normal_form)
    h = hash(z)
    Term.set_normal_form(cartesian_normal_form)
    assert  h == hash(z)
    Term.set_normal_form(old)


def test_regression_alphanum_variable():
    """Regression test to only allow alphanumeric variable names in the Complex
    theory. This ensures that variable names can be converted into RCF variables
    during quantifier elimination.
    """
    valid = VV['a_1']
    with pytest.raises(ValueError):
        invalid = VV['1a']
    with pytest.raises(ValueError):
        invalid = VV['a+1']