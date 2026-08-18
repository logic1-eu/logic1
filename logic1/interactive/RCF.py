from string import ascii_lowercase

from gmpy2 import mpq

from logic1.firstorder import *
from logic1.theories.RCF import *

for _v in ascii_lowercase:
    globals()[_v] = VV[_v]
