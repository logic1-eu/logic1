from string import ascii_lowercase

from logic1.firstorder import *
from logic1.theories.RCF import *

for v in ascii_lowercase:
    globals()[v] = VV[v]
