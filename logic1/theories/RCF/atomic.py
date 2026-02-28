# try:

#     import __main__

#     if __main__.choose_polylib == 'SAGE':
#         from logic1.theories.RCF.atomic_sage import *
#     elif __main__.choose_polylib == 'FLINT':
#         from logic1.theories.RCF.atomic_flint import *
#     else:
#         raise ValueError(f'illegal value {__main__.choose_polylib=}')

# except AttributeError:

#     from logic1.theories.RCF.atomic_flint import *

from logic1.theories.RCF.atomic_sage import *