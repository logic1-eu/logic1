# type: ignore

from setuptools import setup, Extension
from Cython.Build import cythonize

setup(ext_modules=cythonize([Extension(
    'logic1.theories.RCF.range',
    sources=['logic1/theories/RCF/range.pyx'],
    include_dirs=['/Users/sturm/miniforge3/envs/logic1_dev/lib/python3.12/site-packages/gmpy2'],
    # extra_compile_args=['-DCYTHON_WITHOUT_ASSERTIONS'],
    )],
    annotate=True))
