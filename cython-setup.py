# type: ignore

from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize([
    'logic1/theories/RCF/substitution.pyx']))
