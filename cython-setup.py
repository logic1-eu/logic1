# type: ignore

from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    ext_modules=cythonize([
        Extension(
            'logic1.theories.RCF.range',
            sources=['logic1/theories/RCF/range.pyx'],
        )
    ],
    annotate=True)
)
