# Usage: python setup.py bdist_wheel

import setuptools  # type: ignore

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='logic1',
    version='0.1.1',
    author="Thomas Sturm",
    author_email="tsturm@me.com",
    description="Interpreted first-order logic in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thomas-sturm/logic1",
    packages=setuptools.find_packages(
        exclude=['logic1.theories', 'logic1.abc']
    ),
    install_requires=[
        'sympy',
        'pyeda',
        'typing_extensions'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-2-Clause",
        "Operating System :: OS Independent",
    ],
)
