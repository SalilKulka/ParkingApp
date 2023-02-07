from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("cythonizedFunctions.pyx", annotate = "true",compiler_directives={'language_level' : "3"}),
)