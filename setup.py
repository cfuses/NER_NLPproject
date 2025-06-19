from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = cythonize([
    Extension(
        name="skseq.sequences.sp_helpers",
        sources=["skseq/sequences/sp_helpers.pyx"],
        include_dirs=[numpy.get_include()]
    )
])

setup(
    name='skseq',
    ext_modules=extensions,
)
