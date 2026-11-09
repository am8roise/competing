from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
	ext_modules=cythonize([
		Extension("ed", ["ed.pyx"], include_dirs=[np.get_include()])
	],
	compiler_directives={'language_level': "3"})
)