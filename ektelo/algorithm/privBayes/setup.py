from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(ext_modules = cythonize(Extension(
           "privBayesSelect",                           # the extension name
           sources=["privBayesSelect.pyx", 
           "lib/methods.cpp",
           "lib/table.cpp", 
           "lib/translator.cpp", 
           "lib/noise.cpp",
           "lib/privBayes_model.cpp"],            # the Cython source and
                                                  # additional C++ source files
           language="c++",                        # generate and compile C++ code
           extra_compile_args=["-std=c++11"],
           extra_link_args=["-std=c++11"],
           include_dirs=[numpy.get_include()]
      )))