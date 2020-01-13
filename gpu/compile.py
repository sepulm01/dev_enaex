from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [
	Extension("multiserver2",  ["multiserver2.py"]),
    Extension("drk",  ["drk.py"]),
    Extension("pnet",  ["pnet.py"]),
#   ... all your modules that need be compiled ...
]
setup(
    name = 'Multiserver 2',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)