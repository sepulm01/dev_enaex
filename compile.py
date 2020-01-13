from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [
	Extension("dev_enaex_v0",  ["dev_enaex_v0.py"]),

#   ... all your modules that need be compiled ...
]
setup(
    name = 'Dev Enaex',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)