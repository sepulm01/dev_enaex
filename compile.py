from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [
    Extension("detector_mul_rq",  ["detector_mul_rq.py"]), 
    Extension("common",  ["common.py"]), 
    Extension("centroidtracker",  ["centroidtracker.py"]),
#   ... all your modules that need be compiled ...
]
setup(
    name = 'Worker',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)