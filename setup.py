from distutils.core import setup
from Cython.Build import cythonize
import numpy
import distutils.sysconfig
from distutils.extension import Extension
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# Remove the "-Wstrict-prototypes" compiler option, which isn't valid for C++.
cfg_vars = distutils.sysconfig.get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")

#setup(
#    ext_modules = cythonize("data.pyx", language='c++'),
#    include_dirs=[numpy.get_include()]
#)


#setup(
#    name = 'pymcinv',
#    ext_modules = cythonize(["*.pyx"], language='c++'), 
#    include_dirs=[numpy.get_include()],
#    extra_compile_args=["-std=c++11"]
#)

ext_modules = [
     Extension(
         'rffunc',
         ["rffunc.pyx"]
     )
]

setup(
    name = 'rffunc',
    ext_modules = cythonize(ext_modules), 
    include_dirs=[numpy.get_include()]
)


