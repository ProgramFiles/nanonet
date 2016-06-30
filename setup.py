import os
import sys
import platform
import re
import numpy
import subprocess
from glob import glob
from setuptools import setup, find_packages, Extension, Command
import pkg_resources

print """
*********************************************************************************
Proprietary and confidential information of Oxford Nanopore Technologies, Limited
All rights reserved; (c)2016: Oxford Nanopore Technologies, Limited
*********************************************************************************
"""

# Get the version number from __init__.py
verstrline = open(os.path.join('nanonet', '__init__.py'), 'r').read()
vsre = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(vsre, verstrline, re.M)
if mo:
    version = mo.group(1)
else:
    raise RuntimeError('Unable to find version string in "nanonet/__init__.py".')


OPTIMISATION = [ '-O3', '-DNDEBUG', '-fstrict-aliasing' ]
c_compile_args = ['-pedantic', '-Wall', '-std=c99'] + OPTIMISATION
cpp_compile_args = ['-std=c++0x'] + OPTIMISATION

boost_python = 'boost_python'

pkg_path = os.path.join(os.path.dirname(__file__), 'nanonet')


system = platform.system()
print "System is {}".format(system)

main_include = os.path.join(os.path.dirname(__file__), 'nanonet', 'include')
include_dirs = [main_include]
boost_inc = []
boost_lib_path = []
boost_libs = []
if system == 'Darwin':
    # Could make this better
    print "Adding OSX compile/link options"
    boost_inc = ['/opt/local/include/']
    boost_libs.append('boost_python-mt')
elif system == 'Windows':
    print "Adding windows compile/link options"
    cpp_compile_args += ['/EHsc']
    include_dirs.append(os.path.join(main_include, 'extras'))
    boost_lib_path = ['c:\\boost_1_55_0\\lib64-msvc-9.0']
    boost_inc = ['c:\\boost_1_55_0']
    # no boost_python
    cpp_compile_args += ['-DBOOST_PYTHON_STATIC_LIB']
else:
    boost_libs.append('boost_python')
#eventdetect = os.path.join(os.path.dirname(__file__), 'nanonet', 'eventdetection')
#decode = os.path.join(os.path.dirname(__file__), 'nanonet')
#maths = os.path.join(os.path.dirname(__file__), 'nanonet', 'fastmath')



extensions = []

extensions.append(Extension(
    'nanonetfilters',
    sources=[os.path.join(pkg_path, 'eventdetection', 'filters.c')],
    include_dirs=include_dirs,
    extra_compile_args=c_compile_args
))

extensions.append(Extension(
    'nanonetdecode',
    sources=[os.path.join(pkg_path, 'decoding.cpp')],
    include_dirs=include_dirs,
    extra_compile_args=cpp_compile_args
))

caller_2d_path = os.path.join('nanonet', 'caller_2d')
extensions.append(Extension(
    'nanonet.caller_2d.viterbi_2d.viterbi_2d',
    include_dirs=[os.path.join(caller_2d_path, 'viterbi_2d')] +
                 [os.path.join(caller_2d_path, 'common')] +
                 [numpy.get_include()] + boost_inc + include_dirs,
    sources=[os.path.join(caller_2d_path, 'viterbi_2d', x)
             for x in ['viterbi_2d_py.cpp', 'viterbi_2d.cpp']],
    depends=[os.path.join(caller_2d_path, x)
             for x in ['viterbi_2d_py.h', 'viterbi_2d.h']] +
            [os.path.join(caller_2d_path, 'common', x)
             for x in ['bp_tools.h', 'data_view.h', 'utils.h', 'view_numpy_arrays.h']],
    extra_compile_args=cpp_compile_args,
    library_dirs=boost_lib_path,
    libraries=boost_libs
))

extensions.append(Extension(
    'nanonet.caller_2d.pair_align.pair_align',
    include_dirs=[os.path.join(caller_2d_path, 'pair_align')] + boost_inc + include_dirs,
    sources=[os.path.join(caller_2d_path, 'pair_align', x)
             for x in ['pair_align_py.cpp', 'nw_align.cpp', 'mm_align.cpp']],
    depends=[os.path.join(caller_2d_path, 'pair_align', x)
             for x in ['pair_align_py.h', 'pair_align.h', 'nw_align.h', 'mm_align.h']],
    extra_compile_args=cpp_compile_args,
    library_dirs=boost_lib_path,
    libraries=boost_libs
))

extensions.append(Extension(
    'nanonet.caller_2d.common.stub',
    include_dirs=[os.path.join(caller_2d_path, 'common')] +
                 [numpy.get_include()] + boost_inc + include_dirs,
    sources=[os.path.join(caller_2d_path, 'common', 'stub_py.cpp')],
    depends=[os.path.join(caller_2d_path, 'common', x)
             for x in ['bp_tools.h', 'data_view.h', 'utils.h', 'view_numpy_arrays.h']],
    extra_compile_args=cpp_compile_args,
    library_dirs=boost_lib_path,
    libraries=boost_libs
))

opencl_build = False
if opencl_build:
    #TODO: make this work
    extensions.append(Extension(
        'nanonet.caller_2d.viterbi_2d_ocl.viterbi_2d_ocl',
        include_dirs=[opencl_include, os.path.join(caller_2d_path, 'viterbi_2d_ocl'),
                      os.path.join(caller_2d_path, 'common')] +
                     boost_inc + [numpy.get_include()],
        sources=[os.path.join(caller_2d_oath, 'viterbi_2d_ocl', x)
                 for x in ['viterbi_2d_ocl_py.cpp', 'viterbi_2d_ocl.cpp', 'proxyCL.cpp']],
        depends=[os.path.join(caller_2d_path, 'viterbi_2d_ocl', x)
                 for x in ['viterbi_2d_ocl.py.h', 'viterbi_2d_ocl.h', 'proxyCL.h']] +
                [os.path.join(caller_2d_path, 'common', x)
                 for x in ['bp_tools.h', 'data_view.h', 'utils.h', 'view_numpy_arrays.h']],
        extra_compile_args=['-Wall', '-std=c++0x'] + OPTIMISATION + MFPMATH,
        library_dirs=[opencl_lib],
        libraries=['boost_python', 'OpenCL'],
    ))


requires=[
    'h5py',
    'myriad >=0.1.2',
    'numpy',
]
extra_requires = {
    'currennt': ['netCDF4'],
    'watcher': ['watchdog'],
    'opencl': ['pyopencl']
}

setup(
    name='nanonet',
    version=version,
    description='A simple recurrent neural network based basecaller nanopore data.',
    maintainer='Chris Wright',
    maintainer_email='chris.wright@nanoporetech.com',
    url='http://www.nanoporetech.com',
    packages=find_packages(exclude=["*.test", "*.test.*", "test.*", "test"]),
    package_data={'nanonet.data':['nanonet/data/*']},
    include_package_data=True,
    tests_require=requires,
    install_requires=requires,
    extras_require=extra_requires,
    dependency_links=[],
    zip_safe=True,
    ext_modules=extensions,
    test_suite='discover_tests',
    entry_points={
        'console_scripts': [
            'nanonetcall = nanonet.nanonetcall:main',
            'nanonet2d = nanonet.nanonetcall_2d:main',
            'nanonettrain = nanonet.nanonettrain:main'
        ]
    }
)
