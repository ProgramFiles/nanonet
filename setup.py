import os
import re
import subprocess
import sys
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

c_compile_args = [
    '-Wall', '-DNDEBUG', '-std=c99',
    '-fstrict-aliasing', '-O3', '-march=native'
]
cpp_compile_args = [
    a for a in c_compile_args if a != '-std=c99'
]

extensions = []

eventdetect = os.path.join(os.path.dirname(__file__), 'nanonet', 'eventdetection')
decode = os.path.join(os.path.dirname(__file__), 'nanonet')
maths = os.path.join(os.path.dirname(__file__), 'nanonet', 'fastmath')

include_dirs=[eventdetect, decode, maths]
if os.name == 'nt':
    include_dirs.append(os.path.join(eventdetect, 'include'))

extensions.append(Extension(
    'nanonetfilters',
    sources=[os.path.join(eventdetect, 'filters.c')],
    include_dirs=include_dirs,
    extra_compile_args=c_compile_args
))


extensions.append(Extension(
    'nanonetdecode',
    sources=[os.path.join(decode, 'decoding.cpp')],
    include_dirs=include_dirs,
    extra_compile_args=cpp_compile_args
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
            'nanonettrain = nanonet.nanonettrain:main'
        ]
    }
)
