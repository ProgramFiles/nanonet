import os
import re
import subprocess
import sys
from glob import glob
from setuptools import setup, find_packages, Extension
import pkg_resources

c_compile_args = [
    '-Wall', '-DNDEBUG', '-std=c99',
    '-fstrict-aliasing', '-O3', '-march=native'
]

extensions = []
extensions.append(Extension(
    'clib_viterbi',
    sources=[os.path.join(os.path.dirname(__file__), 'nanonet', 'c_log_viterbi.c')],
    extra_compile_args=c_compile_args
))

###
### TODO: other requirements?
###
requires=[
    'h5py',
    'numpy',
    'netCDF4'
]


setup(
    name='Nanonet',
    version=0.1,
    description='A simple recurrent neural network based basecaller nanopore data.',
    maintainer='Chris Wright',
    maintainer_email='chris.wright@nanoporetech.com',
    url='http://www.nanoporetech.com',
    packages=find_packages(exclude=["*.test", "*.test.*", "test.*", "test"]),
    package_data={},
    tests_require=requires,
    install_requires=requires,
    dependency_links=[],
    zip_safe=True,
    ext_modules=extensions,
    #test_suite='discover_tests',
    scripts=glob('bin/*.py')
)
