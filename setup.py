import os
import re
import subprocess
import sys
from glob import glob
from setuptools import setup, find_packages, Extension, Command
import pkg_resources


class EnsureClibs(Command):
    description = 'Ensures C libraries are precompiled and in data folder.'
    user_options = []
    def initialize_options(self):
        self.cwd = None
    def finalize_options(self):
        self.cwd = os.getcwd()
    def run(self):
        assert os.getcwd() == self.cwd, 'Must be in package root'
        # Get all extensions, and check for presence of file
        for ext in self.__dict__['distribution'].__dict__['ext_modules']:
            name = os.path.join('nanonet', 'data', '{}.dll'.format(ext.name))
            if not os.path.isfile(name):
                raise IOError('Library {} not found, see README.md for details of precompiling libraries.'.format(name))
        # Remove extensions so we don't try to compile
        self.__dict__['distribution'].__dict__['ext_modules'] = []
        

c_compile_args = [
    '-Wall', '-DNDEBUG', '-std=c99',
    '-fstrict-aliasing', '-O3', '-march=native'
]

nanonet_dir = os.path.join(os.path.dirname(__file__), 'nanonet')

extensions = []
extensions.append(Extension(
    'clib_viterbi',
    include_dirs=[nanonet_dir],
    sources=[os.path.join(nanonet_dir, 'c_log_viterbi.c')],
    extra_compile_args=c_compile_args
))

requires=[
    'h5py',
    'numpy',
    'netCDF4'
]


setup(
    name='nanonet',
    version=0.1,
    description='A simple recurrent neural network based basecaller nanopore data.',
    maintainer='Chris Wright',
    maintainer_email='chris.wright@nanoporetech.com',
    url='http://www.nanoporetech.com',
    cmdclass={'noext':EnsureClibs},
    packages=find_packages(exclude=["*.test", "*.test.*", "test.*", "test"]),
    package_data={'nanonet.data':['nanonet/data/*']},
    tests_require=requires,
    install_requires=requires,
    dependency_links=[],
    zip_safe=True,
    ext_modules=extensions,
    #test_suite='discover_tests',
    entry_points={
        'console_scripts': [
            'nanonetcall = nanonet.nanonetcall:main',
            'nanonettrain = nanonet.nanonettrain:main'
        ]
    }
)
