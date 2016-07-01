Installation
============

The following contains instructions for a 'full' installation of nanonet
including currennt, as required for training networks. If you wish to use
nanonet only for basecalling please refer to the file README.md.


Binary Installation on Ubuntu
-----------------------------

For convenience Oxford Nanopore Technologies provides binary packages for
Ubuntu. These will perform a full install of all components including currennt.

    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb
    sudo apt-get update
    sudo apt-get -y install cuda-cudart-7-5 cuda-cublas-7-5 netcdf-bin libboost-all-dev

    wget https://github.com/nanoporetech/currennt/releases/download/v0.2-rc1-2/python-netcdf4_1.2.3-1_amd64.deb \
         https://github.com/nanoporetech/currennt/releases/download/v0.2-rc1-2/ont-currennt-0.2.1-3-trusty.deb \
         https://github.com/nanoporetech/nanonet/releases/download/v1.1.2/python-nanonet_1.1.3-1_amd64.deb

    sudo dpkg -i python-netcdf4_1.2.3-1_amd64.deb ont-currennt_0.2.1-2-trusty_amd64.deb python-nanonet_1.1.3-1_amd64.deb
    # expect an error here about missing prerequisite packages, which can be corrected with:
    sudo apt-get -f install


Source Installation on Ubuntu
-----------------------------

The following was tested to work on a clean Ubuntu 14.04 Vagrant virtual machine.

**Installation of currennt**

Oxford Nanopore Technologies have provided a fork of currennt with minor
modifications to that which is available on
https://sourceforge.net/projects/currennt/. The changes serve to allow
currennt to run with current versions of Nvidia graphics cards and libraries.

The modified distribution of currennt may be found at:
https://github.com/nanoporetech/currennt

Installation notes may be found at
https://github.com/nanoporetech/currennt/blob/master/Makefile

**Installation of nanonet**

Nanonet is a pure python library. Installation is complicated only by the
fact that currennt requires as input netCDF files. We must first install
some prequisites:

    sudo apt-get install -y netcdf-bin libhdf5-dev python-h5py python-numpy cython 

Nanonet should then install quite trivially using the standard python
mechanisms:

    cd nanonet
    pip install -e .[currennt]

pip is used here to force installation of the optional dependencies
required to support currennt.

Installation on OSX
-------------------

Installation on Mac OSX mirrors that on Ubuntu. The following was performed on
a Mac OSX 10.11.3 development machine. Additional dependencies may be required
from those listed below. Most Macs do not have a CUDA enabled GPU.

**Installation of currennt**

First download and install NVIDIAs CUDA package:

    http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda_7.5.20_mac.dmg

It is not neccessary to install the samples.

Ensure that you have Xcode and Xcode Command Line Tools installed and that you
have a softlink to the /Developer folder present:

    sudo ln -s /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer /Developer

Next use homebrew to install boost and netcdf:

    brew install boost netcdf

Currennt can then be built as for Ubuntu:

    cd currennt
    mkdir build && cd build
    cmake ..
    make
    sudo cp currennt /usr/local/bin


**Installation of nanonet**

The python components can be installed using the setup script as in
the case of Ubuntu:

    cd nanonet
    pip install -e .[currennt]
    
which will install all dependencies including those required to
support interoperability with currennt.


Installation on Windows
-----------------------
The following build of currennt was performed on Windows 10.

**Boost**

Download prebuild boost libraries from:

    https://sourceforge.net/projects/boost/files/boost-binaries/1.55.0-build2/boost_1_55_0-msvc-12.0-64.exe/download

and install to `c:\boost_1_55_0` (this is not the default). If you choose a
location you will have to modify the Visual Studio project.

**Visual Studio**

Download and install Visual Studio Community 2013 with Update 5 from

    https://www.visualstudio.com/en-us/downloads/download-visual-studio-vs.aspx

Do not download Visual Studio 2015, the NVIDIA CUDA libraries do not support
it.

**NVIDIA CUDA**

Download and install NVIDIA CUDA for Windows from:

    https://developer.nvidia.com/cuda-downloads

If you receive a message that no version of Visual Studio can be found, it is
likely that you installed Visual Studio 2015.
    
**Building currennt**

Open the Visual Studio solution currennt.sln. Ensure that the solution
configuration is set to Release on x64 (in the toolbar right below the
main menu). Compile the whole solution via Build > Build Solution. The
following files will be produced which should be copied to a location
from which you wish to run currennt:

    currennt/Release/cudart32_75.dll
    currennt/Release/cudart64_75.dll
    currennt.exe

To the same location you should copy all .dll files located under the currennt
folder. In order for nanonet to execute currennt you should add this location
to the environment variable `CURRENNT`. In Windows Powershell this can be done
via:

    $env:CURRENNT = "<Folder containing currennt.exe>"

If you prefer you can append the location to your PATH environment variable.

**Installing nanonet**

Various python distributions are available for Windows. Here we will use a
the standard installer available from:

    https://www.python.org/downloads/release/python-2711/

Download and install the x86-64 version of Python 2.7 choosing to install
python into the path (not done by default).

__Installing python libraries__

Nanonet requires libraries for reading and writing HDF5 and netCDF4 files.
These libraries ordinarily must be compiled from source, however Christophe
Golke maintains a repository of compiled python wheels at:

    http://www.lfd.uci.edu/~gohlke/pythonlibs/

From this page locate and download the following packages:

    numpy-1.11.0b3+mkl-cp27-cp27m-win_amd64.whl
    h5py-2.5.0-cp27-none-win_amd64.whl
    netCDF4-1.2.2-cp27-none-win_amd64.whl

For each of these run the following at a command prompt:

    pip install <package>

Following this one can install nanonet with:

    python setup.py install

Unlike the Ubuntu and OSX cases we need not force installation of the
optional dependencies, as these have been handled above.
