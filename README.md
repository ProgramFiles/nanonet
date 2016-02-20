Nanonet
=======
Nanonet provides recurrent neural network basecalling via currennt. Event data
is extracted from .fast5 files to create feature vectors to input into a
pretrained network. Output is as a single .fasta file.

Nanonet leverages currennt to run recurrennt neural networks. Currennt is
generally run with GPUs to aid performance but can be run in a CPU only
environment. 

Installation on Ubuntu
----------------------
The following was tested to work on a clean Ubuntu 14.04 Vagrant virtual machine.

**Installation of currennt**

Oxford Nanopore Technologies have provided a fork of currennt with minor
modifications to that which is available on
https://sourceforge.net/projects/currennt/. The changes serve to allow
currennt to run with current versions of Nvidia graphics cards and libraries.

Compilation of currennt requires a few dependencies to be fulfilled. We must
first install the NVIDIA CUDA libaries. Note that the NVIDIA website has two
versions of this package, a local and a network install. Here we use the
network package.

    # Install NVIDIA's meta package
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1410/x86_64/cuda-repo-ubuntu1410_7.0-28_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1410_7.0-28_amd64.deb

    sudo apt-get update
    sudo apt-get install -y cuda

At this point if you wish to use a GPU with currennt you will need to restart
your machine. Upon restart issuing the command:

    nvidia-smi

should show information about your GPU.

Building currennt has some further, more standard, requirements:

    sudo apt-get install libboost-all-dev libnetcdf-dev netcdf-bin cmake

after which we can build currennt:

    cd currennt
    mkdir build && cd build
    cmake ..
    make
    sudo cp currennt /usr/local/bin

A successful build will result in a single executable file named `currennt`.

**Installation of nanonet**

Nanonet is a mostly python with a single C library for performing Viterbi
decoding on the probability matrix output by currennt. Installation is
complicated only by the fact that currennt requires as input netCDF files.
We must first install some prequisites:

    sudo apt-get install -y netcdf-bin libhdf5-dev python-h5py python-numpy cython 

Nanonet should then install quite trivially using the standard python
mechanisms:

    python setup.py install --user


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

Currennt can then be built as for Ubuntu:

    cd currennt
    mkdir build && cd build
    cmake ..
    make
    sudo cp currennt /usr/local/bin


**Installation of nanonet**
The easiest way to install the netCDF dependencies is via homebrew:

    brew install hdf5 netcdf

The python components can be installed using the setup script:

    python setup.py install --user



Peforming basecalling
---------------------



