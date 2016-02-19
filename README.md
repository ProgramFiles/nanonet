
Nanonet
=======
Nanonet provides recurrent neural network basecalling via currennt. Event data
is extracted from .fast5 files to create feature vectors to input into a
pretrained network. Output is as a single .fasta file.


Installation
------------
Nanonet leverages currennt to run recurrennt neural networks. Currennt is
generally run with GPUs to aid performance but can be run in a CPU only
environment. 


**Installation of currennt**

Oxford Nanopore Technologies have provided a fork of currennt with minor
modifications to that which is available on
https://sourceforge.net/projects/currennt/. The changes serve to allow
currennt to run with current versions of Nvidia graphics cards and libraries.

Compilation of currennt requires a few dependencies to be fulfilled. On Ubuntu
14.04 these can be fulfilled via the following process. We must first install
the Nvidia CUDA libaries:

    # Install nvidias meta package
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1410/x86_64/cuda-repo-ubuntu1410_7.0-28_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1410_7.0-28_amd64.deb

    sudo apt-get install libboost-all-dev libnetcdf-dev netcdf-bin cmake
    sudo apt-get update
    sudo apt-get install -y cuda

At this point if you wish to use a GPU with currennt you will need to restart
your machine. Upon restart issuing the command:

    nvidia-smi

should show information about your GPU.

Building currennt has some further, more standard, requirements:

    sudo apt-get install libboost-all-dev libnetcdf-dev cmake

after which we can build currennt:

    cd currennt
    mkdir build && cd build
    cmake ..
    make

A successful build will result in a single executable file named `currennt`.


**Installation of nanonet**

Nanonet is a mostly python with a single C library for performing Viterbi
decoding on the probability matrix output by currennt. Installation is via
the standard mechanism for python packages:

    python setup.py install

This should build the C module and install the basecalling program.


Peforming basecalling
---------------------



