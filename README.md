*************************************************************************************
**Proprietary and confidential information of Oxford Nanopore Technologies, Limited**

**All rights reserved; (c)2016: Oxford Nanopore Technologies, Limited**
*************************************************************************************


Nanonet
=======

Nanonet provides recurrent neural network basecalling via currennt. Event data
is extracted from .fast5 files to create feature vectors to input into a
pretrained network. Output is as a single .fasta file.

Nanonet leverages currennt to run recurrent neural networks. Currennt is
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

The modified distribution of currennt may be found at:
https://github.com/nanoporetech/currennt

Installation notes may be found at
https://github.com/nanoporetech/currennt/blob/master/Makefile

**Installation of nanonet**

Nanonet is a mostly python with a single C library for performing Viterbi
decoding on the probability matrix output by currennt. Installation is
complicated only by the fact that currennt requires as input netCDF files.
We must first install some prequisites:

    sudo apt-get install -y netcdf-bin libhdf5-dev python-h5py python-numpy cython 

Nanonet should then install quite trivially using the standard python
mechanisms:

    cd nanonet
    python setup.py install


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

The python components can be installed using the setup script:

    cd nanonet
    python setup.py develop --user
    
to perform an inplace install.


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

__Compiling c_log_viterbi.c and setting-up nanonet__

Nanonet contains small amounts of C code which must be compiled. The simplest
way to have this compilation performed is to install the Microsoft Visual C++
Compiler for Python 2.7. This can be found here:

    https://www.microsoft.com/en-gb/download/details.aspx?id=44266

Having installed this compiler nanonet one can simply run:

    python setup.py install --user

to install nanonet.

If having two Visual Studio compilers on your system seems like overkill you
can make use of the existing Visual Studio 2013 installed to compile currennt.
To do this from a command prompt run:

    cd nanonet/nanonet
    "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64\vcvarsx86_amd64.bat"
    "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64\cl.exe" /LD c_log_viterbi.c /link /out:data\clib_viterbi.dll

Using this method you should run:

    python setup.py noext install --user

to instruct setuptools not to try compiling the C library itself. In this case
the library will be loaded from the package data folder at runtime.

Peforming basecalling
---------------------

Nanonet provides a single program for basecalling Oxford Nanopore Technolgies'
reads from .fast5 files. The output is to stdout as fasta formatted sequences.

If you followed the instructions above the program `nanonetcall` should be on
your path. It requires a single argument:

    nanonetcall {input_folder} > {output.fasta}

To test your installation several .fast5 files are provided in the
`sample_data` folder of the source distribution, as a concrete example the
command:

    nanonetcall --network_jobs 5 --decoding_jobs 1 --batch 1 sample_data > basecalls.fa

will produced output along the following lines:

    Creating currennt input NetCDF(s): /tmp/DBRMWZ/basecall_features_{}.netcdf
    Running basecalling for template sections. Network is running on 5 CPU(s) with decoding running on 1 CPU(s). Batch size is 1.
    Finished neural network processing for batch 0.
    Finished basecalling batch 0.
    Finished neural network processing for batch 1.
    Finished basecalling batch 1.
    Finished neural network processing for batch 4.
    Finished neural network processing for batch 3.
    Finished neural network processing for batch 2.
    Finished basecalling batch 2.
    Finished basecalling batch 3.
    Finished basecalling batch 4.
    Processed 5 reads (25464 bases) in 12.3650121689s

The order of the output may differ including the ordering or records in the
output basecalls.fa, though the content of the fasta records should be
deterministic and identical to that provided.


**Using a GPU**

By default nanonetcall will not use a GPU to run the neural network. To enable
use of a GPU specify the `--cuda` argument. In doing so one should also specify
the `--nseqs <n>` option to alter how many inputs the GPU processes in parallel.
For a GPU with 4GB a value of 25 is likely optimal. Note laptop versions of GPUs
will not likely outperform their CPU equivalents.

**Using multiple CPUs**

By default nanonetcall will use a maximum of two CPUs, one each for running the
neural network and running Viterbi decoding of the results of the network. The
extent to which all requested CPUs are utilised depends on the dataset. To
increase performance first increase the number of CPUs used for the network
with the `--network_jobs <n>`. If you find that you are exhausting the system
memory but have remaining CPU resource, you may also wish to utilise the option
`--decoding_jobs <n>`.

__Batching__

By default the input dataset is split into batches determined by the option
`--network_jobs`. If you wish to make improved use of the `--network_jobs`
and `--decoding_jobs` you should specify also the `--batch <n>` options. This
instructs nanonetcall to process at most `<n>` reads in a single batch.

Using batch has the added benefit that final basecalls will be produced in a
more streamed fashion, useful if you wish to pipe them to e.g. an alignment
program.

Training a network
------------------

The package provides also an interface to currennt for training networks from
.fast5 files. The type of neural networks implemented by currennt require
labelled input data. **Nanonet does not provide a method for labelling data**.
It does however provide a hook to create labelled data.

To run the trainer we specify training data, and validation data. The former
will be used to train the network whilst the latter is used to check that the
network does not become overtrained to the test data.

    nanonettrain --train <training_data> --val <validation_data> \
        --output <output_model_prefix> --model <input_model_spec>

An example input model can be found in `nanonet/data/default_model.tmpl`.
Currently the only supported models are those based on three-mers. (Though most
of the code supported generic kmers, the C code for Viterbi decoding does not).
The only other consideration is that the size of the first ("input") layer of
the network must correspond to the featur vectors created by
`nanonet.features.events_to_features`. The nanonettrain program will try to
enforce these considerations. In contructing models one should assign the
input layer a size of `<n_features>` and the final two layers as `<n_states>`,
as in the example.

Training is an intensive process, even on a GPU expect it to take hours not
minutes. It is not recommended to attempt training models without GPU support.


Trouble Shooting
----------------

If you performed a user install (`setup.py install --user`) the `nanonetcall`
program may not be on your path. This is because the location into which
setuptools installs programs for users is not often a default item in the
user's path. On OSX the location is typically:

    /Users/<username>/Library/Python/2.7/bin/

whilst on Ubuntu it is:

    ~/.local/bin

If `nanonet` complains that it cannot locate the `currennt` executable you will
need to set the `CURRENNT` environment variable to the location of the
executable.


