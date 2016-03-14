*************************************************************************************
**Proprietary and confidential information of Oxford Nanopore Technologies, Limited**

**All rights reserved; (c)2016: Oxford Nanopore Technologies, Limited**
*************************************************************************************


Nanonet
=======

Nanonet provides recurrent neural network basecalling via currennt. Event data
is extracted from .fast5 files to create feature vectors to input into a
pretrained network. Output is as a single .fasta file.

For training networks, Nanonet leverages currennt to run recurrent neural
networks. Currennt is generally run with GPUs to aid performance but can be run
in a CPU only environment. The basecaller does not require currennt, and is
written in pure python with minimal requirements.


Installation
------------
The basecalling component of nanonet should install quite trivially using the
standard python mechanism on most platforms:

    python setup.py install

The only required dependencies are numpy and h5py. Under Linux and OSX these
will be downloaded and installed/compiled automatically. For Windows one
can download and install the Microsoft Visual C++ Compiler for Python 2.7 from:

    https://www.microsoft.com/en-gb/download/details.aspx?id=44266

before executing the above. Alternatively Christophe Golke maintains a
repository of compiled python wheels at:

    http://www.lfd.uci.edu/~gohlke/pythonlibs/

See the full installation instructions for further details where instructions
to perform a binary installation under Ubuntu can also be found.


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

    nanonetcall --decoding_jobs 5 sample_data > basecalls.fa

will produced output along the following lines:

    Processed 5 reads (24927 bases) in 20.6830949783s
    Feature generation: 0.0864291191101
    Load network: 0.0629456043243
    Run network: 13.4354393482
    Decoding: 7.09250092506


**Using multiple CPUs**

By default nanonetcall will use a maximum of one CPU. This can be altered
through use of the `--decoding_jobs` option. In using this option be aware that
higher numbers will lead to increased memory usage.


Training a network
------------------

The package provides also an interface to currennt for training networks from
.fast5 files. The type of neural networks implemented by currennt require
labelled input data. **Nanonet does not provide a method for labelling data**.
It does however provide a hook to create labelled data. To install currennt
please refer to the full installation instructions in INSTALLATION.md.

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


