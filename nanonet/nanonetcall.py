#!/usr/bin/env python
import argparse
import json
import os
import re
import math
import sys
import shutil
import tempfile
import timeit
import subprocess
import pkg_resources

import numpy as np

from nanonet import decoding, nn
from nanonet.fast5 import Fast5, iterate_fast5
from nanonet.util import random_string, conf_line, FastaWrite, tang_imap, all_nmers, kmers_to_sequence
from nanonet.cmdargs import FileExist, CheckCPU, AutoBool
from nanonet.features import make_basecall_input_multi

import warnings
warnings.simplefilter("ignore")


def get_parser():
    parser = argparse.ArgumentParser(
        description="""A simple ANN 3-mer basecaller, works only on HMM basecall mapped data.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("input", action=FileExist,
        help="A path to fast5 files or a single netcdf file.")
    parser.add_argument("--section", default=None, choices=('template', 'complement'),
        help="Section of read for which to produce basecalls, will override that stored in model file.")

    parser.add_argument("--output", type=str,
        help="Output name, output will be in fasta format.")
    parser.add_argument("--strand_list", default=None, action=FileExist,
        help="List of reads to process.")
    parser.add_argument("--limit", default=None, type=int,
        help="Limit the number of input for processing.")
    parser.add_argument("--min_len", default=500, type=int,
        help="Min. read length (events) to basecall.")
    parser.add_argument("--max_len", default=15000, type=int,
        help="Max. read length (events) to basecall.")
    
    parser.add_argument("--model", type=str, action=FileExist,
        default=pkg_resources.resource_filename('nanonet', 'data/template_model.npy'),
        help="Trained ANN.")
    parser.add_argument("--decoding_jobs", default=1, type=int, action=CheckCPU,
        help="No of decoding jobs to run in parallel.")

    parser.add_argument("--trans", type=float, nargs=3, default=None,
        metavar=('stay', 'step', 'skip'), help='Base transition probabilities')
    parser.add_argument("--window", type=int, nargs='+', default=[-1, 0, 1],
        help="The detailed list of the entire input window, default -1 0 1, tested with R7.3 to be optimal.")

    return parser


def process_read(modelfile, fast5, min_prob=1e-5, trans=None, **kwargs):
    """Run neural network over a set of fast5 files

    :param modelfile: neural network specification.
    :param fast5: read file to process
    :param **kwargs: kwargs of make_basecall_input_multi
    """
    kmer_len = 3 #TODO: parameterise this

    t0 = timeit.default_timer()
    try:
        name, features = make_basecall_input_multi((fast5,), **kwargs).next()
    except Exception as e:
        return None 
    t1 = timeit.default_timer()
    feature_time = t1 - t0

    t0 = timeit.default_timer()
    network = np.load(modelfile).item()
    t1 = timeit.default_timer()
    post = network.run(features.astype(nn.tang_nn_type))
    t2 = timeit.default_timer()

    # Reorder ATGC -> ACGT, models are trained with funny order
    kmers_nn = all_nmers(kmer_len)
    kmers_nn_revmap = {k:i for i, k in enumerate(kmers_nn)}
    kmers_hmm = sorted(kmers_nn)
    nkmers = len(kmers_nn)
    kmer_out_order = np.arange(nkmers)
    kmer_order = np.fromiter(
        (kmers_nn_revmap[k] for k in kmers_hmm),
        dtype=int, count=len(kmers_hmm)
    )

    # Strip out events where XXX most likely, and XXX states entirely 
    max_call = np.argmax(post, axis=1)
    post = post[max_call < nkmers]
    post = post[:, :-1]
    post[:, kmer_out_order] = post[:, kmer_order]
    post /= np.sum(post, axis=1).reshape((-1, 1))

    post = min_prob + (1.0 - min_prob) * post
    trans = decoding.estimate_transitions(post, trans=trans)
    score, states = decoding.decode_profile(post, trans=trans, log=False)

    kmer_path = [kmers_hmm[i] for i in states]
    seq = kmers_to_sequence(kmer_path)
    t3 = timeit.default_timer()

    return (name, seq, score, len(post), (feature_time, t1 - t0, t2 - t1, t3 - t2))


def main():
    if len(sys.argv) == 1:
        sys.argv.append("-h")
    args = get_parser().parse_args()

    modelfile  = os.path.abspath(args.model)
    if args.section is None:
        try:
            args.section = np.load(modelfile).item().meta['section']
        except:
            sys.stderr.write("No 'section' found in modelfile, try specifying --section.\n")
            sys.exit(1)

    fix_args = [
        modelfile
    ]
    fix_kwargs = {
        'window':args.window,
        'min_len':args.min_len,
        'max_len':args.max_len,
        'section':args.section
    }

    t0 = timeit.default_timer()
    n_reads = 0
    n_bases = 0
    n_events = 0
    timings = [0.0, 0.0, 0.0, 0.0]
    fast5_files = iterate_fast5(args.input, paths=True, strand_list=args.strand_list, limit=args.limit)
    with FastaWrite(args.output) as fasta:
        for result in tang_imap(process_read, fast5_files, fix_args=fix_args, fix_kwargs=fix_kwargs, threads=args.decoding_jobs):
            if result is None:
                continue
            name, basecall, _, n_ev, time = result
            fasta.write(*(name, basecall))
            n_reads += 1
            n_bases += len(basecall)
            n_events += n_ev
            timings = [x + y for x, y in zip(timings, time)]
    t1 = timeit.default_timer()
    sys.stderr.write('Basecalled {} reads ({} bases, {} events) in {}s (wall time)\n'.format(n_reads, n_bases, n_events, t1 - t0))
    feature, load, network, decoding = timings
    sys.stderr.write(
        'Profiling\n---------\n'
        'Feature generation: {}\n'
        'Load network: {}\n'
        'Run network: {} ({} kb/s, {} kev/s)\n'
        'Decoding: {} ({} kb/s, {} kev/s)\n'.format(
            feature, load,
            network, n_bases/1000/network, n_events/1000/network,
            decoding, n_bases/1000/decoding, n_events/1000/decoding
        )
    )


if __name__ == "__main__":
    main()
