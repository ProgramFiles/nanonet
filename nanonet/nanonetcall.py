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
import itertools
import numpy as np

from nanonet import decoding, nn
from nanonet.fast5 import Fast5, iterate_fast5
from nanonet.util import random_string, conf_line, FastaWrite, tang_imap, all_nmers, kmers_to_sequence, kmer_overlap, AddFields
from nanonet.cmdargs import FileExist, CheckCPU, AutoBool
from nanonet.features import make_basecall_input_multi

import warnings
warnings.simplefilter("ignore")


__fast5_analysis_name__ = 'Basecall_RNN_1D'
__fast5_section_name__ = 'BaseCalled_{}'
__ETA__ = 1e-300


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
    parser.add_argument("--write_fast5", action=AutoBool, default=False,
        help="Write datasets to .fast5.")
    parser.add_argument("--strand_list", default=None, action=FileExist,
        help="List of reads to process.")
    parser.add_argument("--limit", default=None, type=int,
        help="Limit the number of input for processing.")
    parser.add_argument("--min_len", default=500, type=int,
        help="Min. read length (events) to basecall.")
    parser.add_argument("--max_len", default=15000, type=int,
        help="Max. read length (events) to basecall.")
    
    parser.add_argument("--model", type=str, action=FileExist,
        default=pkg_resources.resource_filename('nanonet', 'data/default_template.npy'),
        help="Trained ANN.")
    parser.add_argument("--jobs", default=1, type=int, action=CheckCPU,
        help="No of decoding jobs to run in parallel.")

    parser.add_argument("--trans", type=float, nargs=3, default=None,
        metavar=('stay', 'step', 'skip'), help='Base transition probabilities')

    return parser


def process_read(modelfile, fast5, min_prob=1e-5, trans=None, post_only=False, write_events=True, **kwargs):
    """Run neural network over a set of fast5 files

    :param modelfile: neural network specification.
    :param fast5: read file to process
    :param post_only: return only the posterior matrix
    :param **kwargs: kwargs of make_basecall_input_multi
    """
    t0 = timeit.default_timer()
    network = np.load(modelfile).item()
    t1 = timeit.default_timer()
    load_time = t1 - t0

    kwargs['window'] = network.meta['window']

    try:
        it = make_basecall_input_multi((fast5,), **kwargs)
        if write_events:
            name, features, events = it.next()
        else:
            name, features, _ = it.next()
    except Exception as e:
        return None
    t2 = timeit.default_timer()
    feature_time = t2 - t1

    post = network.run(features.astype(nn.tang_nn_type))
    t3 = timeit.default_timer()
    network_time = t3 - t2

    kmers = network.meta['kmers']
    # Do we have an XXX kmer? Strip out events where XXX most likely,
    #    and XXX states entirely
    if kmers[-1] == 'X'*len(kmers[-1]):
        bad_kmer = post.shape[1] - 1
        max_call = np.argmax(post, axis=1)
        good_events = (max_call != bad_kmer)
        post = post[good_events]
        post = post[:, :-1]

    weights = np.sum(post, axis=1).reshape((-1,1))
    post /= weights
    if post_only:
        return post

    post = min_prob + (1.0 - min_prob) * post
    trans = decoding.estimate_transitions(post, trans=trans)
    score, states = decoding.decode_profile(post, trans=np.log(__ETA__ + trans), log=False)

    # Form basecall
    kmer_path = [kmers[i] for i in states]
    seq = kmers_to_sequence(kmer_path)
    t4 = timeit.default_timer()
    decode_time = t4 -t3

    # Write events table
    if write_events:
        adder = AddFields(events[good_events])
        adder.add('model_state', kmer_path,
            dtype='>S{}'.format(len(kmers[0])))
        adder.add('p_model_state', np.fromiter(
            (post[i,j] for i,j in itertools.izip(xrange(len(post)), states)),
            dtype=float, count=len(post)))
        adder.add('mp_model_state', np.fromiter(
            (kmers[i] for i in np.argmax(post, axis=1)),
            dtype='>S{}'.format(len(kmers[0])), count=len(post)))
        adder.add('p_mp_model_state', np.max(post, axis=1))
        adder.add('move', np.array(kmer_overlap(kmer_path)), dtype=int)

        mid = len(kmers[0]) / 2
        bases = set(''.join(kmers)) - set('X')
        for base in bases:
            cols = np.fromiter((k[mid] == base for k in kmers),
                dtype=bool, count=len(kmers))
            adder.add('p_{}'.format(base), np.sum(post[:, cols], axis=1), dtype=float)

        events = adder.finalize()

        with Fast5(fast5, 'a') as fh:
           base = fh._join_path(
               fh.get_analysis_new(__fast5_analysis_name__),
               __fast5_section_name__.format(kwargs['section']))
           fh._add_event_table(events, fh._join_path(base, 'Events'))
           try:
               name = fh.get_read(group=True).attrs['read_id']
           except:
               pass # filename inherited from above
           fh._add_string_dataset(
               '@{}\n{}\n+\n{}\n'.format(name, seq, '!'*len(seq)),
               fh._join_path(base, 'Fastq'))

    return (name, seq, score, len(post), (feature_time, load_time, network_time, decode_time))


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
        for result in tang_imap(process_read, fast5_files, fix_args=fix_args, fix_kwargs=fix_kwargs, threads=args.jobs):
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
    if n_reads > 0:
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
