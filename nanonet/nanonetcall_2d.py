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
from functools import partial

from multiprocessing.pool import ThreadPool as Pool

from nanonet.fast5 import Fast5, iterate_fast5, short_names
from nanonet.util import FastaWrite, tang_imap, rc_kmer 
from nanonet.cmdargs import FileExist, CheckCPU, AutoBool
from nanonet.nanonetcall import process_read as process_read_1d
from nanonet.nanonetcall import form_basecall
from nanonet.caller_2d.caller_2d import call_2d

import warnings
warnings.simplefilter("ignore")

now = timeit.default_timer

__fast5_analysis_name__ = 'Basecall_RNN_2D'


def get_parser():
    parser = argparse.ArgumentParser(
        description="""A simple RNN basecaller for Oxford Nanopore data.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("input", action=FileExist, nargs='?', #--list_platforms means this can be absent
        help="A path to fast5 files.")
    parser.add_argument("output_prefix", type=str, default=None,
        help="Output prefix, output will be in fasta format.")
    parser.add_argument("--fastq", action=AutoBool, default=False,
        help="Output fastq rather than fasta.")

    parser.add_argument("--watch", default=None, type=int,
        help="Switch to watching folder, argument value used as timeout period.")
    parser.add_argument("--section", default=None, choices=('template', 'complement'),
        help="Section of read for which to produce basecalls, will override that stored in model file.")
    parser.add_argument("--event_detect", default=True, action=AutoBool,
        help="Perform event detection, else use existing event data")

    parser.add_argument("--write_events", action=AutoBool, default=False,
        help="Write event datasets to .fast5.")
    parser.add_argument("--strand_list", default=None, action=FileExist,
        help="List of reads to process.")
    parser.add_argument("--limit", default=None, type=int,
        help="Limit the number of input for processing.")
    parser.add_argument("--min_len", default=500, type=int,
        help="Min. read length (events) to basecall.")
    parser.add_argument("--max_len", default=15000, type=int,
        help="Max. read length (events) to basecall.")

    parser.add_argument("--template_model", type=str, action=FileExist,
        default=pkg_resources.resource_filename('nanonet', 'data/default_template.npy'),
        help="Trained ANN.")
    parser.add_argument("--complement_model", type=str, action=FileExist,
        default=pkg_resources.resource_filename('nanonet', 'data/default_complement.npy'),
        help="Trained ANN.")

    parser.add_argument("--jobs", default=1, type=int, action=CheckCPU,
        help="No of decoding jobs to run in parallel.")
    parser.add_argument("--opencl_2d", default=False, action=AutoBool,
        help="Use OpenCL for 2D calls.")

    parser.add_argument("--trans", type=float, nargs=3, default=None,
        metavar=('stay', 'step', 'skip'), help='Base transition probabilities')
    parser.add_argument("--fast_decode", action=AutoBool, default=False,
        help="Use simple, fast decoder with no transition estimates.")

    return parser


def process_read_sections(fast5, modelfiles, jobs=2, **kwargs):
    # Placeholder function for processing two 1D parts of a read.
    #    TODO: improve scheduling of template, complement and 2D. Here we
    #          simply use a thread pool to give a small benefit (python code
    #          still subject to GIL)
    sections = ('template', 'complement')
    results = []
    worker = partial(process_read_1d, **kwargs)
    pool = Pool(jobs)
    async = (pool.apply_async(worker, args=(modelfiles[s], fast5), kwds={'section':s}) for s in sections)
    for res in async:
        try:
            results.append(res.get())
        except:
            results.append(None)
    pool.close()
    pool.join()
    return {s:r for s, r in zip(sections, results)}


def process_read_2d(modelfiles, fast5, min_prob=1e-5, trans=None, write_events=True, fast_decode=False, opencl_2d=False, **kwargs):
    """Perform 2D call for a single read. We process the two 1D reads in
    parallel. For CPU only use this may conflict with --jobs option of program.

    """
    sections = ('template', 'complement')
    kwargs.update({
       'min_prob':min_prob, 'trans':trans, 'for_2d':True,
       'write_events':write_events, 'fast_decode':fast_decode,
    })
    #TODO: see comments in the below function
    results = process_read_sections(fast5, modelfiles, jobs=2, **kwargs)
    if any(v is None for v in results.values()):
        results['2d'] = None
    else:
        posts = [results[x][2][0] for x in sections]
        kmers = [results[x][2][1] for x in sections]
        transitions = [results[x][2][2].tolist() for x in sections]
        allkmers = [x for x in results[sections[0]][2][3] if 'X' not in x]

        try:
            t0 = now()
            results_2d = call_2d(
                posts, kmers, transitions, allkmers, call_band=10, chunk_size=500, use_opencl=opencl_2d, cpu_id=0)
            time_2d = now() - t0
        except Exception as e:
            results['2d'] = None
        else:
            sequence, qstring, out_kmers, out_align = results_2d

            results['2d'] = (sequence, qstring), time_2d
            if write_events:
                write_to_file(fast5, sequence, qual, out_align)

    for section in sections:
        if results[section] is not None:
            results[section] = results[section][0:2]       
    return results


def write_to_file(fast5, seq, qual, alignment):
    with Fast5(fast5, 'a') as fh:
       base = fh.get_analysis_new(__fast5_analysis_name__)
       fh[fh._join_path(base, 'Alignment')] = alignment
       try:
           name = fh.get_read(group=True).attrs['read_id']
       except:
           name = fh.filename_short
       fh._add_string_dataset(
           '@{}\n{}\n+\n{}\n'.format(name, seq, qual),
           fh._join(path(base, 'Fastq')))


def main():
    if len(sys.argv) == 1:
        sys.argv.append("-h")
    args = get_parser().parse_args()
 
    modelfiles = {
        'template': os.path.abspath(args.template_model),
        'complement': os.path.abspath(args.complement_model)
    }
            
    #TODO: handle case where there are pre-existing files.
    if args.watch is not None:
        # An optional component
        from nanonet.watcher import Fast5Watcher
        fast5_files = Fast5Watcher(args.input, timeout=args.watch)
    else:
        sort_by_size = None
        fast5_files = iterate_fast5(args.input, paths=True, strand_list=args.strand_list, limit=args.limit, sort_by_size=sort_by_size)

    fix_args = [
        modelfiles
    ]
    fix_kwargs = {a: getattr(args, a) for a in ( 
        'min_len', 'max_len', 'section',
        'event_detect', 'fast_decode',
        'write_events', 'opencl_2d'
    )}

    # Define worker functions   
    mapper = tang_imap(
        process_read_2d, fast5_files,
        fix_args=fix_args, fix_kwargs=fix_kwargs,
        threads=args.jobs, unordered=True
    )

    # Off we go
    n_reads = 0
    n_bases = 0
    n_events = 0
    n_bases_2d = 0
    timings = [0.0, 0.0, 0.0]
    t0 = now()
    sections = ('template', 'complement', '2d')
    if args.output_prefix is not None:
        ext = 'fastq' if args.fastq else 'fasta'
        filenames = ['{}_{}.{}'.format(args.output_prefix, x, ext) for x in sections]
    else:
        filenames = ['-'] * 3

    with FastaWrite(filenames[0], args.fastq) as fasta_temp, FastaWrite(filenames[1], args.fastq) as fasta_comp, FastaWrite(filenames[2], args.fastq) as fasta_2d:
        for result in mapper:
            if result['template'] is None:
                continue
            data, time = result['template']
            fname, basecall, _, n_ev = data
            basecall, quality = basecall
            name, _ = short_names(fname)
            if args.fastq:
                fasta_temp.write(name, basecall, quality)
            else:
                fasta_temp.write(name, basecall)
            n_reads += 1
            n_bases += len(basecall)
            n_events += n_ev
            timings = [x + y for x, y in zip(timings, time + (0.0,))]

            if result['complement'] is None:
                continue
            data, time = result['complement']
            _, basecall, _, _ = data
            basecall, quality = basecall
            if args.fastq:
                fasta_comp.write(name, basecall, quality)
            else:
                fasta_comp.write(name, basecall)

            if result['2d'] is None:
                continue
            basecall, time_2d = result['2d']
            basecall, quality = basecall
            if args.fastq:
                fasta_2d.write(name, basecall, quality)
            else:
                fasta_2d.write(name, basecall)
            n_bases_2d += len(basecall)
            timings[2] += time_2d
    t1 = now()

    sys.stderr.write('Processed {} reads in {}s (wall time)\n'.format(n_reads, t1 - t0))
    if n_reads > 0:
        network, decoding, call_2d  = timings
        sys.stderr.write(
            'Template Run network: {:6.2f} ({:6.3f} kb/s, {:6.3f} kev/s)\n'
            'Template Decoding:    {:6.2f} ({:6.3f} kb/s, {:6.3f} kev/s)\n'
            '2D calling:           {:6.2f} ({:6.3f} kb/s)\n'
            .format(
                network, n_bases/1000.0/network, n_events/1000.0/network,
                decoding, n_bases/1000.0/decoding, n_events/1000.0/decoding,
                call_2d, n_bases_2d/1000.0/call_2d
            )
        )


if __name__ == "__main__":
    main()
