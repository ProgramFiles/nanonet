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
import pyopencl as cl
from functools import partial

from nanonet import decoding, nn
from nanonet.fast5 import Fast5, iterate_fast5
from nanonet.util import random_string, conf_line, FastaWrite, tang_imap, all_nmers, kmers_to_sequence, kmer_overlap, group_by_list, AddFields
from nanonet.cmdargs import FileExist, CheckCPU, AutoBool
from nanonet.features import make_basecall_input_multi
from nanonet.jobqueue import JobQueue

import warnings
warnings.simplefilter("ignore")


__fast5_analysis_name__ = 'Basecall_RNN_1D'
__fast5_section_name__ = 'BaseCalled_{}'
__ETA__ = 1e-300


def get_parser():
    parser = argparse.ArgumentParser(
        description="""A simple RNN basecaller for Oxford Nanopore data.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("input", action=FileExist,
        help="A path to fast5 files.")
    parser.add_argument("--watch", default=None, type=int,
        help="Switch to watching folder, argument value used as timeout period.")
    parser.add_argument("--section", default=None, choices=('template', 'complement'),
        help="Section of read for which to produce basecalls, will override that stored in model file.")
    parser.add_argument("--event_detect", default=True, action=AutoBool,
        help="Perform event detection, else use existing event data")

    parser.add_argument("--output", type=str,
        help="Output name, output will be in fasta format.")
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

    parser.add_argument("--model", type=str, action=FileExist,
        default=pkg_resources.resource_filename('nanonet', 'data/default_template.npy'),
        help="Trained ANN.")
    parser.add_argument("--jobs", default=1, type=int, action=CheckCPU,
        help="No of decoding jobs to run in parallel.")

    parser.add_argument("--trans", type=float, nargs=3, default=None,
        metavar=('stay', 'step', 'skip'), help='Base transition probabilities')
    parser.add_argument("--fast_decode", action=AutoBool, default=False,
        help="Use simple, fast decoder with no transition estimates.")

    parser.add_argument("--exc_opencl", action=AutoBool, default=False,
        help="Do not use CPU alongside OpenCL, overrides --jobs.")
    parser.add_argument("--list_platforms", action=AutoBool, default=False,
        help="Output list of available OpenCL GPU platforms.")
    parser.add_argument("--platforms", nargs="+", type=str,
        help="List of OpenCL GPU platforms and devices to be used in a format VENDOR:DEVICE:N_Files space separated, i.e. --platforms nvidia:0:1 amd:0:2 amd:1:2.")

    return parser

class ProcessAttr(object):
    def __init__(self, use_opencl=False, vendor=None, device_id=0):
        self.use_opencl = use_opencl
        self.vendor = vendor
        self.device_id = device_id

def list_opencl_platforms():
    print('\n' + '=' * 60 + '\nOpenCL Platforms and Devices')
    platforms = [p for p in cl.get_platforms() if p.get_devices(device_type=cl.device_type.ALL)]
    for platform in platforms:
        print('=' * 60)
        print('Platform - Name:  ' + platform.name)
        print('Platform - Vendor:  ' + platform.vendor)
        print('Platform - Version:  ' + platform.version)
        for device in platform.get_devices(device_type=cl.device_type.ALL):  # Print each device per-platform
            print('    ' + '-' * 56)
            print('    Device - Name:  ' + device.name)
            print('    Device - Type:  ' + cl.device_type.to_string(device.type))
            print('    Device - Max Clock Speed:  {0} Mhz'.format(device.max_clock_frequency))
            print('    Device - Compute Units:  {0}'.format(device.max_compute_units))
            print('    Device - Local Memory:  {0:.0f} KB'.format(device.local_mem_size/1024))
            print('    Device - Constant Memory:  {0:.0f} KB'.format(device.max_constant_buffer_size/1024))
            print('    Device - Global Memory: {0:.0f} GB'.format(device.global_mem_size/1073741824.0))
    print('\n')


def process_read(modelfile, fast5, min_prob=1e-5, trans=None, post_only=False, write_events=True, fast_decode=False, **kwargs):
    """Run neural network over a set of fast5 files

    :param modelfile: neural network specification.
    :param fast5: read file to process
    :param post_only: return only the posterior matrix
    :param **kwargs: kwargs of make_basecall_input_multi
    """
    #sys.stderr.write("CPU process\n processing {}\n".format(fast5))

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

    post = network.run(features.astype(nn.dtype))
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
        if len(post) == 0:
            return None

    weights = np.sum(post, axis=1).reshape((-1,1))
    post /= weights
    if post_only:
        return post

    post = min_prob + (1.0 - min_prob) * post
    if fast_decode:
        score, states = decoding.decode_homogenous(post, log=False)
    else:
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
    write_time = timeit.default_timer() - t4

    return (name, seq, score, len(features), (feature_time, load_time, network_time, decode_time, write_time))

        
def process_read_opencl(modelfile, pa, fast5_list, min_prob=1e-5, trans=None, post_only=False, write_events=True, fast_decode=False, **kwargs):
    """Run neural network over a set of fast5 files

    :param modelfile: neural network specification.
    :param fast5: read file to process
    :param post_only: return only the posterior matrix
    :param **kwargs: kwargs of make_basecall_input_multi
    """
    #sys.stderr.write("OpenCL process\n processing {}\n{}\n".format(fast5_list, pa.__dict__))

    t0 = timeit.default_timer()
    network = np.load(modelfile).item()
    t1 = timeit.default_timer()
    load_time = t1 - t0

    kwargs['window'] = network.meta['window']
    use_opencl = pa.use_opencl
    
    post_list = []
    features_list = []
    events_list = []
    name_list = []
    
    feature_time_list = []
    network_time_list = []

    for fast5 in fast5_list:
        t1 = timeit.default_timer()
        try:
            it = make_basecall_input_multi((fast5,), **kwargs)
            if write_events:
                name, features, events = it.next()
                events_list.append(events)
            else:
                name, features, _ = it.next()
            features_list.append(features.astype(nn.dtype))
            name_list.append(name)
        except Exception as e:
            print str(e)
            return None
        t2 = timeit.default_timer()
        feature_time_list.append(t2 - t1)
    
        if not use_opencl:
            post = network.run(features.astype(nn.dtype))
            post_list.append(post)
            t3 = timeit.default_timer()
            network_time_list.append(t3 - t2)
            
    ctx = None
    queue_list = []
    max_workgroup_size = 256        
    t2 = timeit.default_timer()
    if use_opencl:
        platforms = [p for p in cl.get_platforms() if p.get_devices(device_type=cl.device_type.GPU) and pa.vendor.lower() in p.get_info(cl.platform_info.NAME).lower()]
        platform = platforms[0]
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        device = devices[pa.device_id]
        max_workgroup_size = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
        ctx = cl.Context([device]) 
        for x in xrange(len(features_list)):
            queue_list.append(cl.CommandQueue(ctx))
        post_list = network.run(features_list, ctx, queue_list)
        t3 = timeit.default_timer()
        for x in xrange(len(features_list)):
            network_time_list.append((t3 - t2)/len(features_list))

    
    trans_copy = trans
    trans_list = []
    good_events_list = []
    t3 = timeit.default_timer()
    score_list = [None] * len(fast5_list)
    states_list = [None] * len(fast5_list)
    for x in xrange(len(fast5_list)):
        post = post_list[x]
        trans = trans_copy
        
        kmers = network.meta['kmers']
        # Do we have an XXX kmer? Strip out events where XXX most likely,
        #    and XXX states entirely
        if kmers[-1] == 'X'*len(kmers[-1]):
            bad_kmer = post.shape[1] - 1
            max_call = np.argmax(post, axis=1)
            good_events = (max_call != bad_kmer)
            good_events_list.append(good_events)
            post = post[good_events]
            post = post[:, :-1]
    
        weights = np.sum(post, axis=1).reshape((-1,1))
        post /= weights
        if post_only:
            return post
    
        post = min_prob + (1.0 - min_prob) * post
        post_list[x] = post
        if fast_decode:
            score_list[x], states_list[x] = decoding.decode_homogenous(post_list[x], log=False)
        else:
            trans = decoding.estimate_transitions(post_list[x], trans=trans)
            if use_opencl:
                trans_list.append(np.log(__ETA__ + trans))    
            else:
                score_list[x], states_list[x] = decoding.decode_profile(post_list[x], trans=np.log(__ETA__ + trans), log=False)
    
    if not fast_decode and use_opencl:
        score_list, states_list = decoding.decode_profile_opencl(ctx, queue_list, post_list, trans_list=trans_list, log=False, max_workgroup_size=max_workgroup_size)
            
    # Form basecall
    kmer_path_list = []
    seq_list = []
    for x in xrange(len(fast5_list)):
        kmer_path = [kmers[i] for i in states_list[x]]
        seq = kmers_to_sequence(kmer_path)
        kmer_path_list.append(kmer_path)
        seq_list.append(seq)

    decode_time = timeit.default_timer() - t3
    decode_time_list = []
    for x in xrange(len(fast5_list)):
        decode_time_list.append(decode_time/len(fast5_list))
    
    ret = []
    write_time_list = [0] * len(fast5_list)
    # Write events table
    if write_events:
        for x in xrange(len(fast5_list)):
            t4 = timeit.default_timer()
            events = events_list[x]
            name = name_list[x]
            post = post_list[x]
            kmer_path = kmer_path_list[x]
            seq = seq_list[x]
            states = states_list[x]
            good_events = good_events_list[x] 
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
 
            write_time_list[x] = timeit.default_timer() - t4
   
    for x in xrange(len(fast5_list)):
        ret.append((name, seq, score_list[x], len(features), (feature_time_list[x], load_time, network_time_list[x], decode_time_list[x], write_time_list[x])))
    return ret

def main():
    if len(sys.argv) == 1:
        sys.argv.append("-h")
    args = get_parser().parse_args()
    
    if args.list_platforms:
        list_opencl_platforms() 
        sys.exit(0)
        
    modelfile  = os.path.abspath(args.model)
    if args.section is None:
        try:
            args.section = np.load(modelfile).item().meta['section']
        except:
            sys.stderr.write("No 'section' found in modelfile, try specifying --section.\n")
            sys.exit(1)
                 

            
    #TODO: handle case where there are pre-existing files.
    if args.watch is not None:
        # An optional component
        from nanonet.watcher import Fast5Watcher
        fast5_files = Fast5Watcher(args.input, timeout=args.watch)
    else:
        sort_by_size = 'desc' if args.platforms is not None else None
        fast5_files = iterate_fast5(args.input, paths=True, strand_list=args.strand_list, limit=args.limit, sort_by_size=sort_by_size)

    # files_pattern controls grouping of files. First N entries
    #    are for the N opencl devices. files are sent singly to
    #    CPUs whilst in groups of args.opencl_input_files to 
    #    other devices.
    #files_pattern = [1] * args.jobs
    #if args.opencl:
    #    files_pattern[:len(args.platforms)] = [args.opencl_input_files] * len(args.platforms)
    #fast5_groups = group_by_list(fast5_files, files_pattern) 

    #def create_processes():
    #    for i, ff in enumerate(fast5_groups):
    #        if args.opencl and i % args.jobs in xrange(len(args.platforms)):
    #            vendor,device_id = args.platforms[i%args.jobs].split(':')
    #            yield ProcessAttr(ff, use_opencl=True, vendor=vendor, device_id=int(device_id))
    #        else:
    #            yield ProcessAttr(ff)
    #pa_gen = create_processes()

    fix_args = [
        modelfile
    ]
    fix_kwargs = {a: getattr(args, a) for a in ( 
        'min_len', 'max_len', 'section',
        'event_detect', 'fast_decode',
        'write_events'
    )}
   
    workers = [] 
    if not args.exc_opencl:
        cpu_function = partial(process_read, *fix_args, **fix_kwargs)
        workers.extend([(cpu_function, None)] * args.jobs)
    if args.platforms is not None:
        for platform in args.platforms:
            vendor, device_id, n_files = platform.split(':')
            pa = ProcessAttr(use_opencl=True, vendor=vendor, device_id=int(device_id))
            fix_args.append(pa)
            opencl_function = partial(process_read_opencl, *fix_args, **fix_kwargs)
            workers.append(
                (opencl_function, int(n_files))
            )

    t0 = timeit.default_timer()
    n_reads = 0
    n_bases = 0
    n_events = 0
    timings = [0.0, 0.0, 0.0, 0.0, 0.0]

    with FastaWrite(args.output) as fasta:
        for result in JobQueue(fast5_files, workers):
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
        feature, load, network, decoding, events_writing = timings
        feature /= args.jobs
        load /= args.jobs
        network /= args.jobs
        decoding /= args.jobs
        events_writing /= args.jobs
        sys.stderr.write(
            #'Profiling\n---------\n'
            #'Feature generation: {:4.1f}\n'
            #'Load network: {:4.1f}\n'
            'Run network: {:6.2f} ({:6.3f} kb/s, {:6.3f} kev/s)\n'
            'Decoding:    {:6.2f} ({:6.3f} kb/s, {:6.3f} kev/s)\n'
            #'Write events: {:6.2f}\n'
            .format(
                #feature, load,
                network, n_bases/1000.0/network, n_events/1000.0/network,
                decoding, n_bases/1000.0/decoding, n_events/1000.0/decoding,
                #events_writing
            )
        )


if __name__ == "__main__":
    main()
