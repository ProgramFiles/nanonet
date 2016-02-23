#!/usr/bin/env python
import argparse
import os
import re
import math
import sys
import shutil
import tempfile
import timeit
import subprocess
import pkg_resources

from nanonet import run_currennt
from nanonet.fast5 import Fast5, iterate_fast5
from nanonet.util import random_string, conf_line, FastaWrite, tang_imap
from nanonet.cmdargs import FileExist, CheckCPU, AutoBool
from nanonet.parse_currennt import CurrenntParserCaller
from nanonet.features import make_currennt_basecall_input_multi

import warnings
warnings.simplefilter("ignore")


def get_parser():
    parser = argparse.ArgumentParser(
        description="""A simple ANN 3-mer basecaller, works only on HMM basecall mapped data.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("input", action=FileExist,
        help="A path to fast5 files or a single netcdf file.")

    parser.add_argument("--output", type=str,
        help="Output name, output will be in fasta format.")
    parser.add_argument("--strand_list", default=None, action=FileExist,
        help="List of reads to process.")
    parser.add_argument("--limit", default=None, type=int,
        help="Limit the number of input for processing.")
    parser.add_argument('--workspace', default=None,
        help='Workspace directory')
    parser.add_argument("--min_len", default=500, type=int,
        help="Min. read length (events) to basecall.")
    parser.add_argument("--max_len", default=15000, type=int,
        help="Max. read length (events) to basecall.")
    
    parser.add_argument("--model", type=str, action=FileExist,
        default=pkg_resources.resource_filename('nanonet', 'data/default_model.jsn'),
        help="Trained ANN.")
    parser.add_argument("--device", type=int, default=0,
        help="ID of CUDA device to use." )
    parser.add_argument("--cuda", default=False, action=AutoBool,
        help="Use CPU for neural network calculations.")
    parser.add_argument("--network_jobs", default=1, type=int, action=CheckCPU,
        help="No of neural network jobs to run in parallel, only valid with --no-cuda.")
    parser.add_argument("--decoding_jobs", default=1, type=int, action=CheckCPU,
        help="No of Viterbi decoding jobs to run in parallel.")
    parser.add_argument("--nseqs", default=1, type=int,
        help="No. of sequences for currennt to process simultaneously. The upper limit is determined by --max_len and ANN size." )
    parser.add_argument("--batch", default=None, type=int,
        help="No. of sequences to include in each processing batch.")

    parser.add_argument("--trans", type=float, nargs='+', default=[0.1153, 0.6890, 0.1881, 0.0077],
        help="Transition parameters, stay, step1, step2 and step3, to enable this, use --use_trans.")
    parser.add_argument("--window", type=int, nargs='+', default=[-1, 0, 1],
        help="The detailed list of the entire input window, default -1 0 1, tested with R7.3 to be optimal.")
    parser.add_argument("--trans_free", action=AutoBool, default=True,
        help="Estimate transition parameters by magic, the magic may suffer from some numerical issues.")
    parser.add_argument("--use_trans", dest="trans_free", action="store_false",
        help="Use fixed input transition parameters, no magic.")
    parser.add_argument("--cache_path", default=tempfile.gettempdir(),
        help="Path for currennt cache files.")

    return parser


def process_reads(workspace, modelfile, cache_path, device, cuda, nseqs, inputfile, **kwargs):
    """Run neural network over a set of fast5 files

    :param workspace: directory in which to write intermediates and neural
        network output.
    :param: modelfile: neural network specification.
    :param cache_path: cache path for currennt.
    :param device: ID of CUDA device to use.
    :param cuda: use cuda?
    :param nseqs: no. of sequences to process in parallel.
    """

    batch, fast5s, netcdf = inputfile
    reads_written = make_currennt_basecall_input_multi(fast5s, netcdf_file=netcdf, **kwargs)
    if reads_written == 0:
        sys.stderr.write('All reads filtered out in batch {}.\n'.format(batch))
        return batch, None

    # Currennt config file
    currennt_cfg = os.path.join(workspace, 'currennt_{}.cfg'.format(batch))
    currennt_out = os.path.join(workspace, 'currennt_{}.out'.format(batch))

    with open(currennt_cfg, 'w') as cfg:    
        cfg.write(conf_line('network', modelfile))
        cfg.write(conf_line('ff_input_file', netcdf))
        cfg.write(conf_line('ff_output_file', currennt_out))
        cfg.write(conf_line('ff_output_format', 'single_csv'))
        cfg.write(conf_line('input_noise_sigma', 0.0))
        cfg.write(conf_line('parallel_sequences', nseqs))
        cfg.write(conf_line('cache_path', cache_path))
        if not cuda:
            cfg.write(conf_line('cuda', 'false'))   
 
    # Run Currennt
    run_current(currennt_cfg, device)
    sys.stderr.write('Finished neural network processing for batch {}.\n'.format(batch))
    return batch, currennt_out


def main():
    if len(sys.argv) == 1:
        sys.argv.append("-h")
    args = get_parser().parse_args()

    modelfile  = os.path.abspath(args.model)

    if args.cuda:
        args.network_jobs = 1
    else:
        args.nseqs = 1

    # User-defined workspace or use system tmp
    workspace = args.workspace
    if workspace is None:
        workspace = os.path.join(tempfile.gettempdir(), random_string())
    if not os.path.exists(workspace):
        os.makedirs(workspace)

    # Define currennt input(s)
    inputfile_tmpl = os.path.join(workspace, 'basecall_features_{}.netcdf')
    sys.stderr.write("Creating currennt input NetCDF(s): {}\n".format(inputfile_tmpl))
    fast5_files = list(iterate_fast5(args.input, paths=True, strand_list=args.strand_list, limit=args.limit))
    if args.batch is None:
        batch_size = int(math.ceil(float(len(fast5_files)) / args.network_jobs))
    else:
        batch_size = args.batch
    inputs = []
    for i, group in enumerate(fast5_files[i:i+batch_size] for i in xrange(0, len(fast5_files), batch_size)):
        inputfile = inputfile_tmpl.format(i)
        inputs.append((i, group, os.path.abspath(inputfile)))         

    sys.stderr.write("Running basecalling. Network is running on {} {}(s) with"
        " decoding running on {} CPU(s). Batch size is {}.\n".format(
        args.network_jobs, 'GPU' if args.cuda else 'CPU',
        args.decoding_jobs, batch_size)
    )

    fix_args = [
        workspace, modelfile, args.cache_path,
        args.device, args.cuda,
        args.nseqs
    ]
    fix_kwargs = {
        'window':args.window,
        'min_len':args.min_len,
        'max_len':args.max_len
    }

    pstay  = args.trans[0]
    pstep1 = args.trans[1]/4.0
    pstep2 = args.trans[2]/16.0
    pstep3 = args.trans[3]/44.0

    t0 = timeit.default_timer()
    n_reads = 0
    n_bases = 0
    with FastaWrite(args.output) as fasta:
        for batch, currennt_out in tang_imap(process_reads, inputs, fix_args=fix_args, fix_kwargs=fix_kwargs, threads=args.network_jobs):
            if currennt_out == None:
                continue
            # Viterbi calls
            cpc = CurrenntParserCaller(
                fin=currennt_out, trans_free=args.trans_free,
                pstay=pstay, pstep1=pstep1, pstep2=pstep2, pstep3=pstep3
            )
            for result in cpc.basecalls(ncpus=args.decoding_jobs):
                fasta.write(*result)
                n_reads += 1
                n_bases += len(result[1])
            sys.stderr.write('Finished basecalling batch {}.\n'.format(batch))
    t1 = timeit.default_timer()
    sys.stderr.write('Processed {} reads ({} bases) in {}s\n'.format(n_reads, n_bases, t1 - t0))

    # Clean up, should use a context manager...
    if args.workspace is None:
        shutil.rmtree(workspace)


if __name__ == "__main__":
    main()
