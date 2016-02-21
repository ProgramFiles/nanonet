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

from nanonet import __currennt_exe__
from nanonet.fast5 import Fast5, iterate_fast5
from nanonet.util import random_string, tang_imap
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
    
    parser.add_argument("--input", action=FileExist, help="A path to fast5 files or a single netcdf file.", required=True)
    parser.add_argument("--strand_list", default=None, action=FileExist, help="List of reads to process.")
    parser.add_argument("--limit", default=None, type=int, help="Limit the number of input for processing.")
    parser.add_argument('--workspace', default=None, help='Workspace directory')
    parser.add_argument("--min_len", default=1000, type=int, help="Min. read length.")
    parser.add_argument("--max_len", default=9000, type=int, help="Max. read length.")
    parser.add_argument("--phase", default="T", choices=["T", "C"], help="Choice of phase.")
    
    parser.add_argument("--output", type=str, required=True,
        help="Output name, output will be in fasta format.")
    parser.add_argument("--model", type=str, action=FileExist, required=True,
        help="Trained ANN.")
    parser.add_argument("--cuda", type=int, default=0,
        help="CUDA device number to use." )
    parser.add_argument("--nocuda", default=False, action='store_true',
        help="Use CPU for neural network calculations.")
    parser.add_argument("--network_jobs", default=1, type=int, action=CheckCPU,
        help="No of neural network jobs to run in parallel, only valid with --nocuda.")
    parser.add_argument("--decoding_jobs", default=1, type=int, action=CheckCPU,
        help="No of Viterbi decoding jobs to run in parallel.")
    parser.add_argument("--nseqs", default=20, type=int,
        help="No. of sequences for currennt to process simultaneously. The upper limit is determined by --max_len and ANN size." )
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


def process_features(workspace, modelfile, cache_path, cuda, nocuda, nseqs, inputfile):
    i, inputfile = inputfile

    # Currennt config file
    currennt_cfg = os.path.join(workspace, 'currennt_{}.cfg'.format(i))
    currennt_out = os.path.join(workspace, 'currennt_{}.out'.format(i))
    with open(currennt_cfg, 'w') as cfg:    
        cfg.write("network              = " + modelfile + "\n")
        cfg.write("ff_input_file        = " + inputfile + "\n")
        cfg.write("ff_output_file       = " + currennt_out + "\n")
        cfg.write("ff_output_format     = single_csv\n")
        cfg.write("input_noise_sigma    = 0.0\n")
        cfg.write("parallel_sequences   = {}\n".format(nseqs))
        cfg.write("cache_path           = " + cache_path + "\n")
        if nocuda:
            cfg.write("cuda             = false\n")   
 
    # Run Currennt
    os.environ["CURRENNT_CUDA_DEVICE"]="{}".format(cuda)
    cmd = [__currennt_exe__, currennt_cfg]
    with open(os.devnull, 'wb') as devnull:
        #subprocess.check_call(cmd)#, stdout=devnull, stderr=devnull)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=devnull)
        stdout, _ = p.communicate()
        p.wait()
        if p.returncode != 0:
            # On windows currennt fails to remove the cache file. Check for
            #   this and move on, else raise an error.
            e = subprocess.CalledProcessError(2, ' '.join(cmd))
            if os.name != 'nt':
                raise e
            else:
                cache_file = re.match(
                    '(FAILED: boost::filesystem::remove.*: )"(.*)"',
                    stdout.splitlines()[-1])
                if cache_file is not None:
                    cache_file = cache_file.group(2)
                    sys.stderr.write('currennt failed to clear its cache, cleaning up {}\n'.format(cache_file))
                    os.unlink(cache_file)
                else:
                    raise e

    return currennt_out


def main():
    if len(sys.argv) == 1:
        sys.argv.append("-h")
    args = get_parser().parse_args()

    modelfile  = os.path.abspath(args.model)
    outputfile = os.path.abspath(args.output)

    if args.nocuda:
        args.nseqs = 1
    else:
        args.network_jobs = 1

    # User-defined workspace or use system tmp
    workspace = args.workspace
    if workspace is None:
        workspace = os.path.join(tempfile.gettempdir(), random_string())
    if not os.path.exists(workspace):
        os.makedirs(workspace)

    # Create currennt input(s)
    inputs = []
    if os.path.isdir(args.input):
        inputfile = os.path.join(workspace, 'basecall_features_{}.netcdf')
        print "Creating currennt input NetCDF(s): {}".format(inputfile)
        fast5_files = list(iterate_fast5(args.input, paths=True, strand_list=args.strand_list, limit=args.limit))
        per_group = int(math.ceil(float(len(fast5_files)) / args.network_jobs))
        for i, group in enumerate(fast5_files[i:i+per_group] for i in xrange(0, len(fast5_files), per_group)):
            inputfile = inputfile.format(i)
            inputs.append((i, os.path.abspath(inputfile)))
            make_currennt_basecall_input_multi(
                fast5_files=group,
                netcdf_file=inputfile,
                phase=args.phase,
                window=args.window,
                min_len=args.min_len,
                max_len=args.max_len)
    else:    
        inputs.append((0, os.path.abspath(inputfile)))
        print "Using precomputed feature data: {}".format(inputs[0])

    fix_args = [
        workspace, modelfile, args.cache_path,
        args.cuda, args.nocuda,
        args.nseqs
    ]

    pstay  = args.trans[0]
    pstep1 = args.trans[1]/4.0
    pstep2 = args.trans[2]/16.0
    pstep3 = args.trans[3]/44.0

    print "Running basecalling. Network is running on {} {}(s) with decoding running on {} CPU(s)".format(
        args.network_jobs, 'CPU' if args.nocuda else 'GPU', args.decoding_jobs)
    with open(args.output, 'w') as fasta:
        for currennt_out in tang_imap(process_features, inputs, fix_args=fix_args, threads=args.network_jobs):
            # Viterbi calls
            cpc = CurrenntParserCaller(
                fin=currennt_out, limit=args.limit,
                trans_free=args.trans_free, pstay=pstay, pstep1=pstep1, pstep2=pstep2, pstep3=pstep3
            )
            for result in cpc.basecalls(ncpus=args.decoding_jobs):
                fasta.write(">{}\n{}\n".format(*result))

    # Clean up, should use a context manager...
    if args.workspace is None:
        shutil.rmtree(workspace)


if __name__ == "__main__":
    main()
