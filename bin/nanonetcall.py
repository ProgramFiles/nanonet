#!/usr/bin/env python
import argparse
import os
import sys
import shutil
import tempfile
import timeit
import subprocess

###
### TODO: pull in these imports
###
from tang.util.cmdargs import FileExist, CheckCPU, AutoBool
from tang.fast5 import fast5, iterate_fast5

from nanonet import __currennt_exe__
from nanonet.util import random_string
from nanonet.parse_currennt import CurrenntParserCaller
from nanonet.features import make_currennt_basecall_input_multi

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
    help="CUDA device number, on AWS, it is one of 0, 1, 2, 3, check with nvidia-smi first, default 0." )
parser.add_argument("--jobs", default=8, type=int, action=CheckCPU,
    help="No of Viterbi decoding jobs to run in parallel, using more than 1 cpus together with --use_trans will slow things down significantly.")
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


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("-h")
    args = parser.parse_args()

    modelfile  = os.path.abspath(args.model)
    outputfile = os.path.abspath(args.output)

    # User-defined workspace or use system tmp
    workspace = args.workspace
    if workspace is None:
        workspace = os.path.join(tempfile.gettempdir(), random_string())
    if not os.path.exists(workspace):
        os.makedirs(workspace)

    # Create currennt input
    t0 = timeit.default_timer()
    inputfile = None
    if os.path.isdir(args.input):
        inputfile = os.path.join(workspace, 'basecall_features.netcdf')
        print "Creating training data NetCDF: {}".format(inputfile)
        fast5_files = list(iterate_fast5(args.input, paths=True, strand_list=args.strand_list, limit=args.limit))
        make_currennt_basecall_input_multi(
            fast5_files=fast5_files,
            netcdf_file=inputfile,
            phase=args.phase,
            window=args.window,
            min_len=args.min_len,
            max_len=args.max_len)
    else:
        print "Using precomputed feature data: {}".format(inputfile)
    inputfile = os.path.abspath(inputfile)
    t1 = timeit.default_timer()
    print "Feature generation took {}s.".format(t1-t0)

    # Currennt config file
    currennt_cfg = os.path.join(workspace, 'currennt.cfg')
    currennt_out = os.path.join(workspace, 'currennt.out')
    with open(currennt_cfg, 'w') as cfg:    
        cfg.write("network              = " + modelfile + "\n")
        cfg.write("ff_input_file        = " + inputfile + "\n")
        cfg.write("ff_output_file       = " + currennt_out + "\n")
        cfg.write("ff_output_format     = single_csv\n")
        cfg.write("input_noise_sigma    = 0.0\n")
        cfg.write("parallel_sequences   = {}\n".format(args.nseqs))
        cfg.write("cache_path           = " + args.cache_path + "\n")
    
    # Run Currennt
    os.environ["CURRENNT_CUDA_DEVICE"]="{}".format(args.cuda)
    cmd = [__currennt_exe__, currennt_cfg]
    print "Running: {}".format(' '.join(cmd))
    subprocess.check_call(cmd)

    # Viterbi calls
    print "Running decoding"
    pstay   = args.trans[0]
    pstep1  = args.trans[1]/4.0
    pstep2  = args.trans[2]/16.0
    pstep3  = args.trans[3]/44.0
    
    cpc = CurrenntParserCaller(fin=currennt_out, limit=args.limit, pstay=pstay, pstep1=pstep1, pstep2=pstep2, pstep3=pstep3)
    with open(args.output, 'w') as fasta:
        for result in cpc.viterbi_basecalls(ncpus=args.jobs, trans_free=args.trans_free):
            fasta.write(">{}\n{}\n".format(*result))   
 
    # Clean up, should use a context manager...
    if args.workspace is None:
        shutil.rmtree(workspace)
