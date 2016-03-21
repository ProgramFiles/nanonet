#!/usr/bin/env python
from __future__ import print_function
import argparse
import h5py
import os
import sys
import time
import numpy as np
import pkg_resources

from chimaera.common.utilities import load_substitution_matrix

from dragonet.basecall.align_kmers import align_basecalls
from dragonet.basecall.viterbi_2d import viterbi_2d
from dragonet.bio import seq_tools
from dragonet.util.model import Model

from tang.fast5 import iterate_fast5
from tang.hmm import decoding
from tang.util.cmdargs import (display_version_and_exit, FileExist, CheckCPU,
                               probability, Positive, TypeOrNone, Vector)
from tang.util.io import numpy_genfromtsv
from tang.util.tang_iter import tang_imap

from nanonet.nanonetcall import process_read as process_read_1d

__hp_sm_name__ = 'model/rtc_mismatch_scores.txt'
__hairpin_align_substitution_matrix__ = os.path.join('/opt/chimaera',__hp_sm_name__)
if not os.path.isfile(__hairpin_align_substitution_matrix__):
    try:
        __hairpin_align_substitution_matrix__ = os.path.join(os.environ['CHIMAERA'], 'data', __hp_sm_name__)
    except:
        raise RuntimeError('Cannot find hp alignment sub. matrix file')

# This is here, not in main to allow documentation to be built
parser = argparse.ArgumentParser(
    description='Skeleton program to iterate through fast5 files.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

hp_grp = parser.add_argument_group('Alignment', "Options specific to 'hair-pin' aligner")
hp_grp.add_argument('--gap', default= 500, type=Positive(int), metavar='penalty',
    help='Gap penalty')
hp_grp.add_argument('--ratio', default=[0.5, 2.0], nargs=2, metavar=('min', 'max'),
    help='Ratio of template to complement events to try calling')
hp_grp.add_argument('--subsmatrix', default=__hairpin_align_substitution_matrix__,
    action=FileExist, help='Substituion matrix to use for hairpin alignment')

bc1d_grp = parser.add_argument_group('Basecall 1D', 'options fot 1d basecalling')
bc1d_grp.add_argument('--min_prob', metavar='probability', default=1e-5,
    type=probability, help='Minimum allowed probabiility for basecalls')
bc1d_grp.add_argument('--trans', default=None, action=Vector(probability), nargs=3,
    metavar=('stay', 'step', 'skip'), help='Base transition probabilities')
bc1d_grp.add_argument("--min_len", default=500, type=int,
    help="Min. read length (events) to basecall.")
bc1d_grp.add_argument("--max_len", default=15000, type=int,
    help="Max. read length (events) to basecall.")
bc1d_grp.add_argument("--bc_window", type=int, nargs='+', default=[-1, 0, 1],
    help="The detailed list of the entire input window.")
bc1d_grp.add_argument("--template_model", type=str, action=FileExist,
    default=pkg_resources.resource_filename('nanonet', 'data/template_model.npy'),
    help="Trained RNN for template.")
bc1d_grp.add_argument("--complement_model", type=str, action=FileExist,
    default=pkg_resources.resource_filename('nanonet', 'data/complement_model.npy'),
    help="Trained RNN for complement.")


bc2d_grp = parser.add_argument_group('Basecall 2D', 'Options for 2D basecalling')
bc2d_grp.add_argument('--band', metavar='size', default=15, type=Positive(int),
    help='Band-size around alignment for 2D calling')
bc2d_grp.add_argument('--max_length', default=20000, metavar='number', type=Positive(int),
    help='Maximum length of 2D call')
bc2d_grp.add_argument('--max_nodes', default=500000, metavar='number', type=Positive(int),
    help='Maximum number of nodes for 2D call')
bc2d_grp.add_argument('--trans_comp', default=None, nargs=3,
    action=Vector(probability), metavar=('stay', 'step', 'skip'),
    help='Template transition probabilities for 2D')
bc2d_grp.add_argument('--trans_temp', default=None, nargs=3,
    action=Vector(probability), metavar=('stay', 'step', 'skip'),
    help='Template transition probabilities for 2D')
bc2d_grp.add_argument('--window', metavar='size', default=101, type=Positive(int),
    help='Window size for estimating offsets')

parser.add_argument('--jobs', default=8, type=Positive(int), action=CheckCPU,
    help='Number of jobs to run in parallel.')
parser.add_argument('--kmer', default=3, type=Positive(int), metavar='length',
    help='Kmer length')
parser.add_argument("--strand_list", default=None, action=FileExist,
    help="List of reads to process.")
parser.add_argument("--limit", default=None, type=int,
    help="Limit the number of input for processing.")

parser.add_argument("input", action=FileExist,
    help="A path to fast5 files or a single netcdf file.")
parser.add_argument('output_prefix', help='Prefix for output fasta files')


LINELEN = 50
def basecall2d(args, fast5):
    # Load data and do 1D basecalls

    kwargs = {
        'window':args.bc_window,
        'min_len':args.min_len,
        'max_len':args.max_len,
    }
    
    try:
        kwargs['section'] = 'template'
        post_tmp = process_read_1d(args.template_model, fast5, post_only=True, **kwargs)
        if post_tmp is None:
            raise RuntimeError('Could not form template posterior')
    except Exception as e:
        return (fast5,)
    else:
        post_tmp = args.min_prob + (1.0 - args.min_prob) * post_tmp
        trans_tmp = decoding.estimate_transitions(post_tmp, trans=args.trans)
        score_tmp, calls_tmp = decoding.decode_profile(post_tmp, trans=trans_tmp, log=False)
        stay_tmp = np.ones(len(post_tmp))
        if args.trans_temp is None:
            args.trans_temp = np.mean(trans_tmp, axis=0)
            stay_tmp = trans_tmp[:,0] / args.trans_temp[0]
    try:
        kwargs['section'] = 'complement'
        post_comp = process_read_1d(args.complement_model, fast5, post_only=True, **kwargs)
        if post_comp is None:
            raise RuntimeError('Could not form complement posterior')
    except:
        return (fast5,)
    else:
        post_comp = args.min_prob + (1.0 - args.min_prob) * post_comp
        trans_comp = decoding.estimate_transitions(post_comp, trans=args.trans)
        score_comp, calls_comp = decoding.decode_profile(post_comp, trans=trans_comp, log=False)
        stay_comp = np.ones(len(post_comp))
        if args.trans_comp is None:
            args.trans_comp = np.mean(trans_comp, axis=0)
            stay_comp = trans_comp[:,0] / args.trans_comp[0]


    # 'Hairpin alignment'
    kmers = np.array(seq_tools.all_kmers(length=args.kmer))
    alignment, score = align_basecalls(kmers[calls_tmp], kmers[calls_comp],
                                       args.subsmatrix, args.gap)

    # -- Prepare for 2D call --
    # Trim input and reset alignment so coordinate are relative to trimmed
    # arrays.
    post_tmp = post_tmp[alignment['pos0'][0] : alignment['pos0'][-1]]
    stay_tmp = stay_tmp[alignment['pos0'][0] : alignment['pos0'][-1]]
    post_comp = post_comp[alignment['pos1'][-1] : alignment['pos1'][0]]
    stay_comp = stay_comp[alignment['pos1'][-1] : alignment['pos1'][0]]
    alignment['pos0'] = np.where(alignment['pos0'] >=0, alignment['pos0'] - alignment['pos0'][0], -1)
    alignment['pos1'] = np.where(alignment['pos1'] >=0, alignment['pos1'][0] - alignment['pos1'], -1)
    align_in = []
    for elt in alignment:
        align_in.append((elt['pos0'], elt['pos1']))
    # Parameters
    params = {'band_size': args.band,
              'kmer_len': args.kmer,
              'seq2_is_rc': True,
              'use_sd': False,
              'max_nodes': args.max_nodes,
              'max_len': args.max_length,
              'stay1': args.trans_temp[0],
              'step1': args.trans_temp[1],
              'skip1': args.trans_temp[2],
              'stay2': args.trans_comp[0],
              'step2': args.trans_comp[1],
              'skip2': args.trans_comp[2]}
    prior = None

    # Fake model is to satisfy current implementation of Viterbi code. Not actually used.
    fake_model = Model.EmptyModel(args.kmer)
    vit_model = {name: np.copy(fake_model[name], order='C') for name in ['level_mean', 'level_stdv',
                                                                     'sd_mean', 'sd_stdv']}
    vit_model['kmer'] = list(fake_model['kmer'])
    vit_model2 = {name: np.copy(fake_model[name], order='C') for name in ['level_mean', 'level_stdv',
                                                                     'sd_mean', 'sd_stdv']}
    vit_model2['kmer'] = list(fake_model['kmer'])

    # Reorder complement probabilities
    rc_order = np.argsort(map(seq_tools.rc_kmer, kmers))
    post_comp = post_comp[:, rc_order]
    post_comp = post_comp[::-1]
    stay_comp = stay_comp[::-1]

    vit = viterbi_2d.Viterbi2D(vit_model, vit_model2, params)
    res = vit.call_post(post_tmp.astype(np.float32), post_comp.astype(np.float32),
                        stay_tmp, stay_comp,
                        align_in, prior)

    kmers = seq_tools.all_kmers(length=args.kmer)
    calls_tmp = seq_tools.kmers_to_sequence([kmers[st] for st in calls_tmp])
    calls_comp = seq_tools.kmers_to_sequence([kmers[st] for st in calls_comp])
    calls_2d = seq_tools.kmers_to_sequence(res['kmers'])

    return os.path.splitext(os.path.basename(fast5))[0], score_tmp, score_comp, calls_tmp, calls_comp, calls_2d

if __name__ == '__main__':
    #  Turn off output buffering
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    args = parser.parse_args()

    #  Load substitution matrix
    args.subsmatrix = load_substitution_matrix(args.subsmatrix)

    #  Gap penalties in Chimaera format
    args.gap = {'open0': args.gap, 'open1': args.gap,
                'start0': args.gap, 'start1': args.gap,
                'end0': args.gap, 'end1': args.gap,
                'extend0': args.gap, 'extend1': args.gap}

    fast5_files = iterate_fast5(args.input, paths=True, strand_list=args.strand_list, limit=args.limit)

    kmers = seq_tools.all_kmers(length=args.kmer)
    #  Main loop
    n2dbase = 0
    count = 0
    with open(args.output_prefix + '_template.fa', 'w') as temp_fh, open(args.output_prefix + '_complement.fa', 'w') as comp_fh, open(args.output_prefix + '_2d.fa', 'w') as profile2d_fh:
        time_in = time.time()
        for result in tang_imap(basecall2d, fast5_files, fix_args=[args], threads=args.jobs, unordered=True):
            if len(result) == 1:
                continue
            fn, score1, score2, calls1, calls2, calls_2d = result
            temp_fh.write('>{} {} {}\n'.format(fn, len(calls1), score1))
            temp_fh.write('{}\n'.format(calls1))

            comp_fh.write('>{} {} {}\n'.format(fn, len(calls2), score2))
            comp_fh.write('{}\n'.format(calls2))

            profile2d_fh.write('>{} {}\n'.format(fn, len(calls_2d)))
            profile2d_fh.write('{}\n'.format(calls_2d))
            n2dbase += len(calls_2d)
            count += 1

            print('.', end='')
            if count % LINELEN == 0:
                dtime = time.time() - time_in
                print('    {:6d} reads {:6d} bases/s'.format(count, int(round(n2dbase / dtime))))
        if count % LINELEN != 0:
            print()

        time_out = time.time()
    dtime = time_out - time_in
    print('Called {} bases in {:.1f} s ({} bases/s)\n'.format(n2dbase, dtime, int(round(n2dbase / dtime))))
