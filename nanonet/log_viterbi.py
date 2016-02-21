import imp
import os
import numpy as np 
from ctypes import c_int
from numpy.ctypeslib import ndpointer
from ctypes import cdll, WinDLL

clib = cdll.LoadLibrary(imp.find_module('clib_viterbi')[1])


def make_log_trans(stay, kmers, step1, step2, step3, null_kmer=None):
    """Create a dense transition matrix from sparse parameters and kmer list.

    :param kmers: kmers for which to calculate possible transtions.
    :param step1:
    :param step2:
    :param step3:
    :param null_kmer: identity of null kmer
    """
    nkmers = len(kmers)
    transm = np.zeros((nkmers, nkmers))
    for i in range(nkmers):
        for j in range(nkmers):
            kmeri = kb.digit2str(i)
            kmerj = kb.digit2str(j)
            if kmeri == kmerj:
                transm[i][j] = stay
            elif kmeri[1:] == kmerj[:2]:
                transm[i][j] = step1
            elif kmeri[2] == kmerj[0]:
                transm[i][j] = step2
            elif kmeri == null_kmer or kmerj == null_kmer:
                transm[i][j] = step1
            else:
                transm[i][j] = step3
    return transm


def c_viterbi(log_emmision_m, log_trans_p, num_states, num_obs, vp):
    """Python wrapper for basecall_viterbi function of clib_crossroads.so
    """
    func = clib.log_viterbi
    func.restype = None
    func.argtypes = [
        ndpointer(dtype='f8', flags='CONTIGUOUS'),
        ndpointer(dtype='f8', flags='CONTIGUOUS'),
        c_int,
        c_int, 
        ndpointer(dtype='i4', flags='CONTIGUOUS')
    ]
    func(log_emmision_m, log_trans_p, num_states, num_obs, vp)


def c_viterbi_trans_free(log_emmision_m, num_states, num_obs, vp):
    """Python wrapper for basecall_viterbi function of clib_crossroads.so
    """
    func = clib.log_viterbi_trans_free
    func.restype = None
    func.argtypes = [
        ndpointer(dtype='f8', flags='CONTIGUOUS'),
        c_int,
        c_int, 
        ndpointer(dtype='i4', flags='CONTIGUOUS')
    ]
    func(log_emmision_m, num_states, num_obs, vp)

