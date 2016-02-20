import numpy as np
from itertools import product
from copy import copy

###
### TODO: pull in tang_imap to ann_util
###
from tang.util.tang_iter import tang_imap

from nanonet.util import kmers_to_sequence
###
### TODO: change this import
###
from log_viterbi import make_log_trans, c_viterbi, c_viterbi_trans_free

def all_nmers(n=3):
    return [''.join(x) for x in product('ATGC', repeat=n)]
__eta__ = 1e-100 


def basecall_read(currennt_output, kmers=all_nmers(3), null_state=True, trans_free=False, pstay=0.1153, pstep1=0.17225, pstep2=0.01176, pstep3=0.00018):
    """Form basecall for a single read from a line of currennt output.

    :param currennt_output: line of output from currennt containing kmer-probabilities for all events.
    :param kmers: identity of the kmers.
    :param null_state: whether a null (error) state is included.
    :param trans_free:
    :param pstay:
    :param pstep1:
    :param pstep2:
    :param pstep3:
    """
    
    null_kmer = 'X'*len(kmers[0])
    if null_state:
        kmers = kmers + [null_kmer]
    num_kmers = len(kmers)
    scores = np.array(currennt_output.strip().split(';'))
    seq_label = scores[0]
    scores = scores[1:].astype('f8')

    # Take log before decoding
    post = np.reshape(scores, (num_kmers, len(scores) / num_kmers), order='F')
    np.add(__eta__, post, post)
    np.log(post, post)
    post = np.ascontiguousarray(post)

    # Run decoding
    num_obs = post.shape[1]
    v_path  = np.zeros(num_obs, dtype='i4')
    if trans_free:
        c_viterbi_trans_free(log_emmision_m=post, num_states=num_kmers, num_obs=num_obs, vp=v_path)
    else:
        log_trans = np.ascontiguousarray(make_log_trans(
            stay=np.log(pstay), step1=np.log(pstep1), step2=np.log(pstep2), step3=np.log(pstep3), null_kmer=null_kmer),
        dtype='f8')
        c_viterbi(log_emmision_m=post, log_trans_p=log_trans, num_states=num_kmers, num_obs=num_obs, vp=v_path)
   
    # Form basecall from kmers, the filter here removes null_state kmer
    calls = (kmers[r] for r in v_path)
    basecall = kmers_to_sequence([k for k in calls if k != null_kmer])
    return seq_label, basecall


class CurrenntParserCaller(object):
    def __init__(self, fin, limit=None, kmers=all_nmers(3), null_state=True, trans_free=False, pstay=0.1153, pstep1=0.17225, pstep2=0.01176, pstep3=0.00018):
        """Helper class for parsing and basecalling currennt output.

        :param fin: file containing current class-probability output
        :param limit: number of lines to process

        ..note:
            Other kwargs are as :func:`basecall_read`.
        """

        self.currennt_file = fin
        self.limit = limit
        self.pstay = pstay
        self.pstep1 = pstep1
        self.pstep2 = pstep2
        self.pstep3 = pstep3
        self.all_data = None
        self.caller_kwargs = {
            'kmers':kmers, 'null_state':null_state, 'trans_free':trans_free, 'pstay':self.pstay,
            'pstep1':self.pstep1, 'pstep2':self.pstep2, 'pstep3':self.pstep3
        }


    def currennt_data(self):
        """Lazily read all of the input data.""" 

        limit = -1 if self.limit is None else -1
        if self.all_data is None:
            with open(self.currennt_file) as f:
                for cnt, line in enumerate(f):
                    print "processing CURRENNT output for read {}".format(cnt)
                    if cnt == limit:
                        break
                    l = line.strip()
                    yield l
        else:
            for x in self.all_data:
                yield x

    
    def basecalls(self, ncpus=1):
        """Performing Viterbi decoding of currennt output to produce basecalls.

        :param npus: number of reads to process in parallel
        """

        results = tang_imap(basecall_read, self.currennt_data(), fix_kwargs=self.caller_kwargs, threads=ncpus, unordered=True)
        for res in results:
            if res is not None:
                yield res

