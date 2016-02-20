from itertools import tee, izip, izip_longest
import random
import string
import math

def random_string(length=6):
    """Return a random upper-case string of given length.

    :param length: length of string to return.
    """

    return ''.join(random.choice(string.ascii_uppercase) for _ in range(length))


def window(iterable, size):
    """Create an iterator returning a sliding window from another iterator.

    :param iterable: iterable object.
    :param size: size of window.
    """

    iters = tee(iterable, size)
    for i in xrange(1, size):
        for each in iters[i:]:
            next(each, None)
    return izip(*iters)


def kmer_overlap(kmers, moves=None, it=False):
    """From a list of kmers return the character shifts between them.
    (Movement from i to i+1 entry, e.g. [AATC,ATCG] returns [0,1]).

    :param kmers: sequence of kmer strings.
    :param moves: allowed movements, if None all movements to length of kmer
        are allowed.
    :param it: yield values instead of returning a list.

    Allowed moves may be specified in moves argument in order of preference.
    """

    if it:
        return kmer_overlap_gen(kmers, moves)
    else:
        return list(kmer_overlap_gen(kmers, moves))


def kmer_overlap_gen(kmers, moves=None):
    """From a list of kmers return the character shifts between them.
    (Movement from i to i+1 entry, e.g. [AATC,ATCG] returns [0,1]).
    Allowed moves may be specified in moves argument in order of preference.

    :param moves: allowed movements, if None all movements to length of kmer
        are allowed.
    """

    first = True
    yield 0
    for last_kmer, this_kmer in window(kmers, 2):
        if first:
            if moves is None:
                l = len(this_kmer)
                moves = range(l + 1)
            first = False

        l = len(this_kmer)
        for j in moves:
            if j < 0:
                if last_kmer[:j] == this_kmer[-j:]:
                    yield j
                    break
            elif j > 0 and j < l:
                if last_kmer[j:l] == this_kmer[0:-j]:
                    yield j
                    break
            elif j == 0:
                if last_kmer == this_kmer:
                    yield 0
                    break
            else:
                yield l
                break


def kmers_to_call(kmers, moves):
    """From a list of kmers and movements, produce a basecall.

    :param kmers: iterable of kmers
    :param moves: iterbale of character overlaps between kmers
    """

    # We use izip longest to check that iterables are same length
    bases = None
    for kmer, move in izip_longest(kmers, moves, fillvalue=None):
        if kmer is None or move is None:
            raise RuntimeError('Lengths of kmers and moves must be equal (kmers={} and moves={}.'.format(len(kmers), len(moves)))
        if move < 0 and not math.isnan(x):
            raise RuntimeError('kmers_to_call() cannot perform call when backward moves are present.')

        if bases  is None:
            bases = kmer
        else:
            if math.isnan(move):
                bases = bases + 'N' + kmer
            else:
                bases = bases + kmer[len(kmer) - int(move):len(kmer)]
    return bases


def kmers_to_sequence(kmers):
    """Convert a sequence of kmers into a contiguous symbol string.

    :param kmers: list of kmers from which to form a sequence

    .. note:
       This is simply a convenient synthesis of :func:`kmer_overlap`
       and :func:`kmers_to_call`
    """
    return kmers_to_call(kmers, kmer_overlap(kmers))
