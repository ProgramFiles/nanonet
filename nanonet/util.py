import sys
from itertools import tee, imap, izip, izip_longest, product
from functools import partial
from multiprocessing import Pool
import random
import string
import math

__eta__ = 1e-100

# N.B. this defines the order of our states, it is not lexographical!
def all_nmers(n=3):
    return [''.join(x) for x in product('ATGC', repeat=3)]


def random_string(length=6):
    """Return a random upper-case string of given length.

    :param length: length of string to return.
    """

    return ''.join(random.choice(string.ascii_uppercase) for _ in range(length))


def conf_line(option, value, pad=30):
    return '{} = {}\n'.format(option.ljust(pad), value)


class ReplWrapper(object):
    def __init__(self, replacement, occurrence):
        """Replace Nth (or all if occurrence=0) of regex match with
        replacement in a string.

        e.g.
        data = re.sub(r'target', ReplWrapper(r'replace', 1)
        """

        self.count = 0
        self.replacement = replacement
        self.occurrence = occurrence
    def repl(self, match):
        self.count += 1
        if self.occurrence == 0 or self.occurrence == self.count:
            return match.expand(self.replacement)
        else:
            try:
                return match.group(0)
            except IndexError:
                return match.group(0)


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


def docstring_parameter(*sub):
    """Allow docstrings to contain parameters."""
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*sub)
        return obj
    return dec


class FastaWrite(object):
    def __init__(self, filename=None):
        """Simple Fasta writer to file or stdout. The only task this
        class achieves is formatting sequences into fixed line lengths.

        :param filename: if `None` or '-' output is written to stdout
            else it is written to a file opened with name `filename`.
        :param mode: mode for opening file.
        """
        self.filename = filename

    def __enter__(self):
        if self.filename is not None and self.filename != '-':
            self.fh = open(self.filename, 'w')
        else:
            self.fh = sys.stdout
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.fh is not sys.stdout:
            self.fh.close()

    def write(self, name, seq, meta=None, line_length=80):
        if meta is None:
            self.fh.write(">{}\n".format(name))
        else:
            self.fh.write(">{}\n".format(name))
        
        for chunk in (seq[i:i+line_length] for i in xrange(0, len(seq), line_length)):
            self.fh.write('{}\n'.format(chunk))


def _try_except_pass(func, *args, **kwargs):
    """Implementation of try_except_pass below. When wrapping a function we
    would ordinarily form a closure over a (sub)set of the inputs. Such
    closures cannot be pickled however since the wrapper name is not
    importable. We get around this by using functools.partial (which is
    pickleable). The result is that we can decorate a function to mask
    exceptions thrown by it.
    """

    # Strip out "our" arguments, this slightly perverse business allows
    #    us to call the target function with multiple arguments.
    recover = kwargs.pop('recover', None)
    recover_fail = kwargs.pop('recover_fail', False)
    try:
        return func(*args, **kwargs)
    except:
        exc_info = sys.exc_info()
        try:
            if recover is not None:
                recover(*args, **kwargs)
        except Exception as e:
            sys.stderr.write("Unrecoverable error.")
            if recover_fail:
                raise e
            else:
                traceback.print_exc(sys.exc_info()[2])
        # print the original traceback
        traceback.print_tb(exc_info[2])
        return None


def try_except_pass(func, recover=None, recover_fail=False):
    """Wrap a function to mask exceptions that it may raise. This is
    equivalent to::

        def try_except_pass(func):
            def wrapped()
                try:
                    func()
                except Exception as e:
                    print str(e)
            return wrapped

    in the simplest sense, but the resulting function can be pickled.

    :param func: function to call
    :param recover: function to call immediately after exception thrown in
        calling `func`. Will be passed same args and kwargs as `func`.
    :param recover_fail: raise exception if recover function raises?

    ..note::
        See `_try_except_pass` for implementation, which is not locally
        scoped here because we wish for it to be pickleable.

    ..warning::
        Best practice would suggest this to be a dangerous function. Consider
        rewriting the target function to better handle its errors. The use
        case here is intended to be ignoring exceptions raised by functions
        when mapped over arguments, if failures for some arguments can be
        tolerated.

    """
    return partial(_try_except_pass, func, recover=recover, recover_fail=recover_fail)


class __NotGiven(object):
    def __init__(self):
        """Some horrible voodoo"""
        pass


def tang_imap(
    function, args, fix_args=__NotGiven(), fix_kwargs=__NotGiven(),
    threads=1, unordered=False, chunksize=1,
    pass_exception=False, recover=None, recover_fail=False,
):
    """Wrapper around various map functions

    :param function: the function to apply, must be pickalable for multiprocess
        mapping (problems will results if the function is not at the top level
        of scope).
    :param args: iterable of argument values of function to map over
    :param fix_args: arguments to hold fixed
    :param fix_kwargs: keyword arguments to hold fixed
    :param threads: number of subprocesses
    :param unordered: use unordered multiprocessing map
    :param chunksize: multiprocessing job chunksize
    :param pass_exception: ignore exceptions thrown by function?
    :param recover: callback for recovering from exceptions in function
    :param recover_fail: reraise exceptions when recovery fails?

    .. note::
        This function is a generator, the caller will need to consume this.

    If fix_args or fix_kwargs are given, these are first used to create a
    partially evaluated version of function.

    The special :class:`__NotGiven` is used here to flag when optional arguments
    are to be used.
    """

    my_function = function
    if not isinstance(fix_args, __NotGiven):
        my_function = partial(my_function, *fix_args)
    if not isinstance(fix_kwargs, __NotGiven):
        my_function = partial(my_function, **fix_kwargs)

    if pass_exception:
        my_function = try_except_pass(my_function, recover=recover, recover_fail=recover_fail)

    if threads == 1:
        for r in imap(my_function, args):
            yield r
    else:
        pool = Pool(threads)
        if unordered:
            mapper = pool.imap_unordered
        else:
            mapper = pool.imap
        for r in mapper(my_function, args, chunksize=chunksize):
            yield r
        pool.close()
        pool.join()


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
