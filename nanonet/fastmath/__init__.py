import numpy as np

from numpy.ctypeslib import ndpointer
from ctypes import c_size_t

from nanonet.util import get_shared_lib
nanonetmaths = get_shared_lib('nanonetmaths')

_tanh = nanonetmaths.fast_tanh
_tanh.restype = None
_tanh.argtypes = [
    ndpointer(dtype='f4', flags='CONTIGUOUS'),
    ndpointer(dtype='f4', flags='CONTIGUOUS'),
    c_size_t
]

def fast_tanh(x):
    if not x.flags['OWNDATA']:
        return np.tanh(x)
    if x.dtype != 'f4':
        return np.tanh(x)

    length = reduce(lambda x,y: x*y, iter(x.shape), 1)
    if length % 4 != 0:
        return np.tanh(x)

    out = np.empty_like(x, dtype='f4')
    _tanh(x, out, length)
    return out
