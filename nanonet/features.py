import os
import random
import string
import numpy as np 
from netCDF4 import Dataset

###
### TODO: remove this
###
from tang.fast5 import fast5





###
### TODO: replace this with a simple class that can just extract event data
###
class MappingFast5(fast5):
    """Basically an alternative interface to Fast5 files"""
    __phase_section__   = {'T':'template', 'C':'complement', '2d':'2d'}
    
    def get_events(self, phase='T'):
        return self.get_section_events(self.__phase_section__[phase])
    
    def get_mapped_read(self, phase='T'):
        events, _ = self.get_any_mapping_data(self.__phase_section__[phase])
        return events
    
    def get_mapping_accuracy(self, direction="T"):
        base = self.get_analysis_latest(self.__default_alignment_analysis__)
        attr_path = self._join_path(base, self.__default_basecall_mapping_alignment_summary__.format(self.__phase_section__[direction]))

        try:
            attri = dict(self[attr_path].attrs)
            return attri.get("accuracy", 0.0)
        except:
            return 0.0
 
    def get_ref(self, phase='T'):
        try:
            self.get_reference_fasta(self.__phase_section__[phase])
        except:
            return None


def padded_offset_array(array, pos):
    """Offset an array and pad with zeros.
    :param array: the array to offset.
    :param pos: offset size, positive values correspond to shifting the
       original array to the left (and padding the end of the output).
    """
    out = np.empty(len(array))
    if pos == 0:
        out = array
    elif pos > 0:
        out[:-pos] = array[pos:]
        out[-pos:] = 0.0
    else:
        out[:-pos] = 0.0
        out[-pos:] = array[0:pos]
    return out


def scale_array(X, with_mean=True, with_std=True, copy=True):
    """Standardize an array
    Center to the mean and component wise scale to unit variance.
    :param X: the data to center and scale.
    :param with_mean: center the data before scaling.
    :param with_std: scale the data to unit variance.
    :param copy: copy data (or perform in-place)
    """    
    X = np.asarray(X)
    if copy:
        X = X.copy()
    if with_mean:
        mean_ = np.mean(X)
        X -= mean_
        mean_1 = X.mean()
        if not np.allclose(mean_1, 0.0):
            X -= mean_1
    if with_std:
        scale_ = np.std(X)
        if scale == 0.0:
            scale = 1.0
        X /= scale_
        if with_mean:
            mean_2 = X.mean()
            if not np.allclose(mean_2, 0.0):
                X -= mean_2
    return X


def basecall_features(filename, window=[-1, 0, 1], phase='T', trim=10):
    """Read events from a .fast5 and return feature vectors.

    :param filename: path of file to read.
    :param window: list specifying event offset positions from which to
        derive features. A short centered window is used by default.
    :param phase: 'T' (template) or 'C' (complement).
    :param trim: number of feature vectors to trim from ends.
    """

    with MappingFast5(filename) as f:
        events = f.get_events(phase=phase)
    
    fg = SquiggleFeatureGenerator(events)
    for pos in window:
        fg.add_mean_pos(pos)
        fg.add_sd_pos(pos)
        fg.add_dwell_pos(pos)
        fg.add_mean_diff_pos(pos)
    X = fg.to_numpy()
    if trim > 0:
       X = X[trim:-trim]
    return X


def make_currennt_basecall_input_multi(fast5_files, netcdf_file, window=[-1, 0, 1], num_kmers=64, phase="T", trim=10, min_len=1000, max_len=9000):
     """Prepare a .netcdf file for input to currennt from .fast5 files.

     :param fast5_files:
     :param netcdf_file:
     :param window:
     :param num_kmers:
     :param phase:
     :param trim:
     :param min_len:
     :param max_lan:
     """

     # We need to know ahead of time how wide our feature vector is, lets generate one and take a peek
     X = basecall_features(fast5_files[0], window=window, phase=phase, trim=0)
     inputPattSize = X.shape[1]

     with Dataset(netcdf_file, 'w', format='NETCDF4') as ncroot:
         # Set dimensions
         ncroot.createDimension('numSeqs', None)
         ncroot.createDimension('numLabels', num_kmers + 1)
         ncroot.createDimension('maxSeqTagLength', 500)
         ncroot.createDimension('numTimesteps', None)
         ncroot.createDimension('inputPattSize', inputPattSize)
         
         # Set variables
         seqTags = ncroot.createVariable('seqTags', 'S1',  ('numSeqs', 'maxSeqTagLength'))
         seqLengths = ncroot.createVariable('seqLengths', 'i4', ('numSeqs',))
         inputs = ncroot.createVariable('inputs', 'f4', ('numTimesteps', 'inputPattSize'))
         targetClasses = ncroot.createVariable('targetClasses', 'i4', ('numTimesteps',))

         for f in fast5_files:
             filename = os.path.basename(f)
             X = basecall_features(f, window=window, phase=phase, trim=trim)
             if len(X) < min_len or len(X) > max_len:
                 continue

             numTimesteps = X.shape[0]             
             _seqTags = np.zeros(500, dtype = "S1")
             _seqTags[:len(filename)] = list(filename)
             # Assign values to variables for each input file
             curr_numSeqs = len(ncroot.dimensions["numSeqs"])
             curr_numTimesteps = len(ncroot.dimensions["numTimesteps"])
             
             seqTags[curr_numSeqs] = _seqTags
             seqLengths[curr_numSeqs] = numTimesteps
             inputs[curr_numTimesteps:] = X


class SquiggleFeatureGenerator(object):
    def __init__(self, events, labels=None):
        """Feature vector generation from events.

        :param events: standard event array.
        :param labels: labels for events, only required for training

        ..note:
            The order in which the feature adding methods is called should be the
            same for both training and basecalling.
        """
        self.events = events
        self.labels = labels
        self.features = {}
        self.feature_order = []

        # Augment events 
        for field in ('mean', 'stdv', 'length',):
            scale_array(self.events[field], copy=False)
        delta = np.edifff1d(self.events['mean'], to_begin=0)
        scale_array(delta, with_mean=False, copy = False)
        self.events = nprf.append_fields(events, 'delta', delta)
 
    def to_numpy(self):
        out = np.empty((self.end - self.start, len(self.feature_order)))
        for j, key in enumerate(self.feature_order):
            out[:, j] = self.features[key]
        return out

    def _add_field_pos(self, field, pos):
        tag = "{}[{}]".format(field, pos)
        if tag in self.features:
            return self
        self.feature_order.append(tag)
        self.features[tag] = self._padded_offset_array(self.events[field], pos)
        return self

    def add_mean_pos(self, pos):
        self._add_field_pos('mean', pos)
        return self

    def add_sd_pos(self, pos):
        self._add_field_pos('stdv', pos)
        return self

    def add_dwell_pos(self, pos):
        self._add_field_pos('length', pos)
        return self

    def add_mean_diff_pos(self, pos):
        self._add_field_pos('delta', pos)
        return self

