import os
import sys
from glob import glob
import subprocess
import shutil
import re
import h5py
import numpy as np

class Fast5(h5py.File):
    """Class for grabbing data from single read fast5 files. Many attributes/
    groups are assumed to exist currently (we're concerned mainly with reading).
    Needs some development to make robust and for writing.

    """
    __base_analysis__ = '/Analyses'
    __event_detect_name__ = 'EventDetection'
    __raw_path__ = '/Raw/Reads'
    __event_path__ = '{}/{}_000/Reads/'.format(__base_analysis__, __event_detect_name__)
    __channel_meta_path__ = '/UniqueGlobalKey/channel_id'
    __tracking_id_path__ = 'UniqueGlobalKey/tracking_id'
    __context_tags_path__ = 'UniqueGlobalKey/context_tags'

    __default_basecall_2d_analysis__ = 'Basecall_2D'
    __default_basecall_1d_analysis__ = 'Basecall_1D'

    __default_seq_section__ = '2D'
    __default_basecall_fastq__ = 'BaseCalled_{}/Fastq'
    __default_basecall_1d_events__ = 'BaseCalled_{}/Events'
    __default_basecall_1d_model__ = 'BaseCalled_{}/Model'
    __default_basecall_1d_summary__ = 'Summary/basecall_1d_{}'
    __default_section__ = 'template'

    def __init__(self, fname, read='r'):
        super(Fast5, self).__init__(fname, read)

        # Attach channel_meta as attributes, slightly redundant
        for k, v in self[self.__channel_meta_path__].attrs.iteritems():
            setattr(self, k, v)
        # Backward compat.
        self.sample_rate = self.sampling_rate

        self.filename_short = os.path.splitext(os.path.basename(self.filename))[0]
        short_name_match = re.search(re.compile(r'ch\d+_file\d+'), self.filename_short)
        self.name_short = self.filename_short
        if short_name_match:
            self.name_short = short_name_match.group()


    def _join_path(self, *args):
        return '/'.join(args)


    @property
    def writable(self):
        """Can we write to the file."""
        if self.mode is 'r':
            return False
        else:
            return True


    def assert_writable(self):
        assert self.writable, "File not writable, opened with {}.".format(self.mode)


    @property
    def channel_meta(self):
        """Channel meta information as python dict"""
        return dict(self[self.__channel_meta_path__].attrs)


    @property
    def attributes(self):
        """Attributes for a read, assumes one read in file"""
        return dict(self.get_read(group = True).attrs)


    def summary(self, rename=True, delete=True, scale=True):
        """A read summary, assumes one read in file"""
        to_rename = zip(
            ('start_mux', 'abasic_found', 'duration', 'median_before'),
            ('mux', 'abasic', 'strand_duration', 'pore_before')
        )
        to_delete = ('read_number', 'scaling_used')

        data = deepcopy(self.attributes)
        data['filename'] = os.path.basename(self.filename)
        data['channel'] = self.channel_meta['channel_number']
        if scale:
            data['duration'] /= self.channel_meta['sampling_rate']
            data['start_time'] /= self.channel_meta['sampling_rate']
        if rename:
            for i,j in to_rename:
                try:
                    data[j] = data[i]
                    del data[i]
                except KeyError:
                    pass
        if delete:
            for i in to_delete:
                try:
                    del data[i]
                except KeyError:
                    pass

        return data


    def strip_analyses(self, keep=('{}_000'.format(__event_detect_name__), __raw_path__)):
        """Remove all analyses from file

        :param keep: whitelist of analysis groups to keep

        """
        analyses = self[self.__base_analysis__]
        for name in analyses.keys():
            if name not in keep:
                del analyses[name]


    def repack(self):
        """Run h5repack on the current file. Returns a fresh object."""
        path = os.path.abspath(self.filename)
        path_tmp = '{}.tmp'.format(path)
        mode = self.mode
        self.close()
        subprocess.call(['h5repack', path, path_tmp])
        shutil.move(path_tmp, path)
        return Fast5(path, mode)


    def get_reads(self, group=False, raw=False, read_numbers=None):
        """Iterator across event data for all reads in file

        :param group: return hdf group rather than event data
        """
        if not raw:
            reads = self[self.__event_path__]
        else:
            try:
                reads = self[self.__raw_path__]
            except:
                yield self.get_raw()[0]

        if read_numbers is None:
            it = reads.keys()
        else:
            it = (k for k in reads.keys()
                  if reads[k].attrs['read_number'] in read_numbers)

        if group == 'all':
            for read in it:
                yield reads[read], read
        elif group:
            for read in it:
                yield reads[read]
        else:
            for read in it:
                if not raw:
                    yield self._get_read_data(reads[read])
                else:
                    yield self._get_read_data_raw(reads[read])


    def get_read(self, group=False, raw=False, read_number=None):
        """Like get_reads, but only the first read in the file

        :param group: return hdf group rather than event/raw data
        """
        if read_number is None:
            return self.get_reads(group, raw).next()
        else:
            return self.get_reads(group, raw, read_numbers=[read_number]).next() 


    def _get_read_data(self, read, indices=None):
        """Private accessor to read event data"""
        # We choose the following to always be floats
        float_fields = ('start', 'length', 'mean', 'stdv')

        events = read['Events']

        # We assume that if start is an int or uint the data is in samples
        #    else it is in seconds already.
        needs_scaling = False
        if events['start'].dtype.kind in ['i', 'u']:
            needs_scaling = True

        dtype = np.dtype([(
            d[0], 'float') if d[0] in float_fields else d
            for d in events.dtype.descr
        ])
        data = None
        with events.astype(dtype):
            if indices is None:
                data = events[()]
            else:
                try:
                    data = events[indices[0]:indices[1]]
                except:
                    raise ValueError(
                        'Cannot retrieve events using {} as indices'.format(indices)
                    )

        # File spec mentions a read.attrs['scaling_used'] attribute,
        #    its not clear what this is. We'll ignore it and hope for
        #    the best.
        if needs_scaling:
            data['start'] /= self.sample_rate
            data['length'] /= self.sample_rate
        return data


    def _get_read_data_raw(self, read, indices=None, scale=True):
        """Private accessor to read raw data"""
        raw = read['Signal']
        dtype = float if scale else int

        data = None
        with raw.astype(dtype):
            if indices is None:
                data = raw[()]
            else:
                try:
                    data = raw[indices[0]:indices[1]]
                except:
                    raise ValueError(
                        'Cannot retrieve events using {} as indices'.format(indices)
                    )

        # Scale data to pA
        if scale:
            meta = self.channel_meta
            raw_unit = meta['range'] / meta['digitisation']
            data = (data + meta['offset']) * raw_unit
        return data


    def get_read_stats(self):
        """Combines stats based on events with output of .summary, assumes a
        one read file.

        """
        data = deepcopy(self.summary())
        read = self.get_read()
        sorted_means = np.sort(read['mean'])
        n_events = len(sorted_means)
        n10 = int(0.1*n_events)
        n90 = int(0.9*n_events)
        data['range_current'] = sorted_means[n90] - sorted_means[n10]
        data['median_current'] = sorted_means[int(0.5*n_events)] # could be better
        data['num_events'] = n_events
        data['median_sd'] = np.median(read['stdv'])
        return data


    def get_analysis_latest(self, name):
        """Get group of latest (present) analysis with a given base path.

        :param name: Get the (full) path of newest analysis with a given base
            name.
        """
        try:
            return self._join_path(
                self.__base_analysis__,
                sorted(filter(
                    lambda x: name in x, self[self.__base_analysis__].keys()
                ))[-1]
            )
        except (IndexError, KeyError):
            raise IndexError('No analyses with name {} present.'.format(name))


    def get_analysis_new(self, name):
        """Get group path for new analysis with a given base name.

        :param name: desired analysis name
        """

        # Formatted as 'base/name_000'
        try:
            latest = self.get_analysis_latest(name)
            root, counter = latest.rsplit('_', 1)
            counter = int(counter) + 1
        except IndexError:
            # Nothing present
            root = self._join_path(
                self.__base_analysis__, name
            )
            counter = 0
        return '{}_{:03d}'.format(root, counter)



def iterate_fast5(path, strand_list=None, paths=False, mode='r', limit=None):
    """Iterate over directory or list of .fast5 files.

    :param path: Directory in which single read fast5 are located or filename.
    :param strand_list: list of filenames to iterate, will be combined with path.
    :param paths: yield file paths instead of fast5 objects.
    :param mode: mode for opening files.
    :param limit: limit number of files to consider.
    """
    if strand_list is None:
        #  Could make glob more specific to filename pattern expected
        if os.path.isdir(path):
            files = glob(os.path.join(path, '*.fast5'))
        else:
            files = [path]
    else:
        files = [os.path.join(path, x) for x in strand_list]
        
    for f in files[:limit] :
        if not os.path.exists(f):
            sys.stderr.write('File {} does not exist, skipping\n'.format(f))
            continue
        if not paths:
            fh = Fast5(f, read=mode)
            yield fh
            fh.close()
        else:
            yield os.path.abspath(f)
