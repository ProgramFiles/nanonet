import argparse
import os
import multiprocessing


class FileExist(argparse.Action):
    """Check if the input file exist."""
    def __call__(self, parser, namespace, values, option_string=None):
        if values is not None and not os.path.exists(values):
             raise RuntimeError("File/path for '{}' does not exist, {}".format(self.dest, values))
        setattr(namespace, self.dest, values)


class CheckCPU(argparse.Action):
    """Make sure people do not overload the machine"""
    def __call__(self, parser, namespace, values, option_string=None):
        num_cpu = multiprocessing.cpu_count()
        if int(values) <= 0 or int(values) > num_cpu:
            raise RuntimeError('Number of jobs can only be in the range of {} and {}'.format(1, num_cpu))
        setattr(namespace, self.dest, values)


class AutoBool(argparse.Action):
    def __init__(self, option_strings, dest, default=None, required=False, help=None):
        """Automagically create --foo / --no-foo argument pairs"""

        if default is None:
            raise ValueError('You must provide a default with AutoBool action')
        if len(option_strings)!=1:
            raise ValueError('Only single argument is allowed with AutoBool action')
        opt = option_strings[0]
        if not opt.startswith('--'):
            raise ValueError('AutoBool arguments must be prefixed with --')

        opt = opt[2:]
        opts = ['--' + opt, '--no-' + opt]
        if default:
            default_opt = opts[0]
        else:
            default_opt = opts[1]
        super(AutoBool, self).__init__(opts, dest, nargs=0, const=None,
                                       default=default, required=required,
                                       help='{} (Default: {})'.format(help, default_opt))
    def __call__(self, parser, namespace, values, option_strings=None):
        if option_strings.startswith('--no-'):
            setattr(namespace, self.dest, False)
        else:
            setattr(namespace, self.dest, True)
