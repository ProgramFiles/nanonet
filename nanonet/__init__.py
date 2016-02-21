import os
import subprocess

__currennt_exe__ = 'currennt'

try:
    __currennt_exe__ = os.path.abspath(os.environ['CURRENNT'])
except KeyError:
    __currennt_exe__ = 'currennt'

# Check we can run currennt
try:
    with open(os.devnull, 'w') as devnull:
        subprocess.call([__currennt_exe__, '-h'], stdout=devnull, stderr=devnull)
except OSError:
    raise OSError("Cannot execute currennt, it must be in your path as 'currennt' or set via the environment variable 'CURRENNT'.")
