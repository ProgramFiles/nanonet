import os

__currennt_exe__ = 'currennt'

try:
    __currennt_exe__ = os.environ['CURRENNT']
except KeyError:
    __currennt_exe__ = 'currennt'
