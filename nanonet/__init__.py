import os

try:
    __currennt_exe__ = os.environ['CURRENNT']
except KeyError:
    __currennt_exe__ = 'currennt'
