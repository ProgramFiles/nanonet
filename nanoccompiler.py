from distutils.cygwinccompiler import *

class Mingw64CCompiler(CygwinCCompiler):
    # Set some not so prehistoric compiler/linker options
    compiler_type = 'mingw32'

    def __init__ (self, verbose=0, dry_run=0, force=0):
        CygwinCCompiler.__init__ (self, verbose, dry_run, force)

        self.set_executables(
            compiler='gcc -O -Wall',
            compiler_so='gcc -mdll -O -Wall',
            compiler_cxx='g++ -O -Wall',
            linker_exe='gcc',
            linker_so='{} -shared'.format(self.linker_dll)
        )
        self.dll_libraries=[]


