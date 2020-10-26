#!/usr/bin/env python3
""" package-routine.py - package up a single routine into a distributable package
See 'package-routine.web' for how to use multiple files together

Example:

$ ./tools/package-routine.py dgesv

$ ./tools/package-routine.py dgesv

$ ./tools/package-routine.py -h

"""
## imports (all std)
import argparse
import io
import os
import glob
import re
import shutil
import time
import errno
import tarfile

# for 'blas' endings
import magmasubs

# construct & parse given arguments
parser = argparse.ArgumentParser(description='Package a single MAGMA routine into a folder')

parser.add_argument('routine', help='Routine to package up (i.e. sgemm, dgemm, etc)')
parser.add_argument('-o', '--output', default=None, help='Destination tar archive (leave empty for a default)')
parser.add_argument('--interface', default='cuda', choices=['hip', 'cuda'], help='Which interface/backend to use?')

args = parser.parse_args()

if args.routine.startswith('magma'):
    args.routine = args.routine
else:
    #pass
    args.routine = 'magma_' + args.routine

# generate output folder
if not args.output:
    args.output = "magma_" + args.interface + '_' + args.routine.replace('magmablas_', '').replace('magma_', '') + '.tar.gz'

print (f"""Packaging routine: {args.routine} and storing in: {args.output}""")

# escape sequence, so it can be in an fstring
_n = '\n'

# Package for a given interface
if args.interface == 'cuda':
    pass
elif args.interface == 'hip':
    pass
else:
    raise Exception(f"Unknown interface requested: {args.interface}")


# -*- Regex Definitions -*-

# regex for calling a MAGMA function
re_call = re.compile(r" *((?:magma(?:blas)?_|\w*_kernel)\w+)\s*(?:<<<.*>>>)?\s*\(")

# regex for a function definition, i.e. not just a declaration (must be multiline due to how many functions are declared)
#re_funcdef = re.compile(r"(?:extern \"C\"|static inline)? ?\n?(?:[\w\* ]+ *?)\n?(magma(?:blas)?_\w+)( \n)*\([\w\[\]\* ,\n]*\)\n? *\n?\{", re.MULTILINE)
re_funcdef = re.compile(r"(?:extern \"C\"|static inline)? ?\n?[\w\* ]+\s+(\w+)( \n)*\([\w\[\]\* ,\n]*\)\n? *\n?\{", re.MULTILINE)

# regex for a macro definition
re_macdef = re.compile(r"#define  *(magma(?:blas)?_\w+)\(")

# regex for an include statement
re_include = re.compile(r"\#include (?:\"|\<)(magma[\w\.]+)(?:\"|\<)")


# -*- Initialization -*-


# all files possible
allfiles = set(glob.glob("src/*.cpp") + glob.glob("control/*.cpp") + glob.glob("include/*.h") + glob.glob(f"interface_{args.interface}/*.cpp") + (glob.glob(f"magmablas/*.cpp") + glob.glob(f"magmablas/*.cu") + glob.glob(f"magmablas/*.cuh") + glob.glob(f"magmablas/*.h")) if args.interface == "cuda" else (glob.glob(f"magmablas_hip/*.cpp") + glob.glob(f"magmablas_hip/*.hpp") + glob.glob(f"magmablas_hip/*.h")))

#print (allfiles)


# set of all files
set_c = { 'control/pthread_barrier.h', 'control/affinity.h', 'control/trace.h', 'control/batched_kernel_param.h', 'include/magma_v2.h', f"interface_{args.interface}/error.h" }

# what functions are requested as part of the package (this may grow to other functions called recursively)
funcs_requested = set()

# list of defined functions (i.e. start with an empty set)
funcs_defined = set()

# functions that will emit a warning, due to some special case
funcs_warn = set()

# if these special cases are encountered, add it to warns
funcs_special_cases = {

}


# errored functions
funcs_err = set()

# functions to ignore
funcs_ignore = { 
    'magma_warn_leaks', 
    #'magma_dgetf2_native_fused', 'magma_dgetf2trsm_2d_native',

    #'magma_zlaswp_rowparallel_native', 'magma_claswp_rowparallel_native', 'magma_dlaswp_rowparallel_native', 'magma_slaswp_rowparallel_native',
    #'magma_zlaswp_columnserial', 'magma_claswp_columnserial', 'magma_dlaswp_columnserial', 'magma_slaswp_columnserial',
    #'magma_zlaswp_rowserial_native', 'magma_claswp_rowserial_native', 'magma_dlaswp_rowserial_native', 'magma_slaswp_rowserial_native',
    
    # TODO handle these?
    #'magma_zgetmatrix_1D_col_bcyclic', 'magma_cgetmatrix_1D_col_bcyclic', 'magma_dgetmatrix_1D_col_bcyclic', 'magma_sgetmatrix_1D_col_bcyclic', 
    #'magma_zgetmatrix_1D_row_bcyclic', 'magma_cgetmatrix_1D_row_bcyclic', 'magma_dgetmatrix_1D_row_bcyclic', 'magma_sgetmatrix_1D_row_bcyclic',
    #'magma_zsetmatrix_1D_col_bcyclic', 'magma_csetmatrix_1D_col_bcyclic', 'magma_dsetmatrix_1D_col_bcyclic', 'magma_ssetmatrix_1D_col_bcyclic', 
    #'magma_zsetmatrix_1D_row_bcyclic', 'magma_csetmatrix_1D_row_bcyclic', 'magma_dsetmatrix_1D_row_bcyclic', 'magma_ssetmatrix_1D_row_bcyclic',
}


# set of BLAS routines requested
blas_requested = set()


# -*- Utility Functions

# filter through files, and only return those that exist and have not yet been included
def newfiles(*fls):
    for fl in fls:
        if fl not in set_c and os.path.exists(fl):
            yield fl

# Read entire file
def readall(fl):
    # read entire file
    src_file = open(fl, 'r')
    src = src_file.read()
    src_file.close()
    return src

# return a set of matches pf 'regex' in 'src'
# NOTE: most are default `group==1`, so that is defaulted
def matches(regex, src, group=1):
    ret = set()
    for match in re.finditer(regex, src):
        ret.add(match.group(group))
    return ret


# return a set of functions still needed
def needed_funcs():
    return funcs_requested - funcs_defined - funcs_err - funcs_warn - funcs_ignore

_p_files = {}

def p_file(fname, mode):
    if mode not in _p_files:
        _p_files[mode] = set()
    if fname in _p_files[mode]:
        #print ("Warning: file checked multiple times: ", fname)
        pass
    print ("[", mode, "] Checking file:", fname, " " * 80, end='\r')
    _p_files[mode].add(fname)

# -*- Search through routines -*-

# attempt to resolve each one
for func in [args.routine]:
    funcs_requested.add(func)

    ct = 0
    fnm = func.replace('magmablas_', '').replace('magma_', '')

    # check a list of files
    for fl in newfiles(f"src/{fnm}.cpp", f"src/{fnm}_gpu.cpp", f"src/{fnm}2.cpp", f"src/{fnm}2_mgpu", *allfiles):
        src = readall(fl)

        # add functions which were defined
        defs = matches(re_funcdef, src)
        if func in defs:
            ct += 1

            funcs_defined.update(defs)
            print (func, fl)

            # add references to subroutines & other functions called
            funcs_requested.update(matches(re_call, src))

            set_c.add(fl)
            break

    if ct < 1:
        raise Exception(f"Unknown routine '{func}'")

print ("Checking for functions:", funcs_requested)

# while there are needed functions to resolve
while needed_funcs():
    # get first one
    func = next(iter(needed_funcs()))

    # ensure it is a magma function
    if not ('magma' in func or '_kernel' in func):
        raise Exception(f"Need function '{func}', which is not part of MAGMA!")

    # turn it into just the MAGMA name (no prefix)
    magma_name = func.replace('magma_', '').replace('magmablas_', '')

    # iterate through files the routine probably needs
    for fl in newfiles(f"src/{magma_name}.cpp", f"src/{''.join([i for i in magma_name if not i.isdigit()])}.cpp"):
        p_file(fl, 'defs')
        src = readall(fl)

        # get matches and see if this file works
        defs = matches(re_funcdef, src) | matches(re_macdef, src)

        if func in defs:
            # found it, so we need to include this file
            set_c.add(fl)
            funcs_defined.update(defs)

            # we need to see what else is requested
            funcs_requested.update(matches(re_call, src))

            # we found the requested function, so stop looking for it
            break

    if func not in funcs_defined:
        # we haven't found anything valid yet
        isFound = False

        if func in funcs_special_cases:
            funcs_warn.add(func)
            isFound = True

        if not isFound:
            if 'opencl' in func and 'opencl' not in args.interface:
                # we don't care about OpenCL functions
                funcs_ignore.add(func)
                isFound = True

        if not isFound:
            # check if it is a BLAS routine (in which case, it should
            #   be provided by someone else)
            for rout in magmasubs.blas:
                for prout in rout:
                    if prout in func:
                        funcs_defined.add(func)
                        blas_requested.add(magma_name)
                        isFound = True
                        break
            
        if not isFound:
            #print ("not yet found:", func)

            # not a BLAS routine, so now just search everywhere for it
            for fl in newfiles(*allfiles):
                p_file(fl, 'defs')
                src = readall(fl)

                # get matches and see if this file works
                defs = matches(re_funcdef, src) | matches(re_macdef, src)
                
                if func in defs:
                    # found it, so we need to include this file
                    funcs_defined.update(defs)

                    # we need to see what else is requested (if not a header)
                    if fl[fl.index('.'):] not in ('.h', '.hh', '.hpp', '.cuh',):
                        funcs_requested.update(matches(re_call, src))
                    set_c.add(fl)
                    isFound = True
                    break
        
            if not isFound:
                funcs_err.add(func)
                #raise Exception(f"Could not find '{func}'")

if funcs_err:
    raise Exception(f"Could not find functions: {funcs_err}")


print ("Checking for included files", set_c)

# new includse
keepGoing = True
while keepGoing:

    new_includes = set()

    for fl in set_c:
        p_file(fl, 'includes')
        src = readall(fl)
        for incfl in matches(re_include, src):
            possible = [
                f"include/{incfl}",
                f"control/{incfl}",
                f"magmablas/{incfl}" if args.interface == "cuda" else f"magmablas_{args.interface}/{incfl}",
            ]

            isFound = False
            for pos in possible:
                if pos in set_c:
                    isFound = True
                    break
            
            if isFound:
                continue

            # we need to find
            for pos in possible:
                if os.path.exists(pos):
                    new_includes.add(pos)
                    isFound = True
                    break
            if isFound:
                continue

            # not found
            raise Exception(f"Could not find included file '{incfl}'")

    set_c.update(new_includes)
    keepGoing = bool(new_includes)


# -*- Output -*-

# make output tarfile
tarcomp = args.output[args.output.rindex('.')+1:]
tf = tarfile.open(args.output, 'w:' + tarcomp)

def addfile(name, src):
    bs = src.encode()
    fp = io.BytesIO(bs)
    info = tarfile.TarInfo(name)
    info.size = len(bs)
    fp.seek(0)
    tf.addfile(info, fp)

addfile('README.md', f"""# MAGMA Package Routine

This is a generic README; check the `.mf` files (manifest files) for specific information about packaged routines.

  * FUNCS.mf
    * Contains a list of defined functions
  * BLAS.mf
    * Contains a list of required BLAS
  * WARNINGS.mf
    * Contains a list of possibly problematic functions

## CUDA Interface

To build with CUDA, source files that end in `.cu` should be compiled with `nvcc`, i.e. the NVIDIA CUDA compiler. Given as makefile rules, you should have (approximately):

(keep in mind, throughout these examples, that some variables are just illustrative; you will have to define or supplement them with the relevant files/definitions in your build system)

```makefile

# rule to compile single object file
%.o: %.cu $(magma_H)
\t$(NVCC) -std=c++11 $< -Xcompiler "-fPIC" -o $@

```

And, to compile MAGMA into your own library (say `libmine.so`), you would modify your existing rule:

```makefile

# rule to compile your library (including MAGMA objects from this folder)
libmine.so: $(MAGMA_CU_O) $(MINE_C_O)
\t$(CC) $^ -lcublas -lcusparse -lcudart -lcudadevrt -shared -o $@

```

Assuming `MAGMA_CU_O` are the object files from MAGMA, and `MINE_C_O` are the object files from your library, this should link them together and create your shared library

## HIP Interface

To build with HIP, source files that end in `.hip.cpp` should be compiled with `hipcc`, i.e. the HIP device compiler.

(keep in mind, throughout these examples, that some variables are just illustrative; you will have to define or supplement them with the relevant files/definitions in your build system)

```makefile

# rule to compile single object file
%.o: %.cu $(magma_H)
\t$(HIPCC) -DHAVE_HIP -std=c++11 -fno-gpu-rdc $< -fPIC -o $@

```

And, to compile MAGMA into your own library (say `libmine.so`), you would modify your existing rule:

```makefile

# rule to compile your library (including MAGMA objects from this folder)
libmine.so: $(MAGMA_HIP_O) $(MINE_C_O)
\t$(CC) $^ -lhipsparse -lhipblas -shared -o $@

```

Assuming `MAGMA_HIP_O` are the object files from MAGMA, and `MINE_C_O` are the object files from your library, this should link them together and create your shared library

""")



addfile('Makefile', f"""# -*- Makefile - generated by `package-routine.py`

# variables
NVCC       ?= nvcc

# recursive wildcard
rwildcard = $(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

# source files
MAGMA_C    := $(call rwildcard,.,*.c)

# object files
MAGMA_O    := $(patsubst %.cpp,%.o,$(MAGMA_C))

MAGMA_CFLAGS := -std=c++11 -DADD_ -DMIN_CUDA_ARCH=600 { {'hip': '-DHAVE_HIP', 'cuda': '-DHAVE_CUDA -DHAVE_CUBLAS'}[args.interface] }


default: libmagma_pkg.so test

# single file
%.o: %.cpp
\t$(NVCC) $(CFLAGS) -I./include -I./control $(MAGMA_CFLAGS) $< -Xcompiler "-fPIC" -c -o $@

# compile magma embedded
# (i.e. `magmapkg`)
libmagma_pkg.so: $(MAGMA_O)
\t$(CC) $^ $(LDFLAGS) -lcublas -lcusparse -lcudart -lcudadevrt -shared -o $@

test: test.c libmagma_pkg.so
\t$(CC) $(CFLAGS) -I./include -I./control $(MAGMA_CFLAGS) $^ $(LDFLAGS) -L./ -lmagma_pkg -o $@

clean: FORCE
\trm -f $(wildcard libmagma_pkg.so control/*.o src/*.o interface_{args.interface}/*.o)

FORCE:

.PHONY: default clean FORCE

""")
addfile('test.c', f"""/* test.c - GENERATED test file to ensure magma compiles & can execute
 *
 * Generated by `package-routine.py`
 *
 * @author: Cade Brown <cade@utk.edu>
 */
 
#include <magma_v2.h>
#include <stdio.h>

int main(int argc, char** argv) {{
    // initialize
    int st;
    if ((st = magma_init()) != MAGMA_SUCCESS) {{
        fprintf(stderr, "magma_init() failed! (code: %i)\\n", st);
        return -1;
    }}
    
    if ((st = magma_finalize()) != MAGMA_SUCCESS) {{
        fprintf(stderr, "magma_finalize() failed! (code: %i)\\n", st);
        
        return -1;
    }}
    // success
    return 0;
}} 

""")

addfile('FUNCS.mf', "\n".join(funcs_defined))
addfile('WARNINGS.mf', "\n".join(funcs_warn))
addfile('BLAS.mf', "\n".join(blas_requested))

for fl in set_c:
    addfile(fl, readall(fl))

tf.close()
