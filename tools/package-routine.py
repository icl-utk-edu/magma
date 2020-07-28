#!/usr/bin/env python3
""" package-routine.py - package up a single routine into a distributable package

"""
## imports (all std)
import argparse
import sys
import os
import glob
import re
import shutil
import errno
import time

# construct & parse given arguments
parser = argparse.ArgumentParser(description='Package a single MAGMA routine into a folder')

parser.add_argument('routines', nargs='+', help='Routines to package up (i.e. sgemm, dgemm, etc)')
parser.add_argument('-o', '--output', default=None, help='Destination folder (leave empty for a default)')
parser.add_argument('--interface', default='cuda', choices=['hip', 'cuda'], help='Which interface/backend to use?')

args = parser.parse_args()

#assert len(args.routines) == 1 and "only 1 routine supported at this time!"


if not args.output:
    args.output = "magma_pkg_" + "_".join(args.routines)

print (f"""Output: {args.output}""")

try:
    os.makedirs(args.output)
except:
    pass

# common files required by all routines
common_files = glob.glob("include/magma_*.h") + glob.glob("control/*.cpp") + glob.glob(f"interface_{args.interface}/*.cpp") + glob.glob(f"interface_{args.interface}/*.h")

CC = "cc"

# C flags
CFLAGS = [

]

# Linker flags (to ld)
LDFLAGS = [

]


# ensure precisions are generated
if "include/magma_s.h" not in common_files:
    raise Exception("Build files have not been properly generated! Run `make generate` in the MAGMA repo!")

# Set ENV vars
if args.interface == 'cuda':
    CC = "nvcc"
    CFLAGS += [
        '-DHAVE_HIP',
        '-Xcompiler "-fPIC"'
        '-std=c++11',
    ]

    LDFLAGS += [
        '-lcublas -lcusparse -lcudart -lcudadevrt'
    ]

elif args.interface == 'hip':
    CC = "hipcc"

    CFLAGS += [
        '-DHAVE_HIP',
        '-std=c++11 -fPIC -fno-gpu-rdc'
    ]

    LDFLAGS += [
        '-L/opt/rocm/lib -L/opt/rocm/hip/lib',
        '-lhipsparse -lhipblas'
    ]

else:
    raise Exception(f"Unknown interface requested: {args.interface}")



# Utility Funcs

# Print iterations progress
def show_progress (iteration, total, prefix = '', suffix = '', decimals = 1, length = 80, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix} {" " * 20}', end = printEnd)
    # Print New Line on Complete
    if iteration == total - 1: 
        print("Done! " + " " * (40 + length))


# Read entire file
def readall(fl):
    # read entire file
    src_file = open(fl, 'r')
    src = src_file.read()
    src_file.close()
    return src


# Regex Patterns

# calling another MAGMA function
reg_call_magma = re.compile(r" *(magma_\w+)\(")

# defining a MAGMA function
reg_def_magma = re.compile(r"^(?:extern \"C\"|static inline)? ?\n?(?:[\w\*]+ *?)\n?(magma_\w+)( \n)*\([\w\* ,\n]*\)\n? *\{", re.MULTILINE)

# macro definnition
reg_macdef_magma = re.compile(r"#define  *(magma_\w+)\(")

# set of defined functions
# NOTE: these defaults are macros or other special cases that are defined in common/headers
defined_funcs = set()

# set of functions called
requested_funcs = set()

# -*- Step 1: Iterate through common files -*-

print (" -- COMMON FILES -- ")

# go through all files in the common folder
for i, fl in enumerate(common_files):
    # print out progress
    show_progress(i, len(common_files), fl)
    
    # read file
    src = readall(fl)

    # iterate through all functions defined by MAGMA
    for match in re.finditer(reg_def_magma, src):
        defined_funcs.add(match.group(1))

    # iterate through all macros defined by MAGMA
    for match in re.finditer(reg_macdef_magma, src):
        print (match.group(1))

        defined_funcs.add(match.group(1))




print (" -- ROUTINES -- ")

# go through each routine
for i, rout in enumerate(args.routines):
    # assume src/ path
    fl = f"src/{rout}.cpp"
    # read file
    src = readall(fl)

    # printout progress
    show_progress(i, len(args.routines), rout)

    # iterate through all functions defined by MAGMA
    for match in re.finditer(reg_def_magma, src):
        defined_funcs.add(match.group(1))

    # iterate through all macros defined by MAGMA
    for match in re.finditer(reg_macdef_magma, src):
        defined_funcs.add(match.group(1))

    # add requested function for this routine
    requested_funcs.add(f"magma_{rout}")

    # set of subroutines it calls
    subroutines = set()

    # now, search through other functions called by this routine
    for match in re.finditer(reg_call_magma, src):
        subroutines.add(match.group(1))

    # add all results in
    requested_funcs.update(subroutines)

#print (defined_funcs)

# what functions do we need to still find
needed_funcs = requested_funcs - defined_funcs



if needed_funcs:
    print (f"Still need {needed_funcs}")
else:
    # found them all
    print ("Done with no other files traversed!")


# copy a file, creating destination folder
def copy(src, dst):
    try:
        os.makedirs(os.path.dirname(dst))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
    shutil.copy(src, dst)

# copy in everything

try:
    os.makedirs(f"{args.output}/common")
except:
    pass

for fl in common_files:
    copy(fl, f"{args.output}/common/{fl}")

    