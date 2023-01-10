#!/usr/bin/env python
#
# MAGMA (version 2.0) --
# Univ. of Tennessee, Knoxville
# Univ. of California, Berkeley
# Univ. of Colorado, Denver
# @date

## @file
#  @author Mark Gates
#  @author Hartwig Anzt
#
# Script to run testers with various matrix sizes.
# Small sizes are chosen around block sizes (e.g., 30...34 around 32) to
# detect bugs that occur at the block size, and the switch over from
# LAPACK to MAGMA code.
# Tall and wide sizes are chosen to exercise different aspect ratios,
# e.g., nearly square, 2:1, 10:1, 1:2, 1:10.
# The -h or --help option provides a summary of the options.
#
# Batch vs. interactive mode
# --------------------------
# When output is redirected to a file, it runs in batch mode, printing a
# short summary to stderr on the console and all other output to the file.
# For example:
#
#       ./run_tests.py --lu --precision s --small > lu.txt
#       testing_sgesv_gpu -c                      ok
#       testing_sgetrf_gpu -c2                    ok
#       testing_sgetf2_gpu -c                     ok
#       testing_sgetri_gpu -c                     ** 45 tests failed
#       testing_sgetrf_mgpu -c2                   ok
#       testing_sgesv -c                          ok
#       testing_sgetrf -c2                        ok
#
#       ****************************************************************************************************
#       summary
#       ****************************************************************************************************
#         282 tests in 7 commands passed
#          45 tests failed accuracy test
#           0 errors detected (crashes, CUDA errors, etc.)
#       routines with failures:
#           testing_sgetri_gpu -c
#
# When output is to console (tty), it runs in interactive mode, pausing after
# each test. At the pause, typing "M" re-makes and re-runs that tester,
# while typing enter goes to the next tester.
# For example (some output suppressed with ... for brevity):
#
#       ./run_tests.py --lu --precision s --small
#       ****************************************************************************************************
#       ./testing_sgesv_gpu -c --range 1:20:1 ...
#       ****************************************************************************************************
#           N  NRHS   CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||B - AX|| / N*||A||*||X||
#       ================================================================================
#           1     1     ---   (  ---  )      0.00 (   0.00)   9.26e-08   ok
#           2     1     ---   (  ---  )      0.00 (   0.00)   1.32e-08   ok
#           3     1     ---   (  ---  )      0.00 (   0.00)   8.99e-09   ok
#       ...
#         ok
#       [enter to continue; M to make and re-run]
#
#       ****************************************************************************************************
#       ./testing_sgetri_gpu -c --range 1:20:1 ...
#       ****************************************************************************************************
#       % MAGMA 1.4.0 svn compiled for CUDA capability >= 3.0
#       % CUDA runtime 6000, driver 6000. MAGMA not compiled with OpenMP.
#       % device 0: GeForce GT 750M, 925.5 MHz clock, 2047.6 MB memory, capability 3.0
#       Usage: ./testing_sgetri_gpu [options] [-h|--help]
#
#           N   CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||R||_F / (N*||A||_F)
#       =================================================================
#           1      0.00 (   0.00)      0.00 (   0.00)   6.87e+01   failed
#           2      0.00 (   0.00)      0.00 (   0.00)   2.41e+00   failed
#           3      0.01 (   0.00)      0.00 (   0.00)   1.12e+00   failed
#       ...
#         ** 45 tests failed
#       [enter to continue; M to make and re-run]
#
#       ...
#
#       ****************************************************************************************************
#       summary
#       ****************************************************************************************************
#         282 tests in 7 commands passed
#          45 tests failed accuracy test
#           0 errors detected (crashes, CUDA errors, etc.)
#       routines with failures:
#           testing_sgetri_gpu -c
#
#
# What tests are run
# ------------------
# The --blas, --aux, --chol, --lu, --qr, --syev, --sygv, --geev, --svd options run
# particular sets of tests. By default, all tests are run.
#
# The --start option skips all testers before the given one, then continues
# with testers from there. This is helpful to restart a batch. For example:
#
#       ./run_tests.py --start testing_spotrf
#
# If specific testers are named on the command line, only those are run.
# For example:
#
#       ./run_tests.py testing_spotrf testing_sgetrf
#
# The -p/--precision option controls what precisions are tested, the default
# being "sdcz" for all four precisions. For example, to run single and double:
#
#       ./run_tests.py -p sd
#
# The -s/--small, -m/--medium, -l/--large options control what sizes are tested,
# the default being all three sets.
#       -s/--small  does small  tests, N < 300.
#       -m/--medium does medium tests, N < 1000.
#       -l/--large  does large  tests, N > 1000.
# For example, running small and medium tests:
#
#       ./run_tests.py -s -m
#
#
# What is checked
# ------------------
# The --memcheck option runs cuda-memcheck. This is very helpful for finding
# memory bugs (reading & writing outside allocated memory). It is, however, slow.
#
# The --tol option sets the tolerance to verify accuracy. This is 30 by default,
# which may be too tight for some testers. Setting it somewhat higher
# (e.g., 50 or 100) filters out spurious accuracy failures.

import os
import re
import sys
import time

import subprocess
from subprocess import PIPE, STDOUT

from optparse import OptionParser

# on a TTY screen, stop after each test for user input
# when redirected to file ("batch mode"), don't stop
batch = not sys.stdout.isatty()

parser = OptionParser()
parser.add_option('-p', '--precisions',  action='store',      dest='precisions', help='run given precisions', default='sdcz' )
parser.add_option(      '--start',       action='store',      dest='start',      help='start with given routine; useful to restart an interupted run')
parser.add_option(      '--memcheck',    action='store_true', dest='memcheck',   help='run with cuda-memcheck (slow)')
parser.add_option(      '--tol',         action='store',      dest='tol',        help='set tolerance')
                                         
parser.add_option('-s', '--small',       action='store_true', dest='small',      help='run small  tests, N < 300')
parser.add_option('-m', '--medium',      action='store_true', dest='med',        help='run medium tests, N < 1000')
parser.add_option('-l', '--large',       action='store_true', dest='large',      help='run large  tests, N > 1000')
parser.add_option('-n', '--nonsym',      action='store_true', dest='nonsym',     help='run nonsym  tests')

parser.add_option(      '--sparse-blas', action='store_true', dest='sparse_blas', help='run sparse BLAS tests')
parser.add_option(      '--solver',      action='store_true', dest='solver',      help='run sparse solvers')
parser.add_option(      '--control',     action='store_true', dest='control',     help='run sparse IO, copy, etc.')

parser.add_option(      '--csr',         action='store_true', dest='csr',         help='run CSR matrix format')
parser.add_option(      '--ell',         action='store_true', dest='ell',         help='run ELL matrix format')
parser.add_option(      '--sellp',       action='store_true', dest='sellp',       help='run SELLP matrix format')
parser.add_option(      '--csr5',        action='store_true', dest='csr5',        help='run CSR5 matrix format')

parser.add_option(      '--cg'               , action='store_true', dest='cg'            , help='run cg'            )
parser.add_option(      '--cg_merge'         , action='store_true', dest='cg_merge'      , help='run cg_merge'      )
parser.add_option(      '--pcg'              , action='store_true', dest='pcg'           , help='run pcg'           )
parser.add_option(      '--bicgstab'         , action='store_true', dest='bicgstab'      , help='run bicgstab'      )
parser.add_option(      '--bicgstab_merge'   , action='store_true', dest='bicgstab_merge', help='run bicgstab_merge')
parser.add_option(      '--pbicgstab'        , action='store_true', dest='pbicgstab'     , help='run pbicgstab'     )
parser.add_option(      '--gmres'            , action='store_true', dest='gmres'         , help='run gmres'         )
parser.add_option(      '--pgmres'           , action='store_true', dest='pgmres'        , help='run pgmres'        )
parser.add_option(      '--lobpcg'           , action='store_true', dest='lobpcg'        , help='run lobpcg'        )
parser.add_option(      '--iterref'          , action='store_true', dest='iterref'       , help='run iterref'       )
parser.add_option(      '--jacobi'           , action='store_true', dest='jacobi'        , help='run jacobi'        )
parser.add_option(      '--ba'               , action='store_true', dest='ba'            , help='run ba-iter'       )
parser.add_option(      '--idr'              , action='store_true', dest='idr'           , help='run idr'           )
parser.add_option(      '--idr_merge'        , action='store_true', dest='idr_merge'     , help='run idr_merge'     )
parser.add_option(      '--pidr'             , action='store_true', dest='pidr'          , help='run pidr'          )
parser.add_option(      '--cgs'              , action='store_true', dest='cgs'           , help='run cgs'           )
parser.add_option(      '--cgs_merge'        , action='store_true', dest='cgs_merge'     , help='run cgs_merge'     )
parser.add_option(      '--qmr'              , action='store_true', dest='qmr'           , help='run qmr'           )
parser.add_option(      '--qmr_merge'        , action='store_true', dest='qmr_merge'     , help='run qmr_merge'     )
parser.add_option(      '--pcgs'             , action='store_true', dest='pcgs'          , help='run pcgs'          )
parser.add_option(      '--pcgs_merge'       , action='store_true', dest='pcgs_merge'    , help='run pcgs_merge'    )
parser.add_option(      '--pqmr'             , action='store_true', dest='pqmr'          , help='run pqmr'          )
parser.add_option(      '--pqmr_merge'       , action='store_true', dest='pqmr_merge'    , help='run pqmr_merge'    )   
parser.add_option(      '--bombard'          , action='store_true', dest='bombard'       , help='run bombard'       )
parser.add_option(      '--bombard_merge'    , action='store_true', dest='bombard_merge' , help='run bombard_merge' )
parser.add_option(      '--lsqr'             , action='store_true', dest='lsqr'          , help='run lsqr'          )
parser.add_option(      '--bicg'             , action='store_true', dest='bicg'          , help='run bicg'          )
parser.add_option(      '--pbicg'            , action='store_true', dest='pbicg'         , help='run pbicg'         )

                                                                                           
parser.add_option(      '--jacobi-prec'      , action='store_true', dest='jacobi_prec'   , help='run Jacobi preconditioner')
parser.add_option(      '--ilu-prec'         , action='store_true', dest='ilu_exact_prec', help='run ILU + exact solve preconditioner')
parser.add_option(      '--ilu-jac'          , action='store_true', dest='ilu_jac_prec',   help='run ILU + Jacobi solve preconditioner')
parser.add_option(      '--ilu-bjac'         , action='store_true', dest='ilu_bjac_prec',  help='run ILU + Block Jacobi solve preconditioner')
parser.add_option(      '--ilu-isai-prec'    , action='store_true', dest='ilu_isai_prec' , help='run ILU + ISAI preconditioner')
parser.add_option(      '--ilut-prec'        , action='store_true', dest='ilut_prec',      help='run threshold ILU + exact solve preconditioner')

(opts, args) = parser.parse_args()

# default if no sizes given is all sizes
if ( not opts.small and not opts.med and not opts.large and not opts.nonsym ):
    opts.small  = True
    opts.med    = False
    opts.large  = False
    opts.nonsym = True
# end

# default if no groups given is all groups
if (     not opts.sparse_blas
     and not opts.solver
     and not opts.control
     and not opts.csr
     and not opts.ell
     and not opts.sellp ):
    opts.sparse_blas = True
    opts.solver      = True
    opts.control     = True
    opts.csr         = True
    opts.ell         = True
    opts.sellp       = True
    opts.csr5        = True
# end

# default if no solvers given is all solvers
if (     not opts.cg
     and not opts.cg_merge
     and not opts.pcg
     and not opts.bicgstab
     and not opts.bicgstab_merge
     and not opts.pbicgstab
     and not opts.gmres
     and not opts.pgmres
     and not opts.lobpcg
     and not opts.iterref
     and not opts.jacobi
     and not opts.ba
     and not opts.idr
     and not opts.ba
     and not opts.idr_merge
     and not opts.qmr
     and not opts.qmr_merge
     and not opts.cgs
     and not opts.cgs_merge
     and not opts.pqmr
     and not opts.pqmr_merge
     and not opts.pcgs
     and not opts.pcgs_merge
     and not opts.bombard
     and not opts.bombard_merge
     and not opts.bicg
     and not opts.pbicg
     and not opts.lsqr
     and not opts.pidr ):
    opts.cg             = True
    opts.cg_merge       = True
    opts.pcg            = True
    opts.bicgstab       = True
    opts.bicgstab_merge = True
    opts.pbicgstab      = True
    opts.gmres          = True
    opts.pgmres         = True
    opts.lobpcg         = True
    opts.iterref        = True
    opts.jacobi         = True
    opts.ba             = True
    opts.idr            = True
    opts.idr_merge      = True
    opts.pidr           = True
    opts.cgs            = True
    opts.cgs_merge      = True
    opts.qmr            = True
    opts.qmr_merge      = True
    opts.pcgs           = True
    opts.pcgs_merge     = True
    opts.pqmr           = True
    opts.pqmr_merge     = True
    opts.bombard        = True
    opts.bombard_merge  = True
    opts.bicg           = True
    opts.pbicg          = True
    opts.lsqr           = True
# end

# default if no preconditioners given all
if (     not opts.jacobi_prec
     and not opts.ilu_exact_prec 
     and not opts.ilut_prec
     and not opts.ilu_jac_prec
     and not opts.ilu_bjac_prec
     and not opts.ilu_isai_prec ):
    opts.jacobi_prec      = True
    opts.ilu_prec         = True
    opts.ilu_jac_prec     = True
    opts.ilu_isai_prec    = True
    opts.ilu_bjac_prec    = True
    opts.ilut_prec        = True
# end

# default if no sizes given is all sizes
if ( not opts.small and not opts.med and not opts.large and not opts.nonsym ):
    opts.small = True
    opts.med   = True
    opts.large = True
    opts.nonsym = True
# end

print 'opts', opts
print 'args', args


# ----------------------------------------------------------------------
    
# looping over formats
formats = []
if ( opts.csr ):
    formats += ['--format CSR']
# end
if ( opts.ell ):
    formats += ['--format ELL']
# end
if ( opts.sellp ):
    formats += ['--format SELLP']
# end
if ( opts.csr5 ):
    formats += ['--format CSR5']
# end

# looping over solvers
solvers = []
if ( opts.cg ):
    solvers += ['--solver CG --basic']
# end
if ( opts.cg_merge ):
    solvers += ['--solver CG']
# end
if ( opts.bicgstab ):
    solvers += ['--solver BICGSTAB --basic']
# end
if ( opts.bicgstab_merge ):
    solvers += ['--solver BICGSTAB']
# end
if ( opts.gmres ):
    solvers += ['--solver GMRES']
# end
if ( opts.lobpcg ):
    solvers += ['--solver LOBPCG']
# end
if ( opts.jacobi ):
    solvers += ['--solver JACOBI']
# end
if ( opts.ba ):
    solvers += ['--solver BA']
# end
if ( opts.idr ):
    solvers += ['--solver IDR --basic']
# end
if ( opts.idr_merge ):
    solvers += ['--solver IDR']
# end
if ( opts.cgs ):
    solvers += ['--solver CGS --basic']
# end
if ( opts.cgs_merge ):
    solvers += ['--solver CGS --basic']
# end
if ( opts.qmr ):
    solvers += ['--solver QMR --basic']
# end
if ( opts.qmr_merge ):
    solvers += ['--solver QMR']
# end
if ( opts.lsqr ):
    solvers += ['--solver LSQR']
# end
if ( opts.bicg ):
    solvers += ['--solver BICG']
# end
if ( opts.bombard ):
    solvers += ['--solver BOMBARDMENT']
# end
if ( opts.bombard_merge ):
    solvers += ['--solver BOMBARDMENT --basic']
# end


# looping over precsolvers
precsolvers = []
if ( opts.pcg ):
    precsolvers += ['--solver PCG ']
# end
if ( opts.pbicgstab ):
    precsolvers += ['--solver PBICGSTAB ']
# end
if ( opts.pgmres ):
    precsolvers += ['--solver PGMRES ']
# end
if ( opts.pidr ):
    precsolvers += ['--solver PIDR ']
# end
if ( opts.pqmr ):
    precsolvers += ['--solver PQMR ']
# end
if ( opts.pidr ):
    precsolvers += ['--solver PCGS ']
# end
if ( opts.pbicg ):
    precsolvers += ['--solver PBICG ']
# end
if ( opts.lsqr ):
    precsolvers += ['--solver PLSQR ']
# end



# looping over eigensolvers
IR = []
if ( opts.iterref ):
    IR += ['--solver LOBPCG ']
# end



# looping over preconditioners
precs = ['--precond NONE ']
if ( opts.jacobi_prec ):
    precs += ['--precond JACOBI ']
# end
if ( opts.ilu_exact_prec ):
    precs += ['--precond ILU ']
# end
if ( opts.ilut_prec ):
    precs += ['--precond PARILUT --prestart 1 --psweeps 5 --plevels 0 --prtol 0.05 --patol 0.2 ']
# end
if ( opts.ilu_jac_prec ):
    precs += ['--precond ILU --trisolver JACOBI --ppattern 1 --piters 5 ']
# end
if ( opts.ilu_bjac_prec ):
    precs += ['--precond ILU --trisolver JACOBI --ppattern 4 --piters 3 ']
# end
if ( opts.ilu_isai_prec ):
    precs += ['--precond ILU --trisolver ISAI --ppattern 1 --piters 1 ']
# end


# looping over preconditioners for Iter-Ref
IRprecs = []
if ( opts.iterref ):
    IRprecs += ['--precond CG ']
    IRprecs += ['--precond CGS ']
    IRprecs += ['--precond BICGSTAB ']
    IRprecs += ['--precond BICG ']
    IRprecs += ['--precond GMRES ']
    IRprecs += ['--precond QMR ']
# end



# ----------------------------------------------------------------------
# blocksizes
# ----------
blocksizes = []
blocksizes += ['--blocksize 4']
blocksizes += ['--blocksize 8']
blocksizes += ['--blocksize 16']
#end


# ----------------------------------------------------------------------
# alignments
# ----------
alignments = []
alignments += ['--alignment 4']
alignments += ['--alignment 8']
alignments += ['--alignment 16']
alignments += ['--alignment 32']


# ----------------------------------------------------------------------
# problem sizes
# n    is squared
# ----------
sizes = []
if opts.small:
    sizes += ['LAPLACE2D 47']
if opts.med:
    sizes += ['LAPLACE2D 95']
if opts.large:
    sizes += ['LAPLACE2D 317']
if opts.nonsym:
    sizes += ['test_matrices/ani5_crop.mtx']
#end


# ----------------------------------------------------------------------
precisions = (
    's', 'd', 'c', 'z'
)

subs = (
    ('',              'dlag2s',      '',              'zlag2c'    ),
    ('ssy',           'dsy',         'che',           'zhe'       ),
    ('sor',           'dor',         'cun',           'zun'       ),
    ('sy2sb',         'sy2sb',       'he2hb',         'he2hb'     ),
    ('',              'testing_ds',  '',              'testing_zc'),
    ('testing_s',     'testing_d',   'testing_c',     'testing_z' ),
    ('lansy',         'lansy',       'lanhe',         'lanhe'     ),
    ('blas_s',        'blas_d',      'blas_c',        'blas_z'    ),
)

# ----------
# simple precision generation
def substitute( txt, pfrom, pto ):
    if ( pfrom != pto ):
        ifrom = precisions.index( pfrom )
        ito   = precisions.index( pto )
        for sub in subs:
            txt = re.sub( sub[ifrom], sub[ito], txt )
    # end
    return txt
# end


# ----------------------------------------------------------------------
# test problems
#
# Each test has 4 fields:  (program, options, sizes, comments).
# If program begins with #, that test is disabled and will be printed as such.
# In some cases, the line is commented out, so nothing is printed.
#
# These match the order in the Makefile,
# except in some cases the "d" version from the Makefile isn't required here
# To compare with Makefile:
# grep '^\s+testing_\w+' Makefile >! a
# grep "\('#?testing_\w\w+" run_tests.py | perl -pe "s/#?\('#?//; s/',.*/.cpp \\/;" | uniq > ! b
# diff -w a b


tests = []


# ----------------------------------------------------------------------
if ( opts.control):
    for precision in opts.precisions:
        for size in sizes:
            # precision generation
            cmd = substitute( 'testing_zio', 'z', precision )
            tests.append( [cmd, '', size, ''] )

# ----------------------------------------------------------------------
if ( opts.control):
    for precision in opts.precisions:
        for size in sizes:
            # precision generation
            cmd = substitute( 'testing_zmatrix', 'z', precision )
            tests.append( [cmd, '', size, ''] )

# ----------------------------------------------------------------------
if ( opts.control):
    for precision in opts.precisions:
        for size in sizes:
            # precision generation
            cmd = substitute( 'testing_zmconverter', 'z', precision )
            tests.append( [cmd, '', size, ''] )

# ----------------------------------------------------------------------
if ( opts.control):
    for precision in opts.precisions:
        for size in sizes:
            # precision generation
            cmd = substitute( 'testing_zmcompressor', 'z', precision )
            tests.append( [cmd, '', size, ''] )

# ----------------------------------------------------------------------
if ( opts.control):
    for precision in opts.precisions:
        for size in sizes:
            # precision generation
            cmd = substitute( 'testing_zmadd', 'z', precision )
            tests.append( [cmd, '', size + ' ' + size, ''] )


# ----------------------------------------------------------------------
if ( opts.sparse_blas):
    for precision in opts.precisions:
        for size in sizes:
            for blocksize in blocksizes:
                for alignment in alignments:
                    # precision generation
                    cmd = substitute( 'testing_zspmv', 'z', precision )
                    tests.append( [cmd, alignment + ' ' + blocksize, size, ''] )


# ----------------------------------------------------------------------
if ( opts.sparse_blas):
    for precision in opts.precisions:
        for size in sizes:
            for blocksize in blocksizes:
                for alignment in alignments:
                    # precision generation
                    cmd = substitute( 'testing_zspmm', 'z', precision )
                    tests.append( [cmd, alignment + ' ' + blocksize, size, ''] )


# ----------------------------------------------------------------------
for solver in solvers:
    for size in sizes:
        for precision in opts.precisions:
            # precision generation
            cmd = substitute( 'testing_zsolver', 'z', precision )
            tests.append( [cmd, solver, size, ''] )


# ----------------------------------------------------------------------
for solver in precsolvers:
    for precond in precs:
        for size in sizes:
            for precision in opts.precisions:
                # precision generation
                cmd = substitute( 'testing_zsolver', 'z', precision )
                tests.append( [cmd, solver + ' ' + precond, size, ''] )


# ----------------------------------------------------------------------
for solver in IR:
    for precond in IRprecs:
        for size in sizes:
            for precision in opts.precisions:
                # precision generation
                cmd = substitute( 'testing_zsolver', 'z', precision )
                tests.append( [cmd, solver + ' ' + precond, size, ''] )








# ----------------------------------------------------------------------
#print 'tests'
#for t in tests:
#    print t


# ----------------------------------------------------------------------
# runs command in a subprocess.
# returns list (okay, fail, errors, status)
# okay   is count of "ok"     in output.
# fail   is count of "failed" in output.
# error  is count of indications of other errors (exit, CUDA error, etc.).
# status is exit status of the command.
def run( cmd ):
    words = re.split( ' +', cmd.strip() )
    
    # stdout & stderr are merged
    p = subprocess.Popen( words, bufsize=1, stdout=PIPE, stderr=STDOUT )
    
    okay  = 0
    fail  = 0
    nospd = 0
    error = 0
    nosupport = 0
    slowconv = 0
    # read unbuffered ("for line in p.stdout" will buffer)
    while True:
        line = p.stdout.readline()
        if not line:
            break
        print line.rstrip()
        if re.search( r'\bok *$', line ):
            okay += 1
        if re.search( 'failed', line ):
            fail += 1
        if re.search( 'exit|memory mapping error|CUDA runtime error|illegal value|ERROR SUMMARY: [1-9]', line ):
            error += 1
        m = re.search( 'solver info: (-?\d+)', line )
        if ( m ):
            info = int( m.group(1) )
            if ( info == 0 ):
                okay += 1
            if ( info == -103 ):
                nosupport += 1
            if ( info == -201 ):
                slowconv += 1
            if ( info == -202 ):
                fail += 1
            if ( info == -203 ):
                fail += 1
        # end
    # end
    
    status = p.wait()
    return (okay, fail, error, status)
# end


# ----------------------------------------------------------------------
ntest  = 0
nokay  = 0
nfail  = 0
nerror = 0
failures = {}

start = None
if ( opts.start ):
    start = re.compile( opts.start + r'\b' )

seen  = {}
pause = 0

last_cmd = None

for test in tests:
    (cmd, options, sizes, comments) = test
    
    make = False
    disabled = (cmd[0] == '#')
    if ( disabled ):
        cmd = cmd[1:]
    
    # command to run
    cmd_args = './' + cmd +' '+ options +' '+ sizes
    cmd_args = re.sub( '  +', ' ', cmd_args )  # compress spaces
    
    # skip tests before start
    if ( start and not start.search( cmd_args )):
        continue
    start = None
    
    # skip tests not in args, or duplicates, or non-existing
    #if not os.path.exists( cmd ):
    #    print >>sys.stderr, cmd, cmd, "doesn't exist"
    if (    (args and not cmd in args)
         or (not os.path.exists( cmd ))
         or (seen.has_key( cmd_args )) ):
        #print "skipping", cmd_args
        continue
    # end
    seen[ cmd_args ] = True
    
    if ( opts.memcheck ):
        cmd_args = 'cuda-memcheck ' + cmd_args
    
    go = True
    while( go ):
        if pause > 0:
            time.sleep( pause )
            pause = 0
        # end
        
        print
        print '*'*100
        print cmd_args
        print '*'*100
        sys.stdout.flush()
        
        if ( batch ):
            if ( last_cmd and cmd != last_cmd ):
                sys.stderr.write( '\n' )
            last_cmd = cmd
            sys.stderr.write( '%-70s' % cmd_args )
            sys.stderr.flush()
        # end
        
        if ( disabled ):
            if ( comments ):
                sys.stderr.write( '  (disabled: ' + comments + ')\n' )
            else:
                sys.stderr.write( '  (disabled)\n' )
            go = False
            continue
        # end
        
        if ( make ):
            m = 'make lib ' + cmd
            print m
            run( m )
        # end
        
        t = time.time()
        (okay, fail, error, status) = run( cmd_args )
        t = time.time() - t
        
        # count stats
        ntest  += 1
        nokay  += okay
        nfail  += fail
        nerror += error
        
        errmsg = ''
        if ( fail > 0 ):
            errmsg += '  ** %d tests failed' % (fail)
        if ( error > 0 ):
            errmsg += '  ** %d errors' % (error)
        if ( status < 0 ):
            errmsg += '  ** exit with signal %d' % (-status)
            nerror += 1  # count crash as an error
        
        if ( errmsg != '' ):
            if ( batch ):
                sys.stderr.write( errmsg + '\n' )  # to console
            sys.stdout.write( errmsg + '\n' )  # to file
            failures[ cmd_args ] = True
        else:
            sys.stderr.write( '  ok\n' )
        # end
        
        if ( batch ):
            # set to sleep a few seconds before next test,
            # to allow processor to cool off some between tests.
            pause = min( t, 5. )
            go = False
        else:
            x = raw_input( '[enter to continue; M to make and re-run] ' )
            if ( x in ('m','M')):
                make = True
            else:
                go = False
        # endif
    # end
# end


# print summary
msg  = '\n'
msg += '*'*100   + '\n'
msg += 'summary' + '\n'
msg += '*'*100   + '\n'

if ( nfail == 0 and nerror == 0 ):
    msg += 'all %d tests in %d commands passed!\n' % (nokay, ntest)
else:
    msg += '%5d tests in %d commands passed\n' % (nokay, ntest)
    msg += '%5d tests failed accuracy test\n' % (nfail)
    msg += '%5d errors detected (crashes, CUDA errors, etc.)\n' % (nerror)
    f = failures.keys()
    f.sort()
    msg += 'routines with failures:\n    ' + '\n    '.join( f ) + '\n'
# end

if ( batch ):
    sys.stderr.write( msg )  # to console
sys.stdout.write( msg )  # to file
