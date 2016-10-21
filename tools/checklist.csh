#!/bin/tcsh
#
# Looks for things that need to be fixed before a release, e.g.,
# using {malloc,free} instead of magma_{malloc,free}_cpu
# Run at the top level:
#
#     ./tools/checklist.py > checklist.txt
#
# @author Mark Gates

hg st -mac | perl -pe 's/^\w //' | sort \
    | egrep -v '^\.$$|Makefile\.|\.png|results/v|checklist|^scripts' \
    >! files.txt

setenv FILES        `egrep -v checklist files.txt`
setenv HEADERS      `egrep '\.h$' files.txt`
setenv MAKEFILES    `egrep Makefile files.txt | egrep -v sparse`
setenv MAKEFILES_SP `egrep Makefile files.txt | egrep    sparse`

echo "============================================================ required fixes"

# fixed
echo "========== don't use tabs                                                  *** required fix ***"
grep -P "\t" `grep -v "ReleaseChecklist|Make|Doxy|\.py|\.pl|\.sh|\.F90" files.txt` -l
echo

# fixed
echo "========== don't use MIN_CUDA_ARCH; use __CUDA_ARCH__ and magma_getdevice_arch  *** required fix ***"
egrep MIN_CUDA_ARCH $FILES | egrep -v 'CMakeLists.txt|Makefile|interface.cpp'
echo

# fixed
echo "========== work[0] query uses magma_[sdcz]make_lwork                       *** required fix ***"
egrep 'work\[[0-9]\] *=[^=]' $FILES \
    | egrep -v '\b(magma_[sdcz]make_lwork|c_one|MAGMA_Z_ONE|iwork|ztrevc3|ztrevc3_mt|scripts|tiled|testing)\b'
echo

# fixed
# both foo.cu and foo.cpp -> foo.o; can't add two foo.o to same libbar.a file
# check dense and sparse separately, as they go into different libmagma.a and
# libmagma_sparse.a files, and both have error.o
echo "========== no duplicate object filenames                                   *** required fix ***"
egrep -h '^\s+\w+\.(cpp|cu|f|f90|F90)' $MAKEFILES \
    | perl -pe 's/\.(cpp|cu|f|f90|F90)/.o/; s/[ \t]+/ /g;' | sort | uniq -c | egrep -v '^ +1 '
egrep -h '^\s+\w+\.(cpp|cu|f|f90|F90)' $MAKEFILES_SP \
    | perl -pe 's/\.(cpp|cu|f|f90|F90)/.o/; s/[ \t]+/ /g;' | sort | uniq -c | egrep -v '^ +1 '
echo

# fixed
# use 'svn propdel svn:executable [files]' to remove
echo "========== no execute bit on source files                                  *** required fix ***"
find $FILES -perm +a+x \! -name '*.pl' \! -name '*.csh' \! -name '*.py' \! -name '*.sh'
echo

# fixed
echo "========== no prototypes in cpp files; put in headers                      *** required fix ***"
perl -n0777e 'if ( m/extern "C"\s+\w+\s+(\w+)\([^)]*\) *;/ ) { print "$ARGV: $1\n"; }' $FILES
echo

# fixed
echo "========== headers not protecting against multiple inclusion               *** required fix ***"
egrep '^#ifndef \w+_([hH]_?|HPP)\b' -L $HEADERS | egrep -v 'gemm_stencil_defs\.h'
echo
egrep '^#define \w+_([hH]_?|HPP)\b' -L $HEADERS | egrep -v 'gemm_stencil_defs\.h'
echo

# fixed
echo "========== C malloc, instead of magma_*malloc_cpu                          *** required fix ***"
egrep '\b(malloc|calloc|realloc|reallocf|valloc|strdup) *\(' $FILES | egrep -v 'quark|alloc.cpp|magma_util.cpp'
echo

# fixed
echo "========== C free, instead of magma_free_cpu                               *** required fix ****"
egrep '^ *free *\('                    $FILES | egrep -v 'quark|alloc.cpp|magma_util.cpp'
echo

# fixed except trevc3; needs rewrite to fix
# zheevr has "A new O(n^2)..."
echo "========== C++ new, instead of magma_*malloc_cpu                           *** required fix (currently only thread_queue & trevc3_mt use it) ***"
egrep '= *new\b'                       $FILES | egrep -v 'thread_queue|trevc3_mt|A new O\(n\^2\)'
egrep '\bnew +\w+\('                   $FILES | egrep -v 'thread_queue|trevc3_mt|A new O\(n\^2\)'
echo

# fixed except trevc3; needs rewrite to fix
echo "========== C++ delete,  instead of magma_free_cpu                          *** required fix (currently only thread_queue uses it) ***"
egrep '^ *delete\b'                    $FILES | egrep -v 'thread_queue'
echo

# fixed
echo "========== C++ streams                                                     *** required fix ***"
egrep '\b(fstream|iostream|sstream|cin|cout)\b' $FILES
echo

# fixed
echo "========== define PRECISION_{s, d, c, z} outside s, d, c, z files, resp.   *** required fix ***"
egrep '^ *# *define +PRECISION_s' */[dcz]*.{h,c,cu,cpp} sparse/*/[dcz]*.{h,c,cu,cpp} sparse/*/magma_[dcz]*.{h,c,cu,cpp} -l | egrep -v 'zm[sdc]dot|core_s'
egrep '^ *# *define +PRECISION_d' */[scz]*.{h,c,cu,cpp} sparse/*/[scz]*.{h,c,cu,cpp} sparse/*/magma_[scz]*.{h,c,cu,cpp} -l | egrep -v 'zm[sdc]dot|core_d'
egrep '^ *# *define +PRECISION_c' */[sdz]*.{h,c,cu,cpp} sparse/*/[sdz]*.{h,c,cu,cpp} sparse/*/magma_[sdz]*.{h,c,cu,cpp} -l | egrep -v 'zm[sdc]dot|core_c|scnrm2.cu'
egrep '^ *# *define +PRECISION_z' */[sdc]*.{h,c,cu,cpp} sparse/*/[sdc]*.{h,c,cu,cpp} sparse/*/magma_[sdc]*.{h,c,cu,cpp} -l | egrep -v 'zm[sdc]dot|core_z|dznrm2.cu'
echo

# fixed
echo "========== define {REAL,COMPLEX} outside {[sd],[cz]} files, resp.          *** required fix ***"
egrep '^ *# *define +REAL'    */[cz]*.{h,c,cu,cpp} sparse/*/[cz]*.{h,c,cu,cpp} sparse/*/magma_[cz]*.{h,c,cu,cpp} -l | egrep -v 'cblas_[sd].cpp'
egrep '^ *# *define +COMPLEX' */[sd]*.{h,c,cu,cpp} sparse/*/[sd]*.{h,c,cu,cpp} sparse/*/magma_[sd]*.{h,c,cu,cpp} -l
echo

# fixed
echo "========== using PRECISION_z, REAL, COMPLEX without define                 *** required fix ***"
egrep 'defined\( *PRECISION_\w *\)'    $FILES -l | xargs egrep '^ *#define PRECISION_\w'   -L | egrep -v 'gemm_template_device_defs.cuh'
egrep 'ifdef +PRECISION_\w'            $FILES -l | xargs egrep '^ *#define PRECISION_\w'   -L | egrep -v 'gemm_param_[nt][nt].h'
egrep 'defined\( *(REAL|COMPLEX) *\)'  $FILES -l | xargs egrep '^ *#define (REAL|COMPLEX)' -L
egrep 'ifdef +(REAL|COMPLEX)'          $FILES -l | xargs egrep '^ *#define (REAL|COMPLEX)' -L | egrep -v 'gemm_stencil_defs.h'
echo

# fixed
echo "========== define PRECISION_z, REAL, COMPLEX without undef in headers      *** required fix ***"
egrep '^ *#define +PRECISION_\w'       $HEADERS  -l | xargs egrep '^ *#undef PRECISION_\w' -L
egrep '^ *#define +REAL'               $HEADERS  -l | xargs egrep '^ *#undef REAL'         -L
egrep '^ *#define +COMPLEX'            $HEADERS  -l | xargs egrep '^ *#undef COMPLEX'      -L
echo

# fixed
echo "========== using defunct GPUSHMEM                                          *** required fix ***"
egrep 'GPUSHMEM'                       $FILES
echo

# fixed
echo "========== CUDA driver routines (cu[A-Z]*; MathWorks cannot use these)     *** required fix ***"
egrep 'cu[A-Z]\w+ *\(' $FILES \
    | egrep -v 'make_cuComplex|make_cuDoubleComplex|make_cuFloatComplex|cuComplexDoubleToFloat|cuComplexFloatToDouble|cuGetErrorString|cuConj|cuBLAS|cuC(real|imag|add|sub|mul|div|addf|subf|mulf|divf|fmaf|abs|fma)'
echo

# fixed
echo "========== cuda.h (mostly only needed for CUDA driver routines, cu[A-Z]*)  *** required fix ***"
egrep 'cuda\.h'                        $FILES | egrep -v 'interface_cuda/error.h|fortran2.cpp|for CUDA_VERSION'
echo

# fixed except 2-stage
echo "========== exit, instead of returning error code                           *** required fix ***"
egrep '\bexit *\( *-?\w+ *\)'          $FILES | egrep -v 'trace.cpp|quark|magma_util.cpp|testing|\.py'
echo

# fixed
echo "========== old formulas for ceildiv & roundup                              *** required fix ***"
./tools/checklist_ceildiv.pl           $FILES | egrep -v 'documentation.txt|magma\.h|\.f:|\.F90:'
echo

# fixed
echo "========== sprintf; use snprintf for safety                                *** required fix ***"
egrep sprintf $FILES | egrep -v '\.pl|quark|strlcpy'
echo

# fixed
echo "========== sync_wtime of NULL stream (excluding _mgpu codes)               *** required fix ***"
egrep sync_wtime $FILES | egrep -v 'sync_wtime\( *(opts.queue|queue) *\)'
echo

# fixed
echo "========== passing ints by pointer                                         *** required fix ***"
egrep "int_t *\* *(m|n|k|ld\w|ldd\w|inc\w|l\w*work)\b" include/{magma_z,magma_zc,magmablas_z*}.h
echo

# fixed
echo "========== use standard c_one and c_neg_one names                          *** required fix ***"
egrep "\b(mz_one|z_one|zone|mzone|c_mone)\b" $FILES
egrep "done *= *-?[0-9]" $FILES
echo

# needs lots of work; also lots of exceptions
echo "========== int instead of magma_int_t (in headers)                         *** required fix ***"
egrep '\bint\b' $HEADERS \
    | egrep -v 'int argc|int flock|quark|magma_timer\.h|commonblas_z\.h|magma_templates\.h|magma_winthread\.h|pthread_barrier\.h|cblas\.h|typedef int +magma_|const char\* func, const char\* file, int line|int mm_'
echo

echo
echo
echo "============================================================ should be fixed"

# fixed
echo "========== use @date instead of a specific hard-coded date"
egrep '(January|February|March|April|May|June|July|August|September|October|November|December) +[0-9]{4} *$' $FILES \
    | egrep -v 'testing/(checkdiag|lin|matgen)|blas_fix'
echo

echo "========== cuda, cublas functions                                          *** should be fixed ***"
egrep '(cuda|cublas)[a-zA-Z]\w+ *\('   $FILES \
    | egrep -v '\.f:|interface_cuda|testing_blas_z.cpp|cudaProfilerStart|cudaProfilerStop|cudaFuncSetCacheConfig|cudaCreateChannelDesc|cudaCreateTextureObject|cudaDeviceSetSharedMemConfig|cudaDestroyTextureObject|cublasSetAtomicsMode|cudaGetErrorString|cudaMemcpyToSymbol|cudaMemcpyFromSymbol|cudaDeviceSetCacheConfig|cudaBindTexture|cudaUnbindTexture|cudaEventElapsedTime|cudaEventCreateWithFlags|cudaMemGetInfo|cublas\w+Batched|cublasHandle|cublasCreate|cublasDestroy|fortran2.cpp' \
    | egrep -v 'testing_z([a-z]+)(_\w+)?\.cpp:.*cublasZ\1'
echo

echo "========== system includes                                                 *** should be fixed (with a number of exceptions) ***"
egrep '^ *# *include *<' $FILES | egrep -v 'testing/|\.h: *#include|(plasma|core_blas|cublas_v2|omp)\.h' | egrep -v 'quark'
echo

# fixed
echo "========== device sync, instead of queue sync                              *** should be fixed ***"
egrep '(magma_device_sync|cudaDeviceSynchronize)' $FILES | egrep -v 'interface.cpp'
echo

# in src: dsyevdx_2stage, zhetrf_aasen, ztsqrt, ztstrf need fixing
# lots of exceptions elsewhere
#echo "========== routine name in docs doesn't match filename                     *** should be fixed ***"
#perl -n0777 -e 'while( m|Purpose\s+-+\s+(\w+)|g ) { $a = $1; $a =~ tr/A-Z/a-z/; if ( not $ARGV =~ m/$a/ ) { print "$ARGV\n"; } }' $FILES
#echo


echo "============================================================ informational"

echo "========== define PRECISION_z, REAL, COMPLEX without use                   (NOT errors, just curious)"
egrep '#define +(PRECISION_\w|REAL|COMPLEX)' $FILES -l \
    | xargs egrep 'defined\(PRECISION_\w|REAL|COMPLEX\)|ifdef (PRECISION_\w|REAL|COMPLEX)' -L
echo



echo
echo
echo "============================================================ stylistic"

# mostly fixed; some exceptions
echo "========== extraneous newlines after {"
perl -n0777e 'if ( m/^(.*\{) *\n *\n/m ) { print "$ARGV: $1\n"; }' $FILES
echo

# mostly fixed; some exceptions
echo "========== extraneous newlines before }"
perl -n0777e 'if ( m/\n *\n( *\}.*)/ ) { print "$ARGV: $1\n"; }' $FILES
echo

# mostly fixed; some exceptions
echo "========== large block of newlines (recommend 2 between functions)"
perl -n0777 -e 'print "$ARGV\n" if (m/(\n *){4,}\n/);' $FILES
echo

# fixed
echo "========== lacking newline at EOF"
perl -n0777e 'if ( ! m/\n$/ ) { print "$ARGV\n"; }' $FILES
echo

# fixed
echo "========== extraneous newlines (more than one) at EOF"
perl -n0777e 'if ( m/\n\n+$/     ) { print "$ARGV\n"; }' $FILES
echo

# fixed
echo "========== single-line for"
egrep '^ *for *\(.*?;.*?;.*?\).*;'    $FILES
echo

# fixed
echo "========== space before semicolons"
egrep '\S +;'                         $FILES | egrep -v 'quark'
echo

# fixed
# exclude "strings;" and scripts
echo "========== missing space after semicolons"
egrep ';\S'                           $FILES | egrep -v '".*;.*"|\.(csh|sh|py|pl|html)|quark|Makefile'
echo

# mostly fixed; some exceptions
echo "========== 4 space indent (hard to check; only checks some things like if, else, for)"
grep '^(    )* {1,3}(\{|\}|if *\(|else\b|for *\(|while *\(|do\b|switch *\(|break)' $FILES | grep -v testing
grep '^(    )* {1,3}(blasf77_|lapackf77_|magma_|magmablas_)\w+\(' $FILES
echo

echo "========== 80 character rule lines (TODO: sparse and F77)"
egrep '/\*\*|\*{5}' $FILES | egrep -v ':(/\*{75}//\*\*|/\*{78}/|\*{79}/| \*{79}| \*{78}/)$' | egrep -v 'sparse|f77'
echo
