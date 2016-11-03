Getting started with MAGMA.

This is a simple, standalone example to show how to use MAGMA, once it is
compiled. More involved examples for individual routines are in the testing
directory. The testing code includes some extra utilities that we use for
testing, such as testings.h and libtest.a, which are not required to use MAGMA,
though you may use them if desired.

----------------------------------------
C example

See example_v2.c for sample code.

Include the MAGMA header:

    #include "magma_v2.h"

(For the legacy MAGMA v1 interface, see example_v1.c. It includes magma.h
instead. By default, magma.h includes the legacy cuBLAS v1 interface (cublas.h).
You can include cublas_v2.h before magma.h if desired.)

You may also need BLAS and LAPACK functions, which you can get with:

    #include "magma_lapack.h"

You can also use headers that came with your BLAS and LAPACK library, such as
Intel MKL. However, their definitions, while compatible, may not exactly match
ours. Especially their definition of the COMPLEX type will be different. We use
magmaDoubleComplex, which is a typedef of cuDoubleComplex. You may need to cast
back-and-forth between definitions. We have also added const where appropriate.

In C++, you may want complex-number operators and overloaded functions
(* / + -, conj, real, imag, etc.), which you can get with:

    #include "magma_operators.h"

When MAGMA was compiled, one of ADD_, NOCHANGE, or UPCASE was defined in
make.inc for how Fortran functions are name-mangled on your system. The most
common is ADD_, where Fortran adds an underscore after the function. Usually,
you can tell the convention by using nm to examine your BLAS library:

    nm /mnt/scratch/openblas/lib/libopenblas.a | grep -i dnrm2
    dnrm2.o:
    0000000000000000 T dnrm2_
                     U dnrm2_k

Since dnrm2_ has an underscore, use ADD_. Then add the include paths. For
example, to compile your .c file:

    gcc -DADD_ \
        -I$MAGMADIR/include \
        -I$CUDADIR/include  \
        -c example.c

where $MAGMADIR and $CUDADIR are set to where MAGMA and CUDA are installed,
respectively.

To link, add the library paths and necessary libraries. The order matters:
-lmagma should be before -lcublas and your BLAS/LAPACK libraries.
For example with OpenBLAS:

    gcc -L$MAGMADIR/lib    -lmagma_sparse -lmagma \
        -L$CUDADIR/lib64   -lcublas -lcudart -lcusparse \
        -L$OPENBLASDIR/lib -lopenblas \
        -o example example.o

If you are not using MAGMA sparse routines, you can omit -lmagma_sparse.
However, MAGMA always requires -lcusparse.

If it cannot find the shared libraries:

    ./example_v1 
    ./example_v1: error while loading shared libraries: libmagma.so:
        cannot open shared object file: No such file or directory

you may need to add these paths to your LD_LIBRARY_PATH. For instance, with
sh/bash:

    LD_LIBRARY_PATH=$MAGMADIR/lib:$CUDADIR/lib64:$OPENBLASDIR/lib

or with csh/tcsh:

    setenv LD_LIBRARY_PATH $MAGMADIR/lib:$CUDADIR/lib64:$OPENBLASDIR/lib

----------------------------------------
Fortran example

See example_f.F90 for sample code.

MAGMA provides a Fortran interface, with routines prefixed by magmaf_ instead of
magma_. Most, but not all, MAGMA routines are provided in Fortran. It needs to
know what size pointers are; these are typically 64-bit (8 byte), so kind=8
below, unless you have compiled for an older 32-bit system.

NVIDIA provides a Fortran interface to the legacy cuBLAS v1 (without handles),
in the file $CUDADIR/src/fortran.c in your CUDA installation. We use this
interface for simplicity here. You will need to compile and link with that file.

Alternatively, you can define a Fortran interface to the newer cuBLAS v2 (with
handles), such as demonstrated on OLCF's tutorial:
https://www.olcf.ornl.gov/tutorials/concurrent-kernels-ii-batched-library-calls/#Wrappers

Here is compiling the example, compiling the cuBLAS fortran file, and linking:

    gfortran -I$MAGMADIR/include \
             -Dmagma_devptr_t="integer(kind=8)" \
             -c example_f.F90

    gfortran -DCUBLAS_GFORTRAN \
             -I$CUDADIR/include \
             -c -o fortran.o $CUDADIR/src/fortran.c

    gfortran -L$MAGMADIR/lib    -lmagma_sparse -lmagma \
             -L$CUDADIR/lib64   -lcublas -lcudart -lcusparse \
             -L$OPENBLASDIR/lib -lopenblas \
             -o example_f example_f.o fortran.o


----------------------------------------
Makefile example

The Makefile provided in this directory is a starting point for compiling. You
will need to adjust the MAGMA_CFLAGS and MAGMA_LIBS to reflect your system. See
the MAGMA make.inc file for which libraries are required for BLAS & LAPACK.

Alternatively, you can use pkg-config to get MAGMA's CFLAGS and LIBS
automatically. To use pkg-config, install MAGMA with 'make install', add MAGMA
to $PKG_CONFIG_PATH, e.g. with csh,

    setenv PKG_CONFIG_PATH ${PKG_CONFIG_PATH}:/usr/local/magma/lib/pkgconfig

then use 'pkg-config --cflags magma' and 'pkg-config --libs magma' as shown in
the Makefile.
