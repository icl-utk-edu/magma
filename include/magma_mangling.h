/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
*/

#ifndef MAGMA_MANGLING_H
#define MAGMA_MANGLING_H

#include "magma_mangling_cmake.h"

/* Define how to name mangle Fortran names.
 * If using CMake, it defines MAGMA_GLOBAL in magma_mangling_cmake.h
 * Otherwise, the make.inc file should have one of -DADD_, -DNOCHANGE, or -DUPCASE.
 * If using outside of MAGMA, put one of those in your compiler flags (e.g., CFLAGS).
 * These macros are used in:
 *   include/magma_*lapack.h
 *   control/magma_*f77.cpp
 */
/* Optional BLAS symbol prefix (e.g. -DMAGMA_BLAS_PREFIX=openblas_).
 * Two levels of indirection so the prefix macro is expanded before pasting. */
#define _MAGMA_PASTE(a, b)  a ## b
#define _MAGMA_EXPAND(a, b) _MAGMA_PASTE(a, b)

#ifndef MAGMA_FORTRAN_NAME
    #if defined(MAGMA_GLOBAL)
        #define FORTRAN_NAME(lcname, UCNAME)  MAGMA_GLOBAL( lcname, UCNAME )
    #elif defined(ADD_)
        #ifdef MAGMA_BLAS_PREFIX
            #define FORTRAN_NAME(lcname, UCNAME)  _MAGMA_EXPAND(MAGMA_BLAS_PREFIX, lcname##_)
        #else
            #define FORTRAN_NAME(lcname, UCNAME)  lcname##_
        #endif
    #elif defined(NOCHANGE)
        #ifdef MAGMA_BLAS_PREFIX
            #define FORTRAN_NAME(lcname, UCNAME)  _MAGMA_EXPAND(MAGMA_BLAS_PREFIX, lcname)
        #else
            #define FORTRAN_NAME(lcname, UCNAME)  lcname
        #endif
    #elif defined(UPCASE)
        #ifdef MAGMA_BLAS_PREFIX
            #define FORTRAN_NAME(lcname, UCNAME)  _MAGMA_EXPAND(MAGMA_BLAS_PREFIX, UCNAME)
        #else
            #define FORTRAN_NAME(lcname, UCNAME)  UCNAME
        #endif
    #else
        #error "One of ADD_, NOCHANGE, or UPCASE must be defined to set how Fortran functions are name mangled. For example, in MAGMA, add -DADD_ to CFLAGS, FFLAGS, etc. in make.inc. If using CMake, it defines MAGMA_GLOBAL instead."
        #define FORTRAN_NAME(lcname, UCNAME)  lcname##_error
    #endif
#endif

#endif  // MAGMA_MANGLING_H
