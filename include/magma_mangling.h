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
#ifndef MAGMA_FORTRAN_NAME
    #if defined(MAGMA_GLOBAL)
        #define FORTRAN_NAME(lcname, UCNAME)  MAGMA_GLOBAL( lcname, UCNAME )
    #elif defined(ADD_)
        #define FORTRAN_NAME(lcname, UCNAME)  lcname##_
    #elif defined(NOCHANGE)
        #define FORTRAN_NAME(lcname, UCNAME)  lcname
    #elif defined(UPCASE)
        #define FORTRAN_NAME(lcname, UCNAME)  UCNAME
    #else
        #error "One of ADD_, NOCHANGE, or UPCASE must be defined to set how Fortran functions are name mangled. For example, in MAGMA, add -DADD_ to CFLAGS, FFLAGS, etc. in make.inc. If using CMake, it defines MAGMA_GLOBAL instead."
        #define FORTRAN_NAME(lcname, UCNAME)  lcname##_error
    #endif
#endif

#endif  // MAGMA_MANGLING_H
