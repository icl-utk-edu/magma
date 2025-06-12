/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
*/

#ifndef MAGMA_H
#define MAGMA_H

// =============================================================================
// MAGMA configuration
#include "magma_config.h"


// magma v1 includes cublas.h by default, unless cublas_v2.h has already been included
#ifndef CUBLAS_V2_H_
#if defined(MAGMA_HAVE_CUDA)
#include <cublas.h>
#endif
#endif

#include "magma_v2.h"

#undef  MAGMA_API
#define MAGMA_API 1

#endif // MAGMA_H
