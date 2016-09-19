/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
*/

#ifndef MAGMA_H
#define MAGMA_H

#ifdef MAGMA_NO_V1
#error "Since MAGMA_NO_V1 is defined, magma.h is invalid; use magma_v2.h"
#endif

// magma v1 includes cublas.h by default, unless cublas_v2.h has already been included
#ifndef CUBLAS_V2_H_
#include <cublas.h>
#endif

// Include the MAGMA v2 and v1 APIs,
// then map names to the v1 API (e.g., magma_zgemm => magma_zgemm_v1).
// Some functions (like setmatrix_async) are the same in v1 and v2,
// so are provided by the v2 API.
#include "magma_v2.h"
#include "magmablas_v1.h"
#include "magmablas_v1_map.h"

#undef  MAGMA_API
#define MAGMA_API 1

#endif // MAGMA_H
