/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
*/

#ifndef MAGMABLAS_V1_H
#define MAGMABLAS_V1_H

#ifdef MAGMA_NO_V1
#error "Since MAGMA_NO_V1 is defined, magma.h is invalid; use magma_v2.h"
#endif

#include "magma_copy_v1.h"
#include "magmablas_z.h"
#include "magmablas_z_v1.h"
#include "magmablas_c_v1.h"
#include "magmablas_d_v1.h"
#include "magmablas_s_v1.h"
#include "magmablas_zc_v1.h"
#include "magmablas_ds_v1.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// queue support
// new magma_queue_create adds device
#define magma_queue_create_v1( queue_ptr ) \
        magma_queue_create_v1_internal( queue_ptr, __func__, __FILE__, __LINE__ )

void magma_queue_create_v1_internal(
    magma_queue_t* queue_ptr,
    const char* func, const char* file, int line );


// =============================================================================
// @deprecated

#define MagmaUpperLower     MagmaFull
#define MagmaUpperLowerStr  MagmaFullStr

#define MAGMA_Z_CNJG(a)     MAGMA_Z_CONJ(a)
#define MAGMA_C_CNJG(a)     MAGMA_C_CONJ(a)
#define MAGMA_D_CNJG(a)     MAGMA_D_CONJ(a)
#define MAGMA_S_CNJG(a)     MAGMA_S_CONJ(a)

// device_sync is not portable to OpenCL, and is generally not needed
void magma_device_sync();


// =============================================================================
// Define magma queue
// @deprecated
magma_int_t magmablasSetKernelStream( magma_queue_t queue );
magma_int_t magmablasGetKernelStream( magma_queue_t *queue );
magma_queue_t magmablasGetQueue();

#ifdef __cplusplus
}
#endif

#endif // MAGMABLAS_V1_H
