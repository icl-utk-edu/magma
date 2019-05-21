/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mathieu Faverge
       @author Mark Gates

       Based on PLASMA common.h
*/

// =============================================================================
// MAGMA facilities of interest to both src and magmablas directories

#ifndef MAGMA_INTERNAL_H
#define MAGMA_INTERNAL_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

#if defined( _WIN32 ) || defined( _WIN64 )

    #include "magma_winthread.h"
    #include <windows.h>
    #include <limits.h>
    #include <io.h>

    // functions where Microsoft fails to provide C99 standard
    // (only with Microsoft, not with nvcc on Windows)
    // in both magma_internal.h and testings.h
    #ifndef __NVCC__

        #include <float.h>
        #define copysign(x,y) _copysign(x,y)
        #define isnan(x)      _isnan(x)
        #define isinf(x)      ( ! _finite(x) && ! _isnan(x) )
        #define isfinite(x)   _finite(x)
        // note _snprintf has slightly different semantics than snprintf
        #define snprintf _snprintf

    #endif

#else

    #include <pthread.h>
    #include <unistd.h>
    #include <inttypes.h>

    // our magma_winthread doesn't have pthread_key;
    // assume other platforms (Linux, MacOS, etc.) do.
    #define HAVE_PTHREAD_KEY

#endif

// provide our own support for pthread_barrier on MacOS and Windows
#include "pthread_barrier.h"

#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "magma_threadsetting.h"

/***************************************************************************//**
    Define magma_queue structure, which wraps around CUDA and OpenCL queues.
    In C, this is a simple struct.
    In C++, it is a class with getter member functions.
    For both C/C++, use magma_queue_create() and magma_queue_destroy()
    to create and destroy a queue. Global getter functions exist to query the
    queue.

    @see magma_queue_create
    @see magma_queue_create_v2
    @see magma_queue_destroy
    @see magma_queue_get_device
    @see magma_queue_get_cuda_stream
    @see magma_queue_get_cublas_handle
    @see magma_queue_get_cusparse_handle

    @ingroup magma_queue
*******************************************************************************/
struct magma_queue
{
#ifdef __cplusplus
public:
    /// @return device associated with this queue
    magma_device_t   device()          { return device__;   }

    #ifdef HAVE_CUBLAS
    /// @return CUDA stream associated with this queue; requires CUDA.
    cudaStream_t     cuda_stream()     { return stream__;   }

    /// @return cuBLAS handle associated with this queue; requires CUDA.
    /// MAGMA assumes the handle won't be changed, e.g., its stream won't be modified.
    cublasHandle_t   cublas_handle()   { return cublas__;   }

    /// @return cuSparse handle associated with this queue; requires CUDA.
    /// MAGMA assumes the handle won't be changed, e.g., its stream won't be modified.
    cusparseHandle_t cusparse_handle() { return cusparse__; }

    #endif
    /// @return the pointer array dAarray__.
    void** get_dAarray() {return dAarray__; }

    /// @return the pointer array dBarray__.
    void** get_dBarray() {return dBarray__; }

    /// @return the pointer array dCarray__.
    void** get_dCarray() {return dCarray__; }

    /// @return the pointer array dCarray__.
    magma_int_t get_maxBatch() {return (magma_int_t)maxbatch__; }

protected:
    friend
    void magma_queue_create_internal(
        magma_device_t device, magma_queue_t* queuePtr,
        const char* func, const char* file, int line );

    #ifdef HAVE_CUBLAS
    friend
    void magma_queue_create_from_cuda_internal(
        magma_device_t   device,
        cudaStream_t     stream,
        cublasHandle_t   cublas_handle,
        cusparseHandle_t cusparse_handle,
        magma_queue_t*   queuePtr,
        const char* func, const char* file, int line );
    #endif

    friend
    void magma_queue_destroy_internal(
        magma_queue_t queue,
        const char* func, const char* file, int line );
#endif // __cplusplus

    // protected members -- access through getters
    // bitmask whether MAGMA owns the CUDA stream, cuBLAS and cuSparse handles
    int              own__;
    magma_device_t   device__;      // associated device ID
    int              maxbatch__;    // maximum size of the pointer array
    void**           dAarray__;     // pointer array
    void**           dBarray__;     // pointer array
    void**           dCarray__;     // pointer array

    #ifdef HAVE_CUBLAS
    cudaStream_t     stream__;      // associated CUDA stream; may be NULL
    cublasHandle_t   cublas__;      // associated cuBLAS handle
    cusparseHandle_t cusparse__;    // associated cuSparse handle
    #endif
};

#ifdef __cplusplus
extern "C" {
#endif

// needed for BLAS functions that no longer include magma.h (v1)
magma_queue_t magmablasGetQueue();

#ifdef __cplusplus
}
#endif


// =============================================================================
// Determine if weak symbols are allowed

#if defined(linux) || defined(__linux) || defined(__linux__)
#if defined(__GNUC_EXCL__) || defined(__GNUC__)
#define MAGMA_HAVE_WEAK    1
#endif
#endif


// =============================================================================
// Global utilities
// in both magma_internal.h and testings.h
// These generally require that magma_internal.h be the last header,
// as max() and min() often conflict with system and library headers.

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

/***************************************************************************//**
    Suppress "warning: unused variable" in a portable fashion.
    @ingroup magma_internal
*******************************************************************************/
#define MAGMA_UNUSED(var)  ((void)var)

#endif // MAGMA_INTERNAL_H
