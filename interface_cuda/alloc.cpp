/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
*/

#include <stdlib.h>
#include <stdio.h>

#ifdef DEBUG_MEMORY
#include <map>
#include <mutex>  // requires C++11
#endif

#include <cuda_runtime.h>

#include "magma_v2.h"
#include "error.h"

#ifdef HAVE_CUBLAS


#ifdef DEBUG_MEMORY
std::mutex                g_pointers_mutex;  // requires C++11
std::map< void*, size_t > g_pointers_dev;
std::map< void*, size_t > g_pointers_cpu;
std::map< void*, size_t > g_pointers_pin;
#endif


/***************************************************************************//**
    Allocates memory on the GPU. CUDA imposes a synchronization.
    Use magma_free() to free this memory.

    @param[out]
    ptrPtr  On output, set to the pointer that was allocated.
            NULL on failure.

    @param[in]
    size    Size in bytes to allocate. If size = 0, allocates some minimal size.

    @return MAGMA_SUCCESS
    @return MAGMA_ERR_DEVICE_ALLOC on failure

    Type-safe versions avoid the need for a (void**) cast and explicit sizeof.
    @see magma_smalloc
    @see magma_dmalloc
    @see magma_cmalloc
    @see magma_zmalloc
    @see magma_imalloc
    @see magma_index_malloc

    @ingroup magma_malloc
*******************************************************************************/
extern "C" magma_int_t
magma_malloc( magma_ptr* ptrPtr, size_t size )
{
    // malloc and free sometimes don't work for size=0, so allocate some minimal size
    if ( size == 0 )
        size = sizeof(magmaDoubleComplex);
    if ( cudaSuccess != cudaMalloc( ptrPtr, size )) {
        return MAGMA_ERR_DEVICE_ALLOC;
    }

    #ifdef DEBUG_MEMORY
    g_pointers_mutex.lock();
    g_pointers_dev[ *ptrPtr ] = size;
    g_pointers_mutex.unlock();
    #endif

    return MAGMA_SUCCESS;
}


/***************************************************************************//**
    @fn magma_free( ptr )

    Frees GPU memory previously allocated by magma_malloc().

    @param[in]
    ptr     Pointer to free.

    @return MAGMA_SUCCESS
    @return MAGMA_ERR_INVALID_PTR on failure

    @ingroup magma_malloc
*******************************************************************************/
extern "C" magma_int_t
magma_free_internal( magma_ptr ptr,
    const char* func, const char* file, int line )
{
    #ifdef DEBUG_MEMORY
    g_pointers_mutex.lock();
    if ( ptr != NULL && g_pointers_dev.count( ptr ) == 0 ) {
        fprintf( stderr, "magma_free( %p ) that wasn't allocated with magma_malloc.\n", ptr );
    }
    else {
        g_pointers_dev.erase( ptr );
    }
    g_pointers_mutex.unlock();
    #endif

    cudaError_t err = cudaFree( ptr );
    check_xerror( err, func, file, line );
    if ( err != cudaSuccess ) {
        return MAGMA_ERR_INVALID_PTR;
    }
    return MAGMA_SUCCESS;
}


/***************************************************************************//**
    Allocate size bytes on CPU.
    The purpose of using this instead of malloc is to properly align arrays
    for vector (SSE, AVX) instructions. The default implementation uses
    posix_memalign (on Linux, MacOS, etc.) or _aligned_malloc (on Windows)
    to align memory to a 64 byte boundary (typical cache line size).
    Use magma_free_cpu() to free this memory.

    @param[out]
    ptrPtr  On output, set to the pointer that was allocated.
            NULL on failure.

    @param[in]
    size    Size in bytes to allocate. If size = 0, allocates some minimal size.

    @return MAGMA_SUCCESS
    @return MAGMA_ERR_HOST_ALLOC on failure

    Type-safe versions avoid the need for a (void**) cast and explicit sizeof.
    @see magma_smalloc_cpu
    @see magma_dmalloc_cpu
    @see magma_cmalloc_cpu
    @see magma_zmalloc_cpu
    @see magma_imalloc_cpu
    @see magma_index_malloc_cpu

    @ingroup magma_malloc_cpu
*******************************************************************************/
extern "C" magma_int_t
magma_malloc_cpu( void** ptrPtr, size_t size )
{
    // malloc and free sometimes don't work for size=0, so allocate some minimal size
    if ( size == 0 )
        size = sizeof(magmaDoubleComplex);
#if 1
#if defined( _WIN32 ) || defined( _WIN64 )
    *ptrPtr = _aligned_malloc( size, 64 );
    if ( *ptrPtr == NULL ) {
        return MAGMA_ERR_HOST_ALLOC;
    }
#else
    int err = posix_memalign( ptrPtr, 64, size );
    if ( err != 0 ) {
        *ptrPtr = NULL;
        return MAGMA_ERR_HOST_ALLOC;
    }
#endif
#else
    *ptrPtr = malloc( size );
    if ( *ptrPtr == NULL ) {
        return MAGMA_ERR_HOST_ALLOC;
    }
#endif

    #ifdef DEBUG_MEMORY
    g_pointers_mutex.lock();
    g_pointers_cpu[ *ptrPtr ] = size;
    g_pointers_mutex.unlock();
    #endif

    return MAGMA_SUCCESS;
}


/***************************************************************************//**
    Frees CPU memory previously allocated by magma_malloc_cpu().
    The default implementation uses free(),
    which works for both malloc and posix_memalign.
    For Windows, _aligned_free() is used.

    @param[in]
    ptr     Pointer to free.

    @return MAGMA_SUCCESS
    @return MAGMA_ERR_INVALID_PTR on failure

    @ingroup magma_malloc_cpu
*******************************************************************************/
extern "C" magma_int_t
magma_free_cpu( void* ptr )
{
    #ifdef DEBUG_MEMORY
    g_pointers_mutex.lock();
    if ( ptr != NULL && g_pointers_cpu.count( ptr ) == 0 ) {
        fprintf( stderr, "magma_free_cpu( %p ) that wasn't allocated with magma_malloc_cpu.\n", ptr );
    }
    else {
        g_pointers_cpu.erase( ptr );
    }
    g_pointers_mutex.unlock();
    #endif

#if defined( _WIN32 ) || defined( _WIN64 )
    _aligned_free( ptr );
#else
    free( ptr );
#endif
    return MAGMA_SUCCESS;
}


/***************************************************************************//**
    Allocates memory on the CPU in pinned memory.
    Use magma_free_pinned() to free this memory.

    @param[out]
    ptrPtr  On output, set to the pointer that was allocated.
            NULL on failure.

    @param[in]
    size    Size in bytes to allocate. If size = 0, allocates some minimal size.

    @return MAGMA_SUCCESS
    @return MAGMA_ERR_HOST_ALLOC on failure

    Type-safe versions avoid the need for a (void**) cast and explicit sizeof.
    @see magma_smalloc_pinned
    @see magma_dmalloc_pinned
    @see magma_cmalloc_pinned
    @see magma_zmalloc_pinned
    @see magma_imalloc_pinned
    @see magma_index_malloc_pinned

    @ingroup magma_malloc_pinned
*******************************************************************************/
extern "C" magma_int_t
magma_malloc_pinned( void** ptrPtr, size_t size )
{
    // malloc and free sometimes don't work for size=0, so allocate some minimal size
    // (for pinned memory, the error is detected in free)
    if ( size == 0 )
        size = sizeof(magmaDoubleComplex);
    if ( cudaSuccess != cudaHostAlloc( ptrPtr, size, cudaHostAllocPortable )) {
        return MAGMA_ERR_HOST_ALLOC;
    }

    #ifdef DEBUG_MEMORY
    g_pointers_mutex.lock();
    g_pointers_pin[ *ptrPtr ] = size;
    g_pointers_mutex.unlock();
    #endif

    return MAGMA_SUCCESS;
}


/***************************************************************************//**
    @fn magma_free_pinned( ptr )

    Frees CPU pinned memory previously allocated by magma_malloc_pinned().

    @param[in]
    ptr     Pointer to free.

    @return MAGMA_SUCCESS
    @return MAGMA_ERR_INVALID_PTR on failure

    @ingroup magma_malloc_pinned
*******************************************************************************/
extern "C" magma_int_t
magma_free_pinned_internal( void* ptr,
    const char* func, const char* file, int line )
{
    #ifdef DEBUG_MEMORY
    g_pointers_mutex.lock();
    if ( ptr != NULL && g_pointers_pin.count( ptr ) == 0 ) {
        fprintf( stderr, "magma_free_pinned( %p ) that wasn't allocated with magma_malloc_pinned.\n", ptr );
    }
    else {
        g_pointers_pin.erase( ptr );
    }
    g_pointers_mutex.unlock();
    #endif

    cudaError_t err = cudaFreeHost( ptr );
    check_xerror( err, func, file, line );
    if ( cudaSuccess != err ) {
        return MAGMA_ERR_INVALID_PTR;
    }
    return MAGMA_SUCCESS;
}

#endif // HAVE_CUBLAS
