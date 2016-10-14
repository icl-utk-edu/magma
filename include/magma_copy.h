/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
*/

#ifndef MAGMA_COPY_H
#define MAGMA_COPY_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// copying vectors
// set  copies host   to device
// get  copies device to host
// copy copies device to device
// (with CUDA unified addressing, copy can be between same or different devices)
// Add the function, file, and line for error-reporting purposes.

#define magma_setvector(                 n, elemSize, hx_src, incx, dy_dst, incy, queue ) \
        magma_setvector_internal(        n, elemSize, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_getvector(                 n, elemSize, dx_src, incx, hy_dst, incy, queue ) \
        magma_getvector_internal(        n, elemSize, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_copyvector(                n, elemSize, dx_src, incx, dy_dst, incy, queue ) \
        magma_copyvector_internal(       n, elemSize, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_setvector_async(           n, elemSize, hx_src, incx, dy_dst, incy, queue ) \
        magma_setvector_async_internal(  n, elemSize, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_getvector_async(           n, elemSize, dx_src, incx, hy_dst, incy, queue ) \
        magma_getvector_async_internal(  n, elemSize, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_copyvector_async(          n, elemSize, dx_src, incx, dy_dst, incy, queue ) \
        magma_copyvector_async_internal( n, elemSize, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

void
magma_setvector_internal(
    magma_int_t n, magma_int_t elemSize,
    const void *hx_src, magma_int_t incx,
    magma_ptr   dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void
magma_getvector_internal(
    magma_int_t n, magma_int_t elemSize,
    magma_const_ptr dx_src, magma_int_t incx,
    void           *hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void
magma_copyvector_internal(
    magma_int_t n, magma_int_t elemSize,
    magma_const_ptr dx_src, magma_int_t incx,
    magma_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void
magma_setvector_async_internal(
    magma_int_t n, magma_int_t elemSize,
    const void *hx_src, magma_int_t incx,
    magma_ptr   dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void
magma_getvector_async_internal(
    magma_int_t n, magma_int_t elemSize,
    magma_const_ptr dx_src, magma_int_t incx,
    void           *hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void
magma_copyvector_async_internal(
    magma_int_t n, magma_int_t elemSize,
    magma_const_ptr dx_src, magma_int_t incx,
    magma_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line );


// =============================================================================
// copying sub-matrices (contiguous columns)

#define magma_setmatrix(                 m, n, elemSize, hA_src, lda,  dB_dst, lddb, queue ) \
        magma_setmatrix_internal(        m, n, elemSize, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define magma_getmatrix(                 m, n, elemSize, dA_src, ldda, hB_dst, ldb,  queue ) \
        magma_getmatrix_internal(        m, n, elemSize, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )

#define magma_copymatrix(                m, n, elemSize, dA_src, ldda, dB_dst, lddb, queue ) \
        magma_copymatrix_internal(       m, n, elemSize, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define magma_setmatrix_async(           m, n, elemSize, hA_src, lda,  dB_dst, lddb, queue ) \
        magma_setmatrix_async_internal(  m, n, elemSize, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define magma_getmatrix_async(           m, n, elemSize, dA_src, ldda, hB_dst, ldb,  queue ) \
        magma_getmatrix_async_internal(  m, n, elemSize, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )

#define magma_copymatrix_async(          m, n, elemSize, dA_src, ldda, dB_dst, lddb, queue ) \
        magma_copymatrix_async_internal( m, n, elemSize, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

void
magma_setmatrix_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    const void *hA_src, magma_int_t lda,
    magma_ptr   dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void
magma_getmatrix_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    magma_const_ptr dA_src, magma_int_t ldda,
    void           *hB_dst, magma_int_t ldb,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void
magma_copymatrix_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    magma_const_ptr dA_src, magma_int_t ldda,
    magma_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void
magma_setmatrix_async_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    const void *hA_src, magma_int_t lda,
    magma_ptr   dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void
magma_getmatrix_async_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    magma_const_ptr dA_src, magma_int_t ldda,
    void           *hB_dst, magma_int_t ldb,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void
magma_copymatrix_async_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    magma_const_ptr dA_src, magma_int_t ldda,
    magma_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line );


// =============================================================================
// copying vectors - version for magma_int_t

/// Type-safe version of magma_setvector() for magma_int_t arrays.
/// @ingroup magma_setvector
#define magma_isetvector(                 n, hx_src, incx, dy_dst, incy, queue ) \
        magma_isetvector_internal(        n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_getvector() for magma_int_t arrays.
/// @ingroup magma_getvector
#define magma_igetvector(                 n, dx_src, incx, hy_dst, incy, queue ) \
        magma_igetvector_internal(        n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_copyvector() for magma_int_t arrays.
/// @ingroup magma_copyvector
#define magma_icopyvector(                n, dx_src, incx, dy_dst, incy, queue ) \
        magma_icopyvector_internal(       n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_setvector_async() for magma_int_t arrays.
/// @ingroup magma_setvector
#define magma_isetvector_async(           n, hx_src, incx, dy_dst, incy, queue ) \
        magma_isetvector_async_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_getvector_async() for magma_int_t arrays.
/// @ingroup magma_getvector
#define magma_igetvector_async(           n, dx_src, incx, hy_dst, incy, queue ) \
        magma_igetvector_async_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_copyvector_async() for magma_int_t arrays.
/// @ingroup magma_copyvector
#define magma_icopyvector_async(          n, dx_src, incx, dy_dst, incy, queue ) \
        magma_icopyvector_async_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

static inline void
magma_isetvector_internal(
    magma_int_t n,
    const magma_int_t *hx_src, magma_int_t incx,
    magmaInt_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_setvector_internal( n, sizeof(magma_int_t),
                              hx_src, incx,
                              dy_dst, incy,
                              queue, func, file, line );
}

static inline void
magma_igetvector_internal(
    magma_int_t n,
    magmaInt_const_ptr dx_src, magma_int_t incx,
    magma_int_t       *hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_getvector_internal( n, sizeof(magma_int_t),
                              dx_src, incx,
                              hy_dst, incy,
                              queue, func, file, line );
}

static inline void
magma_icopyvector_internal(
    magma_int_t n,
    magmaInt_const_ptr dx_src, magma_int_t incx,
    magmaInt_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_copyvector_internal( n, sizeof(magma_int_t),
                               dx_src, incx,
                               dy_dst, incy,
                               queue, func, file, line );
}

static inline void
magma_isetvector_async_internal(
    magma_int_t n,
    const magma_int_t *hx_src, magma_int_t incx,
    magmaInt_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_setvector_async_internal( n, sizeof(magma_int_t),
                                    hx_src, incx,
                                    dy_dst, incy,
                                    queue, func, file, line );
}

static inline void
magma_igetvector_async_internal(
    magma_int_t n,
    magmaInt_const_ptr dx_src, magma_int_t incx,
    magma_int_t       *hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_getvector_async_internal( n, sizeof(magma_int_t),
                                    dx_src, incx,
                                    hy_dst, incy,
                                    queue, func, file, line );
}

static inline void
magma_icopyvector_async_internal(
    magma_int_t n,
    magmaInt_const_ptr dx_src, magma_int_t incx,
    magmaInt_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_copyvector_async_internal( n, sizeof(magma_int_t),
                                     dx_src, incx,
                                     dy_dst, incy,
                                     queue, func, file, line );
}


// =============================================================================
// copying sub-matrices - version for magma_int_t

/// Type-safe version of magma_setmatrix() for magma_int_t arrays.
/// @ingroup magma_setmatrix
#define magma_isetmatrix(                 m, n, hA_src, lda,  dB_dst, lddb, queue ) \
        magma_isetmatrix_internal(        m, n, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_getmatrix() for magma_int_t arrays.
/// @ingroup magma_getmatrix
#define magma_igetmatrix(                 m, n, dA_src, ldda, hB_dst, ldb,  queue ) \
        magma_igetmatrix_internal(        m, n, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_copymatrix() for magma_int_t arrays.
/// @ingroup magma_copymatrix
#define magma_icopymatrix(                m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        magma_icopymatrix_internal(       m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_setmatrix_async() for magma_int_t arrays.
/// @ingroup magma_setmatrix
#define magma_isetmatrix_async(           m, n, hA_src, lda,  dB_dst, lddb, queue ) \
        magma_isetmatrix_async_internal(  m, n, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_getmatrix_async() for magma_int_t arrays.
/// @ingroup magma_getmatrix
#define magma_igetmatrix_async(           m, n, dA_src, ldda, hB_dst, ldb,  queue ) \
        magma_igetmatrix_async_internal(  m, n, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_copymatrix_async() for magma_int_t arrays.
/// @ingroup magma_copymatrix
#define magma_icopymatrix_async(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        magma_icopymatrix_async_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

static inline void
magma_isetmatrix_internal(
    magma_int_t m, magma_int_t n,
    const magma_int_t *hA_src, magma_int_t lda,
    magmaInt_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_setmatrix_internal( m, n, sizeof(magma_int_t),
                              hA_src, lda,
                              dB_dst, lddb,
                              queue, func, file, line );
}

static inline void
magma_igetmatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaInt_const_ptr dA_src, magma_int_t ldda,
    magma_int_t       *hB_dst, magma_int_t ldb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_getmatrix_internal( m, n, sizeof(magma_int_t),
                              dA_src, ldda,
                              hB_dst, ldb,
                              queue, func, file, line );
}

static inline void
magma_icopymatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaInt_const_ptr dA_src, magma_int_t ldda,
    magmaInt_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_copymatrix_internal( m, n, sizeof(magma_int_t),
                               dA_src, ldda,
                               dB_dst, lddb,
                               queue, func, file, line );
}

static inline void
magma_isetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    const magma_int_t *hA_src, magma_int_t lda,
    magmaInt_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_setmatrix_async_internal( m, n, sizeof(magma_int_t),
                                    hA_src, lda,
                                    dB_dst, lddb,
                                    queue, func, file, line );
}

static inline void
magma_igetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaInt_const_ptr dA_src, magma_int_t ldda,
    magma_int_t       *hB_dst, magma_int_t ldb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_getmatrix_async_internal( m, n, sizeof(magma_int_t),
                                    dA_src, ldda,
                                    hB_dst, ldb,
                                    queue, func, file, line );
}

static inline void
magma_icopymatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaInt_const_ptr dA_src, magma_int_t ldda,
    magmaInt_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_copymatrix_async_internal( m, n, sizeof(magma_int_t),
                                     dA_src, ldda,
                                     dB_dst, lddb,
                                     queue, func, file, line );
}


// =============================================================================
// copying vectors - version for magma_index_t

/// Type-safe version of magma_setvector() for magma_index_t arrays.
/// @ingroup magma_setvector
#define magma_index_setvector(                 n, hx_src, incx, dy_dst, incy, queue ) \
        magma_index_setvector_internal(        n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_getvector() for magma_index_t arrays.
/// @ingroup magma_getvector
#define magma_index_getvector(                 n, dx_src, incx, hy_dst, incy, queue ) \
        magma_index_getvector_internal(        n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_copyvector() for magma_index_t arrays.
/// @ingroup magma_copyvector
#define magma_index_copyvector(                n, dx_src, incx, dy_dst, incy, queue ) \
        magma_index_copyvector_internal(       n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_setvector_async() for magma_index_t arrays.
/// @ingroup magma_setvector
#define magma_index_setvector_async(           n, hx_src, incx, dy_dst, incy, queue ) \
        magma_index_setvector_async_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_getvector_async() for magma_index_t arrays.
/// @ingroup magma_getvector
#define magma_index_getvector_async(           n, dx_src, incx, hy_dst, incy, queue ) \
        magma_index_getvector_async_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_copyvector_async() for magma_index_t arrays.
/// @ingroup magma_copyvector
#define magma_index_copyvector_async(          n, dx_src, incx, dy_dst, incy, queue ) \
        magma_index_copyvector_async_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

static inline void
magma_index_setvector_internal(
    magma_int_t n,
    const magma_index_t *hx_src, magma_int_t incx,
    magmaIndex_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_setvector_internal( n, sizeof(magma_index_t),
                              hx_src, incx,
                              dy_dst, incy,
                              queue, func, file, line );
}

static inline void
magma_index_getvector_internal(
    magma_int_t n,
    magmaIndex_const_ptr dx_src, magma_int_t incx,
    magma_index_t       *hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_getvector_internal( n, sizeof(magma_index_t),
                              dx_src, incx,
                              hy_dst, incy,
                              queue, func, file, line );
}

static inline void
magma_index_copyvector_internal(
    magma_int_t n,
    magmaIndex_const_ptr dx_src, magma_int_t incx,
    magmaIndex_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_copyvector_internal( n, sizeof(magma_index_t),
                               dx_src, incx,
                               dy_dst, incy,
                               queue, func, file, line );
}

static inline void
magma_index_setvector_async_internal(
    magma_int_t n,
    const magma_index_t *hx_src, magma_int_t incx,
    magmaIndex_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_setvector_async_internal( n, sizeof(magma_index_t),
                                    hx_src, incx,
                                    dy_dst, incy,
                                    queue, func, file, line );
}

static inline void
magma_index_getvector_async_internal(
    magma_int_t n,
    magmaIndex_const_ptr dx_src, magma_int_t incx,
    magma_index_t       *hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_getvector_async_internal( n, sizeof(magma_index_t),
                                    dx_src, incx,
                                    hy_dst, incy,
                                    queue, func, file, line );
}

static inline void
magma_index_copyvector_async_internal(
    magma_int_t n,
    magmaIndex_const_ptr dx_src, magma_int_t incx,
    magmaIndex_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_copyvector_async_internal( n, sizeof(magma_index_t),
                                     dx_src, incx,
                                     dy_dst, incy,
                                     queue, func, file, line );
}

// =============================================================================
// copying vectors - version for magma_uindex_t

/// Type-safe version of magma_setvector() for magma_uindex_t arrays.
/// @ingroup magma_setvector
#define magma_uindex_setvector(                 n, hx_src, incx, dy_dst, incy, queue ) \
        magma_uindex_setvector_internal(        n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_getvector() for magma_uindex_t arrays.
/// @ingroup magma_getvector
#define magma_uindex_getvector(                 n, dx_src, incx, hy_dst, incy, queue ) \
        magma_uindex_getvector_internal(        n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_copyvector() for magma_uindex_t arrays.
/// @ingroup magma_copyvector
#define magma_uindex_copyvector(                n, dx_src, incx, dy_dst, incy, queue ) \
        magma_uindex_copyvector_internal(       n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_setvector_async() for magma_uindex_t arrays.
/// @ingroup magma_setvector
#define magma_uindex_setvector_async(           n, hx_src, incx, dy_dst, incy, queue ) \
        magma_uindex_setvector_async_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_getvector_async() for magma_uindex_t arrays.
/// @ingroup magma_getvector
#define magma_uindex_getvector_async(           n, dx_src, incx, hy_dst, incy, queue ) \
        magma_uindex_getvector_async_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_copyvector_async() for magma_uindex_t arrays.
/// @ingroup magma_copyvector
#define magma_uindex_copyvector_async(          n, dx_src, incx, dy_dst, incy, queue ) \
        magma_uindex_copyvector_async_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

static inline void
magma_uindex_setvector_internal(
    magma_int_t n,
    const magma_uindex_t *hx_src, magma_int_t incx,
    magmaUIndex_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_setvector_internal( n, sizeof(magma_uindex_t),
                              hx_src, incx,
                              dy_dst, incy,
                              queue, func, file, line );
}

static inline void
magma_uindex_getvector_internal(
    magma_int_t n,
    magmaUIndex_const_ptr dx_src, magma_int_t incx,
    magma_uindex_t       *hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_getvector_internal( n, sizeof(magma_uindex_t),
                              dx_src, incx,
                              hy_dst, incy,
                              queue, func, file, line );
}

static inline void
magma_uindex_copyvector_internal(
    magma_int_t n,
    magmaUIndex_const_ptr dx_src, magma_int_t incx,
    magmaUIndex_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_copyvector_internal( n, sizeof(magma_uindex_t),
                               dx_src, incx,
                               dy_dst, incy,
                               queue, func, file, line );
}

static inline void
magma_uindex_setvector_async_internal(
    magma_int_t n,
    const magma_uindex_t *hx_src, magma_int_t incx,
    magmaUIndex_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_setvector_async_internal( n, sizeof(magma_uindex_t),
                                    hx_src, incx,
                                    dy_dst, incy,
                                    queue, func, file, line );
}

static inline void
magma_uindex_getvector_async_internal(
    magma_int_t n,
    magmaUIndex_const_ptr dx_src, magma_int_t incx,
    magma_uindex_t       *hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_getvector_async_internal( n, sizeof(magma_uindex_t),
                                    dx_src, incx,
                                    hy_dst, incy,
                                    queue, func, file, line );
}

static inline void
magma_uindex_copyvector_async_internal(
    magma_int_t n,
    magmaUIndex_const_ptr dx_src, magma_int_t incx,
    magmaUIndex_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_copyvector_async_internal( n, sizeof(magma_uindex_t),
                                     dx_src, incx,
                                     dy_dst, incy,
                                     queue, func, file, line );
}


// =============================================================================
// copying sub-matrices - version for magma_index_t

/// Type-safe version of magma_setmatrix() for magma_index_t arrays.
/// @ingroup magma_setmatrix
#define magma_index_setmatrix(                 m, n, hA_src, lda,  dB_dst, lddb, queue ) \
        magma_index_setmatrix_internal(        m, n, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_getmatrix() for magma_index_t arrays.
/// @ingroup magma_getmatrix
#define magma_index_getmatrix(                 m, n, dA_src, ldda, hB_dst, ldb,  queue ) \
        magma_index_getmatrix_internal(        m, n, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_copymatrix() for magma_index_t arrays.
/// @ingroup magma_copymatrix
#define magma_index_copymatrix(                m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        magma_index_copymatrix_internal(       m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_setmatrix_async() for magma_index_t arrays.
/// @ingroup magma_setmatrix
#define magma_index_setmatrix_async(           m, n, hA_src, lda,  dB_dst, lddb, queue ) \
        magma_index_setmatrix_async_internal(  m, n, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_getmatrix_async() for magma_index_t arrays.
/// @ingroup magma_getmatrix
#define magma_index_getmatrix_async(           m, n, dA_src, ldda, hB_dst, ldb,  queue ) \
        magma_index_getmatrix_async_internal(  m, n, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_copymatrix_async() for magma_index_t arrays.
/// @ingroup magma_copymatrix
#define magma_index_copymatrix_async(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        magma_index_copymatrix_async_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

static inline void
magma_index_setmatrix_internal(
    magma_int_t m, magma_int_t n,
    const magma_index_t *hA_src, magma_int_t lda,
    magmaIndex_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_setmatrix_internal( m, n, sizeof(magma_index_t),
                              hA_src, lda,
                              dB_dst, lddb,
                              queue, func, file, line );
}

static inline void
magma_index_getmatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaIndex_const_ptr dA_src, magma_int_t ldda,
    magma_index_t       *hB_dst, magma_int_t ldb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_getmatrix_internal( m, n, sizeof(magma_index_t),
                              dA_src, ldda,
                              hB_dst, ldb,
                              queue, func, file, line );
}

static inline void
magma_index_copymatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaIndex_const_ptr dA_src, magma_int_t ldda,
    magmaIndex_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_copymatrix_internal( m, n, sizeof(magma_index_t),
                               dA_src, ldda,
                               dB_dst, lddb,
                               queue, func, file, line );
}

static inline void
magma_index_setmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    const magma_index_t *hA_src, magma_int_t lda,
    magmaIndex_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_setmatrix_async_internal( m, n, sizeof(magma_index_t),
                                    hA_src, lda,
                                    dB_dst, lddb,
                                    queue, func, file, line );
}

static inline void
magma_index_getmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaIndex_const_ptr dA_src, magma_int_t ldda,
    magma_index_t       *hB_dst, magma_int_t ldb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_getmatrix_async_internal( m, n, sizeof(magma_index_t),
                                    dA_src, ldda,
                                    hB_dst, ldb,
                                    queue, func, file, line );
}

static inline void
magma_index_copymatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaIndex_const_ptr dA_src, magma_int_t ldda,
    magmaIndex_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_copymatrix_async_internal( m, n, sizeof(magma_index_t),
                                     dA_src, ldda,
                                     dB_dst, lddb,
                                     queue, func, file, line );
}

#ifdef __cplusplus
}
#endif

#endif // MAGMA_COPY_H
