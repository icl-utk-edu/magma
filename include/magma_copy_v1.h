/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
*/

#ifndef MAGMA_COPY_V1_H
#define MAGMA_COPY_V1_H

#ifdef MAGMA_NO_V1
#error "Since MAGMA_NO_V1 is defined, magma.h is invalid; use magma_v2.h"
#endif

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
// async versions are same for v1 and v2; see magmablas_q.h

#define magma_setvector_v1(           n, elemSize, hx_src, incx, dy_dst, incy ) \
        magma_setvector_v1_internal(  n, elemSize, hx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_getvector_v1(           n, elemSize, dx_src, incx, hy_dst, incy ) \
        magma_getvector_v1_internal(  n, elemSize, dx_src, incx, hy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_copyvector_v1(          n, elemSize, dx_src, incx, dy_dst, incy ) \
        magma_copyvector_v1_internal( n, elemSize, dx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

void
magma_setvector_v1_internal(
    magma_int_t n, magma_int_t elemSize,
    const void *hx_src, magma_int_t incx,
    magma_ptr   dy_dst, magma_int_t incy,
    const char* func, const char* file, int line );

void
magma_getvector_v1_internal(
    magma_int_t n, magma_int_t elemSize,
    magma_const_ptr dx_src, magma_int_t incx,
    void           *hy_dst, magma_int_t incy,
    const char* func, const char* file, int line );

void
magma_copyvector_v1_internal(
    magma_int_t n, magma_int_t elemSize,
    magma_const_ptr dx_src, magma_int_t incx,
    magma_ptr       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line );


// =============================================================================
// copying sub-matrices (contiguous columns)

#define magma_setmatrix_v1(           m, n, elemSize, hA_src, lda,  dB_dst, lddb ) \
        magma_setmatrix_v1_internal(  m, n, elemSize, hA_src, lda,  dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define magma_getmatrix_v1(           m, n, elemSize, dA_src, ldda, hB_dst, ldb ) \
        magma_getmatrix_v1_internal(  m, n, elemSize, dA_src, ldda, hB_dst, ldb, __func__, __FILE__, __LINE__ )

#define magma_copymatrix_v1(          m, n, elemSize, dA_src, ldda, dB_dst, lddb ) \
        magma_copymatrix_v1_internal( m, n, elemSize, dA_src, ldda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

void
magma_setmatrix_v1_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    const void *hA_src, magma_int_t lda,
    magma_ptr   dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line );

void
magma_getmatrix_v1_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    magma_const_ptr dA_src, magma_int_t ldda,
    void           *hB_dst, magma_int_t ldb,
    const char* func, const char* file, int line );

void
magma_copymatrix_v1_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    magma_const_ptr dA_src, magma_int_t ldda,
    magma_ptr       dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line );


// =============================================================================
// copying vectors - version for magma_int_t

#define magma_isetvector_v1(           n, hx_src, incx, dy_dst, incy ) \
        magma_isetvector_v1_internal(  n, hx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_igetvector_v1(           n, dx_src, incx, hy_dst, incy ) \
        magma_igetvector_v1_internal(  n, dx_src, incx, hy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_icopyvector_v1(          n, dx_src, incx, dy_dst, incy ) \
        magma_icopyvector_v1_internal( n, dx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

static inline void
magma_isetvector_v1_internal(
    magma_int_t n,
    const magma_int_t *hx_src, magma_int_t incx,
    magmaInt_ptr       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    magma_setvector_v1_internal( n, sizeof(magma_int_t),
                                 hx_src, incx,
                                 dy_dst, incy,
                                 func, file, line );
}

static inline void
magma_igetvector_v1_internal(
    magma_int_t n,
    magmaInt_const_ptr dx_src, magma_int_t incx,
    magma_int_t       *hy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    magma_getvector_v1_internal( n, sizeof(magma_int_t),
                                 dx_src, incx,
                                 hy_dst, incy,
                                 func, file, line );
}

static inline void
magma_icopyvector_v1_internal(
    magma_int_t n,
    magmaInt_const_ptr dx_src, magma_int_t incx,
    magmaInt_ptr       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    magma_copyvector_v1_internal( n, sizeof(magma_int_t),
                                  dx_src, incx,
                                  dy_dst, incy,
                                  func, file, line );
}


// =============================================================================
// copying sub-matrices - version for magma_int_t

#define magma_isetmatrix_v1(           m, n, hA_src, lda,  dB_dst, lddb ) \
        magma_isetmatrix_v1_internal(  m, n, hA_src, lda,  dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define magma_igetmatrix_v1(           m, n, dA_src, ldda, hB_dst, ldb ) \
        magma_igetmatrix_v1_internal(  m, n, dA_src, ldda, hB_dst, ldb, __func__, __FILE__, __LINE__ )

#define magma_icopymatrix_v1(          m, n, dA_src, ldda, dB_dst, lddb ) \
        magma_icopymatrix_v1_internal( m, n, dA_src, ldda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

static inline void
magma_isetmatrix_v1_internal(
    magma_int_t m, magma_int_t n,
    const magma_int_t *hA_src, magma_int_t lda,
    magmaInt_ptr       dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line )
{
    magma_setmatrix_v1_internal( m, n, sizeof(magma_int_t),
                                 hA_src, lda,
                                 dB_dst, lddb,
                                 func, file, line );
}

static inline void
magma_igetmatrix_v1_internal(
    magma_int_t m, magma_int_t n,
    magmaInt_const_ptr dA_src, magma_int_t ldda,
    magma_int_t       *hB_dst, magma_int_t ldb,
    const char* func, const char* file, int line )
{
    magma_getmatrix_v1_internal( m, n, sizeof(magma_int_t),
                                 dA_src, ldda,
                                 hB_dst, ldb,
                                 func, file, line );
}

static inline void
magma_icopymatrix_v1_internal(
    magma_int_t m, magma_int_t n,
    magmaInt_const_ptr dA_src, magma_int_t ldda,
    magmaInt_ptr       dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line )
{
    magma_copymatrix_v1_internal( m, n, sizeof(magma_int_t),
                                  dA_src, ldda,
                                  dB_dst, lddb,
                                  func, file, line );
}


// =============================================================================
// copying vectors - version for magma_index_t

#define magma_index_setvector_v1(           n, hx_src, incx, dy_dst, incy ) \
        magma_index_setvector_v1_internal(  n, hx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_index_getvector_v1(           n, dx_src, incx, hy_dst, incy ) \
        magma_index_getvector_v1_internal(  n, dx_src, incx, hy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_index_copyvector_v1(          n, dx_src, incx, dy_dst, incy ) \
        magma_index_copyvector_v1_internal( n, dx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

static inline void
magma_index_setvector_v1_internal(
    magma_int_t n,
    const magma_index_t *hx_src, magma_int_t incx,
    magmaIndex_ptr       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    magma_setvector_v1_internal( n, sizeof(magma_index_t),
                                 hx_src, incx,
                                 dy_dst, incy,
                                 func, file, line );
}

static inline void
magma_index_getvector_v1_internal(
    magma_int_t n,
    magmaIndex_const_ptr dx_src, magma_int_t incx,
    magma_index_t       *hy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    magma_getvector_v1_internal( n, sizeof(magma_index_t),
                                 dx_src, incx,
                                 hy_dst, incy,
                                 func, file, line );
}

static inline void
magma_index_copyvector_v1_internal(
    magma_int_t n,
    magmaIndex_const_ptr dx_src, magma_int_t incx,
    magmaIndex_ptr       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    magma_copyvector_v1_internal( n, sizeof(magma_index_t),
                                  dx_src, incx,
                                  dy_dst, incy,
                                  func, file, line );
}


// =============================================================================
// copying sub-matrices - version for magma_index_t

#define magma_index_setmatrix_v1(           m, n, hA_src, lda,  dB_dst, lddb ) \
        magma_index_setmatrix_v1_internal(  m, n, hA_src, lda,  dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define magma_index_getmatrix_v1(           m, n, dA_src, ldda, hB_dst, ldb ) \
        magma_index_getmatrix_v1_internal(  m, n, dA_src, ldda, hB_dst, ldb, __func__, __FILE__, __LINE__ )

#define magma_index_copymatrix_v1(          m, n, dA_src, ldda, dB_dst, lddb ) \
        magma_index_copymatrix_v1_internal( m, n, dA_src, ldda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

static inline void
magma_index_setmatrix_v1_internal(
    magma_int_t m, magma_int_t n,
    const magma_index_t *hA_src, magma_int_t lda,
    magmaIndex_ptr       dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line )
{
    magma_setmatrix_v1_internal( m, n, sizeof(magma_index_t),
                                 hA_src, lda,
                                 dB_dst, lddb,
                                 func, file, line );
}

static inline void
magma_index_getmatrix_v1_internal(
    magma_int_t m, magma_int_t n,
    magmaIndex_const_ptr dA_src, magma_int_t ldda,
    magma_index_t       *hB_dst, magma_int_t ldb,
    const char* func, const char* file, int line )
{
    magma_getmatrix_v1_internal( m, n, sizeof(magma_index_t),
                                 dA_src, ldda,
                                 hB_dst, ldb,
                                 func, file, line );
}

static inline void
magma_index_copymatrix_v1_internal(
    magma_int_t m, magma_int_t n,
    magmaIndex_const_ptr dA_src, magma_int_t ldda,
    magmaIndex_ptr       dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line )
{
    magma_copymatrix_v1_internal( m, n, sizeof(magma_index_t),
                                  dA_src, ldda,
                                  dB_dst, lddb,
                                  func, file, line );
}

#ifdef __cplusplus
}
#endif

#endif // MAGMA_COPY_V1_H
