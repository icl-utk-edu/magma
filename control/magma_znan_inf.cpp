/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @precisions normal z -> s d c

*/
#include <limits>

#include "magma_internal.h"

#define COMPLEX


const magmaDoubleComplex MAGMA_Z_NAN
    = MAGMA_Z_MAKE( std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN() );

const magmaDoubleComplex MAGMA_Z_INF
    = MAGMA_Z_MAKE( std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::infinity() );


/***************************************************************************//**
    @param[in] x    Scalar to test.
    @return true if either real(x) or imag(x) is NAN.
    @ingroup magma_nan_inf
*******************************************************************************/
int magma_z_isnan( magmaDoubleComplex x )
{
#ifdef COMPLEX
    return isnan( real( x )) ||
           isnan( imag( x ));
#else
    return isnan( x );
#endif
}


/***************************************************************************//**
    @param[in] x    Scalar to test.
    @return true if either real(x) or imag(x) is INF.
    @ingroup magma_nan_inf
*******************************************************************************/
int magma_z_isinf( magmaDoubleComplex x )
{
#ifdef COMPLEX
    return isinf( real( x )) ||
           isinf( imag( x ));
#else
    return isinf( x );
#endif
}


/***************************************************************************//**
    @param[in] x    Scalar to test.
    @return true if either real(x) or imag(x) is NAN or INF.
    @ingroup magma_nan_inf
*******************************************************************************/
int magma_z_isnan_inf( magmaDoubleComplex x )
{
#ifdef COMPLEX
    return isnan( real( x )) ||
           isnan( imag( x )) ||
           isinf( real( x )) ||
           isinf( imag( x ));
#else
    return isnan( x ) || isinf( x );
#endif
}


/***************************************************************************//**
    Purpose
    -------
    magma_znan_inf checks a matrix that is located on the CPU host
    for NAN (not-a-number) and INF (infinity) values.

    NAN is created by 0/0 and similar.
    INF is created by x/0 and similar, where x != 0.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
            Specifies what part of the matrix A to check.
      -     = MagmaUpper:  Upper triangular part of A
      -     = MagmaLower:  Lower triangular part of A
      -     = MagmaFull:   All of A

    @param[in]
    m       INTEGER
            The number of rows of the matrix A. m >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A. n >= 0.

    @param[in]
    A       COMPLEX_16 array, dimension (lda,n), on the CPU host.
            The m-by-n matrix to be printed.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A. lda >= m.

    @param[out]
    cnt_nan INTEGER*
            If non-NULL, on exit contains the number of NAN values in A.

    @param[out]
    cnt_inf INTEGER*
            If non-NULL, on exit contains the number of INF values in A.

    @return
      -     >= 0:  Returns number of NAN + number of INF values.
      -     <  0:  If it returns -i, the i-th argument had an illegal value,
                   or another error occured, such as memory allocation failed.

    @ingroup magma_nan_inf
*******************************************************************************/
extern "C"
magma_int_t magma_znan_inf(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    const magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *cnt_nan,
    magma_int_t *cnt_inf )
{
    #define A(i_, j_) (A + (i_) + (j_)*lda)
    
    magma_int_t info = 0;
    if (uplo != MagmaLower && uplo != MagmaUpper && uplo != MagmaFull)
        info = -1;
    else if (m < 0)
        info = -2;
    else if (n < 0)
        info = -3;
    else if (lda < m)
        info = -5;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;
    }
    
    int c_nan = 0;
    int c_inf = 0;
    
    if (uplo == MagmaLower) {
        for (int j = 0; j < n; ++j) {
            for (int i = j; i < m; ++i) {  // i >= j
                if      (magma_z_isnan( *A(i,j) )) { c_nan++; }
                else if (magma_z_isinf( *A(i,j) )) { c_inf++; }
            }
        }
    }
    else if (uplo == MagmaUpper) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m && i <= j; ++i) {  // i <= j
                if      (magma_z_isnan( *A(i,j) )) { c_nan++; }
                else if (magma_z_isinf( *A(i,j) )) { c_inf++; }
            }
        }
    }
    else if (uplo == MagmaFull) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                if      (magma_z_isnan( *A(i,j) )) { c_nan++; }
                else if (magma_z_isinf( *A(i,j) )) { c_inf++; }
            }
        }
    }
    
    if (cnt_nan != NULL) { *cnt_nan = c_nan; }
    if (cnt_inf != NULL) { *cnt_inf = c_inf; }
    
    return (c_nan + c_inf);
}


/***************************************************************************//**
    Purpose
    -------
    magma_znan_inf checks a matrix that is located on the CPU host
    for NAN (not-a-number) and INF (infinity) values.

    NAN is created by 0/0 and similar.
    INF is created by x/0 and similar, where x != 0.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
            Specifies what part of the matrix A to check.
      -     = MagmaUpper:  Upper triangular part of A
      -     = MagmaLower:  Lower triangular part of A
      -     = MagmaFull:   All of A

    @param[in]
    m       INTEGER
            The number of rows of the matrix A. m >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A. n >= 0.

    @param[in]
    dA      COMPLEX_16 array, dimension (ldda,n), on the GPU device.
            The m-by-n matrix to be printed.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A. ldda >= m.

    @param[out]
    cnt_nan INTEGER*
            If non-NULL, on exit contains the number of NAN values in A.

    @param[out]
    cnt_inf INTEGER*
            If non-NULL, on exit contains the number of INF values in A.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @return
      -     >= 0:  Returns number of NAN + number of INF values.
      -     <  0:  If it returns -i, the i-th argument had an illegal value,
                   or another error occured, such as memory allocation failed.

    @ingroup magma_nan_inf
*******************************************************************************/
extern "C"
magma_int_t magma_znan_inf_gpu(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magma_int_t *cnt_nan,
    magma_int_t *cnt_inf,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if (uplo != MagmaLower && uplo != MagmaUpper && uplo != MagmaFull)
        info = -1;
    else if (m < 0)
        info = -2;
    else if (n < 0)
        info = -3;
    else if (ldda < m)
        info = -5;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;
    }
    
    magma_int_t lda = m;
    magmaDoubleComplex* A;
    magma_zmalloc_cpu( &A, lda*n );

    magma_zgetmatrix( m, n, dA, ldda, A, lda, queue );
    
    magma_int_t cnt = magma_znan_inf( uplo, m, n, A, lda, cnt_nan, cnt_inf );
    
    magma_free_cpu( A );
    return cnt;
}
