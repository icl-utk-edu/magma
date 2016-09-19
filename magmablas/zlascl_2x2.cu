/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       @author Ichitaro Yamazaki
*/
#include "magma_internal.h"

#define NB 64
#define A(i,j) (A[(i) + (j)*lda])
#define W(i,j) (W[(i) + (j)*ldw])


// each thread block does one NB x n block row of A.
// each thread does one row, starting from left edge and moving right to diagonal.
__global__ void
zlascl_2x2_lower(
    int m,
    const magmaDoubleComplex* W, int ldw,
    magmaDoubleComplex* A, int lda)
{
    int ind = blockIdx.x * NB + threadIdx.x;

    magmaDoubleComplex D21 = W( 1, 0 );
    magmaDoubleComplex D11 = MAGMA_Z_DIV( W( 1, 1 ), D21 );
    magmaDoubleComplex D22 = MAGMA_Z_DIV( W( 0, 0 ), MAGMA_Z_CONJ( D21 ) );
    double T = 1.0 / ( MAGMA_Z_REAL( D11*D22 ) - 1.0 );
    D21 = MAGMA_Z_DIV( MAGMA_Z_MAKE(T,0.0), D21 );

    if (ind < m) {
        A( ind, 0 ) = MAGMA_Z_CONJ( D21 )*( D11*W( 2+ind, 0 )-W( 2+ind, 1 ) );
        A( ind, 1 ) = D21*( D22*W( 2+ind, 1 )-W( 2+ind, 0 ) );
    }
}


// each thread block does one NB x n block row of A.
// each thread does one row, starting from right edge and moving left to diagonal.
__global__ void
zlascl_2x2_upper(
    int m,
    const magmaDoubleComplex *W, int ldw,
    magmaDoubleComplex* A, int lda)
{
    int ind = blockIdx.x * NB + threadIdx.x;

    magmaDoubleComplex D21 = W( m, 1 );
    magmaDoubleComplex D11 = MAGMA_Z_DIV( W( m+1, 1 ), MAGMA_Z_CONJ( D21 ) );
    magmaDoubleComplex D22 = MAGMA_Z_DIV( W( m, 0 ), D21 );
    double T = 1.0 / ( MAGMA_Z_REAL( D11*D22 ) - 1.0 );
    D21 = MAGMA_Z_DIV( MAGMA_Z_MAKE(T,0.0), D21 );

    if (ind < m) {
        A( ind, 0 ) = D21*( D11*W( ind, 0 )-W( ind, 1 ) );
        A( ind, 1 ) = MAGMA_Z_CONJ( D21 )*( D22*W( ind, 1 )-W( ind, 0 ) );
    }
}


/***************************************************************************//**
    Purpose
    -------
    ZLASCL_2x2 scales the M by M complex matrix A by the 2-by-2 pivot.
    TYPE specifies that A may be upper or lower triangular.

    Arguments
    ---------
    @param[in]
    type    magma_type_t
            TYPE indices the storage type of the input matrix A.
            = MagmaLower:  lower triangular matrix.
            = MagmaUpper:  upper triangular matrix.
            Other formats that LAPACK supports, MAGMA does not currently support.

    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    dW      DOUBLE PRECISION vector, dimension (2*lddw)
            The matrix containing the 2-by-2 pivot.

    @param[in]
    lddw    INTEGER
            The leading dimension of the array W.  LDDA >= max(1,M).

    @param[in,out]
    dA      COMPLEX*16 array, dimension (LDDA,N)
            The matrix to be scaled by dW.  See TYPE for the
            storage type.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,M).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_lascl_2x2
*******************************************************************************/
extern "C" void
magmablas_zlascl_2x2(
    magma_type_t type, magma_int_t m,
    magmaDoubleComplex_const_ptr dW, magma_int_t lddw,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info )
{
    *info = 0;
    if ( type != MagmaLower && type != MagmaUpper )
        *info = -1;
    else if ( m < 0 )
        *info = -2;
    else if ( ldda < max(1,m) )
        *info = -4;
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return;  //info;
    }
    
    dim3 threads( NB );
    dim3 grid( magma_ceildiv( m, NB ) );
    
    if (type == MagmaLower) {
        zlascl_2x2_lower <<< grid, threads, 0, queue->cuda_stream() >>> (m, dW, lddw, dA, ldda);
    }
    else {
        zlascl_2x2_upper <<< grid, threads, 0, queue->cuda_stream() >>> (m, dW, lddw, dA, ldda);
    }
}
