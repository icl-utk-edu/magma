/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
*/
#include "magma_internal.h"

#define MB 64
#define NB 160


// each thread block does one NB x n block row of A.
// each thread does one row, starting from left edge and moving right to diagonal.
__global__ void
zlascl_diag_lower(
    int m, int n,
    const magmaDoubleComplex* D, int ldd,
    magmaDoubleComplex*       A, int lda)
{
    int ind_x = blockIdx.x * MB + threadIdx.x;
    int ind_y = blockIdx.y * NB;

    A += ind_x;
    if (ind_x < m) {
        for (int j=ind_y; j < min(ind_y+NB, n); j++ ) {
            A[j*lda] = MAGMA_Z_DIV( A[j*lda], D[j + j*ldd] );
        }
    }
}


// each thread block does one NB x n block row of A.
// each thread does one row, starting from right edge and moving left to diagonal.
__global__ void
zlascl_diag_upper(
    int m, int n,
    const magmaDoubleComplex* D, int ldd,
    magmaDoubleComplex*       A, int lda)
{
    int ind_x = blockIdx.x * MB + threadIdx.x;
    int ind_y = blockIdx.y * NB;

    A += ind_x;
    if (ind_x < m) {
        for (int j=ind_y; j < min(ind_y+NB, n); j++ ) {
            A[j*lda] = MAGMA_Z_DIV( A[j*lda], D[ind_x + ind_x*ldd] );
        }
    }
}


/***************************************************************************//**
    Purpose
    -------
    ZLASCL_DIAG scales the M by N complex matrix A by the real diagonal matrix dD.
    TYPE specifies that A may be upper triangular or lower triangular.

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
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in]
    dD      DOUBLE PRECISION vector, dimension (LDDD,M)
            The matrix storing the scaling factor on its diagonal.

    @param[in]
    lddd    INTEGER
            The leading dimension of the array D.

    @param[in,out]
    dA      COMPLEX*16 array, dimension (LDDA,N)
            The matrix to be scaled by dD.  See TYPE for the
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

    @ingroup magma_lascl_diag
*******************************************************************************/
extern "C" void
magmablas_zlascl_diag(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dD, magma_int_t lddd,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info )
{
    *info = 0;
    if ( type != MagmaLower && type != MagmaUpper )
        *info = -1;
    else if ( m < 0 )
        *info = -2;
    else if ( n < 0 )
        *info = -3;
    else if ( lddd < max(1,m) )
        *info = -5;
    else if ( ldda < max(1,m) )
        *info = -7;
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return;  //info;
    }
    
    dim3 threads( MB );
    dim3 grid( magma_ceildiv( m, MB ), magma_ceildiv( n, NB ) );
    
    if (type == MagmaLower) {
        zlascl_diag_lower
            <<< grid, threads, 0, queue->cuda_stream() >>>
            (m, n, dD, lddd, dA, ldda);
    }
    else if (type == MagmaUpper) {
        zlascl_diag_upper
            <<< grid, threads, 0, queue->cuda_stream() >>>
            (m, n, dD, lddd, dA, ldda);
    }
}
