/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       @author Theo Mary
*/
#include "magma_internal.h"

#define NB 64


// each thread block does one NB x n block row of A.
// each thread does one row, starting from left edge and moving right.
__global__ void
zlascl2_full(int m, int n, const double* D, magmaDoubleComplex* A, int lda)
{
    int ind = blockIdx.x * NB + threadIdx.x;

    double mul = D[ind];
    A += ind;
    if (ind < m) {
        for (int j=0; j < n; j++ )
            A[j*lda] *= mul;
    }
}


// each thread block does one NB x n block row of A.
// each thread does one row, starting from left edge and moving right to diagonal.
__global__ void
zlascl2_lower(int m, int n, const double* D, magmaDoubleComplex* A, int lda)
{
    int ind = blockIdx.x * NB + threadIdx.x;

    int break_d = (ind < n) ? ind : n-1;

    double mul = D[ind];
    A += ind;
    if (ind < m) {
        for (int j=0; j <= break_d; j++ )
            A[j*lda] *= mul;
    }
}


// each thread block does one NB x n block row of A.
// each thread does one row, starting from right edge and moving left to diagonal.
__global__ void
zlascl2_upper(int m, int n, const double *D, magmaDoubleComplex* A, int lda)
{
    int ind = blockIdx.x * NB + threadIdx.x;

    double mul = D[ind];
    A += ind;
    if (ind < m) {
        for (int j=n-1; j >= ind; j--)
            A[j*lda] *= mul;
    }
}


/***************************************************************************//**
    Purpose
    -------
    ZLASCL2 scales the M by N complex matrix A by the real diagonal matrix dD.
    TYPE specifies that A may be full, upper triangular, lower triangular.

    Arguments
    ---------
    @param[in]
    type    magma_type_t
            TYPE indices the storage type of the input matrix A.
            = MagmaFull:   full matrix.
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
    dD      DOUBLE PRECISION vector, dimension (M)
            The diagonal matrix containing the scalar factors. Stored as a vector.

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

    @see magma_zlascl_diag 
    @ingroup magma_lascl_diag
*******************************************************************************/
extern "C" void
magmablas_zlascl2(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dD,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info )
{
    *info = 0;
    if ( type != MagmaLower && type != MagmaUpper && type != MagmaFull )
        *info = -1;
    else if ( m < 0 )
        *info = -2;
    else if ( n < 0 )
        *info = -3;
    else if ( ldda < max(1,m) )
        *info = -5;
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return;  //info;
    }
    
    dim3 grid( magma_ceildiv( m, NB ) );
    dim3 threads( NB );
    
    if (type == MagmaLower) {
        zlascl2_lower <<< grid, threads, 0, queue->cuda_stream() >>> (m, n, dD, dA, ldda);
    }
    else if (type == MagmaUpper) {
        zlascl2_upper <<< grid, threads, 0, queue->cuda_stream() >>> (m, n, dD, dA, ldda);
    }
    else if (type == MagmaFull) {
        zlascl2_full  <<< grid, threads, 0, queue->cuda_stream() >>> (m, n, dD, dA, ldda);
    }
}
