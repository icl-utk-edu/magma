/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah
       
       @precisions normal z -> s d c
*/
#include "magma_internal.h"
#include "batched_kernel_param.h"

/***************************************************************************//**
    Purpose
    -------
    ZGETRI computes the inverse of a matrix using the LU factorization
    computed by ZGETRF. This method inverts U and then computes inv(A) by
    solving the system inv(A)*L = inv(U) for inv(A).
    
    Note that it is generally both faster and more accurate to use ZGESV,
    or ZGETRF and ZGETRS, to solve the system AX = B, rather than inverting
    the matrix and multiplying to form X = inv(A)*B. Only in special
    instances should an explicit inverse be computed with this routine.

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in,out]
    dA_array Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU, dimension (LDDA,N)
            On entry, the factors L and U from the factorization
            A = P*L*U as computed by ZGETRF_GPU.
            On exit, if INFO = 0, the inverse of the original matrix A.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,N).

    @param[in]
    dipiv_array Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (N)
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    dinvA_array Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU, dimension (LDDIA,N)
            It contains the inverse of the matrix
  
    @param[in]
    lddia   INTEGER
            The leading dimension of the array invA_array.  LDDIA >= max(1,N).

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.
                  
    @ingroup magma_getri_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgetri_outofplace_batched( magma_int_t n, 
                  magmaDoubleComplex **dA_array, magma_int_t ldda,
                  magma_int_t **dipiv_array, 
                  magmaDoubleComplex **dinvA_array, magma_int_t lddia,
                  magma_int_t *info_array,
                  magma_int_t batchCount, magma_queue_t queue)
{
#define dAarray(i,j)       dA_array, i, j
#define dinvAarray(i,j)    dinvA_array, i, j

    /* Local variables */
    magma_int_t info = 0;
    if (n < 0)
        info = -1;
    else if (ldda < max(1,n))
        info = -3;
    else if (lddia < max(1,n))
        info = -6;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;
    }

    /* Quick return if possible */
    if ( n == 0 )
        return info;

    magma_int_t ib, j;
    magma_int_t nb = 256; 

    // set dinvdiagA to identity
    magmablas_zlaset_batched( MagmaFull, n, n, MAGMA_Z_ZERO, MAGMA_Z_ONE, dinvA_array, lddia, batchCount, queue );

    for (j = 0; j < n; j += nb) {
        ib = min(nb, n-j);
        // dinvdiagA * Piv' = I * U^-1 * L^-1 = U^-1 * L^-1 * I
        // solve lower
        magmablas_ztrsm_recursive_batched(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
                n-j, ib, MAGMA_Z_ONE, 
                dAarray(j, j), ldda, 
                dinvAarray(j, j), lddia, 
                batchCount, queue);

        // solve upper
        magmablas_ztrsm_recursive_batched( MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                n, ib, MAGMA_Z_ONE, 
                dAarray(0, 0), ldda, 
                dinvAarray(0, j), lddia, 
                batchCount, queue);
    }

    // Apply column interchanges
    magma_zlaswp_columnserial_batched( n, dinvA_array, lddia, max(1,n-1), 1, dipiv_array, batchCount, queue );

    magma_queue_sync(queue);

    return info;
#undef dAarray
#undef dinvAarray
}
