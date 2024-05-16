/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include "magma_internal.h"

// === Define what BLAS to use ============================================
#undef  magma_ztrsm
#define magma_ztrsm magmablas_ztrsm
// === End defining what BLAS to use ======================================

extern "C" magma_int_t
magma_zgetri_expert_gpu_work(
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *ipiv,
    magma_int_t *info,
    magma_mode_t mode,
    void* host_work,   magma_int_t *lwork_host,
    void* device_work, magma_int_t *lwork_device,
    magma_queue_t queue )
{
    #define dA(i, j)  (dA + (i) + (j)*ldda)
    #define dL(i, j)  (dL + (i) + (j)*lddl)

    /* Constants */
    const magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;

    /* Local variables */
    magmaDoubleComplex_ptr dL;
    magma_int_t nb = magma_get_zgetri_nb( n );
    magma_int_t j, jmax, jb, jp, lddl;

    // calculate the required workspace in bytes
    magma_int_t h_workspace_bytes = 0;
    magma_int_t d_workspace_bytes = n * nb;

    // check for workspace query
    if( *lwork_host < 0 || *lwork_device < 0 ) {
        *lwork_host   = h_workspace_bytes;
        *lwork_device = d_workspace_bytes;
        *info  = 0;
        return 0;
    }

    *info = 0;
    if (n < 0)
        *info = -1;
    else if (ldda < max(1,n))
        *info = -3;
    else if ( lwork_device < n*nb )
        *info = -10;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if ( n == 0 )
        return *info;

    if (lwork_device >= ldda*n) {
        lddl = ldda;
    }
    else {
        lddl = n;
    }
    dL = device_work;

    /* Invert the triangular factor U */
    magma_ztrtri_gpu( MagmaUpper, MagmaNonUnit, n, dA, ldda, info );
    if ( *info != 0 )
        return *info;

    jmax = ((n-1) / nb)*nb;
    for( j = jmax; j >= 0; j -= nb ) {
        jb = min( nb, n-j );

        // copy current block column of A to work space dL
        // (only needs lower trapezoid, but we also copy upper triangle),
        // then zero the strictly lower trapezoid block column of A.
        magmablas_zlacpy( MagmaFull, n-j, jb,
                          dA(j,j), ldda,
                          dL(j,0), lddl, queue );
        magmablas_zlaset( MagmaLower, n-j-1, jb, c_zero, c_zero, dA(j+1,j), ldda, queue );

        // compute current block column of Ainv
        // Ainv(:, j:j+jb-1)
        //   = ( U(:, j:j+jb-1) - Ainv(:, j+jb:n) L(j+jb:n, j:j+jb-1) )
        //   * L(j:j+jb-1, j:j+jb-1)^{-1}
        // where L(:, j:j+jb-1) is stored in dL.
        if ( j+jb < n ) {
            magma_zgemm( MagmaNoTrans, MagmaNoTrans, n, jb, n-j-jb,
                         c_neg_one, dA(0,j+jb), ldda,
                                    dL(j+jb,0), lddl,
                         c_one,     dA(0,j),    ldda, queue );
        }
        // TODO use magmablas work interface
        magma_ztrsm( MagmaRight, MagmaLower, MagmaNoTrans, MagmaUnit,
                     n, jb, c_one,
                     dL(j,0), lddl,
                     dA(0,j), ldda, queue );
    }

    // Apply column interchanges
    for( j = n-2; j >= 0; --j ) {
        jp = ipiv[j] - 1;
        if ( jp != j ) {
            magmablas_zswap( n, dA(0,j), 1, dA(0,jp), 1, queue );
        }
    }

    return *info;
}

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
    dA      COMPLEX_16 array on the GPU, dimension (LDDA,N)
            On entry, the factors L and U from the factorization
            A = P*L*U as computed by ZGETRF_GPU.
            On exit, if INFO = 0, the inverse of the original matrix A.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,N).

    @param[in]
    ipiv    INTEGER array, dimension (N)
            The pivot indices from ZGETRF; for 1 <= i <= N, row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    dwork   (workspace) COMPLEX_16 array on the GPU, dimension (MAX(1,LWORK))

    @param[in]
    lwork   INTEGER
            The dimension of the array DWORK.  LWORK >= N*NB, where NB is
            the optimal blocksize returned by magma_get_zgetri_nb(n).
    \n
            Unlike LAPACK, this version does not currently support a
            workspace query, because the workspace is on the GPU.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  if INFO = i, U(i,i) is exactly zero; the matrix is
                  singular and its cannot be computed.

    @ingroup magma_getri
*******************************************************************************/
extern "C" magma_int_t
magma_zgetri_gpu(
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *ipiv,
    magmaDoubleComplex_ptr dwork, magma_int_t lwork,
    magma_int_t *info )
{
    #define dA(i, j)  (dA + (i) + (j)*ldda)
    #define dL(i, j)  (dL + (i) + (j)*lddl)

    /* Constants */
    const magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;

    /* Local variables */
    magmaDoubleComplex_ptr dL;
    magma_int_t nb = magma_get_zgetri_nb( n );
    magma_int_t j, jmax, jb, jp, lddl;

    magma_queue_t queue = NULL;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    // query workspace
    magma_int_t lwork_host[1]   = {-1};
    magma_int_t lwork_device[1] = {-1};
    magma_zgetri_expert_gpu_work(
        n, NULL, ldda, NULL, info, MagmaNative,
        NULL, lwork_host, NULL, lwork_device, queue );

    *info = 0;
    if (n < 0)
        *info = -1;
    else if (ldda < max(1,n))
        *info = -3;
    else if ( lwork < lwork_device[0] )
        *info = -6;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    magma_zgetri_expert_gpu_work(
        n, dA, ldda, ipiv, info, MagmaNative,
        NULL, lwork_host, dwork, &lwork, queue );

    magma_queue_destroy( queue );

    return *info;
}
