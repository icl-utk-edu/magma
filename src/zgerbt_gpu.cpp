/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Adrien REMY
*/
#include "magma_internal.h"


/******************************************************************************/
static void
init_butterfly(
        magma_int_t n,
        magmaDoubleComplex* u, magmaDoubleComplex* v)
{
    magma_int_t i;
    double u1, v1;
    for (i=0; i < n; ++i) {
        u1 = exp( (rand()/(double)RAND_MAX - 0.5)/10 );
        v1 = exp( (rand()/(double)RAND_MAX - 0.5)/10 );
        u[i] = MAGMA_Z_MAKE( u1, u1 );
        v[i] = MAGMA_Z_MAKE( v1, v1 );
    }
}


/***************************************************************************//**
    Purpose
    -------
    ZGERBT solves a system of linear equations
        A * X = B
    where A is a general n-by-n matrix and X and B are n-by-nrhs matrices.
    Random Butterfly Tranformation is applied on A and B, then
    the LU decomposition with no pivoting is
    used to factor A as
        A = L * U,
    where L is unit lower triangular, and U is
    upper triangular.  The factored form of A is then used to solve the
    system of equations A * X = B.

    Arguments
    ---------

    @param[in]
    gen     magma_bool_t
     -         = MagmaTrue:     new matrices are generated for U and V
     -         = MagmaFalse:    matrices U and V given as parameter are used


    @param[in]
    n       INTEGER
            The order of the matrix A.  n >= 0.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  nrhs >= 0.

    @param[in,out]
    dA      COMPLEX_16 array, dimension (LDDA,n).
            On entry, the M-by-n matrix to be factored.
            On exit, the factors L and U from the factorization
            A = L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,n).

    @param[in,out]
    dB      COMPLEX_16 array, dimension (LDDB,nrhs)
            On entry, the right hand side matrix B.
            On exit, the solution matrix X.

    @param[in]
    lddb    INTEGER
            The leading dimension of the array B.  LDDB >= max(1,n).

    @param[in,out]
    U       COMPLEX_16 array, dimension (2,n)
            Random butterfly matrix, if gen = MagmaTrue U is generated and returned as output;
            else we use U given as input.
            CPU memory

    @param[in,out]
    V       COMPLEX_16 array, dimension (2,n)
            Random butterfly matrix, if gen = MagmaTrue V is generated and returned as output;
            else we use U given as input.
            CPU memory

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.

    @ingroup magma_gerbt
*******************************************************************************/
extern "C" magma_int_t
magma_zgerbt_gpu(
    magma_bool_t gen, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magmaDoubleComplex *U, magmaDoubleComplex *V,
    magma_int_t *info)
{
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)

    /* Function Body */
    *info = 0;
    if ( ! (gen == MagmaTrue) &&
         ! (gen == MagmaFalse) ) {
        *info = -1;
    }
    else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (ldda < max(1,n)) {
        *info = -5;
    } else if (lddb < max(1,n)) {
        *info = -7;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (nrhs == 0 || n == 0)
        return *info;

    magmaDoubleComplex_ptr dU=NULL, dV=NULL;
    magma_int_t j;

    /* Allocate memory for the buterfly matrices */
    if (MAGMA_SUCCESS != magma_zmalloc( &dU, 2*n ) ||
        MAGMA_SUCCESS != magma_zmalloc( &dV, 2*n )) {
        magma_free( dU );
        magma_free( dV );
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }

    magma_queue_t queue;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    /* Initialize Butterfly matrix on the CPU */
    if (gen == MagmaTrue)
        init_butterfly( 2*n, U, V );

    /* Copy the butterfly to the GPU */
    magma_zsetvector( 2*n, U, 1, dU, 1, queue );
    magma_zsetvector( 2*n, V, 1, dV, 1, queue );

    /* Perform Partial Random Butterfly Transformation on the GPU */
    magmablas_zprbt( n, dA, ldda, dU, dV, queue );

    /* Compute U^T * b on the GPU*/
    magmablas_zprbt_mtv(n, nrhs, dU, dB, lddb, queue);

    magma_queue_destroy( queue );
    magma_free( dU );
    magma_free( dV );

    return *info;
}
