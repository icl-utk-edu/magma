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
    magma_int_t idx;
    double u1, v1;
    for (idx=0; idx < n; idx++) {
        u1 = exp((((rand() * 1.0)/RAND_MAX)-0.5)/10);
        v1 = exp((((rand() * 1.0)/RAND_MAX)-0.5)/10);
        u[idx] = MAGMA_Z_MAKE(u1,u1);

        v[idx] = MAGMA_Z_MAKE(v1,v1);
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

    This is a batched version that solves batchCount matrices in parallel.
    dA, dB, and info become arrays with one entry per matrix.

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
    dA_array    Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU, dimension (LDDA,N).
            On entry, each pointer is an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= max(1,M).


    @param[in,out]
    dB_array   Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU, dimension (LDDB,N).
            On entry, each pointer is an right hand side matrix B.
            On exit, each pointer is the solution matrix X.


    @param[in]
    lddb    INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).

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

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_gerbt_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgerbt_batched(
    magma_bool_t gen, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magmaDoubleComplex **dB_array, magma_int_t lddb,
    magmaDoubleComplex *U, magmaDoubleComplex *V,
    magma_int_t *info, magma_int_t batchCount, magma_queue_t queue)
{
    magmaDoubleComplex *du, *dv;

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

    /* Allocate memory for the buterfly matrices */
    if (MAGMA_SUCCESS != magma_zmalloc( &du, 2*n )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    if (MAGMA_SUCCESS != magma_zmalloc( &dv, 2*n )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }

    /* Initialize Butterfly matrix on the CPU*/
    if (gen == MagmaTrue)
        init_butterfly(2*n, U, V);

    /* Copy the butterfly to the GPU */
    magma_zsetvector( 2*n, U, 1, du, 1, queue );
    magma_zsetvector( 2*n, V, 1, dv, 1, queue );

    /* Perform Partial Random Butterfly Transformation on the GPU*/
    magmablas_zprbt_batched(n, dA_array, ldda, du, dv, batchCount, queue);

    /* Compute U^T.b on the GPU*/
    magmablas_zprbt_mtv_batched(n, nrhs, du, dB_array, lddb, batchCount, queue);

    magma_free( du );
    magma_free( dv );

    return *info;
}
