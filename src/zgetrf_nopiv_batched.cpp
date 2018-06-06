/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Azzam Haidar
   @author Adrien Remy

   @precisions normal z -> s d c
 */
#include <cuda_runtime.h>

#include "magma_internal.h"
#include "batched_kernel_param.h"

/***************************************************************************//**
    Purpose
    -------
    ZGETRF computes an LU factorization of a general M-by-N matrix A without pivoting

    The factorization has the form
        A = L * U
    where L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    This is a batched version that factors batchCount M-by-N matrices in parallel.
    dA, and info become arrays with one entry per matrix.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of each matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of each matrix A.  N >= 0.

    @param[in,out]
    dA_array    Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU, dimension (LDDA,N).
            On entry, each pointer is an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= max(1,M).

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

    @ingroup magma_getrf_nopiv_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgetrf_nopiv_batched(
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex **dA_array, 
        magma_int_t ldda,
        magma_int_t *info_array, 
        magma_int_t batchCount, magma_queue_t queue)
{
#define dAarray(i_, j_)    dA_array, i_, j_
   
    magma_int_t min_mn = min(m, n);
    cudaMemset(info_array, 0, batchCount*sizeof(magma_int_t));

    /* Check arguments */
    magma_int_t arginfo = 0;
    if (m < 0)
        arginfo = -1;
    else if (n < 0)
        arginfo = -2;
    else if (ldda < max(1,m))
        arginfo = -4;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0)
        if (min_mn == 0 ) return arginfo;

    if ( m >  2048 || n > 2048 ) {
        #ifndef MAGMA_NOWARNING
        printf("=========================================================================================\n"
               "   WARNING batched routines are designed for small sizes. It might be better to use the\n"
               "   Native/Hybrid classical routines if you want good performance.\n"
               "=========================================================================================\n");
        #endif
    }


    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magma_int_t nb, recnb, ib, i, pm;
    magma_get_zgetrf_batched_nbparam(n, &nb, &recnb);

    for (i = 0; i < min_mn; i += nb) 
    {
        ib = min(nb, min_mn-i);
        pm = m-i;

        // panel factorization
        //printf("m,n = %d, %d ... ai,aj = %d,%d \n", pm, ib, i, i);
        magma_zgetf2_nopiv_batched(pm, ib, dAarray(i,i), ldda, info_array, i, batchCount, queue);
        if (arginfo != 0 ) goto fin;

        if ( (i + ib) < n){
            magmablas_ztrsm_recursive_batched( 
            MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit, 
            ib, n-i-ib, c_one, 
            dAarray(i,i   ), ldda, 
            dAarray(i,i+ib), ldda, batchCount, queue );

            if ( (i + ib) < m){    
                magma_zgemm_batched_core( MagmaNoTrans, MagmaNoTrans, m-i-ib, n-i-ib, ib, 
                                         c_neg_one, dAarray(i+ib, i   ), ldda, 
                                                    dAarray(i   , i+ib), ldda, 
                                         c_one,     dAarray(i+ib, i+ib), ldda, 
                                         batchCount, queue );
            } // end of if ( (i + ib) < m) 
        } // end of if ( (i + ib) < n)
    }// end of for

fin:
    magma_queue_sync(queue);
    return arginfo;

    #undef dAarray
}
