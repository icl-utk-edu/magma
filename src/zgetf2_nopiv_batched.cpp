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

#include "magma_internal.h"
#include "batched_kernel_param.h"
#define PRECISION_z
/***************************************************************************//**
    Purpose
    -------
    ZGETF2 computes an LU factorization of a general M-by-N matrix A without pivoting

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
    ai      INTEGER
            Row offset for A.

    @param[in]
    aj      INTEGER
            Column offset for A.

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
    gbstep  INTEGER
            internal use.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getf2_nopiv_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgetf2_nopiv_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, magma_queue_t queue)
{
    #define dAarray(i,j) dA_array, i, j

    magma_int_t arginfo = 0;
    if (m < 0) {
        arginfo = -1;
    } else if (n < 0 ) {
        arginfo = -2;
    } else if (ai < 0 ) {
        arginfo = -4;
    } else if (aj < 0 ) {
        arginfo = -5;
    } else if (ldda < max(1,m)) {
        arginfo = -6;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // Quick return if possible
    if (m == 0 || n == 0) {
        return arginfo;
    }

    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    
    #ifdef PRECISION_z
    // reduce the register pressure in the z precision by reducing the panel width
    magma_int_t nb = (m > 512) ? 4 : 8;
    #else
    magma_int_t nb = (m > 512) ? 16 : 32;
    #endif
    if(n <= nb){
        //panel
        magma_zgetf2_nopiv_internal_batched(m, n, dAarray(ai, aj), ldda, info_array, gbstep, batchCount, queue);
    }
    else{
        magma_int_t n1 = n / 2;
        magma_int_t n2 = n-n1;
        magma_int_t min_mn1 = min(m, n1);
        // recpanel
        magma_zgetf2_nopiv_batched(m, n1, dAarray(ai, aj), ldda, info_array, gbstep, batchCount, queue);
        // trsm
        magmablas_ztrsm_recursive_batched( 
            MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit, 
            min_mn1, n2, c_one, 
            dAarray(ai,aj   ), ldda, 
            dAarray(ai,aj+n1), ldda, batchCount, queue );
        if(m-min_mn1 > 0){
            // gemm
            magma_zgemm_batched_core(
                MagmaNoTrans, MagmaNoTrans, 
                m-min_mn1, n2, n1, 
                c_neg_one, dAarray(ai+n1,aj   ), ldda, 
                           dAarray(ai   ,aj+n1), ldda, 
                c_one,     dAarray(ai+n1,aj+n1), ldda, 
                batchCount, queue );
            // recpanel
            magma_zgetf2_nopiv_batched(m-min_mn1, n2, dAarray(ai+n1, aj+n1), ldda, info_array, gbstep+n1, batchCount, queue);
        }
    }
    return 0;

    #undef dAarray
}


