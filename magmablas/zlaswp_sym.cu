/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       
       @author Stan Tomov
       @author Mathieu Faverge
       @author Ichitaro Yamazaki
       @author Mark Gates
*/
#include "magma_internal.h"

// MAX_PIVOTS is maximum number of pivots to apply in each kernel launch
// NTHREADS is number of threads in a block
// 64 and 256 are better on Kepler; 
//#define MAX_PIVOTS 64
//#define NTHREADS   256
#define MAX_PIVOTS 32
#define NTHREADS   64

typedef struct {
    magmaDoubleComplex *dA;
    int n, lda, j0, npivots;
    int ipiv[MAX_PIVOTS];
} zlaswp_sym_params_t;


// Matrix A is stored row or column-wise in dA.
// Divide matrix A into block-columns of NTHREADS columns each.
// Each GPU block processes one block-column of A.
// Each thread goes down a column of A,
// swapping rows according to pivots stored in params.
__global__ void zlaswp_sym_kernel( zlaswp_sym_params_t params )
{
    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if ( tid < params.n ) {
        for( int ii = params.j0; ii < params.npivots; ++ii ) {
            int i1 = ii;
            int i2 = params.ipiv[ii];
            // swap: i1 <-> i2
            // this thread is responsible for the tid-th element
            magmaDoubleComplex *A1 = NULL, *A2 = NULL;
            if (tid < i1) {
                // row swap: (i1,tid) <-> (i2,tid)
                A1 = params.dA + tid*params.lda + i1;
                A2 = params.dA + tid*params.lda + i2;
            } else if (tid == i1) {
                // diagonal swap: (i1,i1) <-> (i2,i2)
                A1 = params.dA + i1*params.lda + i1;
                A2 = params.dA + i2*params.lda + i2;
            } else if (tid < i2) {
                // row-col swap: (tid,i1) <-> (i2,tid)
                A1 = params.dA + i1*params.lda + tid;
                A2 = params.dA + tid*params.lda + i2;
            } else if (tid == i2) {
                // diagonal swap: done by i1-th thread
            } else if (tid > i2) {
                // column swap: (tid,i1) <-> (tid,i2)
                A1 = params.dA + i1*params.lda + tid;
                A2 = params.dA + i2*params.lda + tid;
            }

            if ( A1 != NULL && A2 != NULL) {
                magmaDoubleComplex temp = *A1;
                *A1 = *A2;
                *A2 = temp;
            }
        }
    }
}


// Launch zlaswp_sym kernel with ceil( n / NTHREADS ) blocks of NTHREADS threads each.
extern "C" void zlaswp_sym( zlaswp_sym_params_t &params, magma_queue_t queue )
{
    int blocks = magma_ceildiv(params.n,  NTHREADS);
    zlaswp_sym_kernel<<< blocks, NTHREADS, 0, queue->cuda_stream() >>>( params );
}


/***************************************************************************//**
    Purpose:
    =============
    ZLASWP_SYM applies a series of symmetric pivoting on a symmetric matrix A.
    Currently, it is only implemented for the lower-triangular part of the matrix.
    
    Arguments:
    ==========
    @param[in]
    n        INTEGER
             The number of columns of the matrix A.
    
    @param[in,out]
    dA       COMPLEX*16 array on GPU, dimension (*,*)
             On entry, the matrix of column dimension N to which the row
             interchanges will be applied.
             On exit, the permuted matrix.
    
    @param[in]
    lda      INTEGER
             Stride between elements in same column.
    
    @param[in]
    k1       INTEGER
             The first element of IPIV for which a row interchange will
             be done. (One based index.)
    
    @param[in]
    k2       INTEGER
             The last element of IPIV for which a row interchange will
             be done. (One based index.)
    
    @param[in]
    ipiv     INTEGER array, on CPU, dimension (K2*abs(INCI))
             The vector of pivot indices.  Only the elements in positions
             K1 through K2 of IPIV are accessed.
             IPIV(K) = L implies rows K and L are to be interchanged.
    
    @param[in]
    inci     INTEGER
             The increment between successive values of IPIV.
             Currently, IPIV > 0.
             TODO: If IPIV is negative, the pivots are applied in reverse order.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_laswp_sym
*******************************************************************************/
extern "C" void
magmablas_zlaswp_sym(
    magma_int_t n, magmaDoubleComplex *dA, magma_int_t lda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( n < 0 )
        info = -1;
    else if ( k1 < 0 )
        info = -4;  
    else if ( k2 < 0 || k2 < k1 )
        info = -5;
    else if ( inci <= 0 )
        info = -7;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    for( int k = k1-1; k < k2; k += MAX_PIVOTS ) {
        int npivots = min( MAX_PIVOTS, k2-k );
        // fields are:                 dA      n       lda       j0      npivots
        zlaswp_sym_params_t params = { dA, int(n), int(lda), int(k), int(k+npivots) };
        for( int j = 0; j < npivots; ++j ) {
            params.ipiv[j] = ipiv[(k+j)*inci] - 1;
        }
        zlaswp_sym( params, queue );
    }
}
