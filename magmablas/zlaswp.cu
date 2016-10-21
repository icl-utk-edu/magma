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
    int npivots;
    int ipiv[MAX_PIVOTS];
} zlaswp_params_t;


// Matrix A is stored row-wise in dAT.
// Divide matrix A into block-columns of NTHREADS columns each.
// Each GPU block processes one block-column of A.
// Each thread goes down a column of A,
// swapping rows according to pivots stored in params.
__global__ void zlaswp_kernel(
    int n,
    magmaDoubleComplex *dAT, int ldda,
    zlaswp_params_t params )
{
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if ( tid < n ) {
        dAT += tid;
        magmaDoubleComplex *A1  = dAT;
        
        for( int i1 = 0; i1 < params.npivots; ++i1 ) {
            int i2 = params.ipiv[i1];
            magmaDoubleComplex *A2 = dAT + i2*ldda;
            magmaDoubleComplex temp = *A1;
            *A1 = *A2;
            *A2 = temp;
            A1 += ldda;  // A1 = dA + i1*ldx
        }
    }
}


/***************************************************************************//**
    Purpose:
    =============
    ZLASWP performs a series of row interchanges on the matrix A.
    One row interchange is initiated for each of rows K1 through K2 of A.
    
    ** Unlike LAPACK, here A is stored row-wise (hence dAT). **
    Otherwise, this is identical to LAPACK's interface.
    
    Arguments:
    ==========
    @param[in]
    n       INTEGER
            The number of columns of the matrix A.
    
    @param[in,out]
    dAT     COMPLEX*16 array on GPU, stored row-wise, dimension (LDDA,M)
            The M-by-N matrix, stored transposed as N-by-M matrix embedded in
            LDDA-by-M array. M is not given; it is implicit.
            On entry, the matrix of column dimension N to which the row
            interchanges will be applied.
            On exit, the permuted matrix.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array A. ldda >= n.
    
    @param[in]
    k1      INTEGER
            The first element of IPIV for which a row interchange will
            be done. (Fortran one-based index: 1 <= k1.)
    
    @param[in]
    k2      INTEGER
            The last element of IPIV for which a row interchange will
            be done. (Fortran one-based index: 1 <= k2.)
    
    @param[in]
    ipiv    INTEGER array, on CPU, dimension (K2*abs(INCI))
            The vector of pivot indices.  Only the elements in positions
            K1 through K2 of IPIV are accessed.
            IPIV(K) = L implies rows K and L are to be interchanged.
    
    @param[in]
    inci    INTEGER
            The increment between successive values of IPIV.
            Currently, INCI > 0.
            TODO: If INCI is negative, the pivots are applied in reverse order.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_laswp
*******************************************************************************/
// used in zgessm, zgetrf_incpiv.
extern "C" void
magmablas_zlaswp(
    magma_int_t n,
    magmaDoubleComplex_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_queue_t queue )
{
    #define dAT(i_, j_) (dAT + (i_)*ldda + (j_))
    
    magma_int_t info = 0;
    if ( n < 0 )
        info = -1;
    else if ( n > ldda )
        info = -3;
    else if ( k1 < 1 )
        info = -4;
    else if ( k2 < 1 )
        info = -5;
    else if ( inci <= 0 )
        info = -7;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    dim3 threads( NTHREADS );
    dim3 grid( magma_ceildiv( n, NTHREADS ) );
    zlaswp_params_t params;
    
    for( int k = k1-1; k < k2; k += MAX_PIVOTS ) {
        int npivots = min( MAX_PIVOTS, k2-k );
        params.npivots = npivots;
        for( int j = 0; j < npivots; ++j ) {
            params.ipiv[j] = ipiv[(k+j)*inci] - k - 1;
        }
        zlaswp_kernel
            <<< grid, threads, 0, queue->cuda_stream() >>>
            ( n, dAT(k,0), ldda, params );
    }
    
    #undef dAT
}


/******************************************************************************/
// Extended version has stride in both directions (ldx, ldy)
// to handle both row-wise and column-wise storage.

// Matrix A is stored row or column-wise in dA.
// Divide matrix A into block-columns of NTHREADS columns each.
// Each GPU block processes one block-column of A.
// Each thread goes down a column of A,
// swapping rows according to pivots stored in params.
__global__ void zlaswpx_kernel(
    int n,
    magmaDoubleComplex *dA, int ldx, int ldy,
    zlaswp_params_t params )
{
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if ( tid < n ) {
        dA += tid*ldy;
        magmaDoubleComplex *A1  = dA;
        
        for( int i1 = 0; i1 < params.npivots; ++i1 ) {
            int i2 = params.ipiv[i1];
            magmaDoubleComplex *A2 = dA + i2*ldx;
            magmaDoubleComplex temp = *A1;
            *A1 = *A2;
            *A2 = temp;
            A1 += ldx;  // A1 = dA + i1*ldx
        }
    }
}


/***************************************************************************//**
    Purpose:
    =============
    ZLASWPX performs a series of row interchanges on the matrix A.
    One row interchange is initiated for each of rows K1 through K2 of A.
    
    ** Unlike LAPACK, here A is stored either row-wise or column-wise,
       depending on ldx and ldy. **
    Otherwise, this is identical to LAPACK's interface.
    
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
    ldx      INTEGER
             Stride between elements in same column.
    
    @param[in]
    ldy      INTEGER
             Stride between elements in same row.
             For A stored row-wise,    set ldx=ldda and ldy=1.
             For A stored column-wise, set ldx=1    and ldy=ldda.
    
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

    @ingroup magma_laswp
*******************************************************************************/
extern "C" void
magmablas_zlaswpx(
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldx, magma_int_t ldy,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_queue_t queue )
{
    #define dA(i_, j_) (dA + (i_)*ldx + (j_)*ldy)
    
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
    
    dim3 threads( NTHREADS );
    dim3 grid( magma_ceildiv( n, NTHREADS ) );
    zlaswp_params_t params;
    
    for( int k = k1-1; k < k2; k += MAX_PIVOTS ) {
        int npivots = min( MAX_PIVOTS, k2-k );
        params.npivots = npivots;
        for( int j = 0; j < npivots; ++j ) {
            params.ipiv[j] = ipiv[(k+j)*inci] - k - 1;
        }
        zlaswpx_kernel
            <<< grid, threads, 0, queue->cuda_stream() >>>
            ( n, dA(k,0), ldx, ldy, params );
    }
    
    #undef dA
}


/******************************************************************************/
// This version takes d_ipiv on the GPU. Thus it does not pass pivots
// as an argument using a structure, avoiding all the argument size
// limitations of CUDA and OpenCL. It also needs just one kernel launch
// with all the pivots, instead of multiple kernel launches with small
// batches of pivots. On Fermi, it is faster than magmablas_zlaswp
// (including copying pivots to the GPU).

__global__ void zlaswp2_kernel(
    int n,
    magmaDoubleComplex *dAT, int ldda,
    int npivots,
    const magma_int_t *d_ipiv, int inci )
{
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if ( tid < n ) {
        dAT += tid;
        magmaDoubleComplex *A1  = dAT;
        
        for( int i1 = 0; i1 < npivots; ++i1 ) {
            int i2 = d_ipiv[i1*inci] - 1;  // Fortran index
            magmaDoubleComplex *A2 = dAT + i2*ldda;
            magmaDoubleComplex temp = *A1;
            *A1 = *A2;
            *A2 = temp;
            A1 += ldda;  // A1 = dA + i1*ldx
        }
    }
}


/***************************************************************************//**
    Purpose:
    =============
    ZLASWP2 performs a series of row interchanges on the matrix A.
    One row interchange is initiated for each of rows K1 through K2 of A.
    
    ** Unlike LAPACK, here A is stored row-wise (hence dAT). **
    Otherwise, this is identical to LAPACK's interface.
    
    Here, d_ipiv is passed in GPU memory.
    
    Arguments:
    ==========
    @param[in]
    n        INTEGER
             The number of columns of the matrix A.
    
    @param[in,out]
    dAT      COMPLEX*16 array on GPU, stored row-wise, dimension (LDDA,*)
             On entry, the matrix of column dimension N to which the row
             interchanges will be applied.
             On exit, the permuted matrix.
    
    @param[in]
    ldda     INTEGER
             The leading dimension of the array A.
             (I.e., stride between elements in a column.)
    
    @param[in]
    k1       INTEGER
             The first element of IPIV for which a row interchange will
             be done. (One based index.)
    
    @param[in]
    k2       INTEGER
             The last element of IPIV for which a row interchange will
             be done. (One based index.)
    
    @param[in]
    d_ipiv   INTEGER array, on GPU, dimension (K2*abs(INCI))
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

    @ingroup magma_laswp
*******************************************************************************/
extern "C" void
magmablas_zlaswp2(
    magma_int_t n,
    magmaDoubleComplex_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    magmaInt_const_ptr d_ipiv, magma_int_t inci,
    magma_queue_t queue )
{
    #define dAT(i_, j_) (dAT + (i_)*ldda + (j_))
    
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
    
    magma_int_t nb = k2-(k1-1);
    
    dim3 threads( NTHREADS );
    dim3 grid( magma_ceildiv( n, NTHREADS ) );
    zlaswp2_kernel<<< grid, threads, 0, queue->cuda_stream() >>>
        ( n, dAT(k1-1,0), ldda, nb, d_ipiv, inci );
}
