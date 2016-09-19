/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds
       @author Mark Gates
*/
#include "magma_internal.h"

// mixed precision generation has issues with COMPLEX, so use PRECISION_z
#define PRECISION_z

#define BLK_X 64
#define BLK_Y 32

// TODO get rid of global variable!
static __device__ int flag = 0;


/*
    Divides matrix into ceil( m/BLK_X ) x ceil( n/BLK_Y ) blocks.
    Each block has BLK_X threads.
    Each thread loops across one row, updating BLK_Y entries.
    
    Code similar to zlat2c and zlaset.
*/
__global__
void zlag2c_kernel(
    int m, int n,
    const magmaDoubleComplex *A, int lda,
    magmaFloatComplex *SA,       int ldsa,
    double rmax )
{
    magmaDoubleComplex tmp;
    double neg_rmax = - rmax;
    
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    /* check if full block-column */
    bool full = (iby + BLK_Y <= n);
    /* do only rows inside matrix */
    if ( ind < m ) {
        A  += ind + iby*lda;
        SA += ind + iby*ldsa;
        if ( full ) {
            // full block-column
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                tmp = A[j*lda];
                if (   (MAGMA_Z_REAL(tmp) < neg_rmax) || (MAGMA_Z_REAL(tmp) > rmax)
                    #if defined(PRECISION_z) || defined(PRECISION_c)
                    || (MAGMA_Z_IMAG(tmp) < neg_rmax) || (MAGMA_Z_IMAG(tmp) > rmax)
                    #endif
                    )
                {
                    flag = 1;
                }
                SA[j*ldsa] = MAGMA_C_MAKE( MAGMA_Z_REAL(tmp), MAGMA_Z_IMAG(tmp) );
            }
        }
        else {
            // partial block-column
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                tmp = A[j*lda];
                if (   (MAGMA_Z_REAL(tmp) < neg_rmax) || (MAGMA_Z_REAL(tmp) > rmax)
                    #if defined(PRECISION_z) || defined(PRECISION_c)
                    || (MAGMA_Z_IMAG(tmp) < neg_rmax) || (MAGMA_Z_IMAG(tmp) > rmax)
                    #endif
                    )
                {
                    flag = 1;
                }
                SA[j*ldsa] = MAGMA_C_MAKE( MAGMA_Z_REAL(tmp), MAGMA_Z_IMAG(tmp) );
            }
        }
    }
}


/***************************************************************************//**
    Purpose
    -------
    ZLAG2C converts a double-complex matrix, A,
                 to a single-complex matrix, SA.
    
    RMAX is the overflow for the single-complex arithmetic.
    ZLAG2C checks that all the entries of A are between -RMAX and
    RMAX. If not, the conversion is aborted and a flag is raised.
    
    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of lines of the matrix A.  m >= 0.
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  n >= 0.
    
    @param[in]
    A       COMPLEX_16 array, dimension (LDA,n)
            On entry, the m-by-n coefficient matrix A.
    
    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,m).
    
    @param[out]
    SA      COMPLEX array, dimension (LDSA,n)
            On exit, if INFO=0, the m-by-n coefficient matrix SA;
            if INFO > 0, the content of SA is unspecified.
    
    @param[in]
    ldsa    INTEGER
            The leading dimension of the array SA.  LDSA >= max(1,m).
    
    @param[out]
    info    INTEGER
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     = 1:  an entry of the matrix A is greater than the COMPLEX
                  overflow threshold, in this case, the content
                  of SA on exit is unspecified.
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_lag2
*******************************************************************************/
extern "C" void
magmablas_zlag2c(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr A, magma_int_t lda,
    magmaFloatComplex_ptr SA,       magma_int_t ldsa,
    magma_queue_t queue,
    magma_int_t *info )
{
    *info = 0;
    if ( m < 0 )
        *info = -1;
    else if ( n < 0 )
        *info = -2;
    else if ( lda < max(1,m) )
        *info = -4;
    else if ( ldsa < max(1,m) )
        *info = -6;
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return; //*info;
    }

    /* quick return */
    if ( m == 0 || n == 0 ) {
        return;
    }
    
    double rmax = (double)lapackf77_slamch("O");

    dim3 threads( BLK_X, 1 );
    dim3 grid( magma_ceildiv( m, BLK_X ), magma_ceildiv( n, BLK_Y ) );
    cudaMemcpyToSymbol( flag, info, sizeof(flag) );    // flag = 0
    
    zlag2c_kernel<<< grid, threads, 0, queue->cuda_stream() >>>( m, n, A, lda, SA, ldsa, rmax );
    
    cudaMemcpyFromSymbol( info, flag, sizeof(flag) );  // info = flag
}
