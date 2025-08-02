/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds

*/
#include "magmasparse_internal.h"

#define blksize 512


// TODO get rid of global variable!
__device__ int flag = 0;

__global__ void
magmaint_clag2z_sparse(  int M, int N,
                  const magmaFloatComplex *SA, int ldsa,
                  magmaDoubleComplex *A,       int lda,
                  double RMAX )
{
    int inner_bsize = blockDim.x;
    int outer_bsize = inner_bsize * 512;
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
            // global thread index

    if( thread_id < M ){
        for( int i= outer_bsize * blockIdx.x  + threadIdx.x;
            i<min( M, outer_bsize * ( blockIdx.x + 1));  i+=inner_bsize){
            A[i] = cuComplexFloatToDouble( SA[i] );
        }
    }
}

/**
    Purpose
    -------
    CLAG2Z converts a COMPLEX matrix SA to a COMPLEX_16
    matrix A.
    
    RMAX is the overflow for the COMPLEX arithmetic.
    CLAG2Z checks that all the entries of A are between -RMAX and
    RMAX. If not the convertion is aborted and a flag is raised.
        
    Arguments
    ---------
    @param[in]
    M       INTEGER
            The number of lines of the matrix A.  M >= 0.
    
    @param[in]
    N       INTEGER
            The number of columns of the matrix A.  N >= 0.
    
    @param[in]
    SA      COMPLEX array, dimension (LDSA,N)
            On entry, the M-by-N coefficient matrix SA.
    
    @param[in]
    ldsa    INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).
    
    @param[out]
    A       COMPLEX_16 array, dimension (LDA,N)
            On exit, if INFO=0, the M-by-N coefficient matrix A; if
            INFO>0, the content of A is unspecified.
    
    @param[in]
    lda     INTEGER
            The leading dimension of the array SA.  LDSA >= max(1,M).
    
    @param[in]
    queue       magma_queue_t
                Queue to execute in.
                
    @param[out]
    info    INTEGER
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     = 1:  an entry of the matrix A is greater than the COMPLEX
                  overflow threshold, in this case, the content
                  of SA in exit is unspecified.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" void
magmablas_clag2z_sparse(
    magma_int_t M, magma_int_t N,
    const magmaFloatComplex *SA, magma_int_t ldsa,
    magmaDoubleComplex *A,       magma_int_t lda,
    magma_queue_t queue,
    magma_int_t *info )
{
    /*
    (TODO note from original dense source)
    
    Note
    ----
          - We have to provide INFO at the end that zlag2c isn't doable now.
          - Transfer a single value TO/FROM CPU/GPU
          - SLAMCH that's needed is called from underlying BLAS
          - Only used in iterative refinement
          - Do we want to provide this in the release?
    */
    
    *info = 0;
    if ( M < 0 )
        *info = -1;
    else if ( N < 0 )
        *info = -2;
    else if ( lda < max(1,M) )
        *info = -4;
    else if ( ldsa < max(1,M) )
        *info = -6;
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        //return *info;
    }
    
    double RMAX = (double)lapackf77_slamch("O");

    int block;
    dim3 dimBlock(blksize); // Number of Threads per Block
    block = (M/blksize)/blksize;
    if (block*blksize*blksize < M)
        block++;
    dim3 dimGrid(block); // Number of Blocks
   

    dim3 threads( blksize );
    dim3 grid( magma_ceildiv( M, blksize ) );
    cudaMemcpyToSymbol( flag, info, sizeof(flag) );    // flag = 0
    magmaint_clag2z_sparse<<< dimGrid , dimBlock, 0, queue->cuda_stream >>>
                                        ( M, N, SA, lda, A, ldsa, RMAX );
    cudaMemcpyFromSymbol( info, flag, sizeof(flag) );  // info = flag
}
