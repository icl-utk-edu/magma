/*                                                                                                     
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date                                                                        

       @precisions d -> s

       @author Stan Tomov
*/
#include "magma_internal.h"
#include "commonblas_d.h"
#include "magma_templates.h"

#define NTHREADS    64
#define NBLOCKS     40

__global__ void
dsiinertia_kernel(int n, magmaDouble_const_ptr dA, int ldda, int *dneig)
{
    const int tx  = threadIdx.x;
    const int blk = blockIdx.x;
    int peig = 0, neig = 0, zeig = 0;

    __shared__ int pe[NTHREADS], ne[NTHREADS], ze[NTHREADS];

    // Each thread computes its part of the intertia
    for(int i=tx + blk*NTHREADS; i<n; i+= NTHREADS*NBLOCKS) {
        double diag = MAGMA_D_REAL(dA[i+i*ldda]);
        if (diag > 0.0)
            peig++;
        else if (diag < 0.0)
            neig++;
        else
            zeig++;
    }
    
    pe[tx] = peig;
    ne[tx] = neig;
    ze[tx] = zeig;

    // The threads within a thread block sum their contributions to the inertia
    magma_sum_reduce< NTHREADS >( tx, pe );
    magma_sum_reduce< NTHREADS >( tx, ne );
    magma_sum_reduce< NTHREADS >( tx, ze );

    __syncthreads();

    // Attomic sum the contributions from all theread blocks (by thread 0)
    if (tx == 0){
        atomicAdd(&dneig[0], pe[0]);
        atomicAdd(&dneig[1], ne[0]);
        atomicAdd(&dneig[2], ze[0]);
    }   
}

/***************************************************************************//**
    Purpose
    -------
    magmablas_ddiinertia computes the inertia of a real diagonal matrix. 
    If matrix entries are real, magmablas_ddiinertia considers the real
    part of the diagonal.                            
                        
    Arguments
    ----------
    @param[in] 
    n       INTEGER.
            On entry, N specifies the order of the matrix A. 
            N must be at least zero.
    
    @param[in]
    dA      DOUBLE PRECISION array of DIMENSION ( LDDA, n ).
            The input matrix A with diagonal entries for which the inertia
            is computed. If dA is real, the computation is done on the
            real  part of the diagonal.
 
    @param[in] 
    ldda    INTEGER.
            On entry, LDDA specifies the leading dimension of A.
            LDDA must be at least max( 1, n ). 

    @param[out]
    dneig   INTEGER array of DIMENSION 3 on the GPU memory.
            The number of positive, negative, and zero eigenvalues
            in this order.

    @param[in]
    queue   magma_queue_t. 
            Queue to execute in.

    @ingroup magma_hetrf
*******************************************************************************/ 

extern "C"
magma_int_t
magmablas_dsiinertia(
    magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda, 
    int *dneig, 
    magma_queue_t queue )
{
    /*
     * Test the input parameters.
     */
    magma_int_t info = 0;

    if ( n < 0 ) {
        info = -1;
    } else if ( ldda < max(1, n) ) {
        info = -3;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;
    }

    /*
     * Quick return if possible.
     */
    if (n == 0) 
        return info;

    dim3 grid( NBLOCKS, 1, 1 );
    dim3 threads( NTHREADS, 1, 1 );
    
    // Set itertia to zero
    cudaMemsetAsync(dneig, 0, 3*sizeof(int), queue->cuda_stream() );

    dsiinertia_kernel<<<grid, threads, 0, queue->cuda_stream() >>>
        (n, dA, ldda, dneig);
    
    return info;
}

// end magmablas_ddiinertia
