/*                                                                                                     
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date                                                                        

       @author Hadeer Farahat
       @author Stan Tomov

       @precisions normal z -> s d c 
*/
#include "magma_internal.h"
#include "commonblas_d.h"
#include "magma_templates.h"

#define NTHREADS    128
#define NBLOCKS      40

__global__ void
zheinertia_upper_kernel(int n, magmaDoubleComplex_const_ptr dA, int ldda, magma_int_t *ipiv, int *dneig)
{
    const int tx  = threadIdx.x;
    const int blk = blockIdx.x;
    int peig = 0, neig = 0, zeig = 0;
    double diag, t=0.0;
    int i=0, k, nk, count, sc = ceil((double)n/(NBLOCKS*NTHREADS)); 

    __shared__ int pe[NTHREADS], ne[NTHREADS], ze[NTHREADS];
    
    // unrolling iteration i=0
    k = (tx + blk*NTHREADS)*sc;
    if (k<n && k-1>=0 )
        if ( ipiv[k]-1 < 0 && ipiv[k-1] == ipiv[k]) {
            count =1;
            nk = k-2;
            // check all the previous pivot values 
            while (nk >=0 && ipiv[nk] == ipiv[k] ){
                count ++;
                nk--;
            }
            // if count is odd, it means that the current pivot is a second element of a 2-by-2 diagonal block
            if ( count%2 == 1 ){
                diag = MAGMA_Z_ABS(dA[(k-1)+k*ldda]);
                t=0.0;
                if (diag > 0.) 
                    peig++;
                else if (diag < 0.)
                    neig++;
                else
                    zeig++;
                i = 1;
            }
        }
    
    // Each thread computes its part of the intertia (sc columns)
    #pragma unroll
    for(i=i; i<sc; i++){
        k=((tx + blk*NTHREADS)*sc)+i;
        if (k>=n)
            break;
        diag = MAGMA_Z_REAL(dA[k+k*ldda]);
        if (ipiv[k]-1 < 0){   
            if (t != 0.) {
                diag = t;
                t = 0.;
            } else {
                t = MAGMA_Z_ABS( dA[k+(k+1)*ldda] ); 
                diag = (diag/t) * MAGMA_Z_REAL( dA[(k+1)*(1+ldda)] ) - t; 
            }
        }

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

__global__ void
zheinertia_lower_kernel(int n, magmaDoubleComplex_const_ptr dA, int ldda, magma_int_t *ipiv, int *dneig)
{
    const int tx  = threadIdx.x;
    const int blk = blockIdx.x;
    int peig = 0, neig = 0, zeig = 0;
    double diag, t=0.0;
    int i=0, k, nk, count, sc = ceil((double)n/(NBLOCKS*NTHREADS));

    __shared__ int pe[NTHREADS], ne[NTHREADS], ze[NTHREADS];

    // unrolling iteration i=0
    k = (tx + blk*NTHREADS)*sc;
    if (k<n && k-1>=0 )
        if ( ipiv[k]-1 < 0 && ipiv[k-1] == ipiv[k]) {
            count =1;
            nk = k-2;
            // check all the previous pivot values
            while (nk >=0 && ipiv[nk] == ipiv[k] ){
                count ++;
                nk--;
            }
            // if count is odd, it means that the current pivot is a second element of a 2-by-2 diagonal block
            if ( count%2 == 1 ){
                diag = MAGMA_Z_ABS(dA[(k-1)*ldda+k]);
                t=0.0;
                if (diag > 0.)
                    peig++;
                else if (diag < 0.)
                    neig++;
                else
                    zeig++;
                i = 1;
            }
        }

    // Each thread computes its part of the intertia (sc columns)
    #pragma unroll
    for(i=i; i<sc; i++){
        k=((tx + blk*NTHREADS)*sc)+i;
        if (k>=n)
            break;
        diag = MAGMA_Z_REAL(dA[k+k*ldda]);
        if (ipiv[k]-1 < 0){
            if (t != 0.) {
                diag = t;
                t = 0.;
            } else {
                t = MAGMA_Z_ABS( dA[k*ldda+(k+1)] );
                diag = (diag/t) * MAGMA_Z_REAL( dA[(k+1)*(1+ldda)] ) - t;
            }
        }

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
    magmablas_zheinertia computes the inertia of a hermitian and block 
    diagonal matrix with 1-by-1 and 2-by-2 diagonal blocks. These are matrices
    comming from the Bunch-Kaufman with diagonal pivoting factorizations 
    (the ZHETRF routine). 
                        
    Arguments
    ----------
    @param[in] 
    n       INTEGER.
            On entry, N specifies the order of the matrix A. 
            N must be at least zero.
    
    @param[in]
    dA      COMPLEX_16 array of DIMENSION ( LDDA, n ).
            The input matrix A with 1-by-1 and 2-by-2 diagonal block entries 
            for which the inertia is computed. 
 
    @param[in] 
    ldda    INTEGER.
            On entry, LDDA specifies the leading dimension of A.
            LDDA must be at least max( 1, n ). 

    @param[in]
    ipiv    INTEGER array, dimension (N) 
            The pivot vector from dsytrf.

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
magmablas_zheinertia(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda, 
    magma_int_t *ipiv,
    int *dneig, 
    magma_queue_t queue )
{
    /*
     * Test the input parameters.
     */
    magma_int_t info = 0;
    bool upper = (uplo == MagmaUpper);
    if (! upper && uplo != MagmaLower) {
        info = -1;
    } else if ( n < 0 ) {
        info = -2;
    } else if ( ldda < max(1, n) ) {
        info = -4;
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

    if (upper)
        zheinertia_upper_kernel<<<grid, threads, 0, queue->cuda_stream() >>>
            (n, dA, ldda, ipiv, dneig);
    else
        zheinertia_lower_kernel<<<grid, threads, 0, queue->cuda_stream() >>>
            (n, dA, ldda, ipiv, dneig);

    return info;
}

// end magmablas_zheinertia
