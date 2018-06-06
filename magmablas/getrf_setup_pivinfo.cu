    /*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Azzam Haidar
       @author Tingxing Dong

*/
#include "magma_internal.h"
#include "batched_kernel_param.h"

// =============================================================================
// Auxiliary routine to compute piv final destination for the current step

/******************************************************************************/
static __device__ void setup_pivinfo_devfunc(magma_int_t *pivinfo, magma_int_t *ipiv, int m, int nb)
{
    int tid = threadIdx.x;   
    int nchunk = magma_ceildiv( m, MAX_NTHREADS );
    
    // initialize pivinfo (could be done in a separate kernel using multiple thread block
    for (int s =0; s < nchunk; s++)
    {
        if ( (tid + s * MAX_NTHREADS < m) && (tid < MAX_NTHREADS) )
            pivinfo[tid + s * MAX_NTHREADS] = tid + s * MAX_NTHREADS + 1;
    }
    __syncthreads();

    if (tid == 0)
    {   
        int i, itsreplacement, mynewrowid;
        for (i=0; i < nb; i++) {
            mynewrowid          = ipiv[i]-1; //-1 to get the index in C
            itsreplacement      = pivinfo[mynewrowid];
            pivinfo[mynewrowid] = pivinfo[i];
            pivinfo[i]          = itsreplacement;
        }
    }
}


/******************************************************************************/
static __device__ void setup_pivinfo_sm_devfunc(magma_int_t *pivinfo, magma_int_t *ipiv, int m, int nb)
{
    const int tx  = threadIdx.x;   
    const int nth = blockDim.x;
    __shared__ int spivinfo[10240];    // 40 KB of shared memory
    
    int nchunk = magma_ceildiv( m, nth);
    int m_ = m - (nchunk-1) * nth;
    
    // initialize spivinfo 
    for(int s = 0; s < m-nth; s+= nth){
        spivinfo[ s + tx ] = s + tx + 1;
    }
    if( tx < m_){
        spivinfo[ (nchunk-1) * nth + tx ] = (nchunk-1) * nth + tx + 1;
    }
    __syncthreads();

    if (tx == 0)
    {   
        int i, itsreplacement, mynewrowid;
        for (i=0; i < nb; i++) {
            mynewrowid           = ipiv[i]-1; //-1 to get the index in C
            itsreplacement       = spivinfo[mynewrowid];
            spivinfo[mynewrowid] = spivinfo[i];
            spivinfo[i]          = itsreplacement;
        }
    }
    __syncthreads();
    // write pivinfo 
    for(int s = 0; s < m-nth; s+= nth){
        pivinfo[ s + tx] = spivinfo[ s + tx ];
    }
    if( tx < m_){
        pivinfo[ (nchunk-1) * nth + tx ] = (magma_int_t)(spivinfo[ (nchunk-1) * nth + tx ]);
    }
}


/******************************************************************************/
__global__ void setup_pivinfo_kernel_batched(magma_int_t **pivinfo_array, magma_int_t **ipiv_array, int ipiv_offset, int m, int nb)
{
    int batchid = blockIdx.x;
    setup_pivinfo_devfunc(pivinfo_array[batchid], ipiv_array[batchid]+ipiv_offset, m, nb);
}


/******************************************************************************/
__global__ void setup_pivinfo_kernel(magma_int_t *pivinfo, magma_int_t *ipiv, int m, int nb)
{
    setup_pivinfo_devfunc(pivinfo, ipiv, m, nb);
}


/******************************************************************************/
__global__ void setup_pivinfo_sm_kernel(magma_int_t *pivinfo, magma_int_t *ipiv, int m, int nb)
{
    setup_pivinfo_sm_devfunc(pivinfo, ipiv, m, nb);
}


/******************************************************************************/
extern "C" void
setup_pivinfo_batched( magma_int_t **pivinfo_array, magma_int_t **ipiv_array, magma_int_t ipiv_offset, 
                         magma_int_t m, magma_int_t nb, 
                         magma_int_t batchCount,
                         magma_queue_t queue)
{
    if (nb == 0 ) return;
    setup_pivinfo_kernel_batched
        <<< batchCount, min(m, MAX_NTHREADS), 0, queue->cuda_stream() >>>
        (pivinfo_array, ipiv_array, ipiv_offset, m, nb);
}


/******************************************************************************/
extern "C" void
setup_pivinfo( magma_int_t *pivinfo, magma_int_t *ipiv, 
                 magma_int_t m, magma_int_t nb, 
                 magma_queue_t queue)
{
    if (nb == 0 ) return;
    if( m > 10240 ){
        setup_pivinfo_kernel<<< 1, min(m, MAX_NTHREADS), 0, queue->cuda_stream() >>>(pivinfo, ipiv, m, nb);
    }
    else{
        setup_pivinfo_sm_kernel<<< 1, min(m, MAX_NTHREADS), 0, queue->cuda_stream() >>>(pivinfo, ipiv, m, nb);
    }
}


// =============================================================================
// Auxiliary routine to adjust ipiv

/******************************************************************************/
static __device__ void adjust_ipiv_devfunc(magma_int_t *ipiv, int m, int offset)
{
    int tid = threadIdx.x;
    if (tid < m)
    {
        //printf("ipiv[%d]: %d -> %d\n", tid, ipiv[tid], ipiv[tid]+offset);
        ipiv[tid] += offset;
    }
}


/******************************************************************************/
__global__ void adjust_ipiv_kernel_batched(magma_int_t **ipiv_array, int ipiv_offset, int m, int offset)
{
    int batchid = blockIdx.x;
    adjust_ipiv_devfunc(ipiv_array[batchid] + ipiv_offset, m, offset);
}


/******************************************************************************/
__global__ void adjust_ipiv_kernel(magma_int_t *ipiv, int m, int offset)
{
    adjust_ipiv_devfunc(ipiv, m, offset);
}


/******************************************************************************/
extern "C" void
adjust_ipiv_batched( magma_int_t **ipiv_array, magma_int_t ipiv_offset, 
                         magma_int_t m, magma_int_t offset, 
                         magma_int_t batchCount, magma_queue_t queue)
{
    if (offset == 0 ) return;
    if ( m  > MAX_NTHREADS) 
    {
        fprintf( stderr, "%s: m=%lld > %lld, not supported\n",
                 __func__, (long long) m, (long long) MAX_NTHREADS );
        return;
    }
    adjust_ipiv_kernel_batched
        <<< batchCount, m, 0, queue->cuda_stream() >>>
        (ipiv_array, ipiv_offset, m, offset);
}


/******************************************************************************/
extern "C" void
adjust_ipiv( magma_int_t *ipiv, 
                 magma_int_t m, magma_int_t offset, 
                 magma_queue_t queue)
{
    if (offset == 0 ) return;
    if ( m  > 1024) 
    {
        fprintf( stderr, "%s: m=%lld > %lld, not supported\n",
                 __func__, (long long) m, (long long) MAX_NTHREADS );
        return;
    }
    adjust_ipiv_kernel
        <<< 1, m, 0, queue->cuda_stream() >>>
        (ipiv, m, offset);
}
