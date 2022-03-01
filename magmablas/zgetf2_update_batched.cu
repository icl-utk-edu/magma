/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/

#include <cuda.h>    // for CUDA_VERSION
#include "magma_internal.h"
#include "magma_templates.h"
#include "batched_kernel_param.h"

#define PRECISION_z

#define SLDA(n)    ( (((n)+1)%4) == 0 ? (n) : (n+1) )
#define spA(i,j)   spA[(j) * slda + (i)]
#define sA(i,j)     sA[(j) * slda + (i)]

////////////////////////////////////////////////////////////////////////////////
//              ZGETF2 update kernel
////////////////////////////////////////////////////////////////////////////////
template<int PN, int NB>
__global__
void
zgetf2_update_kernel_batched(
    int m, int n,
    magmaDoubleComplex **dA_array, int  Ai, int Aj, int ldda,
    magma_int_t **dpivinfo_array, magma_int_t pivinfo_i,
    magma_int_t batchCount )
{
    extern __shared__ magmaDoubleComplex zdata[];
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int nty = blockDim.y;
    const int batchid = blockIdx.x * nty + ty;
    if(batchid >= batchCount) return;

    magmaDoubleComplex* dpA      = dA_array[batchid] + Aj * ldda + Ai;
    magmaDoubleComplex* dA       = dpA + PN * ldda;
    magma_int_t* dpivinfo        = dpivinfo_array[batchid] + pivinfo_i;

    magmaDoubleComplex rA[NB] = {MAGMA_Z_ZERO};
    const int slda = SLDA(m);

    // shared memory pointers
    magmaDoubleComplex* spA  = (magmaDoubleComplex*)(zdata);
    magmaDoubleComplex* sA   = spA + nty * slda * PN;
    magmaDoubleComplex* sU   = sA  + nty * slda * NB;

    spA += ty * slda * PN;
    sA  += ty * slda * NB;
    sU  += ty * NB;

    int ib, rowid;
    magmaDoubleComplex reg;

    // read panel into spA, and pivinfo
    rowid = (int)(dpivinfo[tx] - 1);
    #pragma unroll
    for(int j = 0; j < PN; j++) {
        spA(tx,j) = dpA[j * ldda + tx];
    }
    __syncthreads();

    //////////// main loop ////////////////
    for(ib = 0; ib < (n/NB)*NB; ib+=NB) {
        // read A while swapping
        #pragma unroll
        for(int j = 0; j < NB; j++) {
            rA[j] = dA[ j * ldda + rowid ];
        }

        #if 0
        __syncthreads();
        printf("[%2d]: %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f\n",
                 tx, rA[0], rA[1], rA[2], rA[3], rA[4], rA[5], rA[6], rA[7], rA[8], rA[9], rA[10],
                 rA[11], rA[12], rA[13], rA[14], rA[15]);
        __syncthreads();
        #endif

        // apply loop
        #pragma unroll
        for(int j = 0; j < PN; j++) {
            reg = (tx <= j) ? MAGMA_Z_ZERO : spA(tx,j);

            if(tx == j) {
                #pragma unroll
                for(int jj = 0; jj < NB; jj++) {
                    sU[jj] = rA[jj];
                }
            }
            __syncthreads();

            // rank update
            #pragma unroll
            for(int jj = 0; jj < NB; jj++) {
                rA[jj] -= reg * sU[jj];
            }
            __syncthreads();

        }    // end of apply loop

        // write rA
        #pragma unroll
        for(int j = 0; j < NB; j++) {
            dA[ j * ldda + tx ] = rA[j];
        }

        // advance dA
        dA += NB*ldda;
    }    // end of main loop

    //////////// cleanup section ////////////////
    if(n - ib > 0) {
        int nn = n - ib;
        // read A
        for(int j = 0; j < nn; j++) {
            sA(tx,j) = dA[ j * ldda + rowid ];
        }
        __syncthreads();

        // apply loop
        #pragma unroll
        for(int j = 0; j < PN; j++) {
            reg = (tx <= j) ? MAGMA_Z_ZERO : spA(tx,j);

            // rank update
            #pragma unroll
            for(int jj = 0; jj < NB; jj++) {
                sA(tx,jj) -= reg * sA(j,jj);
            }
        }    // end of apply loop

        // write rA
        for(int j = 0; j < nn; j++) {
            dA[ j * ldda + tx ] = sA(tx,j);
        }
    }    // end of cleanup section
}

////////////////////////////////////////////////////////////////////////////////
//              ZGETF2 update kernel driver
////////////////////////////////////////////////////////////////////////////////
template<int PN, int NB>
static magma_int_t
magma_zgetf2_update_kernel_batched_driver(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magma_int_t **dpivinfo_array, magma_int_t pivinfo_i,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t arginfo = 0;
    const magma_int_t ntcol = max(1, 64/m);

    magma_int_t shmem = 0;
    shmem += SLDA(m) * PN  * sizeof(magmaDoubleComplex);  // spA
    shmem += SLDA(m) * NB  * sizeof(magmaDoubleComplex);  // sA (cleanup)
    shmem += NB            * sizeof(magmaDoubleComplex);  // sU
    shmem *= ntcol;
    magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    magma_int_t nthreads = m;
    dim3 grid(gridx, 1, 1);
    dim3 threads(nthreads, ntcol, 1);

    // get max. dynamic shared memory on the GPU
    magma_int_t nthreads_max, shmem_max = 0;
    cudaDeviceGetAttribute (&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device);
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(zgetf2_update_kernel_batched<PN, NB>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    magma_int_t total_threads = nthreads * ntcol;
    if ( total_threads > nthreads_max || shmem > shmem_max ) {
        //printf("error: kernel %s requires too many threads or too much shared memory\n", __func__);
        arginfo = -100;
        return arginfo;
    }

    void *kernel_args[] = {&m, &n, &dA_array, &Ai, &Aj, &ldda, &dpivinfo_array, &pivinfo_i, &batchCount};
    cudaError_t e = cudaLaunchKernel((void*)zgetf2_update_kernel_batched<PN, NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());
    if( e != cudaSuccess ) {
        //printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
        arginfo = -100;
    }

    return arginfo;
}

////////////////////////////////////////////////////////////////////////////////
//              ZGETF2 update kernel driver
//              instantiations based on nb
////////////////////////////////////////////////////////////////////////////////
template<int PN>
static magma_int_t
magma_zgetf2_update_NB_batched(
    magma_int_t m, magma_int_t n, magma_int_t nb,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magma_int_t **dpivinfo_array, magma_int_t pivinfo_i,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    switch(nb) {
        case 1: arginfo = magma_zgetf2_update_kernel_batched_driver<PN, 1>( m, n, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        case 2: arginfo = magma_zgetf2_update_kernel_batched_driver<PN, 2>( m, n, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        case 4: arginfo = magma_zgetf2_update_kernel_batched_driver<PN, 4>( m, n, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        case 8: arginfo = magma_zgetf2_update_kernel_batched_driver<PN, 8>( m, n, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //#if defined(MAGMA_HAVE_CUDA) && !defined(PRECISION_z)
        case 16: arginfo = magma_zgetf2_update_kernel_batched_driver<PN,16>( m, n, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        case 32: arginfo = magma_zgetf2_update_kernel_batched_driver<PN,32>( m, n, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //#endif
        default: arginfo = -100;
    }
    return arginfo;
}

////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_zgetf2_update_batched(
    magma_int_t m, magma_int_t panel_n, magma_int_t n, magma_int_t nb,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magma_int_t **dpivinfo_array, magma_int_t pivinfo_i,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;

    if (m < nb || m < panel_n)
        arginfo = -1;
    else if (n < 0)
        arginfo = -2;
    else if (ldda < max(1,m))
        arginfo = -4;

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return arginfo;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    switch( panel_n ) {
        #if 0
        case  8: arginfo = magma_zgetf2_update_NB_batched< 8>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        #else
        case  1: arginfo = magma_zgetf2_update_NB_batched< 1>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        case  2: arginfo = magma_zgetf2_update_NB_batched< 2>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //case  3: arginfo = magma_zgetf2_update_NB_batched< 3>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        case  4: arginfo = magma_zgetf2_update_NB_batched< 4>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //case  5: arginfo = magma_zgetf2_update_NB_batched< 5>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //case  6: arginfo = magma_zgetf2_update_NB_batched< 6>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //case  7: arginfo = magma_zgetf2_update_NB_batched< 7>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        case  8: arginfo = magma_zgetf2_update_NB_batched< 8>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //case  9: arginfo = magma_zgetf2_update_NB_batched< 9>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //case 10: arginfo = magma_zgetf2_update_NB_batched<10>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //case 11: arginfo = magma_zgetf2_update_NB_batched<11>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //case 12: arginfo = magma_zgetf2_update_NB_batched<12>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //case 13: arginfo = magma_zgetf2_update_NB_batched<13>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //case 14: arginfo = magma_zgetf2_update_NB_batched<14>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //case 15: arginfo = magma_zgetf2_update_NB_batched<15>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        case 16: arginfo = magma_zgetf2_update_NB_batched<16>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //case 17: arginfo = magma_zgetf2_update_NB_batched<17>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //case 18: arginfo = magma_zgetf2_update_NB_batched<18>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //case 19: arginfo = magma_zgetf2_update_NB_batched<19>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //case 20: arginfo = magma_zgetf2_update_NB_batched<20>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //case 21: arginfo = magma_zgetf2_update_NB_batched<21>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //case 22: arginfo = magma_zgetf2_update_NB_batched<22>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //case 23: arginfo = magma_zgetf2_update_NB_batched<23>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //case 24: arginfo = magma_zgetf2_update_NB_batched<24>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //case 25: arginfo = magma_zgetf2_update_NB_batched<25>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //case 26: arginfo = magma_zgetf2_update_NB_batched<26>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //case 27: arginfo = magma_zgetf2_update_NB_batched<27>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //case 28: arginfo = magma_zgetf2_update_NB_batched<28>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //case 29: arginfo = magma_zgetf2_update_NB_batched<29>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //case 30: arginfo = magma_zgetf2_update_NB_batched<30>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        //case 31: arginfo = magma_zgetf2_update_NB_batched<31>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        case 32: arginfo = magma_zgetf2_update_NB_batched<32>(m, n, nb, dA_array, Ai, Aj, ldda, dpivinfo_array, pivinfo_i, batchCount, queue ); break;
        #endif
        default: arginfo = -100;
    }

    return arginfo;
}
