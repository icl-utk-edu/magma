/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/

#include "magma_internal.h"
#include "magma_templates.h"
#include "shuffle.cuh"
#include "sync.cuh"
#include "atomics.cuh"
#include "batched_kernel_param.h"

#define PRECISION_z

/**
    Purpose
    -------
    LU factorization of m-by-n matrix ( m >= n ).
    Each thread block caches an entire column in register.
    Thread blocks communicate and synchronize through global memory.
    Assumptions:
        1. dA is of size MxN such that N <= M.
        2. Thread block must be 1D, with TX multiple of 32 (warp size)
        3. TX must be >= n
        4. n must be less than the number of SMs on the GPU
**/

/******************************************************************************/
// init kernel
__global__ void
zgetf2_native_init_kernel( int n, int npages, magma_int_t *ipiv, int* update_flags)
{
    const int tx = threadIdx.x;
    if( tx < n){
        ipiv[ tx ] = 0;
    }
    if( tx < max(n,npages) ){
        update_flags[ tx ] = 0;
    }
}

/******************************************************************************/
// the main kernel
template<int TX, int NPAGES>
__global__
__launch_bounds__(TX)
void zgetf2_native_kernel
        ( int m, int n,
          magmaDoubleComplex_ptr dA, int ldda,
          volatile magma_int_t *ipiv, int gbstep,
          volatile int* update_flag,
          volatile magma_int_t *info)
{
#if defined(MAGMA_HAVE_CUDA) || defined(MAGMA_HAVE_HIP)
    const int tx  = threadIdx.x;
    const int bx = blockIdx.x;
    magmaDoubleComplex rA[NPAGES] = {MAGMA_Z_ZERO};
    magmaDoubleComplex rx, rx_max;
    magmaDoubleComplex_ptr da = dA;
    int rx_id, max_id, flag = 0, linfo;
    double  rx_abs = 0.0, rx_abs_max = 0.0;
    const int m_ = m-(NPAGES-1)*TX;
    if( bx >= n ) return;

    // read the info (if it is set to non-zero a previous panel, then we don't set it again)
    linfo = (int)(*info);

    __shared__ magmaDoubleComplex sx[ TX ];
    __shared__ double sabs[ TX ];
    __shared__ int smax_id[ TX ];
    __shared__ magmaDoubleComplex sreg;

    // read
    dA += bx * ldda + tx;
    #pragma unroll
    for(int i = 0; i < NPAGES-1; i++){
        rA[i] = dA[ i * TX ];
    }
    if( tx <  m_){
        rA[NPAGES-1] = dA[ (NPAGES-1) * TX ];
    }

    // main loop
    for(int i = 0; i < n; i++){
        // izamax and write pivot for the ith thread block
        if(bx == i){
            rx_max     = rx     = (tx < i) ? MAGMA_Z_ZERO : rA[0];
            rx_abs_max = rx_abs = fabs(MAGMA_Z_REAL(rx)) + fabs(MAGMA_Z_IMAG(rx));
            max_id = rx_id = tx;
            #pragma unroll
            for(int j = 1; j < NPAGES; j++){
                rx = rA[j];
                rx_abs = fabs(MAGMA_Z_REAL(rx)) + fabs(MAGMA_Z_IMAG(rx));
                if ( rx_abs  > rx_abs_max ){
                    rx_max = rx;
                    rx_abs_max = rx_abs;
                    max_id = j * TX + tx;
                }
            }
            sx[ tx ] = rx_max;
            sabs[ tx ] = rx_abs_max;
            smax_id[ tx ] = max_id;
            __syncthreads();

            // let the first warp do the final reduction step
            if(tx < 32){
                #pragma unroll
                for(int j = 0; j < TX; j+= 32){
                    rx     = sx[ j + tx ];
                    rx_abs = sabs[ j + tx ];
                    rx_id  = smax_id[ j + tx ];
                    if ( rx_abs  > rx_abs_max ){
                        rx_max = rx;
                        rx_abs_max = rx_abs;
                        max_id = rx_id;
                    }
                }
                magmablas_syncwarp();
                sx[ tx ] = rx_max;
                sabs[ tx ] = rx_abs_max;
                smax_id[ tx ] = max_id;
                magmablas_syncwarp();
                #pragma unroll
                for(int j = 0; j < 32; j++){
                    rx     = sx[j];
                    rx_abs = sabs[j];
                    rx_id  = smax_id[j];
                    if ( rx_abs  > rx_abs_max ){
                        rx_abs_max = rx_abs;
                        rx_max = rx;
                        max_id = rx_id;
                    }
                }
            }

            if(tx == 0){
                sx[ 0 ] = rx_max;
                sabs[ 0 ] = rx_abs_max;
                smax_id[ 0 ] = (rx_abs_max == MAGMA_D_ZERO) ? i : max_id;
            }
            __syncthreads();
            rx_max = sx[ 0 ];
            rx_abs_max = sabs[ 0 ];
            max_id = smax_id[ 0 ];
            __syncthreads();

            // now every thread in the i^th block has the maximum
            linfo = (rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (max_id+gbstep+1) : linfo;
            if( tx == 0){
                //printf("[%2d]: bx = %d, max_id, = %d, rx_abs_max = %f, linfo = %d\n", i, bx, max_id, rx_abs_max, linfo);
                magmablas_iatomic_exchange((magma_int_t*)info, (magma_int_t)(linfo) );
                magmablas_iatomic_exchange((magma_int_t*)&ipiv[i], (magma_int_t)(max_id+1) ); // fortran indexing
            }
            __syncthreads();
            //if( rx_abs_max == MAGMA_D_ZERO )return;
        }
        else{ // other thread blocks are waiting
            if(tx == 0){
                max_id = 0;
                while( max_id == 0 ){
                    max_id = ipiv[i];
                };
                smax_id[ 0 ] = max_id;
            }
            __syncthreads();
            max_id = smax_id[ 0 ];
            max_id -= 1; // revert fortran indexing
            linfo = (*info);
            __syncthreads();
            //if( (*info) != 0 ) return;
        }

        // swap
        // swap always happens between page 0 and page x
        // to avoid spilling rA to local memory, we use shared memory
        if( max_id != i){
            // all blocks swap in registers
            // for bx < i, the column is already written in memory,
            // but we have a copy in reg., so continue to swap in reg.,
            // and do one final write to memory
            #pragma unroll
            for(int j = 0; j < NPAGES; j++){
                if( j == (max_id/TX) ){
                    sx[ tx ] = rA[j];
                    __syncthreads();
                    if( tx == i ){
                        magmaDoubleComplex tmp    = sx[ max_id%TX ];
                        sx[ max_id%TX ] = rA[0];
                        rA[0] = tmp;
                    }
                    __syncthreads();
                    if( tx == max_id%TX ){
                        rA[j] = sx[ tx ];
                    }
                    __syncthreads();
                }
            }
            //__syncthreads();
        }

        // the ith block does scal
        if(bx == i){
            magmaDoubleComplex reg = (rx_max == MAGMA_Z_ZERO) ? MAGMA_Z_ONE : MAGMA_Z_DIV(MAGMA_Z_ONE, rx_max );
            // scal
            if( tx > i ){
                rA[0] *= reg;
            }
            #pragma unroll
            for(int j = 1; j < NPAGES; j++){
                rA[j] *= reg;
            }
            // write column i to global memory
            #pragma unroll
            for(int j = 0; j < NPAGES-1; j++){
                dA[ j * TX ] = rA[j];
            }
            if( tx <  m_){
                dA[ (NPAGES-1) * TX ] = rA[NPAGES-1];
            }
            __threadfence(); __syncthreads(); // after cuda 9.0, both are needed, not sure why
            if(tx == 0) magmablas_iatomic_exchange( (int *)&update_flag[ i ], 1);
        }

        // thread blocks with ID larger than i perform ger
        if(bx > i){
            if( tx == i ){
                sreg = rA[0];
            }
            // wait for scal
            if( tx == 0){
                flag = 0;
                while( flag == 0 ){
                    flag = update_flag[ i ];
                };
            }
            __syncthreads();

            magmaDoubleComplex reg = sreg;
            if( NPAGES == 1){
                if(tx > i && tx < m_){
                    rA[0] -= da[ i * ldda + tx ] * reg;
                }
            }else{
                if(tx > i){
                    rA[0] -= da[ i * ldda + tx ] * reg;
                }
            }
            #pragma unroll
            for(int j = 1; j < NPAGES-1; j++){
                rA[j] -= da[ i * ldda + j * TX + tx ] * reg;
            }
            if( NPAGES > 1){
                if( tx < m_ ){
                    rA[ NPAGES-1 ] -= da[ i * ldda + (NPAGES-1)*TX + tx ] * reg;
                }
            }
        }
    }

    // all blocks write their columns again except the last one
    if( bx < n-1 ){
        #pragma unroll
        for(int i = 0; i < NPAGES-1; i++){
            dA[ i * TX ] = rA[i];
        }
        if( tx <  m_){
            dA[ (NPAGES-1) * TX ] = rA[NPAGES-1];
        }
    }

#endif    // MAGMA_HAVE_CUDA
}

/******************************************************************************/
template<int TX, int NPAGES>
static magma_int_t
zgetf2_native_kernel_driver(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv, magma_int_t gbstep,
    int* update_flag,
    magma_int_t *info
    magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    magma_device_t device;
    magma_getdevice( &device );

    dim3 grid(n, 1, 1);
    dim3 threads(ZGETF2_FUSED_NTH, 1, 1);

    // configure shared memory
    int shmem_max = 0;   // not magma_int_t (causes problems with 64bit builds)
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    // the kernel uses communication among thread blocks
    // as a safeguard, force one thread block per multiprocessor
    // by allocating more than half the shared memory
    int shmem = (magma_int_t)(0.75 * shmem_max);

    void *kernel_args[] = {&m, &n, &dA, &ldda, &ipiv, &gbstep, &update_flag, &info};
    cudaError_t e = cudaLaunchKernel((void*)zgetf2_native_kernel<TX, NPAGES>, grid, threads, kernel_args, shmem, queue->cuda_stream());
    if( e != cudaSuccess ) {
        //printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
        arginfo = -100;
    }

    return arginfo;
}
// =============================================================================
extern "C" magma_int_t
magma_zgetf2_native_fused(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv, magma_int_t gbstep,
    magma_int_t *flags,
    magma_int_t *info, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    const magma_int_t ntx   = ZGETF2_FUSED_NTH;

    if( m < n || m > ZGETF2_FUSED_MAX_M ){
        arginfo = -1;
    }
    else if( n > magma_getdevice_multiprocessor_count() ){
        arginfo = -2;
    }
    else if( ldda < max(1, m) ){
        arginfo = -4;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    const magma_int_t npages = magma_ceildiv(m, ntx);
    int *update_flag = (int*) flags;    // update_flag is an int, not magma_int_t
    size_t max_n_npages = max(n,npages);
    zgetf2_native_init_kernel<<< 1, max_n_npages, 0, queue->cuda_stream() >>>( n, npages, ipiv, update_flag);

    // The case statement should cover up to ( xGETF2_CHAIN_MAX_M / ntx )
    switch(npages){
        case  1: arginfo = zgetf2_native_kernel_driver< ntx,  1>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case  2: arginfo = zgetf2_native_kernel_driver< ntx,  2>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case  3: arginfo = zgetf2_native_kernel_driver< ntx,  3>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case  4: arginfo = zgetf2_native_kernel_driver< ntx,  4>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case  5: arginfo = zgetf2_native_kernel_driver< ntx,  5>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case  6: arginfo = zgetf2_native_kernel_driver< ntx,  6>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case  7: arginfo = zgetf2_native_kernel_driver< ntx,  7>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case  8: arginfo = zgetf2_native_kernel_driver< ntx,  8>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case  9: arginfo = zgetf2_native_kernel_driver< ntx,  9>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 10: arginfo = zgetf2_native_kernel_driver< ntx, 10>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 11: arginfo = zgetf2_native_kernel_driver< ntx, 11>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 12: arginfo = zgetf2_native_kernel_driver< ntx, 12>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 13: arginfo = zgetf2_native_kernel_driver< ntx, 13>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 14: arginfo = zgetf2_native_kernel_driver< ntx, 14>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 15: arginfo = zgetf2_native_kernel_driver< ntx, 15>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 16: arginfo = zgetf2_native_kernel_driver< ntx, 16>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 17: arginfo = zgetf2_native_kernel_driver< ntx, 17>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 18: arginfo = zgetf2_native_kernel_driver< ntx, 18>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 19: arginfo = zgetf2_native_kernel_driver< ntx, 19>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 20: arginfo = zgetf2_native_kernel_driver< ntx, 20>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        #if defined(PRECISION_s) || defined(PRECISION_d)
        case 21: arginfo = zgetf2_native_kernel_driver< ntx, 21>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 22: arginfo = zgetf2_native_kernel_driver< ntx, 22>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 23: arginfo = zgetf2_native_kernel_driver< ntx, 23>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 24: arginfo = zgetf2_native_kernel_driver< ntx, 24>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 25: arginfo = zgetf2_native_kernel_driver< ntx, 25>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 26: arginfo = zgetf2_native_kernel_driver< ntx, 26>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 27: arginfo = zgetf2_native_kernel_driver< ntx, 27>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 28: arginfo = zgetf2_native_kernel_driver< ntx, 28>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 29: arginfo = zgetf2_native_kernel_driver< ntx, 29>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 30: arginfo = zgetf2_native_kernel_driver< ntx, 30>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 31: arginfo = zgetf2_native_kernel_driver< ntx, 31>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 32: arginfo = zgetf2_native_kernel_driver< ntx, 32>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 33: arginfo = zgetf2_native_kernel_driver< ntx, 33>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 34: arginfo = zgetf2_native_kernel_driver< ntx, 34>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 35: arginfo = zgetf2_native_kernel_driver< ntx, 35>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 36: arginfo = zgetf2_native_kernel_driver< ntx, 36>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 37: arginfo = zgetf2_native_kernel_driver< ntx, 37>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 38: arginfo = zgetf2_native_kernel_driver< ntx, 38>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 39: arginfo = zgetf2_native_kernel_driver< ntx, 39>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 40: arginfo = zgetf2_native_kernel_driver< ntx, 40>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 41: arginfo = zgetf2_native_kernel_driver< ntx, 41>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 42: arginfo = zgetf2_native_kernel_driver< ntx, 42>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 43: arginfo = zgetf2_native_kernel_driver< ntx, 43>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 44: arginfo = zgetf2_native_kernel_driver< ntx, 44>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 45: arginfo = zgetf2_native_kernel_driver< ntx, 45>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 46: arginfo = zgetf2_native_kernel_driver< ntx, 46>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        #endif // defined(PRECISION_s) || defined(PRECISION_d)
        #if defined(PRECISION_s)
        case 47: arginfo = zgetf2_native_kernel_driver< ntx, 47>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 48: arginfo = zgetf2_native_kernel_driver< ntx, 48>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 49: arginfo = zgetf2_native_kernel_driver< ntx, 49>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 50: arginfo = zgetf2_native_kernel_driver< ntx, 50>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 51: arginfo = zgetf2_native_kernel_driver< ntx, 51>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 52: arginfo = zgetf2_native_kernel_driver< ntx, 52>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 53: arginfo = zgetf2_native_kernel_driver< ntx, 53>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 54: arginfo = zgetf2_native_kernel_driver< ntx, 54>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 55: arginfo = zgetf2_native_kernel_driver< ntx, 55>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 56: arginfo = zgetf2_native_kernel_driver< ntx, 56>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 57: arginfo = zgetf2_native_kernel_driver< ntx, 57>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 58: arginfo = zgetf2_native_kernel_driver< ntx, 58>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 59: arginfo = zgetf2_native_kernel_driver< ntx, 59>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 60: arginfo = zgetf2_native_kernel_driver< ntx, 60>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 61: arginfo = zgetf2_native_kernel_driver< ntx, 61>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 62: arginfo = zgetf2_native_kernel_driver< ntx, 62>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 63: arginfo = zgetf2_native_kernel_driver< ntx, 63>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 64: arginfo = zgetf2_native_kernel_driver< ntx, 64>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 65: arginfo = zgetf2_native_kernel_driver< ntx, 65>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 66: arginfo = zgetf2_native_kernel_driver< ntx, 66>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 67: arginfo = zgetf2_native_kernel_driver< ntx, 67>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 68: arginfo = zgetf2_native_kernel_driver< ntx, 68>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 69: arginfo = zgetf2_native_kernel_driver< ntx, 69>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 70: arginfo = zgetf2_native_kernel_driver< ntx, 70>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 71: arginfo = zgetf2_native_kernel_driver< ntx, 71>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 72: arginfo = zgetf2_native_kernel_driver< ntx, 72>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 73: arginfo = zgetf2_native_kernel_driver< ntx, 73>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 74: arginfo = zgetf2_native_kernel_driver< ntx, 74>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 75: arginfo = zgetf2_native_kernel_driver< ntx, 75>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 76: arginfo = zgetf2_native_kernel_driver< ntx, 76>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 77: arginfo = zgetf2_native_kernel_driver< ntx, 77>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 78: arginfo = zgetf2_native_kernel_driver< ntx, 78>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 79: arginfo = zgetf2_native_kernel_driver< ntx, 79>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        case 80: arginfo = zgetf2_native_kernel_driver< ntx, 80>( m, n, dA, ldda, ipiv, gbstep, update_flag, info, queue); break;
        #endif // defined(PRECISION_s)
        default: printf("size not supported \n");
    }
    return 0;
}
