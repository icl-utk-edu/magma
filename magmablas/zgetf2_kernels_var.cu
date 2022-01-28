/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/

#include "magma_internal.h"
#include "batched_kernel_param.h"
#include "magma_templates.h"
#include "shuffle.cuh"
#include "zgetf2_devicefunc.cuh"

#define PRECISION_z

/******************************************************************************/
__global__ void
izamax_kernel_vbatched(
        int length, magma_int_t *M, magma_int_t *N,
        magmaDoubleComplex **dA_array, int Ai, int Aj, magma_int_t *ldda,
        magma_int_t** ipiv_array, int ipiv_i,
        magma_int_t *info_array, int step, int gbstep )
{
    extern __shared__ double sdata[];

    const int batchid = blockIdx.x;

    // compute the actual length
    int my_M    = (int)M[batchid];
    int my_N    = (int)N[batchid];
    int my_ldda = (int)ldda[batchid];
    // check if offsets produce out-of-bound pointers
    if( my_M <= Ai || my_N <= Aj ) return;

    // compute the length of the vector for each matrix
    my_M -= Ai;
    my_M  = min(my_M, length);

    magmaDoubleComplex *dA = dA_array[batchid] + Aj * my_ldda + Ai;
    magma_int_t *ipiv = ipiv_array[batchid] + ipiv_i;
    int tx = threadIdx.x;

    double *shared_x = sdata;
    int *shared_idx = (int*)(shared_x + zamax);

    izamax_devfunc(my_M, dA, 1, shared_x, shared_idx);

    if (tx == 0) {
        *ipiv = shared_idx[0] + step + 1; // Fortran Indexing & adjust pivot
        if (shared_x[0] == MAGMA_D_ZERO) {
            info_array[batchid] = shared_idx[0] + step + gbstep + 1;
        }
    }
}

/******************************************************************************/
extern "C" magma_int_t
magma_izamax_vbatched(
        magma_int_t length, magma_int_t *M, magma_int_t *N,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magma_int_t** ipiv_array, magma_int_t ipiv_i,
        magma_int_t *info_array, magma_int_t step, magma_int_t gbstep,
        magma_int_t batchCount, magma_queue_t queue)
{
    dim3 grid(batchCount, 1, 1);
    dim3 threads(zamax, 1, 1);

    izamax_kernel_vbatched<<< grid, threads, zamax * (sizeof(double) + sizeof(int)), queue->cuda_stream() >>>
    (length, M, N, dA_array, Ai, Aj, ldda, ipiv_array, ipiv_i, info_array, step, gbstep );

    return 0;
}

/******************************************************************************/
__global__
void zswap_kernel_vbatched(
        int max_n, magma_int_t *M, magma_int_t *N,
        magmaDoubleComplex **dA_array, int Ai, int Aj, magma_int_t* ldda,
        magma_int_t** ipiv_array, int piv_adjustment)
{
    const int batchid = blockIdx.x;
    const int my_ldda = (int)ldda[batchid];
    int my_M          = (int)M[batchid];
    int my_N          = (int)N[batchid];
    int my_minmn      = min(my_M, my_N);

    // check if offsets produce out-of-bound pointers
    if( my_M <= Ai || my_N <= Aj || my_minmn <= Ai ) return;

    my_N -= Aj; // this is the maximum possible width
    my_N = min(my_N, max_n);

    // read the pivot entry at Ai
    magma_int_t *ipiv = ipiv_array[batchid] + Ai;
    __shared__ int jp;
    if (threadIdx.x == 0){
        jp  = ipiv[0] - 1; // roll-back Fortran indexing
        // magma_izamax_vbatched adjusts the pivot, so roll it back
        // because Ai and Aj are offsets that already take care of that
        jp -= piv_adjustment;
    }
    __syncthreads();

    if (jp == 0) return; // no swapping required

    magmaDoubleComplex *dA  = dA_array[batchid] + Aj * my_ldda + Ai;
    magmaDoubleComplex *dA1 = dA;
    magmaDoubleComplex *dA2 = dA + jp;

    zswap_device_v2(my_N, dA1, my_ldda, dA2, my_ldda );
}

/******************************************************************************/
extern "C" magma_int_t
magma_zswap_vbatched(
        magma_int_t max_n, magma_int_t *M, magma_int_t *N,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
        magma_int_t** ipiv_array, magma_int_t piv_adjustment,
        magma_int_t batchCount, magma_queue_t queue)
{
    dim3 grid(batchCount, 1, 1);
    dim3 threads(zamax, 1, 1);

    zswap_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>>
    (max_n, M, N, dA_array, Ai, Aj, ldda, ipiv_array, piv_adjustment);

    return 0;
}

/******************************************************************************/
__global__
void zscal_zgeru_1d_generic_kernel_vbatched(
        int max_m, int max_n,
        magma_int_t *M, magma_int_t *N,
        magmaDoubleComplex **dA_array, int Ai, int Aj, magma_int_t *ldda,
        magma_int_t *info_array, int step, int gbstep)
{
    const int batchid = blockIdx.z;
    int my_M    = (int)M[batchid];
    int my_N    = (int)N[batchid];
    int my_ldda = (int)ldda[batchid];

    if( my_M <= Ai || my_N <= Aj ) return;
    my_M -= Ai; // this is the largest possible m per matrix
    my_N -= Aj; // this is the largest possible n per matrix

    my_M = min(my_M, max_m);
    my_N = min(my_N, max_n);

    magmaDoubleComplex* dA = dA_array[batchid] + Aj * my_ldda + Ai;
    magma_int_t *info = &info_array[batchid];
    zscal_zgeru_generic_device(my_M, my_N, dA, my_ldda, info, step, gbstep);
}


/******************************************************************************/
extern "C"
magma_int_t magma_zscal_zgeru_vbatched(
        magma_int_t max_M, magma_int_t max_N,
        magma_int_t *M, magma_int_t *N,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
        magma_int_t *info_array, magma_int_t step, magma_int_t gbstep,
        magma_int_t batchCount, magma_queue_t queue)
{
    /*
    Specialized kernel which merged zscal and zgeru the two kernels
    1) zscale the first column vector A(1:M-1,0) with 1/A(0,0);
    2) Performe a zgeru Operation for trailing matrix of A(1:M-1,1:N-1) += alpha*x*y**T, where
       alpha := -1.0; x := A(1:M-1,0) and y:= A(0,1:N-1);
    */

    magma_int_t max_batchCount = queue->get_maxBatch();
    const int tbx = 256;
    dim3 threads(tbx, 1, 1);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid(magma_ceildiv(max_M,tbx), 1, ibatch);

        zscal_zgeru_1d_generic_kernel_vbatched<<<grid, threads, 0, queue->cuda_stream()>>>
        (max_M, max_N, M+i, N+i, dA_array+i, Ai, Aj, ldda+i, info_array+i, step, gbstep);
    }
    return 0;
}

/******************************************************************************/
#define dA(i,j)              sA[(j) * my_ldda + (i)]
#define sA(i,j)              sA[(j) * my_M + (i)]
__global__
void
zgetf2_fused_sm_kernel_vbatched(
        int max_M, int max_N, int max_minMN, int max_MxN,
        magma_int_t *M, magma_int_t *N,
        magmaDoubleComplex** dA_array, int Ai, int Aj, magma_int_t* ldda,
        magma_int_t** dipiv_array, int ipiv_i,
        magma_int_t *info,  int gbstep, int batchCount )
{
    extern __shared__ magmaDoubleComplex sdata[];
    const int tx      = threadIdx.x;
    const int ty      = threadIdx.y;
    const int ntx     = blockDim.x;
    const int batchid = (blockIdx.x * blockDim.y) + ty;
    if(batchid >= batchCount) return;

    // read data of assigned problem
    const int my_m         = (int)M[batchid];
    const int my_N         = (int)N[batchid];
    const int my_ldda      = (int)ldda[batchid];
    const int my_minmn     = min(my_M, my_N);
    magmaDoubleComplex* dA = dA_array[batchid] + Aj * my_ldda + Ai;
    magma_int_t* dipiv     = dipiv_array[batchid] + ipiv_i;

    // check offsets
    if( my_M <= Ai || my_N <= Aj || my_minmn <= ipiv_i ) return;
    my_M     -= Ai;
    my_N     -= Aj;
    my_minmn  = min(my_M, my_N);

    magmaDoubleComplex *sA = (magmaDoubleComplex*)(sdata);
    double* dsx = (double*)(sA + blockDim.y * max_MxN);
    int* isx    = (int*)(dsx + blockDim.y * max_M);
    int* sipiv  = (int*)(isx + blockDim.y * max_M);
    dsx   += ty * max_M;
    isx   += ty * max_M;
    sipiv += ty * max_minMN;

    magmaDoubleComplex reg  = MAGMA_Z_ZERO;
    magmaDoubleComplex rTmp = MAGMA_Z_ZERO;

    int max_id, rowid = tx;
    int linfo = (gbstep == 0) ? 0 : *info;
    double rx_abs_max = MAGMA_D_ZERO;

    // init sipiv
    for(int i = tx; i < my_minmn; i+=ntx) {
        sipiv[i] = 0;
    }

    // read
    for(int j = 0; j < my_N; j++){
        for(int i = tx; i < my_M; i+=ntx) {
            sA(i,j) = dA(i,j);
        }
    }
    __syncthreads();

    for(int j = 0; j < my_minmn; j++){
        // izamax and find pivot
        for(int i = tx; i < my_M-j; i+=ntx) {
            dsx[ i ] = fabs(MAGMA_Z_REAL( sA(i,j) )) + fabs(MAGMA_Z_IMAG( sA(i,j) ));
            isx[ i ] = i;
        }
        __syncthreads();
        magma_getidmax_n(my_M-j, tx, dsx, isx);
        // the above devfunc has syncthreads at the end
        rx_abs_max = dsx[0];
        max_id     = isx[0];
        linfo  = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (gbstep+j+1) : linfo;
        if( tx == 0 ) sipiv[ j ] = max_id;
        __syncthreads();

        // swap
        if(max_id != j) {
            for(int i = tx; i < my_N; i+=ntx) {
                reg          = sA(j     ,i);
                sA(i,j)      = sA(max_id,i);
                sA(max_id,i) = reg;
            }
        }
        __syncthreads();

        if( linfo == 0 ) {
            reg = MAGMA_Z_DIV( MAGMA_Z_ONE, sA(j,j) );
            for(int i = (tx+j+1); i < my_M; i+=ntx) {
                rTmp    = reg * sA(i,j);
                sA(i,j) = rTmp;
                for(int jj = j+1; jj < my_N; jj++) {
                    sA(i,jj) -= rTmp * sA(j,jj);
                }
            }
        }
        __syncthreads();
    }

    if(tx == 0){
        (*info) = (magma_int_t)( linfo );
    }

    // write pivot
    for(int i = tx; i < my_minmn; i+=ntx) {
        dipiv[i] = (magma_int_t)(sipiv[i]);
    }

    // write A
    for(int j = 0; j < my_N; j++) {
        for(int i = tx; i < my_M; i+=ntx) {
            dA(i,j) = sA(i,j);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_zgetf2_fused_sm_vbatched(
    magma_int_t max_M, magma_int_t max_N, magma_int_t max_minMN, magma_int_t max_MxN,
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    magma_int_t** dipiv_array, magma_int_t ipiv_i,
    magma_int_t* info_array,
    magma_int_t nthreads, magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    magma_device_t device;
    magma_getdevice( &device );

    nthreads = nthreads <= 0 ? (max_M/2) : nthreads;
    #ifdef MAGMA_HAVE_CUDA
    nthreads = magma_roundup(nthreads, 32);
    #else
    nthreads = magma_roundup(nthreads, 64);
    #endif
    nthreads = min(nthreads, 1024);

    // in a variable-size setting, setting ntcol > 1 may lead to
    // kernel deadlocks due to different thread-groups calling
    // syncthreads at different points
    const magma_int_t ntcol = 1;
    magma_int_t shmem = ( max_MxN   * sizeof(magmaDoubleComplex) );
    shmem            += ( max_M     * sizeof(double) );
    shmem            += ( max_M     * sizeof(int) );
    shmem            += ( max_minMN * sizeof(int) );
    shmem            *= ntcol;
    magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    dim3 grid(gridx, 1, 1);
    dim3 threads( nthreads, ntcol, 1);

    // get max. dynamic shared memory on the GPU
    magma_int_t nthreads_max, shmem_max = 0;
    cudaDeviceGetAttribute (&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device);
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(zgetf2_fused_sm_kernel_vbatched, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    magma_int_t total_threads = nthreads * ntcol;
    if ( total_threads > nthreads_max || shmem > shmem_max ) {
        // printf("error: kernel %s requires too many threads or too much shared memory\n", __func__);
        arginfo = -100;
        return arginfo;
    }

    if( check_launch_only == 1 ) return arginfo;

    void *kernel_args[] = {&max_M, &max_N, &max_minMN, &max_MxN, &m, &n, &dA_array, &Ai, &Aj, &ldda, &dipiv_array, &ipiv_i, &info, &gbstep, &batchCount};
    cudaError_t e = cudaLaunchKernel((void*)zgetf2_fused_sm_kernel_vbatched, grid, threads, kernel_args, shmem, queue->cuda_stream());
    if( e != cudaSuccess ) {
        // printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
        arginfo = -100;
    }

    return arginfo;
}


