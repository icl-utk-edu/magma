/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       @author Ahmad Abdelfattah
*/
#include "magma_internal.h"
#include "batched_kernel_param.h"
#include "zlaswp_device.cuh"

#define PRECISION_z
//#define DBG
#define ib    (3)

/******************************************************************************/
// serial swap that does swapping one row by one row
// this is the vbatched routine, for swapping to the left of the panel
__global__ void zlaswp_left_rowserial_kernel_vbatched(
                int n,
                magma_int_t *M, magma_int_t *N,
                magmaDoubleComplex **dA_array, int Ai, int Aj, magma_int_t *ldda,
                magma_int_t** ipiv_array, int ipiv_i,
                int k1, int k2 )
{
    const int batchid = blockIdx.z;
    const int tid     = threadIdx.x + blockDim.x*blockIdx.x;

    int my_M     = (int)M[batchid];
    int my_N     = (int)N[batchid];
    int my_minmn = min(my_M, my_N);
    int my_ldda  = (int)ldda[batchid];
    magmaDoubleComplex* dA = dA_array[batchid] + Aj * my_ldda + Ai;
    magma_int_t *dipiv     = ipiv_array[batchid] + ipiv_i;

    // check if offsets produce out-of-bound pointers
    if( my_M <= Ai || my_N <= Aj ) return;

    //k1--;
    //k2--;

    // reduce minmn by the pivot offset
    my_minmn -= ipiv_i;

    // check my_minmn & the offsets k1, k2
    if( my_minmn <= 0 || k1 >= my_minmn ) return;
    k2 = min(k2, my_minmn);
    // if we use k2 = min(k2, my_minmn-1),
    // then the for loop below should be until i1 <= k2, not i1 < k2

    // the following explanation is based on the assumption m >= n for all matrices
    // since this is a separate kernel for left-swap, we can calculate the maximum
    // affordable n based on (Ai, Aj).
    // In a left swap, Ai > Aj, which means (Ai, Aj) is on the left of the diagonal element
    // If the diagonal (Ai, Ai) is inside the matrix, then my_max_n is the horizontal
    // distance between (Ai, Aj) and (Ai, Ai). If (Ai, Ai) is outside a given matrix, we
    // terminate the thread-block(s) for this matrix only
    if(my_M < Ai || my_N < Ai) return;
    const int my_max_n = Ai - Aj;
    const int my_n     = min(n, my_max_n);

    #if defined(DBG) && defined(PRECISION_d)
    __syncthreads();
    if(batchid == ib && threadIdx.x == 0) {
        printf("matrix %d: k1 = %d, k2 = %d, my_n = %d\n", ib, k1, k2, my_n);
    }
    __syncthreads();
    #endif

    if (tid < my_n) {
        magmaDoubleComplex A1;

        for (int i1 = k1; i1 < k2; i1++) {
            int i2 = dipiv[i1] - 1;  // Fortran index, switch i1 and i2
            if ( i2 != i1 ) {
                A1 = dA[i1 + tid * my_ldda];
                dA[i1 + tid * my_ldda] = dA[i2 + tid * my_ldda];
                dA[i2 + tid * my_ldda] = A1;
            }
        }
    }
}

/******************************************************************************/
// serial swap that does swapping one row by one row
// this is the vbatched routine, for swapping to the right of the panel
__global__ void zlaswp_right_rowserial_kernel_vbatched(
                int n,
                magma_int_t *M, magma_int_t *N,
                magmaDoubleComplex **dA_array, int Ai, int Aj, magma_int_t *ldda,
                magma_int_t** ipiv_array, int ipiv_i,
                int k1, int k2 )
{
    const int batchid = blockIdx.z;
    const int tid     = threadIdx.x + blockDim.x*blockIdx.x;

    int my_M     = (int)M[batchid];
    int my_N     = (int)N[batchid];
    int my_minmn = min(my_M, my_N);
    int my_ldda  = (int)ldda[batchid];
    magmaDoubleComplex* dA = dA_array[batchid] + Aj * my_ldda + Ai;
    magma_int_t *dipiv     = ipiv_array[batchid] + ipiv_i;

    // check if offsets produce out-of-bound pointers
    if( my_M <= Ai || my_N <= Aj ) return;

    //k1--;
    //k2--;

    // reduce minmn by the pivot offset
    my_minmn -= ipiv_i;

    // check minmn, & the offsets k1, k2
    if( my_minmn <= 0 || k1 >= my_minmn ) return;
    k2 = min(k2, my_minmn);
    // if we use k2 = min(k2, my_minmn-1),
    // then the for loop below should be until i1 <= k2, not i1 < k2


    // check the input scalar 'n'
    const int my_max_n = my_N - Aj;
    const int my_n     = min(n, my_max_n);

    if (tid < my_n) {
        magmaDoubleComplex A1;

        for (int i1 = k1; i1 < k2; i1++) {
            int i2 = dipiv[i1] - 1;  // Fortran index, switch i1 and i2
            if ( i2 != i1 ) {
                A1 = dA[i1 + tid * my_ldda];
                dA[i1 + tid * my_ldda] = dA[i2 + tid * my_ldda];
                dA[i2 + tid * my_ldda] = A1;
            }
        }
    }
}

/******************************************************************************/
__global__
void zlaswp_left_rowparallel_kernel_vbatched(
                                int n, int width,
                                magma_int_t* M, magma_int_t* N,
                                magmaDoubleComplex **dA_array,  int Ai,  int Aj,  magma_int_t* ldda,
                                magma_int_t** pivinfo_array, int pivinfo_i,
                                int k1, int k2)
{
    const int batchid = blockIdx.z;

    int my_M     = (int)M[batchid];
    int my_N     = (int)N[batchid];
    int my_ldda  = (int)ldda[batchid];

    #if defined(DBG) && defined(PRECISION_d)
    __syncthreads();
    if(batchid == ib && threadIdx.x == 0) {
        printf("matrix %d: (%d, %d) -- offsets (%d, %d)\n", ib, my_M, my_N, Ai, Aj);
    }
    __syncthreads();
    #endif

    magmaDoubleComplex* dA = dA_array[batchid]  + Aj  * my_ldda + Ai;
    magma_int_t *pivinfo   = pivinfo_array[batchid] + pivinfo_i;

    // check if offsets produce out-of-bound pointers
    #if defined(DBG) && defined(PRECISION_d)
    __syncthreads();
    if(batchid == ib && threadIdx.x == 0) {
        printf("checking offsets\n");
    }
    __syncthreads();
    #endif
    if( my_M <= Ai || my_N <= Aj ) return;

    //my_M -= Ai;
    //my_N -= Aj;
    int my_minmn = min(my_M, my_N);

    // reduce minmn by the pivot offset
    my_minmn -= pivinfo_i;
    #if defined(DBG) && defined(PRECISION_d)
    __syncthreads();
    if(batchid == ib && threadIdx.x == 0) {
        printf("checking minmn\n");
    }
    __syncthreads();
    #endif
    if( my_minmn <= 0  ) return;
    #if defined(DBG) && defined(PRECISION_d)
    __syncthreads();
    if(batchid == ib && threadIdx.x == 0) {
        printf("checking k1\n");
    }
    __syncthreads();
    #endif
    if( k1 >= my_minmn ) return;
    k2 = min(k2, my_minmn);
    const int my_height = k2-k1;

    // the following explanation is based on the assumption m >= n for all matrices
    // since this is a separate kernel for left-swap, we can calculate the maximum
    // affordable n based on (Ai, Aj).
    // In a left swap, Ai > Aj, which means (Ai, Aj) is on the left of the diagonal element
    // If the diagonal (Ai, Ai) is inside the matrix, then my_max_n is the horizontal
    // distance between (Ai, Aj) and (Ai, Ai). If (Ai, Ai) is outside a given matrix, we
    // terminate the thread-block(s) for this matrix only
    if(my_M < Ai || my_N < Ai) return;
    const int my_max_n = Ai - Aj;
    const int my_n     = min(n, my_max_n);

    #if defined(DBG) && defined(PRECISION_d)
    __syncthreads();
    if(batchid == ib && threadIdx.x == 0) {
        printf("matrix %d: height = %d, my_n = %d\n", ib, my_height, my_n);
    }
    __syncthreads();
    #endif

    zlaswp_rowparallel_devfunc( my_n, width, my_height,
                                dA, my_ldda,
                                dA, my_ldda,
                                pivinfo);
}

/******************************************************************************/
// serial swap that does swapping one row by one row
// this is the vbatched routine, for swapping to the right of the panel
__global__ void zlaswp_right_rowparallel_kernel_vbatched(
                int n, int width,
                magma_int_t *M, magma_int_t *N,
                magmaDoubleComplex **dA_array, int Ai, int Aj, magma_int_t *ldda,
                magma_int_t** pivinfo_array, int pivinfo_i,
                int k1, int k2 )
{
    const int batchid = blockIdx.z;

    int my_M     = (int)M[batchid];
    int my_N     = (int)N[batchid];
    int my_minmn = min(my_M, my_N);
    int my_ldda  = (int)ldda[batchid];
    magmaDoubleComplex* dA = dA_array[batchid] + Aj * my_ldda + Ai;
    magma_int_t *dipivinfo = pivinfo_array[batchid] + pivinfo_i;

    // check if offsets produce out-of-bound pointers
    if( my_M <= Ai || my_N <= Aj ) return;

    // reduce minmn by the pivot offset
    my_minmn -= pivinfo_i;

    // check minmn, & the offsets k1, k2
    if( my_minmn <= 0 || k1 >= my_minmn ) return;
    k2 = min(k2, my_minmn);
    const int my_height = k2-k1;

    // check the input scalar 'n'
    const int my_max_n = my_N - Aj;
    const int my_n     = min(n, my_max_n);

    #if defined(DBG) && defined(PRECISION_d)
    __syncthreads();
    if(batchid == ib && threadIdx.x == 0) {
        printf("matrix %d: (%d, %d), offsets (%d, %d), height = %d, my_n = %d\n", ib, my_M, my_N, Ai, Aj, my_height, my_n);
    }
    __syncthreads();
    #endif

    zlaswp_rowparallel_devfunc( my_n, width, my_height,
                                dA, my_ldda,
                                dA, my_ldda,
                                dipivinfo);
}


/******************************************************************************/
// serial swap that does swapping one row by one row, similar to LAPACK
// K1, K2 are in Fortran indexing
extern "C" void
magma_zlaswp_left_rowserial_vbatched(
        magma_int_t n,
        magma_int_t *M, magma_int_t *N, magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
        magma_int_t **ipiv_array, magma_int_t ipiv_offset,
        magma_int_t k1, magma_int_t k2,
        magma_int_t batchCount, magma_queue_t queue)
{
    if (n == 0) return;

    magma_int_t max_batchCount = queue->get_maxBatch();
    magma_int_t blocks         = magma_ceildiv( n, BLK_SIZE );
    magma_int_t min_BLK_SIZE_n = min(BLK_SIZE, n);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3  grid(blocks, 1, ibatch);

        zlaswp_left_rowserial_kernel_vbatched
        <<< grid, min_BLK_SIZE_n, 0, queue->cuda_stream() >>>
        (n, M, N, dA_array, Ai, Aj, ldda, ipiv_array, ipiv_offset, k1, k2);
    }
}


/******************************************************************************/
// serial swap that does swapping one row by one row, similar to LAPACK
// K1, K2 are in Fortran indexing
extern "C" void
magma_zlaswp_right_rowserial_vbatched(
        magma_int_t n,
        magma_int_t *M, magma_int_t *N, magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
        magma_int_t **ipiv_array, magma_int_t ipiv_offset,
        magma_int_t k1, magma_int_t k2,
        magma_int_t batchCount, magma_queue_t queue)
{
    if (n == 0) return;

    magma_int_t max_batchCount = queue->get_maxBatch();
    magma_int_t blocks         = magma_ceildiv( n, BLK_SIZE );
    magma_int_t min_BLK_SIZE_n = min(BLK_SIZE, n);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3  grid(blocks, 1, ibatch);

        zlaswp_right_rowserial_kernel_vbatched
        <<< grid, min_BLK_SIZE_n, 0, queue->cuda_stream() >>>
        (n, M, N, dA_array, Ai, Aj, ldda, ipiv_array, ipiv_offset, k1, k2);
    }
}

/******************************************************************************/
extern "C" void
magma_zlaswp_left_rowparallel_vbatched(
        magma_int_t n,
        magma_int_t* M, magma_int_t* N,
        magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magma_int_t k1, magma_int_t k2,
        magma_int_t **pivinfo_array, magma_int_t pivinfo_i,
        magma_int_t batchCount, magma_queue_t queue)
{
    if (n == 0 ) return;
    int height = k2-k1;
    if ( height  > 1024) {
        fprintf( stderr, "%s: n=%lld > 1024, not supported\n", __func__, (long long) n );
    }

    int blocks = magma_ceildiv( n, SWP_WIDTH );
    magma_int_t max_batchCount = queue->get_maxBatch();
    magma_int_t width = min(n, SWP_WIDTH);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3  grid(blocks, 1, ibatch);

        size_t shmem = sizeof(magmaDoubleComplex) * height * width;
        zlaswp_left_rowparallel_kernel_vbatched
        <<< grid, height, shmem, queue->cuda_stream() >>>
        ( n, width, M, N, dA_array, Ai, Aj, ldda, pivinfo_array, pivinfo_i, k1, k2);
    }
}

/******************************************************************************/
extern "C" void
magma_zlaswp_right_rowparallel_vbatched(
        magma_int_t n,
        magma_int_t* M, magma_int_t* N,
        magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magma_int_t k1, magma_int_t k2,
        magma_int_t **pivinfo_array, magma_int_t pivinfo_i,
        magma_int_t batchCount, magma_queue_t queue)
{
    if (n == 0 ) return;
    int height = k2-k1;
    if ( height  > 1024) {
        fprintf( stderr, "%s: n=%lld > 1024, not supported\n", __func__, (long long) n );
    }

    int blocks = magma_ceildiv( n, SWP_WIDTH );
    magma_int_t max_batchCount = queue->get_maxBatch();
    magma_int_t width = min(n, SWP_WIDTH);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3  grid(blocks, 1, ibatch);

        size_t shmem = sizeof(magmaDoubleComplex) * height * width;
        zlaswp_right_rowparallel_kernel_vbatched
        <<< grid, height, shmem, queue->cuda_stream() >>>
        ( n, width, M, N, dA_array, Ai, Aj, ldda, pivinfo_array, pivinfo_i, k1, k2);
    }
}

