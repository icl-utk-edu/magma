/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Weifeng Liu

*/

// CSC Sync-Free SpTRSM kernel
// see paper by W. Liu, A. Li, J. D. Hogg, I. S. Duff, and B. Vinter. (2016).
// "A Synchronization-Free Algorithm for Parallel Sparse Triangular Solves". 
// 22nd International European Conference on Parallel and Distributed Computing 
// (Euro-Par '16). pp. 617-630.

#include "magmasparse_internal.h"
#include "atomicopsmagmaDoubleComplex.h"

#include <cuda.h>  // for CUDA_VERSION

//#define MAGMA_CSC_SYNCFREE_WARP_PER_BLOCK 16
#define MAGMA_CSC_SYNCFREE_WARP_SIZE 32

#define MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD  0
#define MAGMA_CSC_SYNCFREE_SUBSTITUTION_BACKWARD 1

#define MAGMA_CSC_SYNCFREE_OPT_WARP_NNZ   1
#define MAGMA_CSC_SYNCFREE_OPT_WARP_RHS   2
#define MAGMA_CSC_SYNCFREE_OPT_WARP_AUTO  3

__global__
void sptrsv_syncfree_analyser(magmaIndex_ptr         d_cscRowIdx,
                              magmaDoubleComplex_ptr d_cscVal,
                              magma_int_t            m,
                              magma_int_t            nnz,
                              magmaIndex_ptr         d_graphInDegree)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x; 
    if (global_id < nnz)
    {
        atomicAdd(&d_graphInDegree[d_cscRowIdx[global_id]], 1);
    }
}

/*__global__
void sptrsv_syncfree_executor(magmaIndex_ptr         d_cscColPtr,
                              magmaIndex_ptr         d_cscRowIdx,
                              magmaDoubleComplex_ptr d_cscVal,
                              magmaIndex_ptr         d_graphInDegree,
                              magma_int_t            m,
                              magma_int_t            substitution,
                              magmaDoubleComplex_ptr d_b,
                              magmaDoubleComplex_ptr d_x)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int global_x_id = global_id / MAGMA_CSC_SYNCFREE_WARP_SIZE;
    if (global_x_id >= m) return;

    // substitution is forward or backward
    global_x_id = substitution == MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD ? 
                  global_x_id : m - 1 - global_x_id;

    volatile __shared__ 
             magma_index_t s_graphInDegree[MAGMA_CSC_SYNCFREE_WARP_PER_BLOCK];
    volatile __shared__ 
             magmaDoubleComplex s_left_sum[MAGMA_CSC_SYNCFREE_WARP_PER_BLOCK];

    // Initialize
    const int local_warp_id = threadIdx.x / MAGMA_CSC_SYNCFREE_WARP_SIZE;
    const int lane_id = (MAGMA_CSC_SYNCFREE_WARP_SIZE - 1) & threadIdx.x;
    int starting_x = (global_id / (MAGMA_CSC_SYNCFREE_WARP_PER_BLOCK 
                                   * MAGMA_CSC_SYNCFREE_WARP_SIZE)) 
                     * MAGMA_CSC_SYNCFREE_WARP_PER_BLOCK;
    starting_x = substitution == MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD ? 
                 starting_x : m - 1 - starting_x;
    
    // Prefetch
    const int pos = substitution == MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD ?
                    d_cscColPtr[global_x_id] : d_cscColPtr[global_x_id+1]-1;
    const magmaDoubleComplex one = MAGMA_Z_MAKE( 1.0, 0.0);
    const magmaDoubleComplex coef = one / d_cscVal[pos];

    if (threadIdx.x < MAGMA_CSC_SYNCFREE_WARP_PER_BLOCK) 
    { 
        s_graphInDegree[threadIdx.x] = 1; 
        s_left_sum[threadIdx.x] = MAGMA_Z_ZERO; 
    }
    __syncthreads();

    clock_t start;
    // Consumer
    do {
        start = clock();
    }
    while (s_graphInDegree[local_warp_id] != d_graphInDegree[global_x_id]);

    magmaDoubleComplex xi = d_x[global_x_id] + s_left_sum[local_warp_id];
    xi = (d_b[global_x_id] - xi) * coef;

    // Producer
    const magma_index_t start_ptr = 
              substitution == MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD ? 
              d_cscColPtr[global_x_id]+1 : d_cscColPtr[global_x_id];
    const magma_index_t stop_ptr  = 
              substitution == MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD ? 
              d_cscColPtr[global_x_id+1] : d_cscColPtr[global_x_id+1]-1;
    for (magma_index_t jj = start_ptr + lane_id; 
                       jj < stop_ptr; jj += MAGMA_CSC_SYNCFREE_WARP_SIZE)
    {
        const magma_index_t j = 
                  substitution == MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD ? 
                  jj : stop_ptr - 1 - (jj - start_ptr);
        const magma_index_t rowIdx = d_cscRowIdx[j];
        const bool cond = 
                  substitution == MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD ? 
                  (rowIdx < starting_x + MAGMA_CSC_SYNCFREE_WARP_PER_BLOCK) : 
                  (rowIdx > starting_x - MAGMA_CSC_SYNCFREE_WARP_PER_BLOCK);
        if (cond) 
        {
            const magma_index_t pos = 
                      substitution == MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD ? 
                      rowIdx - starting_x : starting_x - rowIdx;
            atomicAddmagmaDoubleComplex(&s_left_sum[pos], xi * d_cscVal[j]);
            __threadfence_block();
            atomicAdd((int *)&s_graphInDegree[pos], 1);
        }
        else 
        {
            atomicAddmagmaDoubleComplex(&d_x[rowIdx], xi * d_cscVal[j]);
            __threadfence();
            atomicSub(&d_graphInDegree[rowIdx], 1);
        }
    }

    //finish
    if (!lane_id) d_x[global_x_id] = xi;
}*/

__global__
void sptrsm_syncfree_executor(magmaIndex_ptr         d_cscColPtr,
                              magmaIndex_ptr         d_cscRowIdx,
                              magmaDoubleComplex_ptr d_cscVal,
                              magmaIndex_ptr         d_graphInDegree,
                              magma_int_t            m,
                              magma_int_t            substitution,
                              magma_int_t            rhs,
                              magma_int_t            opt,
                              magmaDoubleComplex_ptr d_b,
                              magmaDoubleComplex_ptr d_x)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int global_x_id = global_id / MAGMA_CSC_SYNCFREE_WARP_SIZE;
    if (global_x_id >= m) return;

    // substitution is forward or backward
    global_x_id = substitution == MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD ? 
                  global_x_id : m - 1 - global_x_id;

    // Initialize
    const int lane_id = (MAGMA_CSC_SYNCFREE_WARP_SIZE - 1) & threadIdx.x;

    // Prefetch
    const int pos = substitution == MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD ?
                d_cscColPtr[global_x_id] : d_cscColPtr[global_x_id+1]-1;
    const magmaDoubleComplex one = MAGMA_Z_MAKE( 1.0, 0.0);
    const magmaDoubleComplex coef = one / d_cscVal[pos];

    clock_t start;
    // Consumer
    do {
        start = clock();
    }
    while (1 != d_graphInDegree[global_x_id]);

    for (int k = lane_id; k < rhs; k += MAGMA_CSC_SYNCFREE_WARP_SIZE)
    {
        const int pos = global_x_id * rhs + k;
        d_x[pos] = (d_b[pos] - d_x[pos]) * coef;
    }

    // Producer
    const magma_index_t start_ptr = 
              substitution == MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD ? 
              d_cscColPtr[global_x_id]+1 : d_cscColPtr[global_x_id];
    const magma_index_t stop_ptr  = 
              substitution == MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD ? 
              d_cscColPtr[global_x_id+1] : d_cscColPtr[global_x_id+1]-1;

    if (opt == MAGMA_CSC_SYNCFREE_OPT_WARP_NNZ)
    {
        for (magma_index_t jj = start_ptr + lane_id; 
                           jj < stop_ptr; jj += MAGMA_CSC_SYNCFREE_WARP_SIZE)
        {
            const magma_index_t j = 
                      substitution == MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD ? 
                      jj : stop_ptr - 1 - (jj - start_ptr);
            const magma_index_t rowIdx = d_cscRowIdx[j];
            for (magma_index_t k = 0; k < rhs; k++)
                atomicAddmagmaDoubleComplex(&d_x[rowIdx * rhs + k], 
                    d_x[global_x_id * rhs + k] * d_cscVal[j]);
            __threadfence();
            atomicSub(&d_graphInDegree[rowIdx], 1);
        }
    }
    else if (opt == MAGMA_CSC_SYNCFREE_OPT_WARP_RHS)
    {
        for (magma_index_t jj = start_ptr; jj < stop_ptr; jj++)
        {
            const magma_index_t j = 
                      substitution == MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD ? 
                      jj : stop_ptr - 1 - (jj - start_ptr);
            const magma_index_t rowIdx = d_cscRowIdx[j];
            for (magma_index_t k = lane_id; 
                               k < rhs; k+=MAGMA_CSC_SYNCFREE_WARP_SIZE)
                atomicAddmagmaDoubleComplex(&d_x[rowIdx * rhs + k], 
                    d_x[global_x_id * rhs + k] * d_cscVal[j]);
            __threadfence();
            if (!lane_id) atomicSub(&d_graphInDegree[rowIdx], 1);
        }
    }
    else if (opt == MAGMA_CSC_SYNCFREE_OPT_WARP_AUTO)
    {
        const magma_index_t len = stop_ptr - start_ptr;

        if ((len <= rhs || rhs > 8) && len < 2048)
        {
            for (magma_index_t jj = start_ptr; jj < stop_ptr; jj++)
            {
                const magma_index_t j = 
                      substitution == MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD ? 
                      jj : stop_ptr - 1 - (jj - start_ptr);
                const magma_index_t rowIdx = d_cscRowIdx[j];
                for (magma_index_t k = lane_id; 
                                   k < rhs; k+=MAGMA_CSC_SYNCFREE_WARP_SIZE)
                    atomicAddmagmaDoubleComplex(&d_x[rowIdx * rhs + k], 
                        d_x[global_x_id * rhs + k] * d_cscVal[j]);
                __threadfence();
                if (!lane_id) atomicSub(&d_graphInDegree[rowIdx], 1);
            }
        }
        else
        {
            for (magma_index_t jj = start_ptr + lane_id; 
                             jj < stop_ptr; jj += MAGMA_CSC_SYNCFREE_WARP_SIZE)
            {
                const magma_index_t j = 
                      substitution == MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD ? 
                      jj : stop_ptr - 1 - (jj - start_ptr);
                const magma_index_t rowIdx = d_cscRowIdx[j];
                for (magma_index_t k = 0; k < rhs; k++)
                    atomicAddmagmaDoubleComplex(&d_x[rowIdx * rhs + k], 
                        d_x[global_x_id * rhs + k] * d_cscVal[j]);
                __threadfence();
                atomicSub(&d_graphInDegree[rowIdx], 1);
            }
        }
    }
}


extern "C" magma_int_t
magma_zgecscsyncfreetrsm_analysis(
    magma_int_t             m, 
    magma_int_t             nnz,
    magmaDoubleComplex_ptr  dval,
    magmaIndex_ptr          dcolptr,
    magmaIndex_ptr          drowind, 
    magmaIndex_ptr          dgraphindegree, 
    magmaIndex_ptr          dgraphindegree_bak, 
    magma_queue_t           queue )
{
    int info = MAGMA_SUCCESS;
    printf("magma_zgecscsyncfreetrsm_analysis is called1\n");

    int num_threads = 128;
    int num_blocks = ceil ((double)nnz / (double)num_threads);
    cudaMemset(dgraphindegree, 0, m * sizeof(magma_index_t));
    sptrsv_syncfree_analyser<<< num_blocks, num_threads >>>
                            (drowind, dval, m, nnz, dgraphindegree);
    printf("magma_zgecscsyncfreetrsm_analysis is called2\n");
    // backup in-degree array
    cudaMemcpy(dgraphindegree_bak, dgraphindegree, 
               m * sizeof(int), cudaMemcpyDeviceToDevice);
    printf("magma_zgecscsyncfreetrsm_analysis is called3\n");
    return info;
}

extern "C" magma_int_t
magma_zgecscsyncfreetrsm_solve(
    magma_int_t             m, 
    magma_int_t             nnz,
    magmaDoubleComplex      alpha,
    magmaDoubleComplex_ptr  dval,
    magmaIndex_ptr          dcolptr,
    magmaIndex_ptr          drowind,
    magmaIndex_ptr          dgraphindegree, 
    magmaIndex_ptr          dgraphindegree_bak, 
    magmaDoubleComplex_ptr  dx,
    magmaDoubleComplex_ptr  db,
    magma_int_t             substitution, 
    magma_int_t             rhs, 
    magma_queue_t           queue )
{
    int info = MAGMA_SUCCESS;
    printf("magma_zgecscsyncfreetrsm_solve is called\n");

    // get an unmodified in-degree array, only for benchmarking use
    cudaMemcpy(dgraphindegree, dgraphindegree_bak, 
               m * sizeof(magma_index_t), cudaMemcpyDeviceToDevice);
        
    // clear d_x for atomic operations
    cudaMemset(dx, 0, sizeof(magmaDoubleComplex) * m * rhs);

    int num_threads, num_blocks;
    //if (rhs == 1)
    //{
    //    num_threads = MAGMA_CSC_SYNCFREE_WARP_PER_BLOCK 
    //                  * MAGMA_CSC_SYNCFREE_WARP_SIZE;
    //    num_blocks = ceil ((double)m / 
    //                     (double)(num_threads/MAGMA_CSC_SYNCFREE_WARP_SIZE));
    //    sptrsv_syncfree_executor<<< num_blocks, num_threads >>>
    //                  (dcolptr, drowind, dval, dgraphindegree.
    //                   m, substitution, db, dx);
    //}
    //else
    //{
        num_threads = 4 * MAGMA_CSC_SYNCFREE_WARP_SIZE;
        num_blocks = ceil ((double)m / 
                         (double)(num_threads/MAGMA_CSC_SYNCFREE_WARP_SIZE));
        sptrsm_syncfree_executor<<< num_blocks, num_threads >>>
                      (dcolptr, drowind, dval, dgraphindegree,
                       m, substitution, rhs, MAGMA_CSC_SYNCFREE_OPT_WARP_AUTO,
                       db, dx);
    //}

    return info;
}
/*
const int           *cscColPtrTR,
                         const int           *cscRowIdxTR,
                         const magmaDoubleComplex    *cscValTR,
                         const int            m,
                         const int            n,
                         const int            nnzTR,
                         const int            substitution,
                         const int            rhs,
                         const int            opt,
                               magmaDoubleComplex    *x,
                         const magmaDoubleComplex    *b,
                         const magmaDoubleComplex    *x_ref,
                               double        *gflops)
{
    if (m != n)
    {
        printf("This is not a square matrix, return.\n");
        return -1;
    }

    // transfer host mem to device mem
    int *d_cscColPtrTR;
    int *d_cscRowIdxTR;
    magmaDoubleComplex *d_cscValTR;
    magmaDoubleComplex *d_b;
    magmaDoubleComplex *d_x;

    // Matrix L
    cudaMalloc((void **)&d_cscColPtrTR, (n+1) * sizeof(int));
    cudaMalloc((void **)&d_cscRowIdxTR, nnzTR  * sizeof(int));
    cudaMalloc((void **)&d_cscValTR,    nnzTR  * sizeof(magmaDoubleComplex));

    cudaMemcpy(d_cscColPtrTR, cscColPtrTR, (n+1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscRowIdxTR, cscRowIdxTR, nnzTR  * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscValTR,    cscValTR,    nnzTR  * sizeof(magmaDoubleComplex),   cudaMemcpyHostToDevice);

    // Vector b
    cudaMalloc((void **)&d_b, m * rhs * sizeof(magmaDoubleComplex));
    cudaMemcpy(d_b, b, m * rhs * sizeof(magmaDoubleComplex), cudaMemcpyHostToDevice);

    // Vector x
    cudaMalloc((void **)&d_x, n * rhs * sizeof(magmaDoubleComplex));
    cudaMemset(d_x, 0, n * rhs * sizeof(magmaDoubleComplex));

    //  - cuda syncfree SpTRSV analysis start!
    printf(" - cuda syncfree SpTRSV analysis start!\n");

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    // malloc tmp memory to generate in-degree
    int *d_graphInDegree;
    int *d_graphInDegree_backup;
    cudaMalloc((void **)&d_graphInDegree, m * sizeof(int));
    cudaMalloc((void **)&d_graphInDegree_backup, m * sizeof(int));

    int num_threads = 128;
    int num_blocks = ceil ((double)nnzTR / (double)num_threads);

    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        cudaMemset(d_graphInDegree, 0, m * sizeof(int));
        sptrsv_syncfree_analyser<<< num_blocks, num_threads >>>
                                      (d_cscRowIdxTR, m, nnzTR, d_graphInDegree);
    }
    cudaDeviceSynchronize();

    gettimeofday(&t2, NULL);
    double time_analysis = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_analysis /= BENCH_REPEAT;

    printf("cuda syncfree SpTRSV analysis on L used %4.2f ms\n", time_analysis);

    //  - cuda syncfree SpTRSV solve start!
    printf(" - cuda syncfree SpTRSV solve start!\n");

    // malloc tmp memory to collect a partial sum of each row
    magmaDoubleComplex *d_left_sum;
    cudaMalloc((void **)&d_left_sum, sizeof(magmaDoubleComplex) * m * rhs);

    // backup in-degree array, only used for benchmarking multiple runs
    cudaMemcpy(d_graphInDegree_backup, d_graphInDegree, m * sizeof(int), cudaMemcpyDeviceToDevice);

    // step 5: solve L*y = x
    double time_solve = 0;

    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        // get a unmodified in-degree array, only for benchmarking use
        cudaMemcpy(d_graphInDegree, d_graphInDegree_backup, m * sizeof(int), cudaMemcpyDeviceToDevice);
        
        // clear left_sum array, only for benchmarking use
        cudaMemset(d_left_sum, 0, sizeof(magmaDoubleComplex) * m * rhs);
        cudaMemset(d_x, 0, sizeof(magmaDoubleComplex) * n * rhs);

        gettimeofday(&t1, NULL);

        if (rhs == 1)
        {
            num_threads = MAGMA_CSC_SYNCFREE_WARP_PER_BLOCK * MAGMA_CSC_SYNCFREE_WARP_SIZE;
            num_blocks = ceil ((double)m / (double)(num_threads/MAGMA_CSC_SYNCFREE_WARP_SIZE));
            sptrsv_syncfree_executor<<< num_blocks, num_threads >>>
                                         (d_cscColPtrTR, d_cscRowIdxTR, d_cscValTR,
                                          d_graphInDegree, d_left_sum,
                                          m, substitution, d_b, d_x);
        }
        else
        {
            num_threads = 4 * MAGMA_CSC_SYNCFREE_WARP_SIZE;
            num_blocks = ceil ((double)m / (double)(num_threads/MAGMA_CSC_SYNCFREE_WARP_SIZE));
            sptrsm_syncfree_executor<<< num_blocks, num_threads >>>
                                         (d_cscColPtrTR, d_cscRowIdxTR, d_cscValTR,
                                          d_graphInDegree, d_left_sum,
                                          m, substitution, rhs, opt,
                                          d_b, d_x);
        }

        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);

        time_solve += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    }

    time_solve /= BENCH_REPEAT;
    double flop = 2*(double)rhs*(double)nnzTR;

    printf("cuda syncfree SpTRSV solve used %4.2f ms, throughput is %4.2f gflops\n",
           time_solve, flop/(1e6*time_solve));
    *gflops = flop/(1e6*time_solve);

    cudaMemcpy(x, d_x, n * rhs * sizeof(magmaDoubleComplex), cudaMemcpyDeviceToHost);

    // validate x
    double accuracy = 1e-4;
    double ref = 0.0;
    double res = 0.0;

    for (int i = 0; i < n * rhs; i++)
    {
        ref += abs(x_ref[i]);
        res += abs(x[i] - x_ref[i]);
        //if (x_ref[i] != x[i]) printf ("[%i, %i] x_ref = %f, x = %f\n", i/rhs, i%rhs, x_ref[i], x[i]);
    }
    res = ref == 0 ? res : res / ref;

    if (res < accuracy)
        printf("cuda syncfree SpTRSV executor passed! |x-xref|/|xref| = %8.2e\n", res);
    else
        printf("cuda syncfree SpTRSV executor _NOT_ passed! |x-xref|/|xref| = %8.2e\n", res);

    // step 6: free resources
    free(while_profiler);

    cudaFree(d_graphInDegree);
    cudaFree(d_graphInDegree_backup);
    cudaFree(d_left_sum);

    cudaFree(d_cscColPtrTR);
    cudaFree(d_cscRowIdxTR);
    cudaFree(d_cscValTR);
    cudaFree(d_b);
    cudaFree(d_x);

    return 0;
}

#endif
*/


