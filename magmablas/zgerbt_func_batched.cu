/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       @author Adrien Remy
       @author Azzam Haidar
*/
#include "magma_internal.h"
#include "zgerbt.h"

#define block_height  32
#define block_width  4
#define block_length 256
#define NB 64

/***************************************************************************//**
    Purpose
    -------
    ZPRBT_MVT compute B = UTB to randomize B

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The number of values of db.  n >= 0.

    @param[in]
    du     COMPLEX_16 array, dimension (n,2)
            The 2*n vector representing the random butterfly matrix V

    @param[in,out]
    db     COMPLEX_16 array, dimension (n)
            The n vector db computed by ZGESV_NOPIV_GPU
            On exit db = du*db

    @param[in]
    queue   magma_queue_t
            Queue to execute in.
*******************************************************************************/
extern "C" void
magmablas_zprbt_mtv_batched(
    magma_int_t n,
    magmaDoubleComplex *du, magmaDoubleComplex **db_array,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t threads = block_length;
    magma_int_t max_batchCount = queue->get_maxBatch();

    for(int i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( magma_ceildiv( n, 4*block_length ), ibatch);

        magmablas_zapply_transpose_vector_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>(n/2, du, n, db_array+i, 0);
        magmablas_zapply_transpose_vector_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>(n/2, du, n+n/2, db_array+i, n/2);

        threads = block_length;
        grid = magma_ceildiv( n, 2*block_length );
        magmablas_zapply_transpose_vector_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>(n, du, 0, db_array+i, 0);
    }
}


/***************************************************************************//**
    Purpose
    -------
    ZPRBT_MV compute B = VB to obtain the non randomized solution

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The number of values of db.  n >= 0.

    @param[in,out]
    db      COMPLEX_16 array, dimension (n)
            The n vector db computed by ZGESV_NOPIV_GPU
            On exit db = dv*db

    @param[in]
    dv      COMPLEX_16 array, dimension (n,2)
            The 2*n vector representing the random butterfly matrix V

    @param[in]
    queue   magma_queue_t
            Queue to execute in.
*******************************************************************************/
extern "C" void
magmablas_zprbt_mv_batched(
    magma_int_t n,
    magmaDoubleComplex *dv, magmaDoubleComplex **db_array,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t threads = block_length;
    magma_int_t max_batchCount = queue->get_maxBatch();

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid ( magma_ceildiv( n, 2*block_length ), ibatch);
        magmablas_zapply_vector_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>(n, dv, 0, db_array+i, 0);

        threads = block_length;
        grid = magma_ceildiv( n, 4*block_length );
        magmablas_zapply_vector_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>(n/2, dv, n, db_array+i, 0);
        magmablas_zapply_vector_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>(n/2, dv, n+n/2, db_array+i, n/2);
    }
}


/***************************************************************************//**
    Purpose
    -------
    ZPRBT randomize a square general matrix using partial randomized transformation

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The number of columns and rows of the matrix dA.  n >= 0.

    @param[in,out]
    dA      COMPLEX_16 array, dimension (n,ldda)
            The n-by-n matrix dA
            On exit dA = duT*dA*d_V

    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDA >= max(1,n).

    @param[in]
    du      COMPLEX_16 array, dimension (n,2)
            The 2*n vector representing the random butterfly matrix U

    @param[in]
    dv      COMPLEX_16 array, dimension (n,2)
            The 2*n vector representing the random butterfly matrix V

    @param[in]
    queue   magma_queue_t
            Queue to execute in.
*******************************************************************************/
extern "C" void
magmablas_zprbt_batched(
    magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magmaDoubleComplex *du, magmaDoubleComplex *dv,
    magma_int_t batchCount, magma_queue_t queue)
{
    du += ldda;
    dv += ldda;

    dim3 threads(block_height, block_width);
    dim3 threads2(block_height, block_width);
    magma_int_t max_batchCount = queue->get_maxBatch();

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( magma_ceildiv( n, 4*block_height ), magma_ceildiv( n, 4*block_width  ), ibatch );

        magmablas_zelementary_multiplication_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>(n/2, dA_array+i,            0, ldda, du,   0, dv,   0);
        magmablas_zelementary_multiplication_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>(n/2, dA_array+i,     ldda*n/2, ldda, du,   0, dv, n/2);
        magmablas_zelementary_multiplication_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>(n/2, dA_array+i,          n/2, ldda, du, n/2, dv,   0);
        magmablas_zelementary_multiplication_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>(n/2, dA_array+i, ldda*n/2+n/2, ldda, du, n/2, dv, n/2);

        dim3 grid2( magma_ceildiv( n, 2*block_height ), magma_ceildiv( n, 2*block_width  ), ibatch );
        magmablas_zelementary_multiplication_kernel_batched<<< grid2, threads2, 0, queue->cuda_stream() >>>(n, dA_array+i, 0, ldda, du, -ldda, dv, -ldda);
    }
}
