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
        int length, magma_int_t *N,
        magmaDoubleComplex **x_array, int xi, int xj, int incx, magma_int_t *lda,
        int step, magma_int_t** ipiv_array, magma_int_t *info_array, int gbstep )
{
    extern __shared__ double sdata[];

    const int batchid = blockIdx.x;

    // compute the actual length
    int my_N   = (int)N[batchid];
    int my_lda = (int)lda[batchid];
    // check if offsets produce out-of-bound pointers
    if( my_N < xi || my_N < xj ) return;
    // compute the maximum allowed value for length based on the input offsets
    my_N -= max( (xi+step), (xj+step) );
    // check the length
    if(my_N <= 0) return;

    magmaDoubleComplex *x_start = x_array[batchid] + xj * lda + xi;
    const magmaDoubleComplex *x = &(x_start[step + step * lda]);

    magma_int_t *ipiv = ipiv_array[batchid] + xi;
    int tx = threadIdx.x;

    double *shared_x = sdata;
    int *shared_idx = (int*)(shared_x + zamax);

    izamax_devfunc(my_N, x, incx, shared_x, shared_idx);

    if (tx == 0) {
        ipiv[step]  = shared_idx[0] + step + 1; // Fortran Indexing
        if (shared_x[0] == MAGMA_D_ZERO) {
            info_array[batchid] = shared_idx[0] + step + gbstep + 1;
        }
    }
}

/******************************************************************************/
extern "C" magma_int_t
magma_izamax_vbatched(
        magma_int_t length, magma_int_t *N,
        magmaDoubleComplex **x_array, magma_int_t xi, magma_int_t xj, magma_int_t incx, magma_int_t* lda,
        magma_int_t step, magma_int_t** ipiv_array, magma_int_t *info_array,
        magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue)
{
    if (length == 0 ) return 0;

    dim3 grid(batchCount, 1, 1);
    dim3 threads(zamax, 1, 1);

    izamax_kernel_vbatched<<< grid, threads, zamax * (sizeof(double) + sizeof(int)), queue->cuda_stream() >>>
    (length, N, x_array, xi, xj, incx, lda, step, ipiv_array, info_array, gbstep);

    return 0;
}
