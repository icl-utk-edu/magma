/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

/******************************************************************************/
/*
 *  Swap diagonal blocks of two matrices.
 *  Each thread block swaps one diagonal block.
 *  Each thread iterates across one row of the block.
 */

void
zswapdblk_batched_kernel( int nb, int n_mod_nb,
                  magmaDoubleComplex **dA_array, int ldda, int inca,
                  magmaDoubleComplex **dB_array, int lddb, int incb ,
                  sycl::nd_item<3> item_ct1)
{
    const int tx = item_ct1.get_local_id(2);
    const int bx = item_ct1.get_group(2);
    const int batchid = item_ct1.get_group(0);

    magmaDoubleComplex *dA = dA_array[batchid];
    magmaDoubleComplex *dB = dB_array[batchid];

    dA += tx + bx * nb * (ldda + inca);
    dB += tx + bx * nb * (lddb + incb);

    magmaDoubleComplex tmp;

    if (bx < item_ct1.get_group_range(2) - 1)
    {
        #pragma unroll
        for( int i = 0; i < nb; i++ ) {
            tmp        = dA[i*ldda];
            dA[i*ldda] = dB[i*lddb];
            dB[i*lddb] = tmp;
        }
    }
    else
    {
        for( int i = 0; i < n_mod_nb; i++ ) {
            tmp        = dA[i*ldda];
            dA[i*ldda] = dB[i*lddb];
            dB[i*lddb] = tmp;
        }
    }
}


/***************************************************************************//**
    Purpose
    -------
    zswapdblk swaps diagonal blocks of size nb x nb between matrices
    dA and dB on the GPU. It swaps nblocks = ceil(n/nb) blocks.
    For i = 1 .. nblocks, submatrices
    dA( i*nb*inca, i*nb ) and
    dB( i*nb*incb, i*nb ) are swapped.

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The number of columns of the matrices dA and dB.  N >= 0.

    @param[in]
    nb      INTEGER
            The size of diagonal blocks.
            NB > 0 and NB <= maximum threads per CUDA block (512 or 1024).

    @param[in,out]
    dA_array Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array dA, dimension (ldda,n)
             The matrix dA.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array dA.
            ldda >= (nblocks - 1)*nb*inca + nb.

    @param[in]
    inca    INTEGER
            The row increment between diagonal blocks of dA. inca >= 0. For example,
            inca = 1 means blocks are stored on the diagonal at dA(i*nb, i*nb),
            inca = 0 means blocks are stored side-by-side    at dA(0,    i*nb).

    @param[in,out]
    dB_array Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array dB, dimension (lddb,n)
             The matrix dB.

    @param[in]
    lddb    INTEGER
            The leading dimension of each array dB.
            lddb >= (nblocks - 1)*nb*incb + nb.

    @param[in]
    incb    INTEGER
            The row increment between diagonal blocks of dB. incb >= 0. See inca.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_swapdblk
*******************************************************************************/
extern "C" void
magmablas_zswapdblk_batched(
    magma_int_t n, magma_int_t nb,
    magmaDoubleComplex **dA_array, magma_int_t ldda, magma_int_t inca,
    magmaDoubleComplex **dB_array, magma_int_t lddb, magma_int_t incb,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t nblocks = magma_ceildiv( n, nb );
    magma_int_t n_mod_nb = n % nb;

    magma_int_t info = 0;
    if (n < 0) {
        info = -1;
    } else if (nb < 1 || nb > 1024) {
        info = -2;
    } else if (ldda < (nblocks-1)*nb*inca + nb) {
        info = -4;
    } else if (inca < 0) {
        info = -5;
    } else if (lddb < (nblocks-1)*nb*incb + nb) {
        info = -7;
    } else if (incb < 0) {
        info = -8;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }

    if (n_mod_nb == 0) nblocks += 1; // a dummy thread block for cleanup code

    sycl::range<3> dimBlock(1, 1, nb);
    magma_int_t max_batchCount = queue->get_maxBatch();
    if ( nblocks > 0 ) {

        for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
            magma_int_t ibatch = min(max_batchCount, batchCount-i);
            sycl::range<3> dimGrid(ibatch, 1, nblocks);

            /*
            DPCT1049:1330: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->cuda_stream()))
                ->parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                               [=](sycl::nd_item<3> item_ct1) {
                                   zswapdblk_batched_kernel(
                                       nb, n_mod_nb, dA_array + i, ldda, inca,
                                       dB_array + i, lddb, incb, item_ct1);
                               });
        }
    }
}
