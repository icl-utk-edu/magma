/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       @author Azzam Haidar
       @author Tingxing Dong
*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "batched_kernel_param.h"
#include "zlaswp_device.dp.hpp"

/******************************************************************************/
// parallel swap the swaped dA(1:nb,i:n) is stored in dout

void zlaswp_rowparallel_kernel(
                                int n, int width, int height,
                                magmaDoubleComplex *dinput, int ldi,
                                magmaDoubleComplex *doutput, int ldo,
                                magma_int_t*  pivinfo,
                                sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    zlaswp_rowparallel_devfunc(n, width, height, dinput, ldi, doutput, ldo,
                               pivinfo, item_ct1, dpct_local);
}


/******************************************************************************/

void zlaswp_rowparallel_kernel_batched(
                                int n, int width, int height,
                                magmaDoubleComplex **input_array, int input_i, int input_j, int ldi,
                                magmaDoubleComplex **output_array, int output_i, int output_j, int ldo,
                                magma_int_t** pivinfo_array,
                                sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    int batchid = item_ct1.get_group(0);
    zlaswp_rowparallel_devfunc(
        n, width, height, input_array[batchid] + input_j * ldi + input_i, ldi,
        output_array[batchid] + output_j * ldo + output_i, ldo,
        pivinfo_array[batchid], item_ct1, dpct_local);
}


/******************************************************************************/
extern "C" void
magma_zlaswp_rowparallel_batched( magma_int_t n,
                       magmaDoubleComplex**  input_array, magma_int_t  input_i, magma_int_t  input_j, magma_int_t ldi,
                       magmaDoubleComplex** output_array, magma_int_t output_i, magma_int_t output_j, magma_int_t ldo,
                       magma_int_t k1, magma_int_t k2,
                       magma_int_t **pivinfo_array,
                       magma_int_t batchCount, magma_queue_t queue)
{
#define  input_array(i,j)  input_array, i, j
#define output_array(i,j) output_array, i, j

    if (n == 0 ) return;
    int height = k2-k1;
    if ( height  > 1024)
    {
        fprintf( stderr, "%s: n=%lld > 1024, not supported\n", __func__, (long long) n );
    }

    int blocks = magma_ceildiv( n, SWP_WIDTH );
    magma_int_t max_batchCount = queue->get_maxBatch();

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, blocks);

        if ( n < SWP_WIDTH) {
            /*
            DPCT1083:1262: The size of local memory in the migrated code may be
            different from the original code. Check that the allocated memory
            size in the migrated code is correct.
            */
            size_t shmem = sizeof(magmaDoubleComplex) * height * n;
            /*
            DPCT1049:1261: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * sycl::range<3>(1, 1, height),
                                          sycl::range<3>(1, 1, height)),
                        [=](sycl::nd_item<3> item_ct1) {
                            zlaswp_rowparallel_kernel_batched(
                                n, n, height, input_array + i, input_i, input_j,
                                ldi, output_array + i, output_i, output_j, ldo,
                                pivinfo_array + i, item_ct1,
                                dpct_local_acc_ct1.get_pointer());
                        });
                });
        }
        else {
            /*
            DPCT1083:1264: The size of local memory in the migrated code may be
            different from the original code. Check that the allocated memory
            size in the migrated code is correct.
            */
            size_t shmem = sizeof(magmaDoubleComplex) * height * SWP_WIDTH;
            /*
            DPCT1049:1263: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * sycl::range<3>(1, 1, height),
                                          sycl::range<3>(1, 1, height)),
                        [=](sycl::nd_item<3> item_ct1) {
                            zlaswp_rowparallel_kernel_batched(
                                n, SWP_WIDTH, height, input_array + i, input_i,
                                input_j, ldi, output_array + i, output_i,
                                output_j, ldo, pivinfo_array + i, item_ct1,
                                dpct_local_acc_ct1.get_pointer());
                        });
                });
        }
    }
#undef  input_array
#undef output_attay
}


/******************************************************************************/
extern "C" void
magma_zlaswp_rowparallel_native(
    magma_int_t n,
    magmaDoubleComplex* input, magma_int_t ldi,
    magmaDoubleComplex* output, magma_int_t ldo,
    magma_int_t k1, magma_int_t k2,
    magma_int_t *pivinfo,
    magma_queue_t queue)
{
    if (n == 0 ) return;
    int height = k2-k1;
    if ( height  > MAX_NTHREADS)
    {
        fprintf( stderr, "%s: height=%lld > %lld, magma_zlaswp_rowparallel_q not supported\n",
                 __func__, (long long) n, (long long) MAX_NTHREADS );
    }

    int blocks = magma_ceildiv( n, SWP_WIDTH );
    sycl::range<3> grid(1, 1, blocks);

    if ( n < SWP_WIDTH)
    {
        /*
        DPCT1083:1266: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        size_t shmem = sizeof(magmaDoubleComplex) * height * n;
        /*
        DPCT1049:1265: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * sycl::range<3>(1, 1, height),
                                      sycl::range<3>(1, 1, height)),
                    [=](sycl::nd_item<3> item_ct1) {
                        zlaswp_rowparallel_kernel(
                            n, n, height, input, ldi, output, ldo, pivinfo,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
    }
    else
    {
        /*
        DPCT1083:1268: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        size_t shmem = sizeof(magmaDoubleComplex) * height * SWP_WIDTH;
        /*
        DPCT1049:1267: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * sycl::range<3>(1, 1, height),
                                      sycl::range<3>(1, 1, height)),
                    [=](sycl::nd_item<3> item_ct1) {
                        zlaswp_rowparallel_kernel(
                            n, SWP_WIDTH, height, input, ldi, output, ldo,
                            pivinfo, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
    }
}


/******************************************************************************/
// serial swap that does swapping one row by one row
void zlaswp_rowserial_kernel_batched( int n, magmaDoubleComplex **dA_array, int lda, int k1, int k2, magma_int_t** ipiv_array ,
                                      sycl::nd_item<3> item_ct1)
{
    magmaDoubleComplex *dA = dA_array[item_ct1.get_group(0)];
    magma_int_t *dipiv = ipiv_array[item_ct1.get_group(0)];

    unsigned int tid = item_ct1.get_local_id(2) +
                       item_ct1.get_local_range(2) * item_ct1.get_group(2);

    k1--;
    k2--;

    if (tid < n) {
        magmaDoubleComplex A1;

        for (int i1 = k1; i1 < k2; i1++)
        {
            int i2 = dipiv[i1] - 1;  // Fortran index, switch i1 and i2
            if ( i2 != i1)
            {
                A1 = dA[i1 + tid * lda];
                dA[i1 + tid * lda] = dA[i2 + tid * lda];
                dA[i2 + tid * lda] = A1;
            }
        }
    }
}


/******************************************************************************/
// serial swap that does swapping one row by one row
void zlaswp_rowserial_kernel_native( int n, magmaDoubleComplex_ptr dA, int lda, int k1, int k2, magma_int_t* dipiv ,
                                     sycl::nd_item<3> item_ct1)
{
    unsigned int tid = item_ct1.get_local_id(2) +
                       item_ct1.get_local_range(2) * item_ct1.get_group(2);

    //k1--;
    //k2--;

    if (tid < n) {
        magmaDoubleComplex A1;

        for (int i1 = k1; i1 < k2; i1++)
        {
            int i2 = dipiv[i1] - 1;  // Fortran index, switch i1 and i2
            if ( i2 != i1)
            {
                A1 = dA[i1 + tid * lda];
                dA[i1 + tid * lda] = dA[i2 + tid * lda];
                dA[i2 + tid * lda] = A1;
            }
        }
    }
}


/******************************************************************************/
// serial swap that does swapping one row by one row, similar to LAPACK
// K1, K2 are in Fortran indexing
extern "C" void
magma_zlaswp_rowserial_batched(magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t lda,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t **ipiv_array,
                   magma_int_t batchCount, magma_queue_t queue)
{
    if (n == 0) return;

    int blocks = magma_ceildiv( n, BLK_SIZE );
    magma_int_t max_batchCount = queue->get_maxBatch();

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, blocks);

        magma_int_t max_BLK_SIZE__n = max(BLK_SIZE, n);
        /*
        DPCT1049:1269: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, max_BLK_SIZE__n),
                                  sycl::range<3>(1, 1, max_BLK_SIZE__n)),
                [=](sycl::nd_item<3> item_ct1) {
                    zlaswp_rowserial_kernel_batched(
                        n, dA_array + i, lda, k1, k2, ipiv_array + i, item_ct1);
                });
    }
}



/******************************************************************************/
// serial swap that does swapping one row by one row, similar to LAPACK
// K1, K2 are in Fortran indexing
extern "C" void
magma_zlaswp_rowserial_native(magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t lda,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t* dipiv, magma_queue_t queue)
{
    if (n == 0) return;

    int blocks = magma_ceildiv( n, BLK_SIZE );
    sycl::range<3> grid(1, 1, blocks);

    size_t max_BLK_SIZE_n = max(BLK_SIZE, n);
    /*
    DPCT1049:1270: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(
            sycl::nd_range<3>(grid * sycl::range<3>(1, 1, max_BLK_SIZE_n),
                              sycl::range<3>(1, 1, max_BLK_SIZE_n)),
            [=](sycl::nd_item<3> item_ct1) {
                zlaswp_rowserial_kernel_native(n, dA, lda, k1, k2, dipiv,
                                               item_ct1);
            });
}



/******************************************************************************/
// serial swap that does swapping one column by one column
void zlaswp_columnserial_devfunc(int n, magmaDoubleComplex_ptr dA, int lda, int k1, int k2, magma_int_t* dipiv ,
                                 sycl::nd_item<3> item_ct1)
{
    unsigned int tid = item_ct1.get_local_id(2) +
                       item_ct1.get_local_range(2) * item_ct1.get_group(2);
    k1--;
    k2--;
    if ( k1 < 0 || k2 < 0 ) return;


    if ( tid < n) {
        magmaDoubleComplex A1;
        if (k1 <= k2)
        {
            for (int i1 = k1; i1 <= k2; i1++)
            {
                int i2 = dipiv[i1] - 1;  // Fortran index, switch i1 and i2
                if ( i2 != i1)
                {
                    A1 = dA[i1 * lda + tid];
                    dA[i1 * lda + tid] = dA[i2 * lda + tid];
                    dA[i2 * lda + tid] = A1;
                }
            }
        } else
        {

            for (int i1 = k1; i1 >= k2; i1--)
            {
                int i2 = dipiv[i1] - 1;  // Fortran index, switch i1 and i2
                if ( i2 != i1)
                {
                    A1 = dA[i1 * lda + tid];
                    dA[i1 * lda + tid] = dA[i2 * lda + tid];
                    dA[i2 * lda + tid] = A1;
                }
            }
        }
    }
}


void zlaswp_columnserial_kernel_batched( int n, magmaDoubleComplex **dA_array, int lda, int k1, int k2, magma_int_t** ipiv_array ,
                                         sycl::nd_item<3> item_ct1)
{
    magmaDoubleComplex *dA = dA_array[item_ct1.get_group(0)];
    magma_int_t *dipiv = ipiv_array[item_ct1.get_group(0)];

    zlaswp_columnserial_devfunc(n, dA, lda, k1, k2, dipiv, item_ct1);
}

void zlaswp_columnserial_kernel( int n, magmaDoubleComplex_ptr dA, int lda, int k1, int k2, magma_int_t* dipiv ,
                                 sycl::nd_item<3> item_ct1)
{
    zlaswp_columnserial_devfunc(n, dA, lda, k1, k2, dipiv, item_ct1);
}

/******************************************************************************/
// serial swap that does swapping one column by one column
// K1, K2 are in Fortran indexing
extern "C" void
magma_zlaswp_columnserial(
    magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t lda,
    magma_int_t k1, magma_int_t k2,
    magma_int_t *dipiv, magma_queue_t queue)
{
    if (n == 0 ) return;

    int blocks = magma_ceildiv( n, ZLASWP_COL_NTH );
    sycl::range<3> grid(1, 1, blocks);

    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(
            sycl::nd_range<3>(grid * sycl::range<3>(1, 1, ZLASWP_COL_NTH),
                              sycl::range<3>(1, 1, ZLASWP_COL_NTH)),
            [=](sycl::nd_item<3> item_ct1) {
                zlaswp_columnserial_kernel(n, dA, lda, k1, k2, dipiv, item_ct1);
            });
}

extern "C" void
magma_zlaswp_columnserial_batched(magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t lda,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t **ipiv_array,
                   magma_int_t batchCount, magma_queue_t queue)
{
    if (n == 0 ) return;

    int blocks = magma_ceildiv( n, ZLASWP_COL_NTH );

    magma_int_t max_batchCount = queue->get_maxBatch();

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, blocks);

        magma_int_t min_ZLASWP_COL_NTH__n = min(ZLASWP_COL_NTH, n);
        /*
        DPCT1049:1271: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(
                sycl::nd_range<3>(
                    grid * sycl::range<3>(1, 1, min_ZLASWP_COL_NTH__n),
                    sycl::range<3>(1, 1, min_ZLASWP_COL_NTH__n)),
                [=](sycl::nd_item<3> item_ct1) {
                    zlaswp_columnserial_kernel_batched(
                        n, dA_array + i, lda, k1, k2, ipiv_array + i, item_ct1);
                });
    }
}
