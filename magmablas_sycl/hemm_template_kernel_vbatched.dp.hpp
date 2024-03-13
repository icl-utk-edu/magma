/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah
*/

#ifndef HEMM_TEMPLATE_KERNEL_VBATCHED_CUH
#define HEMM_TEMPLATE_KERNEL_VBATCHED_CUH

////////////////////////////////////////////////////////////////////////////////////////////////////
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "gemm_template_device_defs.dp.hpp"
#include "hemm_template_device.dp.hpp"
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
static 
void hemm_template_vbatched_ll_kernel(
    magma_int_t *M, magma_int_t *N,
    T const * const * Aarray, magma_int_t *LDA,
    T const * const * Barray, magma_int_t *LDB,
    T**       Carray, magma_int_t *LDC,
    T alpha, T beta,
    int roffA, int coffA, int roffB, int coffB, int roffC, int coffC,
    int specM, int specN, sycl::nd_item<3> item_ct1,
    sycl::local_accessor<T, 2> sA,
    sycl::local_accessor<T, 2> sB)
{
    const int batchid = item_ct1.get_group(0);
    int my_M = (int)M[batchid];
    int my_N = (int)N[batchid];
    // check if offsets produce out-of-bound pointers
    if( my_M < roffA || my_M < coffA ) return;
    if( my_M < roffB || my_N < coffB ) return;
    if( my_M < roffC || my_N < coffC ) return;
    // compute the maximum allowed value for m, n, k based on the input offsets
    my_M -= max ( max(roffA, roffC), max(coffA, roffB) );
    my_N -= max( coffB, coffC );
    // check if the user forces values for m, n, and k
    my_M = ( specM <= 0 ) ? my_M : min( my_M, specM );
    my_N = ( specN <= 0 ) ? my_N : min( my_N, specN );

    if(my_M <= 0 || my_N <= 0 ) return;
    if( Aarray[batchid] == NULL || Barray[batchid] == NULL || Carray[batchid] == NULL ) return;
    if (item_ct1.get_group(2) >= magma_ceildiv(my_M, BLK_M)) return;
    if (item_ct1.get_group(1) >= magma_ceildiv(my_N, BLK_N)) return;

    hemm_template_device_ll<T, DIM, BLK_M, BLK_N, (BLK_M / DIM), (BLK_N / DIM),
                            CONJA>(
        my_M, my_N, Aarray[batchid] + (int)LDA[batchid] * coffA + roffA,
        (int)LDA[batchid], Barray[batchid] + (int)LDB[batchid] * coffB + roffB,
        (int)LDB[batchid], Carray[batchid] + (int)LDC[batchid] * coffC + roffC,
        (int)LDC[batchid], alpha, beta, item_ct1, sA, sB);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
static 
void hemm_template_vbatched_lu_kernel(
    magma_int_t *M, magma_int_t *N,
    T const * const * Aarray, magma_int_t *LDA,
    T const * const * Barray, magma_int_t *LDB,
    T**       Carray, magma_int_t *LDC,
    T alpha, T beta,
    int roffA, int coffA, int roffB, int coffB, int roffC, int coffC,
    int specM, int specN, sycl::nd_item<3> item_ct1,
    sycl::local_accessor<T, 2> sA,
    sycl::local_accessor<T, 2> sB)
{
    const int batchid = item_ct1.get_group(0);
    int my_M = (int)M[batchid];
    int my_N = (int)N[batchid];
    // check if offsets produce out-of-bound pointers
    if( my_M < roffA || my_M < coffA ) return;
    if( my_M < roffB || my_N < coffB ) return;
    if( my_M < roffC || my_N < coffC ) return;
    // compute the maximum allowed value for m, n, k based on the input offsets
    my_M -= max ( max(roffA, roffC), max(coffA, roffB) );
    my_N -= max( coffB, coffC );
    // check if the user forces values for m, n, and k
    my_M = ( specM <= 0 ) ? my_M : min( my_M, specM );
    my_N = ( specN <= 0 ) ? my_N : min( my_N, specN );

    if(my_M <= 0 || my_N <= 0 ) return;
    if( Aarray[batchid] == NULL || Barray[batchid] == NULL || Carray[batchid] == NULL ) return;
    if (item_ct1.get_group(2) >= magma_ceildiv(my_M, BLK_M)) return;
    if (item_ct1.get_group(1) >= magma_ceildiv(my_N, BLK_N)) return;

    hemm_template_device_lu<T, DIM, BLK_M, BLK_N, (BLK_M / DIM), (BLK_N / DIM),
                            CONJA>(
        my_M, my_N, Aarray[batchid] + (int)LDA[batchid] * coffA + roffA,
        (int)LDA[batchid], Barray[batchid] + (int)LDB[batchid] * coffB + roffB,
        (int)LDB[batchid], Carray[batchid] + (int)LDC[batchid] * coffC + roffC,
        (int)LDC[batchid], alpha, beta, item_ct1, sA, sB);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
static 
void hemm_template_vbatched_rl_kernel(
    magma_int_t *M, magma_int_t *N,
    T const * const * Aarray, magma_int_t *LDA,
    T const * const * Barray, magma_int_t *LDB,
    T**       Carray, magma_int_t *LDC,
    T alpha, T beta,
    int roffA, int coffA, int roffB, int coffB, int roffC, int coffC,
    int specM, int specN, sycl::nd_item<3> item_ct1,
    sycl::local_accessor<T, 2> sA,
    sycl::local_accessor<T, 2> sB)
{
    const int batchid = item_ct1.get_group(0);
    int my_M = (int)M[batchid];
    int my_N = (int)N[batchid];
    // check if offsets produce out-of-bound pointers
    if( my_N < roffA || my_N < coffA ) return;
    if( my_M < roffB || my_N < coffB ) return;
    if( my_M < roffC || my_N < coffC ) return;
    // compute the maximum allowed value for m, n, k based on the input offsets
    my_M -= max( roffB, roffC );
    my_N -= max( max(coffB, roffA), max(coffA, coffC) );
    // check if the user forces values for m, n, and k
    my_M = ( specM <= 0 ) ? my_M : min( my_M, specM );
    my_N = ( specN <= 0 ) ? my_N : min( my_N, specN );

    if(my_M <= 0 || my_N <= 0 ) return;
    if( Aarray[batchid] == NULL || Barray[batchid] == NULL || Carray[batchid] == NULL ) return;
    if (item_ct1.get_group(2) >= magma_ceildiv(my_M, BLK_M)) return;
    if (item_ct1.get_group(1) >= magma_ceildiv(my_N, BLK_N)) return;

    hemm_template_device_rl<T, DIM, BLK_M, BLK_N, (BLK_M / DIM), (BLK_N / DIM),
                            CONJA>(
        my_M, my_N, Aarray[batchid] + (int)LDA[batchid] * coffA + roffA,
        (int)LDA[batchid], Barray[batchid] + (int)LDB[batchid] * coffB + roffB,
        (int)LDB[batchid], Carray[batchid] + (int)LDC[batchid] * coffC + roffC,
        (int)LDC[batchid], alpha, beta, item_ct1, sA, sB);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
static 
void hemm_template_vbatched_ru_kernel(
    magma_int_t *M, magma_int_t *N,
    T const * const * Aarray, magma_int_t *LDA,
    T const * const * Barray, magma_int_t *LDB,
    T**       Carray, magma_int_t *LDC,
    T alpha, T beta,
    int roffA, int coffA, int roffB, int coffB, int roffC, int coffC,
    int specM, int specN, sycl::nd_item<3> item_ct1,
    sycl::local_accessor<T, 2> sA,
    sycl::local_accessor<T, 2> sB)
{
    const int batchid = item_ct1.get_group(0);
    int my_M = (int)M[batchid];
    int my_N = (int)N[batchid];
    // check if offsets produce out-of-bound pointers
    if( my_N < roffA || my_N < coffA ) return;
    if( my_M < roffB || my_N < coffB ) return;
    if( my_M < roffC || my_N < coffC ) return;
    // compute the maximum allowed value for m, n, k based on the input offsets
    my_M -= max( roffB, roffC );
    my_N -= max( max(coffB, roffA), max(coffA, coffC) );
    // check if the user forces values for m, n, and k
    my_M = ( specM <= 0 ) ? my_M : min( my_M, specM );
    my_N = ( specN <= 0 ) ? my_N : min( my_N, specN );

    if(my_M <= 0 || my_N <= 0 ) return;
    if( Aarray[batchid] == NULL || Barray[batchid] == NULL || Carray[batchid] == NULL ) return;
    if (item_ct1.get_group(2) >= magma_ceildiv(my_M, BLK_M)) return;
    if (item_ct1.get_group(1) >= magma_ceildiv(my_N, BLK_N)) return;

    hemm_template_device_ru<T, DIM, BLK_M, BLK_N, (BLK_M / DIM), (BLK_N / DIM),
                            CONJA>(
        my_M, my_N, Aarray[batchid] + (int)LDA[batchid] * coffA + roffA,
        (int)LDA[batchid], Barray[batchid] + (int)LDB[batchid] * coffB + roffB,
        (int)LDB[batchid], Carray[batchid] + (int)LDC[batchid] * coffC + roffC,
        (int)LDC[batchid], alpha, beta, item_ct1, sA, sB);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel wrappers
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
void hemm_template_vbatched(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t *m, magma_int_t *n,
    T const * const * dA_array, magma_int_t *ldda,
    T const * const * dB_array, magma_int_t *lddb,
    T**       dC_array, magma_int_t *lddc,
    T alpha, T beta,
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, magma_int_t roffC, magma_int_t coffC,
    magma_int_t specM, magma_int_t specN,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    sycl::range<3> threads(1, DIM, DIM);
    if( side == MagmaLeft ){
        if(uplo == MagmaLower){
            for(magma_int_t i = 0; i < batchCount; i+=max_batchCount){
                magma_int_t ibatch = min(max_batchCount, batchCount-i);
                sycl::range<3> grid(ibatch, magma_ceildiv(max_n, BLK_N),
                                    magma_ceildiv(max_m, BLK_M));

                /*
                DPCT1049:855: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::local_accessor<T, 2>
                            sA_acc_ct1(sycl::range<2>(BLK_M, BLK_M + 1), cgh);
                        sycl::local_accessor<T, 2>
                            sB_acc_ct1(sycl::range<2>(BLK_N, BLK_M + 1), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                hemm_template_vbatched_ll_kernel<T, DIM, BLK_M,
                                                                 BLK_N, CONJA>(
                                    m + i, n + i, dA_array + i, ldda + i,
                                    dB_array + i, lddb + i, dC_array + i,
                                    lddc + i, alpha, beta, roffA, coffA, roffB,
                                    coffB, roffC, coffC, specM, specN, item_ct1,
                                    sA_acc_ct1, sB_acc_ct1);
                            });
                    });
            }
        }else{
            for(magma_int_t i = 0; i < batchCount; i+=max_batchCount){
                magma_int_t ibatch = min(max_batchCount, batchCount-i);
                sycl::range<3> grid(ibatch, magma_ceildiv(max_n, BLK_N),
                                    magma_ceildiv(max_m, BLK_M));

                /*
                DPCT1049:856: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::local_accessor<T, 2>
                            sA_acc_ct1(sycl::range<2>(BLK_M, BLK_M + 1), cgh);
                        sycl::local_accessor<T, 2>
                            sB_acc_ct1(sycl::range<2>(BLK_N, BLK_M + 1), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                hemm_template_vbatched_lu_kernel<T, DIM, BLK_M,
                                                                 BLK_N, CONJA>(
                                    m + i, n + i, dA_array + i, ldda + i,
                                    dB_array + i, lddb + i, dC_array + i,
                                    lddc + i, alpha, beta, roffA, coffA, roffB,
                                    coffB, roffC, coffC, specM, specN, item_ct1,
                                    sA_acc_ct1, sB_acc_ct1);
                            });
                    });
            }
        }
    }else{
        if(uplo == MagmaLower){
            for(magma_int_t i = 0; i < batchCount; i+=max_batchCount){
                magma_int_t ibatch = min(max_batchCount, batchCount-i);
                sycl::range<3> grid(ibatch, magma_ceildiv(max_n, BLK_N),
                                    magma_ceildiv(max_m, BLK_M));

                /*
                DPCT1049:857: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::local_accessor<T, 2>
                            sA_acc_ct1(sycl::range<2>(BLK_N, BLK_N + 1), cgh);
                        sycl::local_accessor<T, 2>
                            sB_acc_ct1(sycl::range<2>(BLK_N, BLK_M + 1), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                hemm_template_vbatched_rl_kernel<T, DIM, BLK_M,
                                                                 BLK_N, CONJA>(
                                    m + i, n + i, dA_array + i, ldda + i,
                                    dB_array + i, lddb + i, dC_array + i,
                                    lddc + i, alpha, beta, roffA, coffA, roffB,
                                    coffB, roffC, coffC, specM, specN, item_ct1,
                                    sA_acc_ct1, sB_acc_ct1);
                            });
                    });
            }
        }else{
            for(magma_int_t i = 0; i < batchCount; i+=max_batchCount){
                magma_int_t ibatch = min(max_batchCount, batchCount-i);
                sycl::range<3> grid(ibatch, magma_ceildiv(max_n, BLK_N),
                                    magma_ceildiv(max_m, BLK_M));

                /*
                DPCT1049:858: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::local_accessor<T, 2>
                            sA_acc_ct1(sycl::range<2>(BLK_N, BLK_N + 1), cgh);
                        sycl::local_accessor<T, 2>
                            sB_acc_ct1(sycl::range<2>(BLK_N, BLK_M + 1), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                hemm_template_vbatched_ru_kernel<T, DIM, BLK_M,
                                                                 BLK_N, CONJA>(
                                    m + i, n + i, dA_array + i, ldda + i,
                                    dB_array + i, lddb + i, dC_array + i,
                                    lddc + i, alpha, beta, roffA, coffA, roffB,
                                    coffB, roffC, coffC, specM, specN, item_ct1,
                                    sA_acc_ct1, sB_acc_ct1);
                            });
                    });
            }
        }
    }
}
#endif //HEMM_TEMPLATE_KERNEL_VBATCHED_CUH
