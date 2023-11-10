/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Azzam Haidar
   @author Ahmad Ahmad

   @precisions normal z -> s d c
 */
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "batched_kernel_param.h"

#define PRECISION_z

#if defined(VERSION31)
    #define ENABLE_COND1
    #define ENABLE_COND2
    #define ENABLE_COND4
    #define ENABLE_COND5
    #define ENABLE_COND6
#endif

#define MAX_NTCOL 8
#if defined(PRECISION_s)
#define NTCOL2   (4)
#define NTCOL1   (8)
#elif defined(PRECISION_d)
#define NTCOL2   (2)
#define NTCOL1   (4)
#else
#define NTCOL2   (1)
#define NTCOL1   (1)
#endif

#include "zpotf2_devicesfunc.dp.hpp"

#define A(i_, j_)  (dA + (i_) + (j_)*ldda)
/******************************************************************************/
void zpotf2_smlpin_fixwidth_kernel(int m, magmaDoubleComplex *dA, int ldda, int localstep, int gbstep, magma_int_t *dinfo,
                                   sycl::nd_item<3> item_ct1,
                                   uint8_t *dpct_local)
{
    #pragma unroll
    for(int i = 0; i < m; i+= POTF2_NB){
        if (item_ct1.get_local_id(2) < m - i) {
            zpotf2_smlpout_fixwidth_device(
                m - i, A(localstep + i, 0), A(localstep + i, localstep + i),
                ldda, localstep + i, gbstep, dinfo, item_ct1, dpct_local);
        }
    }
}
/******************************************************************************/
void zpotf2_smlpin_anywidth_kernel(int m, magmaDoubleComplex *dA, int ldda, int localstep, int gbstep, magma_int_t *dinfo,
                                   sycl::nd_item<3> item_ct1,
                                   uint8_t *dpct_local)
{
    #pragma unroll
    for(int i = 0; i < m; i+= POTF2_NB){
        int ib = min(m-i, POTF2_NB);
        if (item_ct1.get_local_id(2) < m - i) {
            zpotf2_smlpout_anywidth_device(
                m - i, ib, A(localstep + i, 0), A(localstep + i, localstep + i),
                ldda, localstep + i, gbstep, dinfo, item_ct1, dpct_local);
        }
    }
}
/******************************************************************************/
void zpotf2_smlpin_fixwidth_kernel_batched(int m,
        magmaDoubleComplex **dA_array, int ai, int aj, int lda,
        int localstep, int gbstep, magma_int_t *info_array, const int batchCount,
        sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    const int batchid = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                        item_ct1.get_local_id(1);
    magmaDoubleComplex *dA = dA_array[batchid] + aj * lda + ai;
    if (batchid >= batchCount) return;
    #pragma unroll
    for(int i = 0; i < m; i+= POTF2_NB){
        //if(threadIdx.x < m-i){
            zpotf2_smlpout_fixwidth_device(
                m - i, dA + localstep + i,
                dA + localstep + i + (localstep + i) * lda, lda, localstep + i,
                gbstep, &(info_array[batchid]), item_ct1, dpct_local);
        //}
    }
}


/******************************************************************************/
void zpotf2_smlpin_anywidth_kernel_batched(int m,
        magmaDoubleComplex **dA_array, int ai, int aj, int lda,
        int localstep, int gbstep, magma_int_t *info_array, const int batchCount,
        sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    const int batchid = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                        item_ct1.get_local_id(1);
    magmaDoubleComplex *dA = dA_array[batchid] + aj * lda + ai;
    if (batchid >= batchCount) return;
    #pragma unroll
    for(int i = 0; i < m; i+= POTF2_NB){
        int ib = min(m-i, POTF2_NB);
        //if(threadIdx.x < m-i){
            zpotf2_smlpout_anywidth_device(
                m - i, ib, dA + localstep + i,
                dA + localstep + i + (localstep + i) * lda, lda, localstep + i,
                gbstep, &(info_array[batchid]), item_ct1, dpct_local);
        //}
    }
}
/******************************************************************************/
void zpotf2_smlpout_fixwidth_kernel(int m,
        magmaDoubleComplex *dA, int lda,
        int localstep, int gbstep, magma_int_t *dinfo,
        sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    zpotf2_smlpout_fixwidth_device(
        m, dA + localstep, dA + localstep + localstep * lda, lda, localstep,
        gbstep, dinfo, item_ct1, dpct_local);
}


/******************************************************************************/
void zpotf2_smlpout_anywidth_kernel(int m, int n,
        magmaDoubleComplex *dA, int lda,
        int localstep, int gbstep, magma_int_t *dinfo,
        sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    zpotf2_smlpout_anywidth_device(
        m, n, dA + localstep, dA + localstep + localstep * lda, lda, localstep,
        gbstep, dinfo, item_ct1, dpct_local);
}



/******************************************************************************/
void zpotf2_smlpout_fixwidth_kernel_batched(int m,
        magmaDoubleComplex **dA_array, int ai, int aj, int lda,
        int localstep, int gbstep, magma_int_t *info_array, const int batchCount,
        sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    const int batchid = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                        item_ct1.get_local_id(1);
    if (batchid >= batchCount) return;
    magmaDoubleComplex *dA = dA_array[batchid] + aj * lda + ai;
    zpotf2_smlpout_fixwidth_device(
        m, dA + localstep, dA + localstep + localstep * lda, lda, localstep,
        gbstep, &(info_array[batchid]), item_ct1, dpct_local);
}


/******************************************************************************/
void zpotf2_smlpout_anywidth_kernel_batched(int m, int n,
        magmaDoubleComplex **dA_array, int ai, int aj, int lda,
        int localstep, int gbstep, magma_int_t *info_array, const int batchCount,
        sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    const int batchid = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                        item_ct1.get_local_id(1);
    if (batchid >= batchCount) return;
    magmaDoubleComplex *dA = dA_array[batchid] + aj * lda + ai;
    zpotf2_smlpout_anywidth_device(
        m, n, dA + localstep, dA + localstep + localstep * lda, lda, localstep,
        gbstep, &(info_array[batchid]), item_ct1, dpct_local);
}

/******************************************************************************/
extern "C" magma_int_t
magma_zpotrf_lpout_batched(
        magma_uplo_t uplo, magma_int_t n,
        magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda, magma_int_t gbstep,
        magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t m = n;
    magma_int_t arginfo = 0;

    if ( uplo != MagmaUpper && uplo != MagmaLower) {
        arginfo = -1;
    } else if (m < 0 || n < 0 ) {
        arginfo = -2;
    } else if (lda < max(1,m)) {
        arginfo = -4;
    } else if (m < n) {
        arginfo = -10;
    }
    if (uplo == MagmaUpper) {
        fprintf( stderr, "%s: uplo=upper is not yet implemented\n", __func__ );
        arginfo = -1;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // Quick return if possible
    if (m == 0 || n == 0) {
        return arginfo;
    }

    magma_int_t roundup_m = m;
    // rounding up need more investigation since it coul dmodify the matrix out of its bound
    //magma_int_t m8  = magma_roundup( m, 8 );
    //magma_int_t roundup_m = m8 > lda ? m : m8;
    //magma_int_t m32 = magma_roundup( m, 32 );
    //magma_int_t roundup_m = m32 > lda ? m : m32;

    magma_int_t  ib, rows;

    for (magma_int_t j = 0; j < n; j += POTF2_NB) {
        ib   = min(POTF2_NB, n-j);
        rows = roundup_m-j;

        // tuning ntcol
        magma_int_t ntcol;  // for z precision, the best tuning is at NTCOL = 1 for all sizes
	ntcol = 1; // Currently setting ntcol to 1 for SYCL always (avoid divergent thread return/barrier issue)
//        if (rows > 64) ntcol = 1;
//        else if (rows > 32) ntcol = NTCOL2;
//        else ntcol = NTCOL1;
        // end of tuning ntcol

        const magma_int_t nTB = magma_ceildiv( batchCount, ntcol );
        sycl::range<3> dimGrid(1, 1, nTB);
        magma_int_t nbth = rows;
        /*
        DPCT1083:1313: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        magma_int_t shared_mem_size =
            ntcol * (sizeof(magmaDoubleComplex) * (nbth + POTF2_NB) * POTF2_NB);
        sycl::range<3> threads(1, ntcol, nbth);

        if (shared_mem_size > 47000)
        {
            arginfo = -33;
            magma_xerbla( __func__, -(arginfo) );
            return arginfo;
        }

        if (ib == POTF2_NB) {
            /*
            DPCT1049:1312: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        dpct_local_acc_ct1(sycl::range<1>(shared_mem_size),
                                           cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(dimGrid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zpotf2_smlpout_fixwidth_kernel_batched(
                                rows, dA_array, ai, aj, lda, j, gbstep,
                                info_array, batchCount, item_ct1,
                                dpct_local_acc_ct1.get_pointer());
                        });
                });
        }
        else {
            /*
            DPCT1049:1314: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        dpct_local_acc_ct1(sycl::range<1>(shared_mem_size),
                                           cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(dimGrid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zpotf2_smlpout_anywidth_kernel_batched(
                                rows, ib, dA_array, ai, aj, lda, j, gbstep,
                                info_array, batchCount, item_ct1,
                                dpct_local_acc_ct1.get_pointer());
                        });
                });
        }
    }

    return arginfo;
}
/******************************************************************************/
extern "C" magma_int_t
magma_zpotrf_lpin_batched(
        magma_uplo_t uplo, magma_int_t n,
        magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda, magma_int_t gbstep,
        magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t m = n;
    magma_int_t arginfo = 0;

    if ( uplo != MagmaUpper && uplo != MagmaLower) {
        arginfo = -1;
    } else if (m < 0 || n < 0 ) {
        arginfo = -2;
    } else if (lda < max(1,m)) {
        arginfo = -4;
    } else if (m < n) {
        arginfo = -10;
    }
    if (uplo == MagmaUpper) {
        fprintf( stderr, "%s: uplo=upper is not yet implemented\n", __func__ );
        arginfo = -1;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // Quick return if possible
    if (m == 0 || n == 0) {
        return arginfo;
    }
    sycl::range<3> grid(1, 1, batchCount);
    sycl::range<3> threads(1, 1, n);
    /*
    DPCT1083:1316: The size of local memory in the migrated code may be
    different from the original code. Check that the allocated memory size in
    the migrated code is correct.
    */
    magma_int_t shared_mem_size =
        sizeof(magmaDoubleComplex) * (n + POTF2_NB) * POTF2_NB;
    if (shared_mem_size > 47000) {
        arginfo = -33;
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }



    if( n % POTF2_NB == 0){
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shared_mem_size), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zpotf2_smlpin_fixwidth_kernel_batched(
                                         n, dA_array, ai, aj, lda, 0, gbstep,
                                         info_array, batchCount, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
    }
    else{
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shared_mem_size), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zpotf2_smlpin_anywidth_kernel_batched(
                                         n, dA_array, ai, aj, lda, 0, gbstep,
                                         info_array, batchCount, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
    }

    return arginfo;
}


/******************************************************************************/
extern "C" magma_int_t
magma_zpotf2_lpout(
        magma_uplo_t uplo, magma_int_t n,
        magmaDoubleComplex *dA, magma_int_t lda, magma_int_t gbstep,
        magma_int_t *dinfo, magma_queue_t queue)
{
    magma_int_t m = n;
    magma_int_t arginfo = 0;

    if ( uplo != MagmaUpper && uplo != MagmaLower) {
        arginfo = -1;
    } else if (m < 0 || n < 0 ) {
        arginfo = -2;
    } else if (lda < max(1,m)) {
        arginfo = -4;
    } else if (m < n) {
        arginfo = -10;
    }
    if (uplo == MagmaUpper) {
        fprintf( stderr, "%s: uplo=upper is not yet implemented\n", __func__ );
        arginfo = -1;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // Quick return if possible
    if (m == 0 || n == 0) {
        return arginfo;
    }

    magma_int_t roundup_m = m;
    // rounding up need more investigation since it coul dmodify the matrix out of its bound
    //magma_int_t m8  = magma_roundup( m, 8 );
    //magma_int_t roundup_m = m8 > lda ? m : m8;
    //magma_int_t m32 = magma_roundup( m, 32 );
    //magma_int_t roundup_m = m32 > lda ? m : m32;

    magma_int_t  ib, rows;

    for (magma_int_t j = 0; j < n; j += POTF2_NB) {
        ib   = min(POTF2_NB, n-j);
        rows = roundup_m-j;

        sycl::range<3> dimGrid(1, 1, 1);
        magma_int_t nbth = rows;
        /*
        DPCT1083:1319: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        magma_int_t shared_mem_size =
            sizeof(magmaDoubleComplex) * (nbth + POTF2_NB) * POTF2_NB;
        sycl::range<3> threads(1, 1, nbth);

        if (shared_mem_size > 47000)
        {
            arginfo = -33;
            magma_xerbla( __func__, -(arginfo) );
            return arginfo;
        }

        if (ib == POTF2_NB)
        {
            /*
            DPCT1049:1318: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        dpct_local_acc_ct1(sycl::range<1>(shared_mem_size),
                                           cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(dimGrid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zpotf2_smlpout_fixwidth_kernel(
                                rows, dA, lda, j, gbstep, dinfo, item_ct1,
                                dpct_local_acc_ct1.get_pointer());
                        });
                });
        } else {
            /*
            DPCT1049:1320: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        dpct_local_acc_ct1(sycl::range<1>(shared_mem_size),
                                           cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(dimGrid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zpotf2_smlpout_anywidth_kernel(
                                rows, ib, dA, lda, j, gbstep, dinfo, item_ct1,
                                dpct_local_acc_ct1.get_pointer());
                        });
                });
        }
    }

    return arginfo;
}

/******************************************************************************/
extern "C" magma_int_t
magma_zpotf2_lpin(
        magma_uplo_t uplo, magma_int_t n,
        magmaDoubleComplex *dA, magma_int_t ldda, magma_int_t gbstep,
        magma_int_t *dinfo, magma_queue_t queue)
{
    magma_int_t arginfo = 0;
    // Quick return if possible
    if ( n == 0 ) {
        return arginfo;
    }
    sycl::range<3> grid(1, 1, 1);
    sycl::range<3> threads(1, 1, n);
    /*
    DPCT1083:1322: The size of local memory in the migrated code may be
    different from the original code. Check that the allocated memory size in
    the migrated code is correct.
    */
    magma_int_t shared_mem_size =
        sizeof(magmaDoubleComplex) * (n + POTF2_NB) * POTF2_NB;
    if (shared_mem_size > 47000) {
        arginfo = -33;
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if( n % POTF2_NB == 0){
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shared_mem_size), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zpotf2_smlpin_fixwidth_kernel(
                                         n, dA, ldda, 0, gbstep, dinfo,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
    }
    else{
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shared_mem_size), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zpotf2_smlpin_anywidth_kernel(
                                         n, dA, ldda, 0, gbstep, dinfo,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
    }
    return arginfo;
}
