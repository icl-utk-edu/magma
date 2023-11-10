/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Azzam Haidar
*/

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "magma_templates.h"

#define use_gemm_larft

/******************************************************************************/
static  void
zlarft_ztrmv_sm32x32_device(
    int n, int k, magmaDoubleComplex *tau,
    magmaDoubleComplex *Tin, int ldtin,  magmaDoubleComplex *Tout, int ldtout ,
    sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    auto shared_data = (magmaDoubleComplex *)dpct_local;

    int tx = item_ct1.get_local_id(2);
    magmaDoubleComplex *sdata = (magmaDoubleComplex*)shared_data;
    magmaDoubleComplex res;

    // this routine apply a sequence of trmv to update k column of the triangular
    // T starting at n-k to n where T is of size n by n and where the first n-k
    // columns of T are supposed updated previously.
    // So the routine load all of T nxn to the shared memory
    // and apply the sequence of trmv.
    // to update a certain column i, threads go in horizontal fashion where
    // every thread read one row and do it gemv(dot) to generate
    // one element of the column of T then move to the next column

    // read T into shared
    for (int s=0; s < n-k; s++)
    {
        sdata[tx + s*n] = Tin[tx + s * ldtin];
    }

#if defined(use_gemm_larft)
    for (int s=n-k; s < n; s++)
    {
        if (tx == s)
            sdata[tx + s*n] = tau[s];
        else
            sdata[tx + s*n] = -tau[s] * Tin[tx + s * ldtin];
    }
#else
    for (int s=n-k; s < n; s++)
    {
        sdata[tx + s*n] = Tin[tx + s * ldtin];
    }
#endif

    // perform trmv
    for (int i=n-k; i < n; i++)
    {
        /*
        DPCT1065:1210: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        res = MAGMA_Z_ZERO;
        if (tx < i)
        {
            for (int j=tx; j < i; j++)
            {
                res += sdata[tx + j * n] * sdata[j+ i * n];
            }
        }
        /*
        DPCT1065:1211: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (tx < i)
        {
            sdata[tx + i * n] = res;
        }
    }

    /*
    DPCT1065:1209: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // write back the updated block of k column of T
    for (int s=n-k; s < n; s++)
    {
        Tout[tx + s * ldtout] = sdata[tx + s*n];
    }
}


/******************************************************************************/
void
zlarft_ztrmv_sm32x32_kernel(
    int n, int k, magmaDoubleComplex *tau,
    magmaDoubleComplex *Tin, int ldtin,  magmaDoubleComplex *Tout, int ldtout ,
    sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    zlarft_ztrmv_sm32x32_device(n, k, tau, Tin, ldtin, Tout, ldtout, item_ct1,
                                dpct_local);
}


/******************************************************************************/
void
zlarft_ztrmv_sm32x32_kernel_batched(
    int n, int k,
    magmaDoubleComplex **tau_array, int taui,
    magmaDoubleComplex **Tin_array,  int Tini, int Tinj, int ldtin,
    magmaDoubleComplex **Tout_array, int Touti, int Toutj, int ldtout ,
    sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    int batchId = item_ct1.get_group(0);
    magmaDoubleComplex *tau  = tau_array[batchId]  + taui;
    magmaDoubleComplex *Tin  = Tin_array[batchId]  + Tinj  * ldtin + Tini;
    magmaDoubleComplex *Tout = Tout_array[batchId] + Toutj * ldtout + Touti;
    zlarft_ztrmv_sm32x32_device(n, k, tau, Tin, ldtin, Tout, ldtout, item_ct1,
                                dpct_local);
}


/******************************************************************************/
extern "C"
void magmablas_zlarft_ztrmv_sm32x32(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *Tin, magma_int_t ldtin,
    magmaDoubleComplex *Tout, magma_int_t ldtout,
    magma_queue_t queue )
{
    sycl::range<3> grid(1, 1, 1);
    sycl::range<3> threads(1, 1, max(m, 1));
    /*
    DPCT1083:1213: The size of local memory in the migrated code may be
    different from the original code. Check that the allocated memory size in
    the migrated code is correct.
    */
    size_t shmem = sizeof(magmaDoubleComplex) * (m * m);
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             zlarft_ztrmv_sm32x32_kernel(
                                 m, n, tau, Tin, ldtin, Tout, ldtout, item_ct1,
                                 dpct_local_acc_ct1.get_pointer());
                         });
    });
}


/******************************************************************************/
extern "C"
void magmablas_zlarft_ztrmv_sm32x32_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **tau_array, magma_int_t taui,
    magmaDoubleComplex **Tin_array, magma_int_t Tini, magma_int_t Tinj, magma_int_t ldtin,
    magmaDoubleComplex **Tout_array, magma_int_t Touti, magma_int_t Toutj, magma_int_t ldtout,
    magma_int_t batchCount, magma_queue_t queue)
{

    magma_int_t max_batchCount = queue->get_maxBatch();
    sycl::range<3> threads(1, 1, max(m, 1));
    /*
    DPCT1083:1215: The size of local memory in the migrated code may be
    different from the original code. Check that the allocated memory size in
    the migrated code is correct.
    */
    size_t shmem = sizeof(magmaDoubleComplex) * (m * m);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, 1);

        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zlarft_ztrmv_sm32x32_kernel_batched(
                            m, n, tau_array + i, taui, Tin_array + i, Tini,
                            Tinj, ldtin, Tout_array + i, Touti, Toutj, ldtout,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
    }
}


/******************************************************************************/
static void
zlarft_recztrmv_sm32x32_device(
    int m, int n, magmaDoubleComplex *tau,
    magmaDoubleComplex *Trec, int ldtrec, magmaDoubleComplex *Ttri, int ldttri,
    sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    auto shared_data = (magmaDoubleComplex *)dpct_local;

    int tx = item_ct1.get_local_id(2);
    magmaDoubleComplex *sdata = (magmaDoubleComplex*)shared_data;
    magmaDoubleComplex res;

    // to update a certain column i, threads go in horizontal fashion where
    // every thread read one row and do it gemv(dot) to generate
    // one element of the column of T then move to the next column

    // read T into shared
    for (int s=0; s < n; s++)
    {
        sdata[tx + s*n] = Trec[tx + s * ldtrec];
    }
    /*
    DPCT1065:1216: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // perform sequence of n-1 gemv
    for (int i=0; i < n; i++)
    {
        /*
        DPCT1064:1219: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        res = MAGMA_Z_ZERO;
        for (int j=0; j < i; j++)
        {
            res += sdata[tx + j * n] * Ttri[j+ i * ldttri];
        }
        /*
        DPCT1065:1217: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); // a enlever
        sdata[tx + i * n] = -tau[i] * (sdata[tx + i * n] + res);
        /*
        DPCT1065:1218: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    // write back the updated block of k column of T  multiplying by -tau
    for (int s=0; s < n; s++)
    {
        Trec[tx + s * ldtrec] = sdata[tx + s*n];
    }
}


/******************************************************************************/
void
zlarft_recztrmv_sm32x32_kernel(
    int m, int n, magmaDoubleComplex *tau,
    magmaDoubleComplex *Trec, int ldtrec, magmaDoubleComplex *Ttri, int ldttri,
    sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    zlarft_recztrmv_sm32x32_device(m, n, tau, Trec, ldtrec, Ttri, ldttri,
                                   item_ct1, dpct_local);
}


/******************************************************************************/
void
zlarft_recztrmv_sm32x32_kernel_batched(
    int m, int n,
    magmaDoubleComplex **tau_array, int taui,
    magmaDoubleComplex **Trec_array, int Treci, int Trecj, int ldtrec,
    magmaDoubleComplex **Ttri_array, int Ttrii, int Ttrij, int ldttri,
    sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    int batchId = item_ct1.get_group(0);
    magmaDoubleComplex *tau  = tau_array[batchId]  + taui;
    magmaDoubleComplex *Trec = Trec_array[batchId] + Trecj * ldtrec + Treci;
    magmaDoubleComplex *Ttri = Ttri_array[batchId] + Ttrij * ldttri + Ttrii;
    zlarft_recztrmv_sm32x32_device(m, n, tau, Trec, ldtrec, Ttri, ldttri,
                                   item_ct1, dpct_local);
}


/******************************************************************************/
extern "C"
void magmablas_zlarft_recztrmv_sm32x32(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *Trec, magma_int_t ldtrec,
    magmaDoubleComplex *Ttri, magma_int_t ldttri,
    magma_queue_t queue )
{
    sycl::range<3> grid(1, 1, 1);
    sycl::range<3> threads(1, 1, max(m, 1));
    /*
    DPCT1083:1221: The size of local memory in the migrated code may be
    different from the original code. Check that the allocated memory size in
    the migrated code is correct.
    */
    size_t shmem = sizeof(magmaDoubleComplex) * (m * n);
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             zlarft_recztrmv_sm32x32_kernel(
                                 m, n, tau, Trec, ldtrec, Ttri, ldttri,
                                 item_ct1, dpct_local_acc_ct1.get_pointer());
                         });
    });
}


/******************************************************************************/
extern "C"
void magmablas_zlarft_recztrmv_sm32x32_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **tau_array, magma_int_t taui,
    magmaDoubleComplex **Trec_array, magma_int_t Treci, magma_int_t Trecj, magma_int_t ldtrec,
    magmaDoubleComplex **Ttri_array, magma_int_t Ttrii, magma_int_t Ttrij, magma_int_t ldttri,
    magma_int_t batchCount, magma_queue_t queue)
{
    sycl::range<3> threads(1, 1, max(m, 1));
    magma_int_t max_batchCount = queue->get_maxBatch();
    /*
    DPCT1083:1223: The size of local memory in the migrated code may be
    different from the original code. Check that the allocated memory size in
    the migrated code is correct.
    */
    size_t shmem = sizeof(magmaDoubleComplex) * (m * n);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, 1);

        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zlarft_recztrmv_sm32x32_kernel_batched(
                            m, n, tau_array + i, taui, Trec_array + i, Treci,
                            Trecj, ldtrec, Ttri_array + i, Ttrii, Ttrij, ldttri,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
    }
}
