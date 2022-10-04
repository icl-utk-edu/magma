/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

       @author Peng Du
       @author Tingxing Dong
       @author Mark Gates
       @author Azzam Haidar
       
       This file implements lower case, and is called by ztrtri_kernel.cu.
       It's convenient to have separate files for lower & upper, to diff the sources.
*/

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

#define TRTRI_BATCHED
#include "ztrtri.dp.hpp"
#include "ztrtri_lower_device.dp.hpp"

/******************************************************************************/
SYCL_EXTERNAL void ztrtri_diag_lower_kernel_batched(
    magma_diag_t diag, int n, magmaDoubleComplex const *const *dA_array,
    int lda, magmaDoubleComplex **dinvA_array, sycl::nd_item<3> item_ct1,
    magmaDoubleComplex *sB)
{
    int batchid = item_ct1.get_group(0);
    ztrtri_diag_lower_device(diag, n, dA_array[batchid], lda,
                             dinvA_array[batchid], item_ct1, sB);
}


/******************************************************************************/
SYCL_EXTERNAL void triple_zgemm16_part1_lower_kernel_batched(
    int n, magmaDoubleComplex const *const *Ain_array, int lda,
    magmaDoubleComplex **dinvA_array, int jb, int npages,
    sycl::nd_item<3> item_ct1,
    sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write,
                   sycl::access::target::local>
        sB)
{
    int batchid = item_ct1.get_group(0);
    triple_zgemm16_part1_lower_device(n, Ain_array[batchid], lda,
                                      dinvA_array[batchid], jb, npages,
                                      item_ct1, sB);
}


/******************************************************************************/
SYCL_EXTERNAL void triple_zgemm16_part2_lower_kernel_batched(
    int n, magmaDoubleComplex const *const *Ain_array, int lda,
    magmaDoubleComplex **dinvA_array, int jb, int npages,
    sycl::nd_item<3> item_ct1,
    sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write,
                   sycl::access::target::local>
        sB)
{
    int batchid = item_ct1.get_group(0);
    triple_zgemm16_part2_lower_device(n, Ain_array[batchid], lda,
                                      dinvA_array[batchid], jb, npages,
                                      item_ct1, sB);
}


/******************************************************************************/
SYCL_EXTERNAL void triple_zgemm32_part1_lower_kernel_batched(
    int n, magmaDoubleComplex const *const *Ain_array, int lda,
    magmaDoubleComplex **dinvA_array, int jb, int npages,
    sycl::nd_item<3> item_ct1,
    sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write,
                   sycl::access::target::local>
        sB)
{
    int batchid = item_ct1.get_group(0);
    triple_zgemm32_part1_lower_device(n, Ain_array[batchid], lda,
                                      dinvA_array[batchid], jb, npages,
                                      item_ct1, sB);
}


/******************************************************************************/
SYCL_EXTERNAL void triple_zgemm32_part2_lower_kernel_batched(
    int n, magmaDoubleComplex const *const *Ain_array, int lda,
    magmaDoubleComplex **dinvA_array, int jb, int npages,
    sycl::nd_item<3> item_ct1,
    sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write,
                   sycl::access::target::local>
        sB)
{
    int batchid = item_ct1.get_group(0);
    triple_zgemm32_part2_lower_device(n, Ain_array[batchid], lda,
                                      dinvA_array[batchid], jb, npages,
                                      item_ct1, sB);
}


/******************************************************************************/
SYCL_EXTERNAL void triple_zgemm64_part1_lower_kernel_batched(
    int n, magmaDoubleComplex const *const *Ain_array, int lda,
    magmaDoubleComplex **dinvA_array, int jb, int npages,
    sycl::nd_item<3> item_ct1,
    sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write,
                   sycl::access::target::local>
        sB)
{
    int batchid = item_ct1.get_group(0);
    triple_zgemm64_part1_lower_device(n, Ain_array[batchid], lda,
                                      dinvA_array[batchid], jb, npages,
                                      item_ct1, sB);
}


/******************************************************************************/
SYCL_EXTERNAL void triple_zgemm64_part2_lower_kernel_batched(
    int n, magmaDoubleComplex const *const *Ain_array, int lda,
    magmaDoubleComplex **dinvA_array, int jb, int npages,
    sycl::nd_item<3> item_ct1,
    sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write,
                   sycl::access::target::local>
        sB)
{
    int batchid = item_ct1.get_group(0);
    triple_zgemm64_part2_lower_device(n, Ain_array[batchid], lda,
                                      dinvA_array[batchid], jb, npages,
                                      item_ct1, sB);
}


/******************************************************************************/
SYCL_EXTERNAL void triple_zgemm_above64_part1_lower_kernel_batched(
    int n, magmaDoubleComplex const *const *Ain_array, int lda,
    magmaDoubleComplex **dinvA_array, int jb, int npages,
    sycl::nd_item<3> item_ct1,
    sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write,
                   sycl::access::target::local>
        sB)
{
    int batchid = item_ct1.get_group(0);
    triple_zgemm_above64_part1_lower_device(n, Ain_array[batchid], lda,
                                            dinvA_array[batchid], jb, npages,
                                            item_ct1, sB);
}


/******************************************************************************/
SYCL_EXTERNAL void triple_zgemm_above64_part2_lower_kernel_batched(
    int n, magmaDoubleComplex const *const *Ain_array, int lda,
    magmaDoubleComplex **dinvA_array, int jb, int npages,
    sycl::nd_item<3> item_ct1,
    sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write,
                   sycl::access::target::local>
        sB)
{
    int batchid = item_ct1.get_group(0);
    triple_zgemm_above64_part2_lower_device(n, Ain_array[batchid], lda,
                                            dinvA_array[batchid], jb, npages,
                                            item_ct1, sB);
}


/******************************************************************************/
SYCL_EXTERNAL void triple_zgemm_above64_part3_lower_kernel_batched(
    int n, magmaDoubleComplex const *const *Ain_array, int lda,
    magmaDoubleComplex **dinvA_array, int jb, int npages,
    sycl::nd_item<3> item_ct1)
{
    int batchid = item_ct1.get_group(0);
    triple_zgemm_above64_part3_lower_device(
        n, Ain_array[batchid], lda, dinvA_array[batchid], jb, npages, item_ct1);
}


// =============================================================================
// vbatched kernels


/******************************************************************************/
SYCL_EXTERNAL void ztrtri_diag_lower_kernel_vbatched(
    magma_diag_t diag, magma_int_t *n,
    magmaDoubleComplex const *const *dA_array, magma_int_t *lda,
    magmaDoubleComplex **dinvA_array, sycl::nd_item<3> item_ct1,
    magmaDoubleComplex *sB)
{
    const int batchid = item_ct1.get_group(0);
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;

    if (item_ct1.get_group(2) >= magma_ceildiv(my_n, IB)) return;

    ztrtri_diag_lower_device(diag, my_n, dA_array[batchid], (int)lda[batchid],
                             dinvA_array[batchid], item_ct1, sB);
}


// The kernels below have 3D grids
// grid.x and grid.y are independent from my_n
// only grid.y is dependent on my_n, so terminating thread blocks is based on blockIdx.y


/******************************************************************************/
SYCL_EXTERNAL void triple_zgemm16_part1_lower_kernel_vbatched(
    magma_int_t *n, magmaDoubleComplex const *const *Ain_array,
    magma_int_t *lda, magmaDoubleComplex **dinvA_array, int jb, int npages,
    sycl::nd_item<3> item_ct1,
    sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write,
                   sycl::access::target::local>
        sB)
{
    const int batchid = item_ct1.get_group(0);
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if (item_ct1.get_group(1) >= my_npages * (jb / 16)) return;
    triple_zgemm16_part1_lower_device(my_n, Ain_array[batchid],
                                      (int)lda[batchid], dinvA_array[batchid],
                                      jb, my_npages, item_ct1, sB);
}


/******************************************************************************/
SYCL_EXTERNAL void triple_zgemm16_part2_lower_kernel_vbatched(
    magma_int_t *n, magmaDoubleComplex const *const *Ain_array,
    magma_int_t *lda, magmaDoubleComplex **dinvA_array, int jb, int npages,
    sycl::nd_item<3> item_ct1,
    sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write,
                   sycl::access::target::local>
        sB)
{
    const int batchid = item_ct1.get_group(0);
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if (item_ct1.get_group(1) >= my_npages * (jb / 16)) return;
    triple_zgemm16_part2_lower_device(my_n, Ain_array[batchid],
                                      (int)lda[batchid], dinvA_array[batchid],
                                      jb, my_npages, item_ct1, sB);
}


/******************************************************************************/
SYCL_EXTERNAL void triple_zgemm32_part1_lower_kernel_vbatched(
    magma_int_t *n, magmaDoubleComplex const *const *Ain_array,
    magma_int_t *lda, magmaDoubleComplex **dinvA_array, int jb, int npages,
    sycl::nd_item<3> item_ct1,
    sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write,
                   sycl::access::target::local>
        sB)
{
    const int batchid = item_ct1.get_group(0);
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if (item_ct1.get_group(1) >= my_npages * (jb / 16)) return;
    triple_zgemm32_part1_lower_device(my_n, Ain_array[batchid],
                                      (int)lda[batchid], dinvA_array[batchid],
                                      jb, my_npages, item_ct1, sB);
}


/******************************************************************************/
SYCL_EXTERNAL void triple_zgemm32_part2_lower_kernel_vbatched(
    magma_int_t *n, magmaDoubleComplex const *const *Ain_array,
    magma_int_t *lda, magmaDoubleComplex **dinvA_array, int jb, int npages,
    sycl::nd_item<3> item_ct1,
    sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write,
                   sycl::access::target::local>
        sB)
{
    const int batchid = item_ct1.get_group(0);
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if (item_ct1.get_group(1) >= my_npages * (jb / 16)) return;
    triple_zgemm32_part2_lower_device(my_n, Ain_array[batchid],
                                      (int)lda[batchid], dinvA_array[batchid],
                                      jb, my_npages, item_ct1, sB);
}


/******************************************************************************/
SYCL_EXTERNAL void triple_zgemm64_part1_lower_kernel_vbatched(
    magma_int_t *n, magmaDoubleComplex const *const *Ain_array,
    magma_int_t *lda, magmaDoubleComplex **dinvA_array, int jb, int npages,
    sycl::nd_item<3> item_ct1,
    sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write,
                   sycl::access::target::local>
        sB)
{
    const int batchid = item_ct1.get_group(0);
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if (item_ct1.get_group(1) >= my_npages * (jb / 16)) return;
    triple_zgemm64_part1_lower_device(my_n, Ain_array[batchid],
                                      (int)lda[batchid], dinvA_array[batchid],
                                      jb, my_npages, item_ct1, sB);
}


/******************************************************************************/
SYCL_EXTERNAL void triple_zgemm64_part2_lower_kernel_vbatched(
    magma_int_t *n, magmaDoubleComplex const *const *Ain_array,
    magma_int_t *lda, magmaDoubleComplex **dinvA_array, int jb, int npages,
    sycl::nd_item<3> item_ct1,
    sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write,
                   sycl::access::target::local>
        sB)
{
    const int batchid = item_ct1.get_group(0);
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if (item_ct1.get_group(1) >= my_npages * (jb / 16)) return;
    triple_zgemm64_part2_lower_device(my_n, Ain_array[batchid],
                                      (int)lda[batchid], dinvA_array[batchid],
                                      jb, my_npages, item_ct1, sB);
}


/******************************************************************************/
SYCL_EXTERNAL void triple_zgemm_above64_part1_lower_kernel_vbatched(
    magma_int_t *n, magmaDoubleComplex const *const *Ain_array,
    magma_int_t *lda, magmaDoubleComplex **dinvA_array, int jb, int npages,
    sycl::nd_item<3> item_ct1,
    sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write,
                   sycl::access::target::local>
        sB)
{
    const int batchid = item_ct1.get_group(0);
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if (item_ct1.get_group(1) >= my_npages * (jb / 16)) return;
    triple_zgemm_above64_part1_lower_device(
        my_n, Ain_array[batchid], (int)lda[batchid], dinvA_array[batchid], jb,
        my_npages, item_ct1, sB);
}


/******************************************************************************/
SYCL_EXTERNAL void triple_zgemm_above64_part2_lower_kernel_vbatched(
    magma_int_t *n, magmaDoubleComplex const *const *Ain_array,
    magma_int_t *lda, magmaDoubleComplex **dinvA_array, int jb, int npages,
    sycl::nd_item<3> item_ct1,
    sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write,
                   sycl::access::target::local>
        sB)
{
    const int batchid = item_ct1.get_group(0);
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if (item_ct1.get_group(1) >= my_npages * (jb / 16)) return;
    triple_zgemm_above64_part2_lower_device(
        my_n, Ain_array[batchid], (int)lda[batchid], dinvA_array[batchid], jb,
        my_npages, item_ct1, sB);
}


/******************************************************************************/
SYCL_EXTERNAL void triple_zgemm_above64_part3_lower_kernel_vbatched(
    magma_int_t *n, magmaDoubleComplex const *const *Ain_array,
    magma_int_t *lda, magmaDoubleComplex **dinvA_array, int jb, int npages,
    sycl::nd_item<3> item_ct1)
{
    const int batchid = item_ct1.get_group(0);
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if (item_ct1.get_group(1) >= my_npages * (jb / 16)) return;
    triple_zgemm_above64_part3_lower_device(
        my_n, Ain_array[batchid], (int)lda[batchid], dinvA_array[batchid], jb,
        my_npages, item_ct1);
}
