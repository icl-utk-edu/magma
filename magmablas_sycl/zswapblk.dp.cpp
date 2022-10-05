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

#define BLOCK_SIZE 64

typedef struct dpct_type_934773 {
    magmaDoubleComplex *A;
    magmaDoubleComplex *B;
    int n, ldda, lddb, npivots;
    short ipiv[BLOCK_SIZE];
} magmagpu_zswapblk_params_t;


/******************************************************************************/
void magmagpu_zswapblkrm( magmagpu_zswapblk_params_t params ,
                          sycl::nd_item<3> item_ct1)
{
    unsigned int y = item_ct1.get_local_id(2) +
                     item_ct1.get_local_range(2) * item_ct1.get_group(2);
    if ( y < params.n )
    {
        magmaDoubleComplex *A = params.A + y - params.ldda;
        magmaDoubleComplex *B = params.B + y;
      
        for( int i = 0; i < params.npivots; i++ )
        {
            A += params.ldda;
            if ( params.ipiv[i] == -1 )
                continue;
            magmaDoubleComplex  tmp1 = *A;
            magmaDoubleComplex *tmp2 = B + params.ipiv[i]*params.lddb;
            *A    = *tmp2;
            *tmp2 =  tmp1;
        }
    }
}


/******************************************************************************/
void magmagpu_zswapblkcm( magmagpu_zswapblk_params_t params ,
                          sycl::nd_item<3> item_ct1)
{
    unsigned int y = item_ct1.get_local_id(2) +
                     item_ct1.get_local_range(2) * item_ct1.get_group(2);
    unsigned int offset1 = y*params.ldda;
    unsigned int offset2 = y*params.lddb;
    if ( y < params.n )
    {
        magmaDoubleComplex *A = params.A + offset1 - 1;
        magmaDoubleComplex *B = params.B + offset2;
      
        for( int i = 0; i < params.npivots; i++ )
        {
            A++;
            if ( params.ipiv[i] == -1 )
                continue;
            magmaDoubleComplex  tmp1 = *A;
            magmaDoubleComplex *tmp2 = B + params.ipiv[i];
            *A    = *tmp2;
            *tmp2 =  tmp1;
        }
    }
    /*
    DPCT1065:1326: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
}


/***************************************************************************//**
    Blocked version: swap several pairs of lines.
    Used in magma_ztstrf() and magma_zssssm().
    @ingroup magma_swapblk
*******************************************************************************/
extern "C" void 
magmablas_zswapblk(
    magma_order_t order, magma_int_t n, 
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci, magma_int_t offset,
    magma_queue_t queue )
{
    magma_int_t  blocksize = 64;
    sycl::range<3> blocks(1, 1, magma_ceildiv(n, blocksize));
    magma_int_t  k, im;
    
    /* Quick return */
    if ( n == 0 )
        return;
    
    if ( order == MagmaColMajor ) {
        for( k=(i1-1); k < i2; k += BLOCK_SIZE )
        {
            magma_int_t sb = min(BLOCK_SIZE, i2-k);
            magmagpu_zswapblk_params_t params = { dA+k, dB, int(n), int(ldda), int(lddb), int(sb) };
            for( magma_int_t j = 0; j < sb; j++ )
            {
                im = ipiv[(k+j)*inci] - 1;
                if ( (k+j) == im )
                    params.ipiv[j] = -1;
                else
                    params.ipiv[j] = im - offset;
            }
            /*
            DPCT1049:1327: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->sycl_stream()))
                ->parallel_for(
                    sycl::nd_range<3>(blocks * sycl::range<3>(1, 1, blocksize),
                                      sycl::range<3>(1, 1, blocksize)),
                    [=](sycl::nd_item<3> item_ct1) {
                        magmagpu_zswapblkcm(params, item_ct1);
                    });
        }
    }
    else {
        for( k=(i1-1); k < i2; k += BLOCK_SIZE )
        {
            magma_int_t sb = min(BLOCK_SIZE, i2-k);
            magmagpu_zswapblk_params_t params = { dA+k*ldda, dB, int(n), int(ldda), int(lddb), int(sb) };
            for( magma_int_t j = 0; j < sb; j++ )
            {
                im = ipiv[(k+j)*inci] - 1;
                if ( (k+j) == im )
                    params.ipiv[j] = -1;
                else
                    params.ipiv[j] = im - offset;
            }
            /*
            DPCT1049:1328: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->sycl_stream()))
                ->parallel_for(
                    sycl::nd_range<3>(blocks * sycl::range<3>(1, 1, blocksize),
                                      sycl::range<3>(1, 1, blocksize)),
                    [=](sycl::nd_item<3> item_ct1) {
                        magmagpu_zswapblkrm(params, item_ct1);
                    });
        }
    }
}
