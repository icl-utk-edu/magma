/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magmasparse_internal.h"

#define PRECISION_z
#define BLOCKSIZE 256


void
magma_zbajac_csr_ls_kernel(int localiters, int n, 
                            magmaDoubleComplex * valD, 
                            magma_index_t * rowD, 
                            magma_index_t * colD, 
                            magmaDoubleComplex * valR, 
                            magma_index_t * rowR,
                            magma_index_t * colR, 
                            const magmaDoubleComplex *  __restrict__ b,                            
                            magmaDoubleComplex * x , sycl::nd_item<3> item_ct1,
                            magmaDoubleComplex *local_x)
{
    int inddiag = item_ct1.get_group(2) * item_ct1.get_local_range(2);
    int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
    int i, j, start, end;

    if (index < n) {
        start = rowR[index];
        end   = rowR[index+1];

        magmaDoubleComplex zero = MAGMA_Z_ZERO;
        magmaDoubleComplex bl, tmp = zero, v = zero;

        bl = b[index];

        #pragma unroll
        for( i=start; i<end; i++ )
             v += valR[i] * x[ colR[i] ];

        start = rowD[index];
        end   = rowD[index+1];

        #pragma unroll
        for( i=start; i<end; i++ )
            tmp += valD[i] * x[ colD[i] ];

        v =  bl - v;

        /* add more local iterations */

        local_x[item_ct1.get_local_id(2)] =
            x[index] + (v - tmp) / (valD[start]);
        /*
        DPCT1065:228: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

#pragma unroll
        for( j=0; j<localiters-1; j++ )
        {
            tmp = zero;
            #pragma unroll
            for( i=start; i<end; i++ )
                tmp += valD[i] * local_x[ colD[i] - inddiag];

            local_x[item_ct1.get_local_id(2)] += (v - tmp) / (valD[start]);
        }
        x[index] = local_x[item_ct1.get_local_id(2)];
    }
}



void
magma_zbajac_csr_kernel(    
    int n, 
    magmaDoubleComplex * valD, 
    magma_index_t * rowD, 
    magma_index_t * colD, 
    magmaDoubleComplex * valR, 
    magma_index_t * rowR,
    magma_index_t * colR, 
    magmaDoubleComplex * b,                                
    magmaDoubleComplex * x ,
    sycl::nd_item<3> item_ct1)
{
    int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
    int i, start, end;   

    if (index < n) {
        magmaDoubleComplex zero = MAGMA_Z_ZERO;
        magmaDoubleComplex bl, tmp = zero, v = zero;

        bl = b[index];

        start = rowR[index];
        end   = rowR[index+1];

        #pragma unroll
        for( i=start; i<end; i++ )
             v += valR[i] * x[ colR[i] ];

        v =  bl - v;

        start = rowD[index];
        end   = rowD[index+1];

        #pragma unroll
        for( i=start; i<end; i++ )
            tmp += valD[i] * x[ colD[i] ];

        x[index] = x[index] + ( v - tmp ) / (valD[start]); 
    }
}


/**
    Purpose
    -------
    
    This routine is a block-asynchronous Jacobi iteration performing s
    local Jacobi-updates within the block. Input format is two CSR matrices,
    one containing the diagonal blocks, one containing the rest.

    Arguments
    ---------

    @param[in]
    localiters  magma_int_t
                number of local Jacobi-like updates

    @param[in]
    D           magma_z_matrix
                input matrix with diagonal blocks

    @param[in]
    R           magma_z_matrix
                input matrix with non-diagonal parts

    @param[in]
    b           magma_z_matrix
                RHS

    @param[in]
    x           magma_z_matrix*
                iterate/solution

    
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zbajac_csr(
    magma_int_t localiters,
    magma_z_matrix D,
    magma_z_matrix R,
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_queue_t queue )
{
    int blocksize1 = BLOCKSIZE;
    int blocksize2 = 1;

    int dimgrid1 = magma_ceildiv(  D.num_rows, blocksize1 );
    int dimgrid2 = 1;
    int dimgrid3 = 1;

    sycl::range<3> grid(dimgrid3, dimgrid2, dimgrid1);
    sycl::range<3> block(1, blocksize2, blocksize1);
    if ( R.nnz > 0 ) { 
        if ( localiters == 1 )
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    auto x_dval_ct8 = x->dval;

                    cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         magma_zbajac_csr_kernel(
                                             D.num_rows, D.dval, D.drow, D.dcol,
                                             R.dval, R.drow, R.dcol, b.dval,
                                             x_dval_ct8, item_ct1);
                                     });
                });
        else
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::accessor<magmaDoubleComplex, 1,
                                   sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        local_x_acc_ct1(sycl::range<1>(BLOCKSIZE), cgh);

                    auto x_dval_ct9 = x->dval;

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * block, block),
                        [=](sycl::nd_item<3> item_ct1) {
                            magma_zbajac_csr_ls_kernel(
                                localiters, D.num_rows, D.dval, D.drow, D.dcol,
                                R.dval, R.drow, R.dcol, b.dval, x_dval_ct9,
                                item_ct1, local_x_acc_ct1.get_pointer());
                        });
                });
    }
    else {
        printf("error: all elements in diagonal block.\n");
    }

    return MAGMA_SUCCESS;
}
