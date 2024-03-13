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
#include <cmath>

//F. Vazquez, G. Ortega, J.J. Fernandez, E.M. Garzon, Almeria University
void 
zgeellrtmv_kernel_32( 
    int num_rows, 
    int num_cols,
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowlength,
    magmaDoubleComplex * dx,
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy,
    int T,
    int alignment ,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) *
                  item_ct1.get_local_range(2) +
              item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2); // global thread index
    int idb = item_ct1.get_local_id(2); // local thread index
    int idp = idb%T;  // number of threads assigned to one row
    int i = idx/T;  // row index

    auto shared = (magmaDoubleComplex *)dpct_local;

    if (i < num_rows ) {
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        int max_ = magma_ceildiv( drowlength[i], T );  
            // number of elements each thread handles

        for ( int k = 0; k < max_; k++ ) {
            // original code in paper (not working for me)
            //magmaDoubleComplex val = dval[ k*(T*alignment)+(i*T)+idp ];  
            //int col = dcolind [ k*(T*alignment)+(i*T)+idp ];    

            // new code (working for me)        
            magmaDoubleComplex val = dval[ k*(T)+(i*alignment)+idp ];
            int col = dcolind [ k*(T)+(i*alignment)+idp ];

            dot += val * dx[ col ];
        }
        shared[idb]  = dot;
        if ( idp < 16 ) {
            shared[idb] += shared[idb+16];
            if ( idp < 8 ) shared[idb] += shared[idb+8];
            if ( idp < 4 ) shared[idb] += shared[idb+4];
            if ( idp < 2 ) shared[idb] += shared[idb+2];
            if ( idp == 0 ) {
                dy[i] = (shared[idb]+shared[idb+1])*alpha + beta*dy [i];
            }
        }
    }
}

//F. Vazquez, G. Ortega, J.J. Fernandez, E.M. Garzon, Almeria University
void 
zgeellrtmv_kernel_16( 
    int num_rows, 
    int num_cols,
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowlength,
    magmaDoubleComplex * dx,
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy,
    int T,
    int alignment ,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) *
                  item_ct1.get_local_range(2) +
              item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2); // global thread index
    int idb = item_ct1.get_local_id(2); // local thread index
    int idp = idb%T;  // number of threads assigned to one row
    int i = idx/T;  // row index

    auto shared = (magmaDoubleComplex *)dpct_local;

    if (i < num_rows ) {
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        int max_ = magma_ceildiv( drowlength[i], T );  
            // number of elements each thread handles

        for ( int k = 0; k < max_; k++ ) {
            // original code in paper (not working for me)
            //magmaDoubleComplex val = dval[ k*(T*alignment)+(i*T)+idp ];  
            //int col = dcolind [ k*(T*alignment)+(i*T)+idp ];    

            // new code (working for me)        
            magmaDoubleComplex val = dval[ k*(T)+(i*alignment)+idp ];
            int col = dcolind [ k*(T)+(i*alignment)+idp ];

            dot += val * dx[ col ];
        }
        shared[idb]  = dot;
        if ( idp < 8 ) {
            shared[idb] += shared[idb+8];
            if ( idp < 4 ) shared[idb] += shared[idb+4];
            if ( idp < 2 ) shared[idb] += shared[idb+2];
            if ( idp == 0 ) {
                dy[i] = (shared[idb]+shared[idb+1])*alpha + beta*dy [i];
            }
        }
    }
}

//F. Vazquez, G. Ortega, J.J. Fernandez, E.M. Garzon, Almeria University
void 
zgeellrtmv_kernel_8( 
    int num_rows, 
    int num_cols,
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowlength,
    magmaDoubleComplex * dx,
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy,
    int T,
    int alignment ,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) *
                  item_ct1.get_local_range(2) +
              item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2); // global thread index
    int idb = item_ct1.get_local_id(2); // local thread index
    int idp = idb%T;  // number of threads assigned to one row
    int i = idx/T;  // row index

    auto shared = (magmaDoubleComplex *)dpct_local;

    if (i < num_rows ) {
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        int max_ = magma_ceildiv( drowlength[i], T );  
            // number of elements each thread handles

        for ( int k = 0; k < max_; k++ ) {
            // original code in paper (not working for me)
            //magmaDoubleComplex val = dval[ k*(T*alignment)+(i*T)+idp ];  
            //int col = dcolind [ k*(T*alignment)+(i*T)+idp ];    

            // new code (working for me)        
            magmaDoubleComplex val = dval[ k*(T)+(i*alignment)+idp ];
            int col = dcolind [ k*(T)+(i*alignment)+idp ];

            dot += val * dx[ col ];
        }
        shared[idb]  = dot;
        if ( idp < 4 ) {
            shared[idb] += shared[idb+4];
            if ( idp < 2 ) shared[idb] += shared[idb+2];
            if ( idp == 0 ) {
                dy[i] = (shared[idb]+shared[idb+1])*alpha + beta*dy [i];
            }
        }
    }
}



/**
    Purpose
    -------
    
    This routine computes y = alpha *  A *  x + beta * y on the GPU.
    Input format is ELLRT. The ideas are taken from 
    "Improving the performance of the sparse matrix
    vector product with GPUs", (CIT 2010), 
    and modified to provide correct values.

    
    Arguments
    ---------

    @param[in]
    transA      magma_trans_t
                transposition parameter for A
    @param[in]
    m           magma_int_t
                number of rows 

    @param[in]
    n           magma_int_t
                number of columns

    @param[in]
    nnz_per_row magma_int_t
                max number of nonzeros in a row

    @param[in]
    alpha       magmaDoubleComplex
                scalar alpha

    @param[in]
    dval        magmaDoubleComplex_ptr
                val array

    @param[in]
    dcolind     magmaIndex_ptr
                col indices  

    @param[in]
    drowlength  magmaIndex_ptr
                number of elements in each row

    @param[in]
    dx          magmaDoubleComplex_ptr
                input vector x

    @param[in]
    beta        magmaDoubleComplex
                scalar beta

    @param[out]
    dy          magmaDoubleComplex_ptr
                output vector y

    @param[in]
    blocksize   magma_int_t
                threads per block

    @param[in]
    alignment   magma_int_t
                threads assigned to each row

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zblas
    ********************************************************************/

extern "C" magma_int_t
magma_zgeellrtmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowlength,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_int_t alignment,
    magma_int_t blocksize,
    magma_queue_t queue )
{
    int num_blocks = magma_ceildiv( m, blocksize );

    magma_int_t num_threads = alignment*blocksize;
    magma_int_t threads = alignment*blocksize;

    int real_row_length = magma_roundup( nnz_per_row, alignment );

    int nthreads_max = queue->sycl_stream()->get_device()
                            .get_info<sycl::info::device::max_work_group_size>();
    if ( num_threads > nthreads_max)
        printf("error: too many threads requested (%d) for this device (max %d).\n",
               num_threads, nthreads_max);

    int dimgrid1 = int( sqrt( double( num_blocks )));
    int dimgrid2 = magma_ceildiv( num_blocks, dimgrid1 );
    sycl::range<3> grid(1, dimgrid2, dimgrid1);

    /*
    DPCT1083:186: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    int Ms = alignment * blocksize * sizeof(magmaDoubleComplex);
    // printf("launch kernel: %dx%d %d %d\n", grid.x, grid.y, num_threads, Ms);

    if ( alignment == 32 ) {
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * sycl::range<3>(1, 1, threads),
                                      sycl::range<3>(1, 1, threads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgeellrtmv_kernel_32(
                            m, n, alpha, dval, dcolind, drowlength, dx, beta,
                            dy, alignment, real_row_length, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
    }
    else if ( alignment == 16 ) {
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * sycl::range<3>(1, 1, threads),
                                      sycl::range<3>(1, 1, threads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgeellrtmv_kernel_16(
                            m, n, alpha, dval, dcolind, drowlength, dx, beta,
                            dy, alignment, real_row_length, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
    }
    else if ( alignment == 8 ) {
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * sycl::range<3>(1, 1, threads),
                                      sycl::range<3>(1, 1, threads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgeellrtmv_kernel_8(m, n, alpha, dval, dcolind,
                                            drowlength, dx, beta, dy, alignment,
                                            real_row_length, item_ct1,
                                            dpct_local_acc_ct1.get_pointer());
                    });
            });
    }
    else {
        printf("error: alignment %d not supported.\n", int(alignment) );
        return MAGMA_ERR_NOT_SUPPORTED;
    }

    return MAGMA_SUCCESS;
}
