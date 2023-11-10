/*
    -- MAGMA (version 1.1) --
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

#define PRECISION_z
#define COMPLEX
#define BLOCKSIZE 32
#define WARP_SIZE 32
#define WRP 32
#define WRQ 4


void 
magma_zselect_insert_kernel(    
    magma_int_t n,
    magma_int_t p,
    magma_index_t *row,
    magma_index_t *col,
    magmaDoubleComplex *val,
    magma_index_t *rowMT,
    magma_index_t *colMT,
    magmaDoubleComplex *valMT,
    magma_index_t *selection,
    magma_index_t *sizes ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_local_id(2);
    int j = item_ct1.get_group(1) * item_ct1.get_group_range(2) +
            item_ct1.get_group(2);

    magma_index_t select = selection[j];
    // return if no match for this thread block
    if( select != p ){
        return;    
    }
    magma_index_t count = sizes[j];
    
    if( i<count ){
        colMT[ rowMT[j]+i ] = col[ row[j]+i ];
        valMT[ rowMT[j]+i ] = val[ row[j]+i ];
    }
}// kernel 


void 
magma_zselect_rowptr_kernel(    
    magma_int_t n,
    magma_index_t *sizes,
    magma_index_t *rowMT ,
    sycl::nd_item<3> item_ct1)
{
    // unfortunately sequential...
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if( i == 0 ){
        magma_index_t count = 0;
        rowMT[0] = 0;
        magma_index_t j=0;
        for( j=0; j<n; j++ ){
                count = count+sizes[j];
                rowMT[j+1] = count;
        }
    }
}// kernel 


void 
magma_zselect_pattern_kernel(    
    magma_int_t n,
    magma_int_t p,
    magma_index_t *row,
    magma_index_t *selection,
    magma_index_t *sizes ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if( i < n ){
        magma_index_t diff = row[i+1] - row[i];
        if( diff <= WRP ){
             selection[ i ] = p;
             sizes[i] = diff;
        } 
    }
}// kernel 



/**
    Purpose
    -------
    This routine maximizes the pattern for the ISAI preconditioner. Precisely,
    it computes L, L^2, L^3, L^4, L^5 and then selects the columns of M_L 
    such that the nonzer-per-column are the lower max than the 
    implementation-specific limit (32).
    
    The input is the original matrix (row-major)
    The output is already col-major.

    Arguments
    ---------
    
    @param[in,out]
    L           magma_z_matrix
                Incomplete factor.
                
    @param[in,out]
    MT          magma_z_matrix*
                SPAI preconditioner structure, CSR col-major.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zgeisai_maxblock(
    magma_z_matrix L,
    magma_z_matrix *MT,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    int bs1 = 512;
    int bs2 = 1;
    int bs3 = 1;
    int gs1 = magma_ceildiv( L.num_rows, bs1 );
    int gs2 = 1;
    int gs3 = 1;
    sycl::range<3> block(bs3, bs2, bs1);
    sycl::range<3> grid(gs3, gs2, gs1);

    sycl::range<3> block0(1, 1, 1);
    sycl::range<3> grid0(1, 1, 1);

    int blocksize1 = WARP_SIZE;
    int blocksize2 = 1;
    int dimgrid1 = min( int( sqrt( double( L.num_rows ))), 65535 );
    int dimgrid2 = min(magma_ceildiv( L.num_rows, dimgrid1 ), 65535);
    int dimgrid3 = magma_ceildiv( L.num_rows, dimgrid1*dimgrid2 );
    sycl::range<3> block2(1, blocksize2, blocksize1);
    sycl::range<3> grid2(dimgrid3, dimgrid2, dimgrid1);

    magma_z_matrix L2={Magma_CSR}, L3={Magma_CSR}, 
                   L4={Magma_CSR}, L5={Magma_CSR}, T={Magma_CSR};
                   
    magma_index_t *selections_d = NULL, *sizes_d = NULL;
    
    CHECK( magma_index_malloc( &selections_d, L.num_rows ) );
    CHECK( magma_index_malloc( &sizes_d, L.num_rows ) );
    
    magma_int_t nonzeros;
    // generate all pattern that may be considered
            
    // pattern L
    CHECK( magma_z_mtransfer( L, &T, Magma_DEV, Magma_DEV, queue ) );

    // pattern L^2
    /*
    DPCT1064:415: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    CHECK(magma_z_spmm(MAGMA_Z_ONE, L, T, &L2, queue));
    // pattern L^3
    /*
    DPCT1064:416: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    CHECK(magma_z_spmm(MAGMA_Z_ONE, T, L2, &L3, queue));
    // pattern L^4
    /*
    DPCT1064:417: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    CHECK(magma_z_spmm(MAGMA_Z_ONE, T, L3, &L4, queue));
    // pattern L^5
     /*
     DPCT1064:418: Migrated make_cuDoubleComplex call is used in a macro
     definition and is not valid for all macro uses. Adjust the code.
     */
     CHECK(magma_z_spmm(MAGMA_Z_ONE, T, L4, &L5, queue));

    // check for pattern L
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * block, block),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zselect_pattern_kernel(L.num_rows, 1, L.drow,
                                                        selections_d, sizes_d,
                                                        item_ct1);
                       });
    // check for pattern L2
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * block, block),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zselect_pattern_kernel(L.num_rows, 2, L2.drow,
                                                        selections_d, sizes_d,
                                                        item_ct1);
                       });
    // check for pattern L3
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * block, block),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zselect_pattern_kernel(L.num_rows, 3, L3.drow,
                                                        selections_d, sizes_d,
                                                        item_ct1);
                       });
    // check for pattern L4
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * block, block),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zselect_pattern_kernel(L.num_rows, 4, L4.drow,
                                                        selections_d, sizes_d,
                                                        item_ct1);
                       });
    // check for pattern L5
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * block, block),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zselect_pattern_kernel(L.num_rows, 5, L5.drow,
                                                        selections_d, sizes_d,
                                                        item_ct1);
                       });

    //now allocate the roptr for MT
    CHECK( magma_index_malloc( &MT->drow, L.num_rows+1 ) );
    // global nonzero count + generate rowptr
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        auto MT_drow_ct2 = MT->drow;

        cgh.parallel_for(sycl::nd_range<3>(grid0 * block0, block0),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_zselect_rowptr_kernel(L.num_rows, sizes_d,
                                                         MT_drow_ct2, item_ct1);
                         });
    });
    dpct::get_default_queue()
        .memcpy(&nonzeros, MT->drow + L.num_rows, sizeof(magma_index_t))
        .wait();

    //now allocate the memory needed
    CHECK( magma_index_malloc( &MT->dcol, nonzeros ) );
    CHECK( magma_zmalloc( &MT->dval, nonzeros ) );
    
    // fill in some info
    MT->memory_location = Magma_DEV;
    MT->storage_type = Magma_CSR;
    MT->num_rows = L.num_rows;
    MT->num_cols = L.num_cols;
    MT->nnz = nonzeros;
    MT->true_nnz = nonzeros;
    MT->fill_mode = T.fill_mode;

    // now insert the data needed
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        auto MT_drow_ct5 = MT->drow;
        auto MT_dcol_ct6 = MT->dcol;
        auto MT_dval_ct7 = MT->dval;

        cgh.parallel_for(sycl::nd_range<3>(grid2 * block2, block2),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_zselect_insert_kernel(
                                 L.num_rows, 1, L.drow, L.dcol, L.dval,
                                 MT_drow_ct5, MT_dcol_ct6, MT_dval_ct7,
                                 selections_d, sizes_d, item_ct1);
                         });
    });

    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        auto MT_drow_ct5 = MT->drow;
        auto MT_dcol_ct6 = MT->dcol;
        auto MT_dval_ct7 = MT->dval;

        cgh.parallel_for(sycl::nd_range<3>(grid2 * block2, block2),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_zselect_insert_kernel(
                                 L.num_rows, 2, L2.drow, L2.dcol, L2.dval,
                                 MT_drow_ct5, MT_dcol_ct6, MT_dval_ct7,
                                 selections_d, sizes_d, item_ct1);
                         });
    });

    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        auto MT_drow_ct5 = MT->drow;
        auto MT_dcol_ct6 = MT->dcol;
        auto MT_dval_ct7 = MT->dval;

        cgh.parallel_for(sycl::nd_range<3>(grid2 * block2, block2),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_zselect_insert_kernel(
                                 L.num_rows, 3, L3.drow, L3.dcol, L3.dval,
                                 MT_drow_ct5, MT_dcol_ct6, MT_dval_ct7,
                                 selections_d, sizes_d, item_ct1);
                         });
    });

    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        auto MT_drow_ct5 = MT->drow;
        auto MT_dcol_ct6 = MT->dcol;
        auto MT_dval_ct7 = MT->dval;

        cgh.parallel_for(sycl::nd_range<3>(grid2 * block2, block2),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_zselect_insert_kernel(
                                 L.num_rows, 4, L4.drow, L4.dcol, L4.dval,
                                 MT_drow_ct5, MT_dcol_ct6, MT_dval_ct7,
                                 selections_d, sizes_d, item_ct1);
                         });
    });

    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        auto MT_drow_ct5 = MT->drow;
        auto MT_dcol_ct6 = MT->dcol;
        auto MT_dval_ct7 = MT->dval;

        cgh.parallel_for(sycl::nd_range<3>(grid2 * block2, block2),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_zselect_insert_kernel(
                                 L.num_rows, 5, L5.drow, L5.dcol, L5.dval,
                                 MT_drow_ct5, MT_dcol_ct6, MT_dval_ct7,
                                 selections_d, sizes_d, item_ct1);
                         });
    });

cleanup:
    magma_free( sizes_d );
    magma_free( selections_d );
    magma_zmfree( &T, queue );
    magma_zmfree( &L2, queue );
    magma_zmfree( &L3, queue );
    magma_zmfree( &L4, queue );
    magma_zmfree( &L5, queue );
    
    return info;
}
