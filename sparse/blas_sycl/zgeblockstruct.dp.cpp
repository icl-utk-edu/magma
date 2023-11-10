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
#define BLOCKSIZE 256



//      does not yet work at this point!        //

void 
magma_zmisai_blockstruct_row_kernel(    
    magma_int_t n, 
    magma_int_t bs, 
    magma_index_t *row,
    magma_index_t *col,
    magmaDoubleComplex *val ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if( i < n+1 ){
        row[ i ] = i * bs;
    }
}// kernel 


void 
magma_zmisai_blockstruct_fill_l_kernel(    
    magma_int_t n, 
    magma_int_t bs, 
    magma_index_t *row,
    magma_index_t *col,
    magmaDoubleComplex *val ,
    sycl::nd_item<3> item_ct1)
{
    int block = item_ct1.get_group(0) * item_ct1.get_group_range(1) *
                    item_ct1.get_group_range(2) +
                item_ct1.get_group(1) * item_ct1.get_group_range(2) +
                item_ct1.get_group(2);
    int i = item_ct1.get_local_id(2);
    int j = item_ct1.get_local_id(1);
    int lrow = block * bs + i;
    int lcol = j + block * bs;
    int offset = block * bs*bs; 
    int loc = offset + lrow*bs +lcol;
    if( lrow < n ){
        if( lcol < n ){
            // val[loc] = MAGMA_Z_MAKE((double)(lrow+1),(double)(1+lcol));
            // col[loc] = lcol;
            if( lcol<=lrow ){
                /*
                DPCT1064:161: Migrated make_cuDoubleComplex call is used in a
                macro definition and is not valid for all macro uses. Adjust the
                code.
                */
                val[loc] = MAGMA_Z_ONE;
                col[loc] = lcol;
            } else {
                val[loc] = MAGMA_Z_ZERO;
                col[loc] = lcol;
            } 
        } 
        // else {
        //         val[loc] = MAGMA_Z_ZERO;
        //         col[loc] = 0;
        // }
    }
}// kernel 

void 
magma_zmisai_blockstruct_fill_u_kernel(    
    magma_int_t n, 
    magma_int_t bs, 
    magma_index_t *row,
    magma_index_t *col,
    magmaDoubleComplex *val ,
    sycl::nd_item<3> item_ct1)
{
    int block = item_ct1.get_group(0) * item_ct1.get_group_range(1) *
                    item_ct1.get_group_range(2) +
                item_ct1.get_group(1) * item_ct1.get_group_range(2) +
                item_ct1.get_group(2);
    int lrow = block * bs + item_ct1.get_local_id(2);
    int offset = block * bs*bs;
    int j = item_ct1.get_local_id(1);
    int lcol = j + block * bs;
    int loc = offset + item_ct1.get_local_id(2) * bs + item_ct1.get_local_id(1);
    if( lrow < n ){
        if( lcol < n ){
            if( lcol>=lrow ){
                /*
                DPCT1064:162: Migrated make_cuDoubleComplex call is used in a
                macro definition and is not valid for all macro uses. Adjust the
                code.
                */
                val[loc] = MAGMA_Z_ONE;
                col[loc] = lcol;
            } else {
                /*
                DPCT1064:163: Migrated make_cuDoubleComplex call is used in a
                macro definition and is not valid for all macro uses. Adjust the
                code.
                */
                val[loc] = MAGMA_Z_ZERO;
                col[loc] = lcol;
            } 
        } 
        else {
                /*
                DPCT1064:164: Migrated make_cuDoubleComplex call is used in a
                macro definition and is not valid for all macro uses. Adjust the
                code.
                */
                val[loc] = MAGMA_Z_ZERO;
                col[loc] = 0;
        }
    }
}// kernel 


/**
    Purpose
    -------
    Generates a block-diagonal sparsity pattern with block-size bs on the GPU.

    Arguments
    ---------
    
    @param[in]
    n           magma_int_t
                Size of the matrix.
                
    @param[in]
    bs          magma_int_t
                Size of the diagonal blocks.
                
    @param[in]
    offs        magma_int_t
                Size of the first diagonal block.
                
    @param[in]
    uplotype    magma_uplo_t
                lower or upper triangular
                
    @param[in,out]
    A           magma_z_matrix*
                Generated sparsity pattern matrix.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmisai_blockstruct_gpu(
    magma_int_t n,
    magma_int_t bs,
    magma_int_t offs,
    magma_uplo_t uplotype,
    magma_z_matrix *A,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    offs = 0;
    
    A->val = NULL;
    A->col = NULL;
    A->row = NULL;
    A->rowidx = NULL;
    A->blockinfo = NULL;
    A->diag = NULL;
    A->dval = NULL;
    A->dcol = NULL;
    A->drow = NULL;
    A->drowidx = NULL;
    A->ddiag = NULL;
    A->num_rows = n;
    A->num_cols = n;
    A->nnz = n*max(bs,offs);
    A->memory_location = Magma_DEV;
    A->storage_type = Magma_CSR;
    printf(" allocate memory of size %lld and %lld\n", (long long) A->num_rows+1, (long long) A->nnz );
    magma_zmalloc( &A->dval, A->nnz );
    magma_index_malloc( &A->drow, A->num_rows+1 );
    magma_index_malloc( &A->dcol, A->nnz );
        
    int maxbs = 12; //max(offs, bs);
    
    int blocksize1 = BLOCKSIZE;
    int blocksize2 = 1;
    int blocksize3 = 1;
    int dimgrid1 = magma_ceildiv(n, BLOCKSIZE);
    int dimgrid2 = 1;
    int dimgrid3 = 1;

    sycl::range<3> grid(dimgrid3, dimgrid2, dimgrid1);
    sycl::range<3> block(blocksize3, blocksize2, blocksize1);

    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        auto A_num_rows_ct0 = A->num_rows;
        auto A_drow_ct2 = A->drow;
        auto A_dcol_ct3 = A->dcol;
        auto A_dval_ct4 = A->dval;

        cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_zmisai_blockstruct_row_kernel(
                                 A_num_rows_ct0, maxbs, A_drow_ct2, A_dcol_ct3,
                                 A_dval_ct4, item_ct1);
                         });
    });

    blocksize1 = maxbs;
    blocksize2 = maxbs;
    dimgrid1 = min( int( sqrt( double( A->num_rows ))), 65535 );
    dimgrid2 = min(magma_ceildiv( A->num_rows, dimgrid1 ), 65535);
    dimgrid3 = magma_ceildiv( A->num_rows, dimgrid1*dimgrid2 );
    // dimgrid1 = n;
    // dimgrid2 = 1;
    // dimgrid3 = 1;

    sycl::range<3> grid2(dimgrid3, dimgrid2, dimgrid1);
    sycl::range<3> block2(1, blocksize2, blocksize1);

    // for now: no offset
    if( uplotype == MagmaLower ){printf("enter here\n");
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                auto A_num_rows_ct0 = A->num_rows;
                auto A_drow_ct2 = A->drow;
                auto A_dcol_ct3 = A->dcol;
                auto A_dval_ct4 = A->dval;

                cgh.parallel_for(sycl::nd_range<3>(grid2 * block2, block2),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     magma_zmisai_blockstruct_fill_l_kernel(
                                         A_num_rows_ct0, maxbs, A_drow_ct2,
                                         A_dcol_ct3, A_dval_ct4, item_ct1);
                                 });
            });
    } else {
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                auto A_num_rows_ct0 = A->num_rows;
                auto A_drow_ct2 = A->drow;
                auto A_dcol_ct3 = A->dcol;
                auto A_dval_ct4 = A->dval;

                cgh.parallel_for(sycl::nd_range<3>(grid2 * block2, block2),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     magma_zmisai_blockstruct_fill_u_kernel(
                                         A_num_rows_ct0, maxbs, A_drow_ct2,
                                         A_dcol_ct3, A_dval_ct4, item_ct1);
                                 });
            });
    }
    magma_z_mvisu(*A, queue );
    
    return info;
}
