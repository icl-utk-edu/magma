/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magmasparse_internal.h"

#define BLOCK_SIZE 256



// kernel copying everything except the last element
void
magma_zpreselect_gpu0( 
    magma_int_t num_rows,
    magmaIndex_ptr row,
    magmaDoubleComplex *val,
    magmaDoubleComplex *valn,
    sycl::nd_item<3> item_ct1)
{
    int tidx = item_ct1.get_local_id(2);
    int bidx = item_ct1.get_group(2);
    int gtidx = bidx * item_ct1.get_local_range(2) + tidx;

    if (gtidx < num_rows) {
        for (int i=row[gtidx]; i<row[gtidx+1]-1; i++){
            valn[i-gtidx] = val[i];
        }
    }
}

// kernel copying everything except the first element
void
magma_zpreselect_gpu1( 
    magma_int_t num_rows,
    magmaIndex_ptr row,
    magmaDoubleComplex *val,
    magmaDoubleComplex *valn,
    sycl::nd_item<3> item_ct1)
{
    int tidx = item_ct1.get_local_id(2);
    int bidx = item_ct1.get_group(2);
    int gtidx = bidx * item_ct1.get_local_range(2) + tidx;

    if (gtidx < num_rows) {
        for (int i=row[gtidx]+1; i<row[gtidx+1]; i++){
            valn[i-gtidx] = val[i];
        }
    }
}



/***************************************************************************//**
    Purpose
    -------
    This function takes a list of candidates with residuals, 
    and selects the largest in every row. The output matrix only contains these
    largest elements (respectively a zero element if there is no candidate for
    a certain row).

    Arguments
    ---------

    @param[in]
    order       magma_int_t
                order==0 lower triangular
                order==1 upper triangular
                
    @param[in]
    A           magma_z_matrix*
                Matrix where elements are removed.
                
    @param[out]
    oneA        magma_z_matrix*
                Matrix where elements are removed.
                
                

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zpreselect_gpu(
    magma_int_t order,
    magma_z_matrix *A,
    magma_z_matrix *oneA,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    sycl::range<3> block(1, 1, BLOCK_SIZE);
    sycl::range<3> grid(1, 1, magma_ceildiv(A->num_rows, BLOCK_SIZE));

    oneA->num_rows = A->num_rows;
    oneA->num_cols = A->num_cols;
    oneA->nnz = A->nnz - A->num_rows;
    oneA->storage_type = Magma_CSR;
    oneA->memory_location = Magma_DEV;
    
    CHECK( magma_zmalloc( &oneA->dval, oneA->nnz ) );
    
    if( order == 1 ){ // don't copy the first
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                auto A_num_rows_ct0 = A->num_rows;
                auto A_drow_ct1 = A->drow;
                auto A_dval_ct2 = A->dval;
                auto oneA_dval_ct3 = oneA->dval;

                cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     magma_zpreselect_gpu1(
                                         A_num_rows_ct0, A_drow_ct1, A_dval_ct2,
                                         oneA_dval_ct3, item_ct1);
                                 });
            });
        // #pragma omp parallel for
        // for( magma_int_t row=0; row<A->num_rows; row++){
        //     for( magma_int_t i=A->row[row]+1; i<A->row[row+1]; i++ ){
        //         oneA->val[ i-row ] = A->val[i];
        //     }
        // }
    } else { // don't copy the last
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                auto A_num_rows_ct0 = A->num_rows;
                auto A_drow_ct1 = A->drow;
                auto A_dval_ct2 = A->dval;
                auto oneA_dval_ct3 = oneA->dval;

                cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     magma_zpreselect_gpu0(
                                         A_num_rows_ct0, A_drow_ct1, A_dval_ct2,
                                         oneA_dval_ct3, item_ct1);
                                 });
            });
        // #pragma omp parallel for
        // for( magma_int_t row=0; row<A->num_rows; row++){
        //     for( magma_int_t i=A->row[row]; i<A->row[row+1]-1; i++ ){
        //         oneA->val[ i-row ] = A->val[i];
        //     }
        // }            
    }
    
cleanup:
    return info;
}
