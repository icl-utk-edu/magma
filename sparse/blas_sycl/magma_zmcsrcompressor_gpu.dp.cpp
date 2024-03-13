/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magmasparse_internal.h"

#define BLOCK_SIZE1 256
#define BLOCK_SIZE2 1


// copy nonzeros into new structure
void
magma_zmcsrgpu_kernel1( int num_rows,
                 magmaDoubleComplex *A_val,
                 magma_index_t *A_rowptr,
                 magma_index_t *A_colind,
                 magmaDoubleComplex *B_val,
                 magma_index_t *B_rowptr,
                 magma_index_t *B_colind ,
                 sycl::nd_item<3> item_ct1)
{
    int row = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    int j;

    if(row<num_rows){
        /*
        DPCT1064:89: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        magmaDoubleComplex zero = MAGMA_Z_ZERO;
        int start = A_rowptr[ row ];
        int new_location = start;
        int end = A_rowptr[ row+1 ];
        for( j=start; j<end; j++ ){
            if( A_val[j] != zero ){
       //         B_val[new_location] = A_val[j];
       //         B_colind[new_location] = A_colind[j];
                new_location++;
            }
        }
        // this is not a correctr rowpointer! this is nn_z in this row!
        B_rowptr[ row ] = new_location-start;
    }
}


// generate a valid rowpointer
void
magma_zmcsrgpu_kernel2( int num_rows,
                 magma_index_t *B_rowptr,
                 magma_index_t *A_rowptr ,
                 sycl::nd_item<3> item_ct1)
{
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    int j, nnz = 0;

    if( idx == 0 ){
    A_rowptr[ 0 ] = nnz;
        for( j=0; j<num_rows; j++ ){
            nnz+=B_rowptr[ j ];
            A_rowptr[ j+1 ] = nnz;
        }
    }
}



// copy new structure into original matrix
void
magma_zmcsrgpu_kernel3( int num_rows,
                 magmaDoubleComplex *B_val,
                 magma_index_t *B_rowptr,
                 magma_index_t *B_colind,
                 magma_index_t *B2_rowptr,
                 magmaDoubleComplex *A_val,
                 magma_index_t *A_rowptr,
                 magma_index_t *A_colind
                                            ,
                 sycl::nd_item<3> item_ct1)
{
    int row = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    int j, new_location;
    
    if(row<num_rows){
    new_location = A_rowptr[ row ];
        int start = B2_rowptr[ row ];
        int end = B2_rowptr[ row+1 ];
        /*
        DPCT1064:90: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        magmaDoubleComplex zero = MAGMA_Z_ZERO;
        for( j=start; j<end; j++ ){
            if( A_val[j] != zero ){
                B_val[new_location] = A_val[j];
                B_colind[new_location] = A_colind[j];
                new_location++;
            }
               // A_val[ j ] = B_val[ j ];
               // A_colind[ j ] = B_colind[ j ];
        }
    }
}


/**
    Purpose
    -------

    Removes zeros in a CSR matrix. This is a GPU implementation of the
    CSR compressor.

    Arguments
    ---------

    @param[in,out]
    A           magma_z_matrix*
                input/output matrix
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmcsrcompressor_gpu(
    magma_z_matrix *A,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_z_matrix B={Magma_CSR}, B2={Magma_CSR};
    magma_z_matrix dA={Magma_CSR}, CSRA={Magma_CSR};
    magma_index_t *cputmp = NULL;
    
    if ( A->memory_location == Magma_DEV && A->storage_type == Magma_CSR ) {
        CHECK( magma_index_malloc( &B.drow, A->num_rows + 1 ));
        CHECK( magma_index_malloc( &B2.drow, A->num_rows + 1 ));
        
        magma_index_copyvector( (A->num_rows+1), A->drow, 1, B2.drow, 1, queue );

        sycl::range<3> grid1(1, 1, magma_ceildiv(A->num_rows, BLOCK_SIZE1));

        // copying the nonzeros into B and write in B.drow how many there are
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                auto A_num_rows_ct0 = A->num_rows;
                auto A_dval_ct1 = A->dval;
                auto A_drow_ct2 = A->drow;
                auto A_dcol_ct3 = A->dcol;

                cgh.parallel_for(
                    sycl::nd_range<3>(grid1 * sycl::range<3>(1, 1, BLOCK_SIZE1),
                                      sycl::range<3>(1, 1, BLOCK_SIZE1)),
                    [=](sycl::nd_item<3> item_ct1) {
                        magma_zmcsrgpu_kernel1(A_num_rows_ct0, A_dval_ct1,
                                               A_drow_ct2, A_dcol_ct3, B.dval,
                                               B.drow, B.dcol, item_ct1);
                    });
            });

        // correct the row pointer
        sycl::range<3> grid2(1, 1, 1);
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                auto A_num_rows_ct0 = A->num_rows;
                auto A_drow_ct2 = A->drow;

                cgh.parallel_for(
                    sycl::nd_range<3>(grid2 * sycl::range<3>(1, 1, BLOCK_SIZE2),
                                      sycl::range<3>(1, 1, BLOCK_SIZE2)),
                    [=](sycl::nd_item<3> item_ct1) {
                        magma_zmcsrgpu_kernel2(A_num_rows_ct0, B.drow,
                                               A_drow_ct2, item_ct1);
                    });
            });
        // access the true number of nonzeros

        CHECK( magma_index_malloc_cpu( &cputmp, 1 ));

        magma_index_getvector( 1, A->row+(A->num_rows), 1, cputmp, 1, queue );
        A->nnz = (magma_int_t) cputmp[0];

        // reallocate with right size
        CHECK( magma_zmalloc( &B.dval, A->nnz ));
        CHECK( magma_index_malloc( &B.dcol, A->nnz ));
        
        // copy correct values back
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                auto A_num_rows_ct0 = A->num_rows;
                auto A_dval_ct5 = A->dval;
                auto A_drow_ct6 = A->drow;
                auto A_dcol_ct7 = A->dcol;

                cgh.parallel_for(
                    sycl::nd_range<3>(grid1 * sycl::range<3>(1, 1, BLOCK_SIZE1),
                                      sycl::range<3>(1, 1, BLOCK_SIZE1)),
                    [=](sycl::nd_item<3> item_ct1) {
                        magma_zmcsrgpu_kernel3(
                            A_num_rows_ct0, B.dval, B.drow, B.dcol, B2.drow,
                            A_dval_ct5, A_drow_ct6, A_dcol_ct7, item_ct1);
                    });
            });

        magma_free( A->dcol );
        magma_free( A->dval );

        A->dcol = B.dcol;
        A->dval = B.dval;
    }
    else {
        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        CHECK( magma_zmconvert( *A, &CSRA, A->storage_type, Magma_CSR, queue ));
        CHECK( magma_zmtransfer( *A, &dA, A->memory_location, Magma_DEV, queue ));

        CHECK( magma_zmcsrcompressor_gpu( &dA, queue ));

        magma_zmfree( &dA, queue );
        magma_zmfree( A, queue );
        CHECK( magma_zmtransfer( dA, &CSRA, Magma_DEV, A_location, queue ));
        CHECK( magma_zmconvert( CSRA, A, Magma_CSR, A_storage, queue ));
        magma_zmfree( &dA, queue );
        magma_zmfree( &CSRA, queue );
    }
    
cleanup:
    magma_zmfree( &dA, queue );
    magma_zmfree( &CSRA, queue );
    magma_free( B2.drow );
    magma_free( B.drow );
    return info;
}
