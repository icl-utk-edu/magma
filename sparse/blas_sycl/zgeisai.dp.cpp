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
#include "shuffle.cuh"
#include <cmath>

#define PRECISION_z
#define COMPLEX
#define BLOCKSIZE 32
#define WARP_SIZE 32
#define WRP 32
#define WRQ 4


// initialize arrays with zero
void
magma_zgpumemzero_z(
    magmaDoubleComplex * d,
    int n,
    int dim_x,
    int dim_y ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(1) * item_ct1.get_group_range(2) +
            item_ct1.get_group(2);
    int idx = item_ct1.get_local_id(2);

    if( i >= n ){
       return;
    }
    if( idx >= dim_x ){
       return;
    }

    for( int j=0; j<dim_y; j++)
        d[i * dim_x * dim_y + j * dim_y + idx] {0.0, 0.0};
}

void
magma_zlocations_lower_kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaDoubleComplex *val,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_local_id(2);
    int j = item_ct1.get_group(1) * item_ct1.get_group_range(2) +
            item_ct1.get_group(2);

    if( j >= n ){
        return;
    }
    int start = row[j];
    int end = row[j+1];
    int count = end-start;
    if( i == 0 ){
        sizes[j] = count;
        /*
        DPCT1064:486: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        rhs[j * WARP_SIZE] = MAGMA_Z_ONE;
    }

    if ( i<count ){
        locations[ j*WARP_SIZE + i ] = col[ row[j]+i ];
    }
}// kernel


void
magma_zlocations_trunc_lower_kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaDoubleComplex *val,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_local_id(2);
    int j = item_ct1.get_group(1) * item_ct1.get_group_range(2) +
            item_ct1.get_group(2);

    if( j >= n ){
        return;
    }
    int start = row[j];
    int end = row[j+1];
    int count = end-start;

    // normal case
    if( count <= BLOCKSIZE ){ // normal case
        if( i == 0 ){
            sizes[j] = count;
            /*
            DPCT1064:487: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            rhs[j * WARP_SIZE] = MAGMA_Z_ONE;
        }
        if ( i<count ){
            locations[ j*WARP_SIZE + i ] = col[ row[j]+i ];
        }
    }
    else {
        // truncate in this row to the blocksize,
        // take only the 32 elements close to the main diagonal into account
        count = BLOCKSIZE;
        if (i == 0) {
            sizes[j] = count;
            /*
            DPCT1064:488: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            rhs[j * WARP_SIZE] = MAGMA_Z_ONE;
        }

        locations[ j*WARP_SIZE + i ] = col[ row[j+1]-BLOCKSIZE+i ];
    }
}// kernel



void
magma_zlocations_upper_kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaDoubleComplex *val,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_local_id(2);
    int j = item_ct1.get_group(1) * item_ct1.get_group_range(2) +
            item_ct1.get_group(2);

    if( j >= n ){
        return;
    }
    int start = row[j];
    int end = row[j+1];
    int count = end-start;
    if( i == 0 ){
        sizes[j] = count;
        /*
        DPCT1064:489: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        rhs[j * WARP_SIZE + count - 1] = MAGMA_Z_ONE;
    }

    if ( i<count ){
        locations[ j*WARP_SIZE + i ] = col[ row[j]+i ];
    }
}// kernel

void
magma_zlocations_trunc_upper_kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaDoubleComplex *val,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_local_id(2);
    int j = item_ct1.get_group(1) * item_ct1.get_group_range(2) +
            item_ct1.get_group(2);

    if( j >= n ){
        return;
    }
    int start = row[j];
    int end = row[j+1];
    int count = end-start;

    // normal case
    if( count <= BLOCKSIZE ){ // normal case
        if( i == 0 ){
            sizes[j] = count;
            /*
            DPCT1064:490: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            rhs[j * WARP_SIZE + count - 1] = MAGMA_Z_ONE;
        }
        if ( i<count ){
            locations[ j*WARP_SIZE + i ] = col[ row[j]+i ];
        }
    }
    else {
        // truncate in this row to the blocksize,
        // take only the 32 elements close to the main diagonal into account
        count = BLOCKSIZE;
        if (i == 0) {
            sizes[j] = count;
            /*
            DPCT1064:491: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            rhs[j * WARP_SIZE + count - 1] = MAGMA_Z_ONE;
        }

        locations[ j*WARP_SIZE + i ] = col[ row[j]+i ];
    }
}// kernel

void
magma_zfilltrisystems_kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaDoubleComplex *val,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs ,
    sycl::nd_item<3> item_ct1)
{
    int i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) +
             item_ct1.get_local_id(2));

    if ( i>=n ){
        return;
    }
    for( int j=0; j<sizes[ i ]; j++ ){// no need for first
        int k = row[ locations[ j+i*WARP_SIZE ] ];
        int l = i*WARP_SIZE;
        int idx = 0;
        while( k < row[ locations[ j+i*WARP_SIZE ]+1 ] && l < (i+1)*WARP_SIZE ){ // stop once this column is done
            if( locations[ l ] == col[k] ){ //match
                // int loc = i*WARP_SIZE*WARP_SIZE + j*WARP_SIZE + idx;
                trisystems[ i*WARP_SIZE*WARP_SIZE + j*WARP_SIZE + idx ]
                                                        = val[ k ];
                k++;
                l++;
                idx++;
            } else if( col[k] < locations[ l ] ){// need to check next element
                k++;
            } else { // element does not exist, i.e. l < LC.col[k]
                // printf("increment l\n");
                l++; // check next elment in the sparsity pattern
                idx++; // leave this element equal zero
            }
        }
    }
}// kernel


/**
    Purpose
    -------

    This routine prepares the batch of small triangular systems that
    need to be solved for computing the ISAI preconditioner.


    Arguments
    ---------

    @param[in]
    uplotype    magma_uplo_t
                input matrix

    @param[in]
    transtype   magma_trans_t
                input matrix

    @param[in]
    diagtype    magma_diag_t
                input matrix

    @param[in]
    L           magma_z_matrix
                triangular factor for which the ISAI matrix is computed.
                Col-Major CSR storage.

    @param[in]
    LC          magma_z_matrix
                sparsity pattern of the ISAI matrix.
                Col-Major CSR storage.

    @param[in,out]
    sizes       magma_index_t*
                array containing the sizes of the small triangular systems

    @param[in,out]
    locations   magma_index_t*
                array containing the locations in the respective column of L

    @param[in,out]
    trisystems  magmaDoubleComplex*
                batch of generated small triangular systems. All systems are
                embedded in uniform memory blocks of size BLOCKSIZE x BLOCKSIZE

    @param[in,out]
    rhs         magmaDoubleComplex*
                RHS of the small triangular systems

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zmprepare_batched_gpu(
    magma_uplo_t uplotype,
    magma_trans_t transtype,
    magma_diag_t diagtype,
    magma_z_matrix L,
    magma_z_matrix LC,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs,
    magma_queue_t queue )
{
    int blocksize1 = WARP_SIZE;
    int blocksize2 = 1;
    int dimgrid1 = min( int( sqrt( double( LC.num_rows ))), 65535 );
    int dimgrid2 = min(magma_ceildiv( LC.num_rows, dimgrid1 ), 65535);
    int dimgrid3 = magma_ceildiv( LC.num_rows, dimgrid1*dimgrid2 );
    /*
    DPCT1026:492: The call to cudaDeviceSetCacheConfig was removed because DPC++
    currently does not support setting cache config on devices.
    */
    sycl::range<3> grid(dimgrid3, dimgrid2, dimgrid1);
    sycl::range<3> block(1, blocksize2, blocksize1);

    int blocksize21 = BLOCKSIZE;
    int blocksize22 = 1;

    int dimgrid21 = magma_ceildiv( LC.num_rows, blocksize21 );
    int dimgrid22 = 1;
    int dimgrid23 = 1;
    sycl::range<3> grid2(dimgrid23, dimgrid22, dimgrid21);
    sycl::range<3> block2(1, blocksize22, blocksize21);

    /*
    DPCT1049:493: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * block, block),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zgpumemzero_z(trisystems, LC.num_rows,
                                               WARP_SIZE, WARP_SIZE, item_ct1);
                       });

    /*
    DPCT1049:494: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * block, block),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zgpumemzero_z(rhs, LC.num_rows, WARP_SIZE, 1,
                                               item_ct1);
                       });

   // magma_zprint_gpu( 32, 32, L.dval, 32, queue );

    // Runtime API
    // cudaFuncCachePreferShared: shared memory is 48 KB
    // cudaFuncCachePreferEqual: shared memory is 32 KB
    // cudaFuncCachePreferL1: shared memory is 16 KB
    // cudaFuncCachePreferNone: no preference
    //cudaFuncSetCacheConfig(cudaFuncCachePreferShared);


    if( uplotype == MagmaLower ){
        /*
        DPCT1049:496: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(sycl::nd_range<3>(grid * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                               magma_zlocations_lower_kernel(
                                   LC.num_rows, LC.drow, LC.dcol, LC.dval,
                                   sizes, locations, trisystems, rhs, item_ct1);
                           });
    } else {
        /*
        DPCT1049:497: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(sycl::nd_range<3>(grid * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                               magma_zlocations_upper_kernel(
                                   LC.num_rows, LC.drow, LC.dcol, LC.dval,
                                   sizes, locations, trisystems, rhs, item_ct1);
                           });
    }

    // magma_zprint_gpu( 32, 32, L.dval, 32, queue );

    /*
    DPCT1049:495: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid2 * block2, block2),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zfilltrisystems_kernel(
                               L.num_rows, L.drow, L.dcol, L.dval, sizes,
                               locations, trisystems, rhs, item_ct1);
                       });
    //magma_zprint_gpu( 32, 32, L.dval, 32, queue );

    return MAGMA_SUCCESS;
}


void
magma_zbackinsert_kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaDoubleComplex *val,
    magma_index_t *sizes,
    magmaDoubleComplex *rhs ,
    sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_local_id(2);
    int j = item_ct1.get_group(1) * item_ct1.get_group_range(2) +
            item_ct1.get_group(2);
    int end = sizes[j];
    if( j >= n ){
        return;
    }

    if ( i>=end ){
        return;
    }

    val[row[j]+i] = rhs[j*WARP_SIZE+i];
}// kernel



/**
    Purpose
    -------
    Inserts the values into the preconditioner matrix

    Arguments
    ---------


    @param[in]
    uplotype    magma_uplo_t
                lower or upper triangular

    @param[in]
    transtype   magma_trans_t
                possibility for transposed matrix

    @param[in]
    diagtype    magma_diag_t
                unit diagonal or not

    @param[in,out]
    M           magma_z_matrix*
                SPAI preconditioner CSR col-major

    @param[out]
    sizes       magma_int_t*
                Number of Elements that are replaced.

    @param[out]
    locations   magma_int_t*
                Array indicating the locations.

    @param[out]
    trisystems  magmaDoubleComplex*
                trisystems

    @param[out]
    rhs         magmaDoubleComplex*
                right-hand sides

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmbackinsert_batched_gpu(
    magma_uplo_t uplotype,
    magma_trans_t transtype,
    magma_diag_t diagtype,
    magma_z_matrix *M,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    int blocksize1 = WARP_SIZE;
    int blocksize2 = 1;
    int dimgrid1 = min( int( sqrt( double( M->num_rows ))), 65535 );
    int dimgrid2 = min(magma_ceildiv( M->num_rows, dimgrid1 ), 65535);
    int dimgrid3 = magma_ceildiv( M->num_rows, dimgrid1*dimgrid2 );

    sycl::range<3> grid(dimgrid3, dimgrid2, dimgrid1);
    sycl::range<3> block(1, blocksize2, blocksize1);

    /*
    DPCT1049:498: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        auto M_num_rows_ct0 = M->num_rows;
        auto M_drow_ct1 = M->drow;
        auto M_dcol_ct2 = M->dcol;
        auto M_dval_ct3 = M->dval;

        cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_zbackinsert_kernel(
                                 M_num_rows_ct0, M_drow_ct1, M_dcol_ct2,
                                 M_dval_ct3, sizes, rhs, item_ct1);
                         });
    });

    return info;
}
