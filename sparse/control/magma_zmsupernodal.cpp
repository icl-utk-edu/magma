/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/

#include "magmasparse_internal.h"


/***************************************************************************//**
    Purpose
    -------
    Generates a block-diagonal sparsity pattern with block-size bs

    Arguments
    ---------

    @param[in,out]
    max_bs      magma_int_t*
                Size of the largest diagonal block.

    @param[in]
    A           magma_z_matrix
                System matrix.

    @param[in,out]
    S           magma_z_matrix*
                Generated sparsity pattern matrix.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmsupernodal(
    magma_int_t *max_bs,
    magma_z_matrix A,
    magma_z_matrix *S,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_int_t *blocksizes=NULL, *blocksizes2=NULL, *start=NULL, *v=NULL;
    magma_int_t blockcount=0, blockcount2=0;
    
    magma_z_matrix x={Magma_CSR};
    //char *filename = "blocksizes";
    //int nlength = 6752;

    int maxblocksize = *max_bs;
    int current_size = 0;
    int prev_matches = 0;
    
    // make sure the target structure is empty
    magma_zmfree( S, queue );

    CHECK( magma_imalloc_cpu( &v, A.num_rows+10 ));
    CHECK( magma_imalloc_cpu( &start, A.num_rows+10 ));
    CHECK( magma_imalloc_cpu( &blocksizes, A.num_rows+10 ));
    CHECK( magma_imalloc_cpu( &blocksizes2, A.num_rows+10 ));

    v[0] = 1;

    for( magma_int_t i=1; i<A.num_rows; i++ ){
        // pattern matches the pattern of the previous row
        int match = 0; // 0 means match!
        if( prev_matches == maxblocksize ){ // bounded by maxblocksize
            match = 1; // no match
            prev_matches = 0;
        } else if( ((A.row[i+1]-A.row[i])-(A.row[i]-A.row[i-1])) != 0 ){
            match = 1; // no match
            prev_matches = 0;
        } else {
            magma_index_t length = (A.row[i+1]-A.row[i]);
            magma_index_t start1 = A.row[i-1];
            magma_index_t start2 = A.row[i];
            for( magma_index_t j=0; j<length; j++ ){
                if( A.col[ start1+j ] != A.col[ start2+j ] ){
                    match = 1;
                    prev_matches = 0;
                }
            }
            if( match == 0 ){
                prev_matches++; // add one match to the block
            }
        }
        v[ i ] = match;
    }

    // start = find[v];
    blockcount = 0;
    for( magma_int_t i=0; i<A.num_rows; i++ ){
        if( v[i] == 1 ){
            start[blockcount] = i;
            blockcount++;
        }
    }
    start[blockcount] = A.num_rows;

    for( magma_int_t i=0; i<blockcount; i++ ){
        blocksizes[i] = start[i+1] - start[i];
        if( blocksizes[i] > maxblocksize ){
            // maxblocksize = blocksizes[i];
            // printf("%% warning: at i=%5lld blocksize required is %5lld\n",
            //                                                (long long) i, (long long) blocksizes[i] );
        }
    }

    current_size = 0;
    blockcount2=0;
    for( magma_int_t i=0; i<blockcount; i++ ){
        if( current_size + blocksizes[i] > maxblocksize ){
            blocksizes2[ blockcount2 ] = current_size;
            blockcount2++;
            current_size = blocksizes[i]; // form new block
        } else {
            current_size = current_size + blocksizes[i]; // add to previous block
        }
        blocksizes[i] = start[i+1] - start[i];
    }
    blocksizes2[ blockcount2 ] = current_size;
    blockcount2++;
    
    *max_bs = maxblocksize;
    
//     magma_zvread( &x, nlength, filename, queue );
// for( int z=0; z< nlength; z++){
//     blocksizes2[z] = (magma_int_t) MAGMA_Z_REAL((x.val[z]));  
// }
// blockcount2 = nlength;

    CHECK( magma_zmvarsizeblockstruct( A.num_rows, blocksizes2, blockcount2, MagmaLower, S, queue ) );
    
    CHECK( magma_index_malloc_cpu( &S->tile_desc_offset_ptr, blockcount2+1 ));
    S->tile_desc_offset_ptr[ 0 ] = 0;
    blockcount = 0;
    for( magma_int_t i=0; i<blockcount2; i++ ){
    blockcount = blockcount + blocksizes2[ i ];
        S->tile_desc_offset_ptr[ i+1 ] = blockcount ;        
    }
    
    S->numblocks = blockcount2;

cleanup:
    magma_free_cpu( v );
    magma_zmfree(&x, queue );
    magma_free_cpu( blocksizes );
    magma_free_cpu( blocksizes2 );
    magma_free_cpu( start );
    v = NULL;
    blocksizes = NULL;
    blocksizes2 = NULL;
    start = NULL;

    return info;
}


/***************************************************************************//**
    Purpose
    -------
    Generates a block-diagonal sparsity pattern with variable block-size

    Arguments
    ---------

    @param[in]
    n           magma_int_t
                Size of the matrix.

    @param[in]
    bs          magma_int_t*
                Vector containing the size of the diagonal blocks.

    @param[in]
    bsl         magma_int_t
                Size of the vector containing the block sizes.

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
magma_zmvarsizeblockstruct(
    magma_int_t n,
    magma_int_t *bs,
    magma_int_t bsl,
    magma_uplo_t uplotype,
    magma_z_matrix *A,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_int_t i, k, j, nnz = 0, col_start, row;

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
    A->memory_location = Magma_CPU;
    A->storage_type = Magma_CSR;
    A->nnz = 0;

    for( i=0; i<bsl; i++ ){
        A->nnz = A->nnz + bs[i] * bs[i];
    }

    CHECK( magma_zmalloc_cpu( &A->val, A->nnz ));
    CHECK( magma_index_malloc_cpu( &A->row, A->num_rows+1 ));
    CHECK( magma_index_malloc_cpu( &A->col, A->nnz ));
    nnz = 0;
    row = 0;
    col_start = 0;
    for( i=0; i<bsl; i++ ){
        for( j=0; j<bs[i]; j++ ){
            A->row[ row ] = nnz;
            row++;
            for( k=0; k<bs[i]; k++ ){
                A->val[ nnz + k ] = MAGMA_Z_ONE;
                A->col[ nnz + k ] = col_start + k;
            }
            nnz = nnz+bs[i];
        }
        col_start = col_start + bs[i];
    }
    A->row[ row ] = nnz;


    CHECK( magma_zmcsrcompressor( A, queue ) );

    // magma_z_mvisu( *A, queue );

cleanup:

    return info;
}
