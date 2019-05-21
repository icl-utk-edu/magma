/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt
*/
#include "magmasparse_internal.h"

#define THRESHOLD 10e-99


/**
    Purpose
    -------

    Takes a matrix and extracts a slice for solving the system in parallel:
    
        B = A( i:i+n, : ) and ALOC = A(i:i+n,i:i+n) and ANLOCA(0:start - end:n,:)
        
    B is of size n x n, ALOC of size end-start x end-start,
    ANLOC of size end-start x n
        
    The last slice might be smaller. For the non-local parts, B is the identity.
    comm contains 1ess in the locations that are non-local but needed to 
    solve local system.


    Arguments
    ---------
    
    @param[in]
    num_slices  magma_int_t
                number of slices
    
    @param[in]
    slice       magma_int_t
                slice id (0.. num_slices-1)

    @param[in]
    A           magma_z_matrix
                sparse matrix in CSR

    @param[out]
    B           magma_z_matrix*
                sparse matrix in CSR
                
    @param[out]
    ALOC        magma_z_matrix*
                sparse matrix in CSR
                
    @param[out]
    ANLOC       magma_z_matrix*
                sparse matrix in CSR
                
    @param[in,out]
    comm_i      magma_int_t*
                communication plan
 
    @param[in,out]
    comm_v      magmaDoubleComplex*
                communication plan

    @param[in]
    queue       magma_queue_t
                Queue to execute in.
                
    @param[out]
    start       magma_int_t*
                start of slice (row-index)
                
    @param[out]
    end         magma_int_t*
                end of slice (row-index)

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmslice(
    magma_int_t num_slices,
    magma_int_t slice,
    magma_z_matrix A, 
    magma_z_matrix *B,
    magma_z_matrix *ALOC,
    magma_z_matrix *ANLOC,
    magma_index_t *comm_i,
    magmaDoubleComplex *comm_v,
    magma_int_t *start,
    magma_int_t *end,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // make sure the target structure is empty
    magma_zmfree( B, queue );
    
    if( A.num_rows != A.num_cols ){
        printf("%%  error: only supported for square matrices.\n");
        info = MAGMA_ERR_NOT_SUPPORTED;
        goto cleanup;
    }
    
    if ( A.memory_location == Magma_CPU
            && A.storage_type == Magma_CSR ){
        CHECK( magma_zmconvert( A, B, Magma_CSR, Magma_CSR, queue ) );
        magma_free_cpu( B->col );
        magma_free_cpu( B->val );
        CHECK( magma_zmconvert( A, ALOC, Magma_CSR, Magma_CSR, queue ) );
        magma_free_cpu( ALOC->col );
        magma_free_cpu( ALOC->row );
        magma_free_cpu( ALOC->val );
        CHECK( magma_zmconvert( A, ANLOC, Magma_CSR, Magma_CSR, queue ) );
        magma_free_cpu( ANLOC->col );
        magma_free_cpu( ANLOC->row );
        magma_free_cpu( ANLOC->val );
        
        magma_int_t i,j,k, nnz, nnz_loc=0, loc_row = 0, nnz_nloc = 0;
        magma_index_t col;
        magma_int_t size = magma_ceildiv( A.num_rows, num_slices ); 
        magma_int_t lstart = slice*size;
        magma_int_t lend = min( (slice+1)*size, A.num_rows );
        // correct size for last slice
        size = lend-lstart;
        CHECK( magma_index_malloc_cpu( &ALOC->row, size+1 ) );
        CHECK( magma_index_malloc_cpu( &ANLOC->row, size+1 ) );
        
        // count elements for slice - identity for rest
        nnz = A.row[ lend ] - A.row[ lstart ] + ( A.num_rows - size );
        CHECK( magma_index_malloc_cpu( &B->col, nnz ) );
        CHECK( magma_zmalloc_cpu( &B->val, nnz ) );         
        
        // for the communication plan
        for( i=0; i<A.num_rows; i++ ) {
            comm_i[i] = 0;
            comm_v[i] = MAGMA_Z_ZERO;
        }
        
        k=0;
        B->row[i] = 0;
        ALOC->row[0] = 0;
        ANLOC->row[0] = 0;
        // identity above slice
        for( i=0; i<lstart; i++ ) {
            B->row[i+1]   = B->row[i]+1;
            B->val[k] = MAGMA_Z_ONE;
            B->col[k] = i;
            k++;
        }
        
        // slice        
        for( i=lstart; i<lend; i++ ) {
            B->row[i+1]   = B->row[i] + (A.row[i+1]-A.row[i]);
            for( j=A.row[i]; j<A.row[i+1]; j++ ){
                B->val[k] = A.val[j];
                col = A.col[j];
                B->col[k] = col;
                // communication plan
                if( col<lstart || col>=lend ){
                    comm_i[ col ] = 1;
                    comm_v[ col ] = comm_v[ col ] 
                            + MAGMA_Z_MAKE( MAGMA_Z_ABS( A.val[j] ), 0.0 );
                    nnz_nloc++;
                } else {
                    nnz_loc++;   
                }
                k++;
            }
            loc_row++;
            ALOC->row[ loc_row ] = nnz_loc;
            ANLOC->row[ loc_row ] = nnz_nloc;
        }
        CHECK( magma_index_malloc_cpu( &ALOC->col, nnz_loc ) );
        CHECK( magma_zmalloc_cpu( &ALOC->val, nnz_loc ) ); 
        ALOC->num_rows = size;
        ALOC->num_cols = size;
        ALOC->nnz = nnz_loc;
        
        CHECK( magma_index_malloc_cpu( &ANLOC->col, nnz_nloc ) );
        CHECK( magma_zmalloc_cpu( &ANLOC->val, nnz_nloc ) ); 
        ANLOC->num_rows = size;
        ANLOC->num_cols = A.num_cols;
        ANLOC->nnz = nnz_nloc;
        
        nnz_loc = 0;
        nnz_nloc = 0;
        // local/nonlocal matrix        
        for( i=lstart; i<lend; i++ ) {
            for( j=A.row[i]; j<A.row[i+1]; j++ ){
                col = A.col[j];
                // insert only in local part in ALOC, nonlocal in ANLOC
                if( col<lstart || col>=lend ){
                    ANLOC->val[ nnz_nloc ] = A.val[j];
                    ANLOC->col[ nnz_nloc ] = col;  
                    nnz_nloc++;
                } else {
                    ALOC->val[ nnz_loc ] = A.val[j];
                    ALOC->col[ nnz_loc ] = col-lstart;  
                    nnz_loc++;
                }
            }
        }
        
        // identity below slice
        for( i=lend; i<A.num_rows; i++ ) {
            B->row[i+1] = B->row[i]+1;
            B->val[k] = MAGMA_Z_ONE;
            B->col[k] = i;
            k++;
        }
        B->nnz = k;
        *start = lstart;
        *end = lend;
    }
    else {
        printf("error: mslice only supported for CSR matrices on the CPU: %d %d.\n", 
                int(A.memory_location), int(A.storage_type) );
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
cleanup:
    return info;
}
