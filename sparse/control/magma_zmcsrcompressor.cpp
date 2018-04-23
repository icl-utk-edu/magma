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


/**
    Purpose
    -------

    Removes zeros in a CSR matrix.

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
magma_zmcsrcompressor(
    magma_z_matrix *A,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_z_matrix B={Magma_CSR};
    magma_z_matrix hA={Magma_CSR}, CSRA={Magma_CSR};
        
    if (A->ownership) {
        if ( A->memory_location == Magma_CPU && A->storage_type == Magma_CSR ) {
            CHECK( magma_zmconvert( *A, &B, Magma_CSR, Magma_CSR, queue ));
    
            magma_free_cpu( A->row );
            magma_free_cpu( A->col );
            magma_free_cpu( A->val );
            CHECK( magma_z_csr_compressor(&B.val, &B.row, &B.col,
                           &A->val, &A->row, &A->col, &A->num_rows, queue ));
            A->nnz = A->row[A->num_rows];
        }
        else {
            magma_storage_t A_storage = A->storage_type;
            magma_location_t A_location = A->memory_location;
            CHECK( magma_zmtransfer( *A, &hA, A->memory_location, Magma_CPU, queue ));
            CHECK( magma_zmconvert( hA, &CSRA, hA.storage_type, Magma_CSR, queue ));
    
            CHECK( magma_zmcsrcompressor( &CSRA, queue ));
    
            magma_zmfree( &hA, queue );
            magma_zmfree( A, queue );
            CHECK( magma_zmconvert( CSRA, &hA, Magma_CSR, A_storage, queue ));
            CHECK( magma_zmtransfer( hA, A, Magma_CPU, A_location, queue ));
            magma_zmfree( &hA, queue );
            magma_zmfree( &CSRA, queue );
        }
    }
    
cleanup:
    magma_zmfree( &hA, queue );
    magma_zmfree( &CSRA, queue );
    magma_zmfree( &B, queue );
    return info;
}
