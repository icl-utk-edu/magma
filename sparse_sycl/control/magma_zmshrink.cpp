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

    Shrinks a non-square matrix (m < n) to the smaller dimension.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                sparse matrix A
                
    @param[out]
    B           magma_z_matrix*
                sparse matrix A

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmshrink(
    magma_z_matrix A,
    magma_z_matrix *B,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // make sure the target structure is empty
    magma_zmfree( B, queue );

    magma_z_matrix hA={Magma_CSR}, hACSR={Magma_CSR}, hB={Magma_CSR}, hBCSR={Magma_CSR};
     
    if( A.num_rows<=A.num_cols){
        if( A.memory_location == Magma_CPU && A.storage_type == Magma_CSR ){
            CHECK( magma_zmconvert( A, B, Magma_CSR, Magma_CSR, queue ));
            for(magma_int_t i=0; i<A.nnz; i++){
                if( B->col[i] >= A.num_rows ){
                    B->val[i] = MAGMA_Z_ZERO;   
                }
            }
            CHECK( magma_zmcsrcompressor( B, queue ) );
            B->num_cols = B->num_rows;
        } else {
            CHECK( magma_zmtransfer( A, &hA, A.memory_location, Magma_CPU, queue ));
            CHECK( magma_zmconvert( hA, &hACSR, A.storage_type, Magma_CSR, queue ));
            CHECK( magma_zmshrink( hACSR, &hBCSR, queue ));
            CHECK( magma_zmconvert( hBCSR, &hB, Magma_CSR, A.storage_type, queue ));
            CHECK( magma_zmtransfer( hB, B, Magma_CPU, A.memory_location, queue ));
        }
    } else {
        printf("%% error: A has too many rows: m > n.\n");
        info = MAGMA_ERR_NOT_SUPPORTED;
        goto cleanup;
    }
    
cleanup:    
    magma_zmfree( &hA, queue );
    magma_zmfree( &hB, queue );
    magma_zmfree( &hACSR, queue );
    magma_zmfree( &hBCSR, queue );

    return info;
}
