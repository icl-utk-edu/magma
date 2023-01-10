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

    Transposes a vector from col to row major and vice versa.


    Arguments
    ---------

    @param[in]
    x           magma_z_matrix
                input vector

    @param[out]
    y           magma_z_matrix*
                output vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zvtranspose(
    magma_z_matrix x,
    magma_z_matrix *y,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t    m = x.num_rows;
    magma_int_t    n = x.num_cols;
    
    // make sure the target structure is empty
    magma_zmfree( y, queue );
    y->ownership = MagmaTrue;
    
    magma_z_matrix dx={Magma_CSR}, dy={Magma_CSR};
            
    if ( x.memory_location == Magma_DEV ) {
        CHECK( magma_zvinit( y, Magma_DEV, x.num_rows,x.num_cols, MAGMA_Z_ZERO, queue ));
        y->num_rows = x.num_rows;
        y->num_cols = x.num_cols;
        y->storage_type = x.storage_type;
        if ( x.major == MagmaColMajor) {
            y->major = MagmaRowMajor;
            magmablas_ztranspose( m, n, x.val, m, y->val, n, queue );
        }
        else {
            y->major = MagmaColMajor;
            magmablas_ztranspose( n, m, x.val, n, y->val, m, queue );
        }
    } else {
        CHECK( magma_zmtransfer( x, &dx, Magma_CPU, Magma_DEV, queue ));
        CHECK( magma_zvtranspose( dx, &dy, queue ));
        CHECK( magma_zmtransfer( dy, y, Magma_DEV, Magma_CPU, queue ));
    }
    
cleanup:
    if( info != 0 ){
        magma_zmfree( y, queue );
    }
    magma_zmfree( &dx, queue );
    magma_zmfree( &dy, queue );
    return info;
}
