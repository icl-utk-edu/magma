/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt
*/

//  in this file, many routines are taken from
//  the IO functions provided by MatrixMarket

#include "magmasparse_internal.h"


/**
    Purpose
    -------

    Passes a vector to MAGMA (located on DEV).

    Arguments
    ---------

    @param[in]
    m           magma_int_t
                number of rows

    @param[in]
    n           magma_int_t
                number of columns

    @param[in]
    val         magmaDoubleComplex_ptr
                array containing vector entries

    @param[out]
    v           magma_z_matrix*
                magma vector
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zvset_dev(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr val,
    magma_z_matrix *v,
    magma_queue_t queue )
{
    
    // make sure the target structure is empty
    magma_zmfree( v, queue );
    
    v->num_rows = m;
    v->num_cols = n;
    v->nnz = m*n;
    v->memory_location = Magma_DEV;
    v->storage_type = Magma_DENSE;
    v->dval = val;
    v->major = MagmaColMajor;

    return MAGMA_SUCCESS;
}


/**
    Purpose
    -------

    Passes a MAGMA vector back (located on DEV).

    Arguments
    ---------

    @param[in]
    v           magma_z_matrix
                magma vector

    @param[out]
    m           magma_int_t
                number of rows

    @param[out]
    n           magma_int_t
                number of columns

    @param[out]
    val         magmaDoubleComplex_ptr
                array containing vector entries

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zvget_dev(
    magma_z_matrix v,
    magma_int_t *m, magma_int_t *n,
    magmaDoubleComplex_ptr *val,
    magma_queue_t queue )
{
    magma_int_t info =0;
    
    magma_z_matrix v_DEV={Magma_CSR};
    
    if ( v.memory_location == Magma_DEV ) {
        *m = v.num_rows;
        *n = v.num_cols;
        *val = v.dval;
    } else {
        CHECK( magma_zmtransfer( v, &v_DEV, v.memory_location, Magma_DEV, queue ));
        CHECK( magma_zvget_dev( v_DEV, m, n, val, queue ));
    }
    
cleanup:
    magma_zmfree( &v_DEV, queue );
    return info;
}
