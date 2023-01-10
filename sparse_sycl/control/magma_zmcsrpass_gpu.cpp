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

    Passes a CSR matrix to MAGMA (located on DEV).

    Arguments
    ---------

    @param[in]
    m           magma_int_t
                number of rows

    @param[in]
    n           magma_int_t
                number of columns

    @param[in]
    row         magmaIndex_ptr
                row pointer

    @param[in]
    col         magmaIndex_ptr
                column indices

    @param[in]
    val         magmaDoubleComplex_ptr
                array containing matrix entries

    @param[out]
    A           magma_z_matrix*
                matrix in magma sparse matrix format
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zcsrset_gpu(
    magma_int_t m,
    magma_int_t n,
    magmaIndex_ptr row,
    magmaIndex_ptr col,
    magmaDoubleComplex_ptr val,
    magma_z_matrix *A,
    magma_queue_t queue )
{   
    // make sure the target structure is empty
    magma_zmfree( A, queue );
    
    A->num_rows = m;
    A->num_cols = n;
    magma_index_t nnz;
    magma_index_getvector( 1, row+m, 1, &nnz, 1, queue );
    A->nnz = (magma_int_t) nnz;
    A->storage_type = Magma_CSR;
    A->memory_location = Magma_DEV;
    A->dval = val;
    A->dcol = col;
    A->drow = row;
    A->ownership = MagmaFalse;

    return MAGMA_SUCCESS;
}


/**
    Purpose
    -------

    Passes a MAGMA matrix to CSR structure (located on DEV).

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                magma sparse matrix in CSR format

    @param[out]
    m           magma_int_t
                number of rows

    @param[out]
    n           magma_int_t
                number of columns

    @param[out]
    row         magmaIndex_ptr
                row pointer

    @param[out]
    col         magmaIndex_ptr
                column indices

    @param[out]
    val         magmaDoubleComplex_ptr
                array containing matrix entries

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zcsrget_gpu(
    magma_z_matrix A,
    magma_int_t *m,
    magma_int_t *n,
    magmaIndex_ptr *row,
    magmaIndex_ptr *col,
    magmaDoubleComplex_ptr *val,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_z_matrix A_DEV={Magma_CSR}, A_CSR={Magma_CSR};
    
    if ( A.memory_location == Magma_DEV && A.storage_type == Magma_CSR ) {
        *m = A.num_rows;
        *n = A.num_cols;
        *val = A.dval;
        *col = A.dcol;
        *row = A.drow;
        A.ownership = MagmaFalse;
    } else {
        CHECK( magma_zmconvert( A, &A_CSR, A.storage_type, Magma_CSR, queue ));
        CHECK( magma_zmtransfer( A_CSR, &A_DEV, A.memory_location, Magma_DEV, queue ));
        magma_zcsrget_gpu( A_DEV, m, n, row, col, val, queue );
    }
    
cleanup:
    magma_zmfree( &A_CSR, queue );
    magma_zmfree( &A_DEV, queue );
    return info;
}
