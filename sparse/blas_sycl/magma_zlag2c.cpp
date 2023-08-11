/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds
       @author Hartwig Anzt
*/
#include "magmasparse_internal.h"


/**
    Purpose
    -------

    convertes magma_z_matrix from Z to C

    Arguments
    ---------

    @param
    x           magma_z_matrix
                input vector descriptor

    @param
    y           magma_c_vector*
                output vector descriptor
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_vector_zlag2c(
    magma_z_matrix x, magma_c_vector *y,
    magma_queue_t queue )
{
    magma_int_t info;
    if ( x.memory_location == Magma_DEV) {
        y->memory_location = x.memory_location;
        y->num_rows = x.num_rows;
        y->nnz = x.nnz;
        magma_cmalloc( &y->val, x.num_rows );
        magmablas_zlag2c_sparse( x.num_rows, 1, x.dval, x.num_rows, y->val,
                    x.num_rows, &info, queue );
        return MAGMA_SUCCESS;
    }
    else if ( x.memory_location == Magma_CPU ) {
        y->memory_location = x.memory_location;
        y->num_rows = x.num_rows;
        y->nnz = x.nnz;
        magma_cmalloc_cpu( &y->val, x.num_rows );

        magma_int_t one= 1;
        magma_int_t info;
        lapackf77_zlag2c( &x.num_rows, &one,
                       x.dval, &x.num_rows,
                       y->val, &x.num_rows, &info);
        return MAGMA_SUCCESS;
    }
    else
        return MAGMA_ERR_NOT_SUPPORTED;
}



/**
    Purpose
    -------

    convertes magma_z_matrix from Z to C

    Arguments
    ---------

    @param
    A           magma_z_matrix
                input matrix descriptor

    @param
    B           magma_c_sparse_matrix*
                output matrix descriptor
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_sparse_matrix_zlag2c(
    magma_z_matrix A, magma_c_sparse_matrix *B,
    magma_queue_t queue )
{
    magma_int_t info;
    if ( A.memory_location == Magma_DEV) {
        B->storage_type = A.storage_type;
        B->memory_location = A.memory_location;
        B->num_rows = A.num_rows;
        B->num_cols = A.num_cols;
        B->nnz = A.nnz;
        B->max_nnz_row = A.max_nnz_row;
        if ( A.storage_type == Magma_CSR ) {
            magma_cmalloc( &B->val, A.nnz );
            magmablas_zlag2c_sparse( A.nnz, 1, A.dval, A.nnz, B->val,
                    A.nnz, &info, queue );
            B->row = A.drow;
            B->col = A.dcol;
        }
        if ( A.storage_type == Magma_ELLPACK ) {
            magma_cmalloc( &B->val, A.num_rows*A.max_nnz_row );
            magmablas_zlag2c_sparse( A.num_rows*A.max_nnz_row, 1, A.dval,
            A.num_rows*A.max_nnz_row, B->val, A.num_rows*A.max_nnz_row, &info, queue );
            B->col = A.dcol;
        }
        if ( A.storage_type == Magma_ELL ) {
            magma_cmalloc( &B->val, A.num_rows*A.max_nnz_row );
            magmablas_zlag2c_sparse(  A.num_rows*A.max_nnz_row, 1, A.dval,
            A.num_rows*A.max_nnz_row, B->val, A.num_rows*A.max_nnz_row, &info, queue );
            B->col = A.dcol;
        }
        if ( A.storage_type == Magma_DENSE ) {
            magma_cmalloc( &B->val, A.num_rows*A.num_cols );
            magmablas_zlag2c_sparse(  A.num_rows, A.num_cols, A.dval, A.num_rows,
                    B->val, A.num_rows, &info, queue );
        }
        else {
            info = MAGMA_ERR_NOT_SUPPORTED;
        }
    }
    else {
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
    
cleanup:
    return info;
}
