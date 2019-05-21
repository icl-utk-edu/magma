/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt
       @author Mark Gates

*/
#include "magmasparse_internal.h"

/**
    Purpose
    -------
    Transposes a matrix stored in CSR format on the CPU host.


    Arguments
    ---------
    @param[in]
    n_rows      magma_int_t
                number of rows in input matrix

    @param[in]
    n_cols      magma_int_t
                number of columns in input matrix

    @param[in]
    nnz         magma_int_t
                number of nonzeros in input matrix

    @param[in]
    values      magmaDoubleComplex*
                value array of input matrix

    @param[in]
    rowptr      magma_index_t*
                row pointer of input matrix

    @param[in]
    colind      magma_index_t*
                column indices of input matrix

    @param[in]
    new_n_rows  magma_index_t*
                number of rows in transposed matrix

    @param[in]
    new_n_cols  magma_index_t*
                number of columns in transposed matrix

    @param[in]
    new_nnz     magma_index_t*
                number of nonzeros in transposed matrix

    @param[in]
    new_values  magmaDoubleComplex**
                value array of transposed matrix

    @param[in]
    new_rowptr  magma_index_t**
                row pointer of transposed matrix

    @param[in]
    new_colind  magma_index_t**
                column indices of transposed matrix

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
z_transpose_csr(
    magma_int_t n_rows,
    magma_int_t n_cols,
    magma_int_t nnz,
    magmaDoubleComplex *values,
    magma_index_t *rowptr,
    magma_index_t *colind,
    magma_int_t *new_n_rows,
    magma_int_t *new_n_cols,
    magma_int_t *new_nnz,
    magmaDoubleComplex **new_values,
    magma_index_t **new_rowptr,
    magma_index_t **new_colind,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // easier to keep names straight if we convert CSR to CSC,
    // which is the same as tranposing CSR.
    magmaDoubleComplex *csc_values=NULL;
    magma_index_t *csc_colptr=NULL, *csc_rowind=NULL;
    
    // i, j are actual row & col indices (0 <= i < nrows, 0 <= j < ncols).
    // k is index into col and values (0 <= k < nnz).
    magma_int_t i, j, k, total, tmp;
    
    CHECK( magma_zmalloc_cpu( &csc_values, nnz ) );
    CHECK( magma_index_malloc_cpu( &csc_colptr, n_cols + 1 ) );
    CHECK( magma_index_malloc_cpu( &csc_rowind, nnz ) );
    
    // example matrix
    // [ x x 0 x ]
    // [ x 0 x x ]
    // [ x x 0 0 ]
    // rowptr = [ 0 3 6, 8 ]
    // colind = [ 0 1 3; 0 2 3; 0 1 ]
    
    // sum up nnz in each original column
    // colptr = [ 3 2 1 2, X ]
    for( j=0; j < n_cols; ++j ) {
        csc_colptr[ j ] = 0;
    }
    for( k=0; k < nnz; ++k ) {
        csc_colptr[ colind[k] ]++;
    }
    
    // running sum to convert to new colptr
    // colptr = [ 0 3 5 6, 8 ]
    total = 0;
    for( j=0; j < n_cols; ++j ) {
        tmp = csc_colptr[ j ];
        csc_colptr[ j ] = total;
        total += tmp;
    }
    csc_colptr[ n_cols ] = total;
    assert( total == nnz );
    
    // copy row indices and values
    // this increments colptr until it effectively shifts left one
    // colptr = [ 3 5 6 8, 8 ]
    // rowind = [ 0 1 2; 0 2; 1; 0 1 ]
    for( i=0; i < n_rows; ++i ) {
        for( k=rowptr[i]; k < rowptr[i+1]; ++k ) {
            j = colind[k];
            csc_rowind[ csc_colptr[ j ] ] = i;
            csc_values[ csc_colptr[ j ] ] = values[k];
            csc_colptr[ j ]++;
        }
    }
    assert( csc_colptr[ n_cols-1 ] == nnz );
    
    // shift colptr right one
    // colptr = [ 0 3 5 6, 8 ]
    for( j=n_cols-1; j > 0; --j ) {
        csc_colptr[j] = csc_colptr[j-1];
    }
    csc_colptr[0] = 0;

    // save into output variables
    *new_n_rows = n_cols;
    *new_n_cols = n_rows;
    *new_nnz    = nnz;
    *new_values = csc_values;
    *new_rowptr = csc_colptr;
    *new_colind = csc_rowind;
    
cleanup:
    magma_free_cpu( csc_values );
    magma_free_cpu( csc_colptr );
    magma_free_cpu( csc_rowind );
    return info;
}


/**
    Purpose
    -------

    Interface to cuSPARSE transpose.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix (CSR)

    @param[out]
    B           magma_z_matrix*
                output matrix (CSR)
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/
    
    
extern "C" magma_int_t
magma_zmtranspose(
    magma_z_matrix A, magma_z_matrix *B,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // make sure the target structure is empty
    magma_zmfree( B, queue );
    if( A.memory_location == Magma_DEV ){
        CHECK( magma_z_cucsrtranspose( A, B, queue ));
    } else {
        CHECK( magma_zmtranspose_cpu(A, B, queue) );
    }
    
cleanup:
    return info;
}


/**
    Purpose
    -------

    Helper function to transpose CSR matrix.
    Using the CUSPARSE CSR2CSC function.


    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix (CSR)

    @param[out]
    B           magma_z_matrix*
                output matrix (CSR)
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_z_cucsrtranspose(
    magma_z_matrix A,
    magma_z_matrix *B,
    magma_queue_t queue )
{
    // for symmetric matrices: convert to csc using cusparse
    
    magma_int_t info = 0;
    cusparseHandle_t handle=NULL;
    cusparseMatDescr_t descrA=NULL;
    cusparseMatDescr_t descrB=NULL;
    
    
    magma_z_matrix ACSR={Magma_CSR}, BCSR={Magma_CSR};
    magma_z_matrix dA={Magma_CSR}, dB={Magma_CSR};
    
    // make sure the target structure is empty
    magma_zmfree( B, queue );

    if( A.storage_type == Magma_CSR && A.memory_location == Magma_DEV ) {
        // fill in information for B
        B->storage_type    = A.storage_type;
        B->diagorder_type  = A.diagorder_type;
        B->memory_location = Magma_DEV;
        B->num_rows        = A.num_cols;  // transposed
        B->num_cols        = A.num_rows;  // transposed
        B->nnz             = A.nnz;
        B->true_nnz = A.true_nnz;
        B->ownership = MagmaTrue;
        if ( A.fill_mode == MagmaFull ) {
            B->fill_mode = MagmaFull;
        }
        else if ( A.fill_mode == MagmaLower ) {
            B->fill_mode = MagmaUpper;
        }
        else if ( A.fill_mode == MagmaUpper ) {
            B->fill_mode = MagmaLower;
        }
        
        B->dval = NULL;
        B->drow = NULL;
        B->dcol = NULL;
        
        // memory allocation
        CHECK( magma_zmalloc( &B->dval, B->nnz ));
        CHECK( magma_index_malloc( &B->drow, B->num_rows + 1 ));
        CHECK( magma_index_malloc( &B->dcol, B->nnz ));
        
        // CUSPARSE context //
        CHECK_CUSPARSE( cusparseCreate( &handle ));
        CHECK_CUSPARSE( cusparseSetStream( handle, queue->cuda_stream() ));
        CHECK_CUSPARSE( cusparseCreateMatDescr( &descrA ));
        CHECK_CUSPARSE( cusparseCreateMatDescr( &descrB ));
        CHECK_CUSPARSE( cusparseSetMatType( descrA, CUSPARSE_MATRIX_TYPE_GENERAL ));
        CHECK_CUSPARSE( cusparseSetMatType( descrB, CUSPARSE_MATRIX_TYPE_GENERAL ));
        CHECK_CUSPARSE( cusparseSetMatIndexBase( descrA, CUSPARSE_INDEX_BASE_ZERO ));
        CHECK_CUSPARSE( cusparseSetMatIndexBase( descrB, CUSPARSE_INDEX_BASE_ZERO ));
        CHECK_CUSPARSE(
        cusparseZcsr2csc( handle, A.num_rows, A.num_cols, A.nnz,
                          A.dval, A.drow, A.dcol, B->dval, B->dcol, B->drow,
                          CUSPARSE_ACTION_NUMERIC,
                          CUSPARSE_INDEX_BASE_ZERO) );
    } else if ( A.storage_type == Magma_CSR && A.memory_location == Magma_CPU ){
        CHECK( magma_zmtransfer( A, &dA, A.memory_location, Magma_DEV, queue ));
        CHECK( magma_z_cucsrtranspose( dA, &dB, queue ));
        CHECK( magma_zmtransfer( dB, B, Magma_DEV, A.memory_location, queue ));
    } else {
        CHECK( magma_zmconvert( A, &ACSR, A.storage_type, Magma_CSR, queue ));
        CHECK( magma_z_cucsrtranspose( ACSR, &BCSR, queue ));
        CHECK( magma_zmconvert( BCSR, B, Magma_CSR, A.storage_type, queue ));
    }
cleanup:
    cusparseDestroyMatDescr( descrA );
    cusparseDestroyMatDescr( descrB );
    cusparseDestroy( handle );
    magma_zmfree( &dA, queue );
    magma_zmfree( &dB, queue );
    magma_zmfree( &ACSR, queue );
    magma_zmfree( &BCSR, queue );
    if( info != 0 ){
        magma_zmfree( B, queue );
    }
    return info;
}


/**
    Purpose
    -------

    This function forms the transpose conjugate of a matrix. For a real-value
    matrix, the output is the transpose.


    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix (CSR)

    @param[out]
    B           magma_z_matrix*
                output matrix (CSR)
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmtransposeconjugate(
    magma_z_matrix A,
    magma_z_matrix *B,
    magma_queue_t queue )
{
    // for symmetric matrices: convert to csc using cusparse
    
    magma_int_t info = 0;
    cusparseHandle_t handle=NULL;
    cusparseMatDescr_t descrA=NULL;
    cusparseMatDescr_t descrB=NULL;
    
    magma_z_matrix ACSR={Magma_CSR}, BCSR={Magma_CSR};
    magma_z_matrix dA={Magma_CSR}, dB={Magma_CSR};
    
    // make sure the target structure is empty
    magma_zmfree( B, queue );

    if( A.storage_type == Magma_CSR && A.memory_location == Magma_DEV ) {
        // fill in information for B
        B->storage_type    = A.storage_type;
        B->diagorder_type  = A.diagorder_type;
        B->memory_location = Magma_DEV;
        B->num_rows        = A.num_cols;  // transposed
        B->num_cols        = A.num_rows;  // transposed
        B->nnz             = A.nnz;
        B->true_nnz = A.true_nnz;
        B->ownership = MagmaTrue;
        if ( A.fill_mode == MagmaFull ) {
            B->fill_mode = MagmaFull;
        }
        else if ( A.fill_mode == MagmaLower ) {
            B->fill_mode = MagmaUpper;
        }
        else if ( A.fill_mode == MagmaUpper ) {
            B->fill_mode = MagmaLower;
        }
        B->dval = NULL;
        B->drow = NULL;
        B->dcol = NULL;
        
        // memory allocation
        CHECK( magma_zmalloc( &B->dval, B->nnz ));
        CHECK( magma_index_malloc( &B->drow, B->num_rows + 1 ));
        CHECK( magma_index_malloc( &B->dcol, B->nnz ));
        // CUSPARSE context //
        CHECK_CUSPARSE( cusparseCreate( &handle ));
        CHECK_CUSPARSE( cusparseSetStream( handle, queue->cuda_stream() ));
        CHECK_CUSPARSE( cusparseCreateMatDescr( &descrA ));
        CHECK_CUSPARSE( cusparseCreateMatDescr( &descrB ));
        CHECK_CUSPARSE( cusparseSetMatType( descrA, CUSPARSE_MATRIX_TYPE_GENERAL ));
        CHECK_CUSPARSE( cusparseSetMatType( descrB, CUSPARSE_MATRIX_TYPE_GENERAL ));
        CHECK_CUSPARSE( cusparseSetMatIndexBase( descrA, CUSPARSE_INDEX_BASE_ZERO ));
        CHECK_CUSPARSE( cusparseSetMatIndexBase( descrB, CUSPARSE_INDEX_BASE_ZERO ));
        CHECK_CUSPARSE(
        cusparseZcsr2csc( handle, A.num_rows, A.num_cols, A.nnz,
                          A.dval, A.drow, A.dcol, B->dval, B->dcol, B->drow,
                          CUSPARSE_ACTION_NUMERIC,
                          CUSPARSE_INDEX_BASE_ZERO) );
        CHECK( magma_zmconjugate( B, queue ));
    } else if ( A.memory_location == Magma_CPU ){
        CHECK( magma_zmtransfer( A, &dA, A.memory_location, Magma_DEV, queue ));
        CHECK( magma_zmtransposeconjugate( dA, &dB, queue ));
        CHECK( magma_zmtransfer( dB, B, Magma_DEV, A.memory_location, queue ));
    } else {
        CHECK( magma_zmconvert( A, &ACSR, A.storage_type, Magma_CSR, queue ));
        CHECK( magma_zmtransposeconjugate( ACSR, &BCSR, queue ));
        CHECK( magma_zmconvert( BCSR, B, Magma_CSR, A.storage_type, queue ));
    }
cleanup:
    cusparseDestroyMatDescr( descrA );
    cusparseDestroyMatDescr( descrB );
    cusparseDestroy( handle );
    magma_zmfree( &dA, queue );
    magma_zmfree( &dB, queue );
    magma_zmfree( &ACSR, queue );
    magma_zmfree( &BCSR, queue );
    if( info != 0 ){
        magma_zmfree( B, queue );
    }
    return info;
}
