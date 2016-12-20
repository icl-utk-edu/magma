/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt
       @author Goran Flegar
*/
#include "magmasparse_internal.h"


/**
    Purpose
    -------

    Generate a symmetric n x n CSR matrix for a stencil.

    Arguments
    ---------

    @param[in]
    n           magma_int_t
                number of rows

    @param[in]
    offdiags    magma_int_t
                number of offdiagonals

    @param[in]
    diag_offset magma_int_t*
                array containing the offsets

                                                (length offsets+1)
    @param[in]
    diag_vals   magmaDoubleComplex*
                array containing the values

                                                (length offsets+1)
    @param[out]
    A           magma_z_matrix*
                matrix to generate
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zmgenerator(
    magma_int_t n,
    magma_int_t offdiags,
    magma_index_t *diag_offset,
    magmaDoubleComplex *diag_vals,
    magma_z_matrix *A,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_z_matrix B={Magma_CSR};
    
    // make sure the target structure is empty
    magma_zmfree( A, queue );
    
    B.val = NULL;
    B.col = NULL;
    B.row = NULL;
    B.rowidx = NULL;
    B.blockinfo = NULL;
    B.diag = NULL;
    B.dval = NULL;
    B.dcol = NULL;
    B.drow = NULL;
    B.drowidx = NULL;
    B.ddiag = NULL;
    B.list = NULL;
    B.dlist = NULL;
    
    B.num_rows = n;
    B.num_cols = n;
    B.fill_mode = MagmaFull;
    B.memory_location = Magma_CPU;
    B.storage_type = Magma_ELLPACKT;
    B.max_nnz_row = (2*offdiags+1);

    CHECK( magma_zmalloc_cpu( &B.val, B.max_nnz_row*n ));
    CHECK( magma_index_malloc_cpu( &B.col, B.max_nnz_row*n ));
    
    for( int i=0; i<n; i++ ) { // stride over rows
        // stride over the number of nonzeros in each row
        // left of diagonal
        for( int j=0; j<offdiags; j++ ) {
            B.val[ i*B.max_nnz_row + j ] = diag_vals[ offdiags - j ];
            B.col[ i*B.max_nnz_row + j ] = -1 * diag_offset[ offdiags-j ] + i;
        }
        // elements on the diagonal
        B.val[ i*B.max_nnz_row + offdiags ] = diag_vals[ 0 ];
        B.col[ i*B.max_nnz_row + offdiags ] = i;
        // right of diagonal
        for( int j=0; j<offdiags; j++ ) {
            B.val[ i*B.max_nnz_row + j + offdiags +1 ] = diag_vals[ j+1 ];
            B.col[ i*B.max_nnz_row + j + offdiags +1 ] = diag_offset[ j+1 ] + i;
        }
    }

    // set invalid entries to zero
    for( int i=0; i<n; i++ ) { // stride over rows
        for( int j=0; j<B.max_nnz_row; j++ ) { // nonzeros in every row
            if ( (B.col[i*B.max_nnz_row + j] < 0) ||
                    (B.col[i*B.max_nnz_row + j] >= n) ) {
                B.val[ i*B.max_nnz_row + j ] = MAGMA_Z_MAKE( 0.0, 0.0 );
            }
        }
    }

    B.nnz = 0;

    for( int i=0; i<n; i++ ) { // stride over rows
        for( int j=0; j<B.max_nnz_row; j++ ) { // nonzeros in every row
            if ( MAGMA_Z_REAL( B.val[i*B.max_nnz_row + j]) != 0.0 )
                B.nnz++;
        }
    }
    B.true_nnz = B.nnz;
    // converting it to CSR will remove the invalit entries completely
    CHECK( magma_zmconvert( B, A, Magma_ELLPACKT, Magma_CSR, queue ));

cleanup:
    if( info != 0 ){
        magma_zmfree( &B, queue );
    }
    return info;
}


/**
    Purpose
    -------

    Generate a 27-point stencil for a 3D FD discretization.

    Arguments
    ---------

    @param[in]
    n           magma_int_t
                number of rows

    @param[out]
    A           magma_z_matrix*
                matrix to generate
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zm_27stencil(
    magma_int_t n,
    magma_z_matrix *A,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t i,j,k;
    magma_z_matrix hA={Magma_CSR};

    
    // generate matrix of desired structure and size (3d 27-point stencil)
    magma_int_t nn = n*n*n;
    magma_int_t offdiags = 13;
    magma_index_t *diag_offset=NULL;
    magmaDoubleComplex *diag_vals=NULL;
    CHECK( magma_zmalloc_cpu( &diag_vals, offdiags+1 ));
    CHECK( magma_index_malloc_cpu( &diag_offset, offdiags+1 ));
    
    // make sure the target structure is empty
    magma_zmfree( A, queue );

    diag_offset[0] = 0;
    diag_offset[1] = 1;
    diag_offset[2] = n-1;
    diag_offset[3] = n;
    diag_offset[4] = n+1;
    diag_offset[5] = n*n-n-1;
    diag_offset[6] = n*n-n;
    diag_offset[7] = n*n-n+1;
    diag_offset[8] = n*n-1;
    diag_offset[9] = n*n;
    diag_offset[10] = n*n+1;
    diag_offset[11] = n*n+n-1;
    diag_offset[12] = n*n+n;
    diag_offset[13] = n*n+n+1;

    diag_vals[0] = MAGMA_Z_MAKE( 26.0, 0.0 );
    diag_vals[1] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[2] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[3] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[4] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[5] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[6] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[7] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[8] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[9] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[10] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[11] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[12] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[13] = MAGMA_Z_MAKE( -1.0, 0.0 );
    CHECK( magma_zmgenerator( nn, offdiags, diag_offset, diag_vals, &hA, queue ));

    // now set some entries to zero (boundary...)
    for( i=0; i < n*n; i++ ) {
        for( j=0; j < n; j++ ) {
            magma_index_t row = i*n+j;
            for( k=hA.row[row]; k<hA.row[row+1]; k++) {
                if ((hA.col[k] == row-1 ||
                    hA.col[k] == row-n-1 ||
                    hA.col[k] == row+n-1 ||
                    hA.col[k] == row-n*n+n-1 ||
                    hA.col[k] == row+n*n-n-1 ||
                    hA.col[k] == row-n*n-1 ||
                    hA.col[k] == row+n*n-1 ||
                    hA.col[k] == row-n*n-n-1 ||
                    hA.col[k] == row+n*n+n-1 ) && (row+1)%n == 1 )
                        
                        hA.val[k] = MAGMA_Z_MAKE( 0.0, 0.0 );
    
                if ((hA.col[k] == row+1 ||
                    hA.col[k] == row-n+1 ||
                    hA.col[k] == row+n+1 ||
                    hA.col[k] == row-n*n+n+1 ||
                    hA.col[k] == row+n*n-n+1 ||
                    hA.col[k] == row-n*n+1 ||
                    hA.col[k] == row+n*n+1 ||
                    hA.col[k] == row-n*n-n+1 ||
                    hA.col[k] == row+n*n+n+1 ) && (row)%n ==n-1 )
                        
                        hA.val[k] = MAGMA_Z_MAKE( 0.0, 0.0 );
            }
        }
    }
    hA.true_nnz = hA.nnz;
    CHECK( magma_zmconvert( hA, A, Magma_CSR, Magma_CSR, queue ));

cleanup:
    magma_free_cpu( diag_vals );
    magma_free_cpu( diag_offset );
    magma_zmfree( &hA, queue );
    return info;
}



/**
    Purpose
    -------

    Generate a 5-point stencil for a 2D FD discretization.

    Arguments
    ---------

    @param[in]
    n           magma_int_t
                number of rows

    @param[out]
    A           magma_z_matrix*
                matrix to generate
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zm_5stencil(
    magma_int_t n,
    magma_z_matrix *A,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t i,j,k;
    magma_z_matrix hA={Magma_CSR};
    
    // make sure the target structure is empty
    magma_zmfree( A, queue );
    
    // generate matrix of desired structure and size (2d 5-point stencil)
    magma_int_t nn = n*n;
    magma_int_t offdiags = 2;
    magma_index_t *diag_offset=NULL;
    magmaDoubleComplex *diag_vals=NULL;
    CHECK( magma_zmalloc_cpu( &diag_vals, offdiags+1 ));
    CHECK( magma_index_malloc_cpu( &diag_offset, offdiags+1 ));

    diag_offset[0] = 0;
    diag_offset[1] = 1;
    diag_offset[2] = n;
    
    #define COMPLEX
    
    #ifdef COMPLEX
        // complex case
        diag_vals[0] = MAGMA_Z_MAKE( 4.0, 4.0 );
        diag_vals[1] = MAGMA_Z_MAKE( -1.0, -1.0 );
        diag_vals[2] = MAGMA_Z_MAKE( -1.0, -1.0 );
        
    #else
        // real case
        diag_vals[0] = MAGMA_Z_MAKE( 4.0, 0.0 );
        diag_vals[1] = MAGMA_Z_MAKE( -1.0, 0.0 );
        diag_vals[2] = MAGMA_Z_MAKE( -1.0, 0.0 );
    #endif
    CHECK( magma_zmgenerator( nn, offdiags, diag_offset, diag_vals, &hA, queue ));

    // now set some entries to zero (boundary...)
    for( i=0; i<n; i++ ) {
        for( j=0; j<n; j++ ) {
            magma_index_t row = i*n+j;
            for( k=hA.row[row]; k<hA.row[row+1]; k++) {
                if ((hA.col[k] == row-1 ) && (row+1)%n == 1 )
                    hA.val[k] = MAGMA_Z_MAKE( 0.0, 0.0 );
    
                if ((hA.col[k] == row+1 ) && (row)%n ==n-1 )
                    hA.val[k] = MAGMA_Z_MAKE( 0.0, 0.0 );
            }
        }
    }

    CHECK( magma_zmconvert( hA, A, Magma_CSR, Magma_CSR, queue ));
    magma_zmcsrcompressor( A, queue );
    A->true_nnz = A->nnz;
    
cleanup:
    magma_free_cpu( diag_vals );
    magma_free_cpu( diag_offset );
    magma_zmfree( &hA, queue );
    return info;
}


namespace {

inline void
matrix_clear(
        magma_z_matrix *A)
{
    A->storage_type = Magma_CSR;
    A->memory_location = Magma_CPU;
    A->sym = Magma_GENERAL;
    A->diagorder_type = Magma_ORDERED;
    A->fill_mode = MagmaFull;
    A->num_rows = 0;
    A->num_cols = 0;
    A->nnz = 0;
    A->max_nnz_row = 0;
    A->diameter = 0;
    A->true_nnz = 0;
    A->val = nullptr;
    A->dval = nullptr;
    A->diag = nullptr;
    A->ddiag = nullptr;
    A->rowidx = nullptr;
    A->row = nullptr;
    A->drow = nullptr;
    A->drowidx = nullptr;
    A->col = nullptr;
    A->dcol = nullptr;
    A->list = nullptr;
    A->dlist = nullptr;
    A->tile_ptr = nullptr;
    A->dtile_ptr = nullptr;
    A->tile_desc = nullptr;
    A->dtile_desc = nullptr;
    A->tile_desc_offset_ptr = nullptr;
    A->dtile_desc_offset_ptr = nullptr;
    A->tile_desc_offset = nullptr;
    A->dtile_desc_offset = nullptr;
    A->calibrator = nullptr;
    A->dcalibrator = nullptr;
    A->blockinfo = nullptr;
    A->blocksize = 0;
    A->numblocks = 0;
    A->alignment = 0;
    A->csr5_sigma = 0;
    A->csr5_bit_y_offset = 0;
    A->csr5_bit_scansum_offset = 0;
    A->csr5_num_packets = 0;
    A->csr5_p = 0;
    A->csr5_num_offsets = 0;
    A->csr5_tail_tile_start = 0;
    A->major = MagmaColMajor;
    A->ld = 0;
}


void
matrix_create_csr_info(
        magma_z_matrix *A,
        magma_int_t num_rows,
        magma_int_t num_cols,
        magma_int_t num_nzeros)
{
    matrix_clear(A);
    A->storage_type = Magma_CSR;
    A->fill_mode = MagmaFull;
    A->num_rows = num_rows;
    A->num_cols = num_cols;
    A->nnz = num_nzeros;
}


}  // namespace


/**
    Purpose
    -------

    TODO(Goran): Write docs!

 */
extern "C" magma_int_t
magma_zmatrix_create_csr_cpu(
        magma_z_matrix *A,
        magma_int_t num_rows,
        magma_int_t num_cols,
        magma_int_t num_nzeros)
{
    matrix_create_csr_info(A, num_rows, num_cols, num_nzeros);
    A->memory_location = Magma_CPU;
    try {
        magma::massert(magma_zmalloc_cpu(&(A->val), A->nnz));
        magma::massert(magma_index_malloc_cpu(&(A->row), A->num_rows + 1));
        magma::massert(magma_index_malloc_cpu(&(A->col), A->nnz));
    } catch (const magma::runtime_error &err) {
        magma_free_cpu(A->val);
        magma_free_cpu(A->row);
        magma_free_cpu(A->col);
        return err.info();
    }
    for (magma_int_t i = 0; i <= A->num_rows; ++i) {
        A->row[i] = 0;
    }
    return MAGMA_SUCCESS;
}


/**
    Purpose
    -------

    TODO(Goran): Write docs!

 */
extern "C" magma_int_t
magma_zmatrix_create_csr(
        magma_z_matrix *A,
        magma_int_t num_rows,
        magma_int_t num_cols,
        magma_int_t num_nzeros,
        magma_queue_t queue)
{
    matrix_create_csr_info(A, num_rows, num_cols, num_nzeros);
    magma_index_t *tmp = nullptr;
    A->memory_location = Magma_DEV;
    try {
        magma::massert(magma_zmalloc(&(A->val), A->nnz));
        magma::massert(magma_index_malloc(&(A->row), A->num_rows + 1));
        magma::massert(magma_index_malloc(&(A->col), A->nnz));
        magma::massert(magma_index_malloc_cpu(&tmp, A->num_rows + 1));
    } catch (const magma::runtime_error &err) {
        magma_free(A->val);
        magma_free(A->row);
        magma_free(A->col);
        magma_free_cpu(tmp);
        return err.info();
    }
    for (magma_int_t i = 0; i <= A->num_rows; ++i) {
        tmp[i] = 0;
    }
    magma_index_setvector(A->num_rows + 1, tmp, 1, A->row, 1, queue);
    magma_free_cpu(tmp);
    return MAGMA_SUCCESS;
}

