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

    Checks the maximal number of nonzeros in a row of matrix A.
    Inserts the data into max_nnz_row.


    Arguments
    ---------

    @param[in,out]
    A           magma_z_matrix*
                sparse matrix
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zrowentries(
    magma_z_matrix *A,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_index_t *length=NULL;
    magma_index_t i,j, maxrowlength=0;
    
    // check whether matrix on CPU
    if ( A->memory_location == Magma_CPU ) {
        // CSR
        if ( A->storage_type == Magma_CSR ) {
            CHECK( magma_index_malloc_cpu( &length, A->num_rows));
            for( i=0; i<A->num_rows; i++ ) {
                length[i] = A->row[i+1]-A->row[i];
                if (length[i] > maxrowlength)
                     maxrowlength = length[i];
            }
            A->max_nnz_row = maxrowlength;
        }
        // Dense
        else if ( A->storage_type == Magma_DENSE ) {
            CHECK( magma_index_malloc_cpu( &length, A->num_rows));

            for( i=0; i<A->num_rows; i++ ) {
                length[i] = 0;
                for( j=0; j<A->num_cols; j++ ) {
                    if ( MAGMA_Z_REAL( A->val[i*A->num_cols + j] ) != 0. )
                        length[i]++;
                    }
                if (length[i] > maxrowlength)
                     maxrowlength = length[i];
            }
            A->max_nnz_row = maxrowlength;
        }
    } // end CPU case

    else {
        printf("error: matrix not on CPU.\n");
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
cleanup:
    magma_free( length );
    return info;
}


/**
    Purpose
    -------

    Computes the diameter of a sparse matrix and stores the value in diameter.


    Arguments
    ---------

    @param[in,out]
    A           magma_z_matrix*
                sparse matrix
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/
extern "C" magma_int_t
magma_zdiameter(
    magma_z_matrix *A,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_index_t i, j, tmp,  *dim=NULL, maxdim=0;
    
    // check whether matrix on CPU
    if ( A->memory_location == Magma_CPU ) {
        // CSR
        if ( A->storage_type == Magma_CSR ) {
            CHECK( magma_index_malloc_cpu( &dim, A->num_rows));
            for( i=0; i<A->num_rows; i++ ) {
                dim[i] = 0;
                for( j=A->row[i]; j<A->row[i+1]; j++ ) {
                   // if ( MAGMA_Z_REAL(A->val[j]) > THRESHOLD ) {
                        tmp = abs( i - A->col[j] );
                        if ( tmp > dim[i] )
                            dim[i] = tmp;
                   // }
                }
                if ( dim[i] > maxdim )
                     maxdim = dim[i];
            }
            A->diameter = maxdim;
        }
        // Dense
        else if ( A->storage_type == Magma_DENSE ) {
            CHECK( magma_index_malloc_cpu( &dim, A->num_rows));
            for( i=0; i<A->num_rows; i++ ) {
                dim[i] = 0;
                for( j=0; j<A->num_cols; j++ ) {
                    if ( MAGMA_Z_REAL( A->val[i*A->num_cols + j] ) !=  0.0 ) {
                        tmp = abs( i -j );
                        if ( tmp > dim[i] )
                            dim[i] = tmp;
                    }
                }
                if ( dim[i] > maxdim )
                     maxdim = dim[i];
            }
            A->diameter = maxdim;
        }
        // ELLPACK
        else if ( A->storage_type == Magma_ELL ) {
            CHECK( magma_index_malloc_cpu( &dim, A->num_rows));
            for( i=0; i<A->num_rows; i++ ) {
                dim[i] = 0;
                for( j=i*A->max_nnz_row; j<(i+1)*A->max_nnz_row; j++ ) {
                    if ( MAGMA_Z_REAL( A->val[j] ) > THRESHOLD ) {
                        tmp = abs( i - A->col[j] );
                        if ( tmp > dim[i] )
                            dim[i] = tmp;
                    }
                }
                if ( dim[i] > maxdim )
                     maxdim = dim[i];
            }
            A->diameter = maxdim;
        }
        // ELL
        else if ( A->storage_type == Magma_ELL ) {
            printf("error:format not supported.\n");
            info = MAGMA_ERR_NOT_SUPPORTED;
        }
    } // end CPU case

    else {
        printf("error: matrix not on CPU.\n");
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
cleanup:
    magma_free( &dim );
    return info;
}
